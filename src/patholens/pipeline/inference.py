"""
PathoLens Inference Pipeline — End-to-end WSI analysis.

Orchestrates all modules in sequence:
  WSI → Preprocess → Embed → Encode → Retrieve → Extract → Explain → Report
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from patholens.config import Config
from patholens.logger import get_logger

log = get_logger(__name__)


@dataclass
class PipelineResult:
    """Full output of a single WSI analysis."""

    slide_id: str
    # Report
    fhir_report: Dict[str, Any] = field(default_factory=dict)
    fhir_json: str = ""
    # Retrieval
    similar_cases: List[Dict] = field(default_factory=list)
    # Heatmap
    heatmap_path: Optional[str] = None
    # Evidence
    evidence_coverage: float = 0.0
    # Metadata
    num_patches: int = 0
    elapsed_seconds: float = 0.0
    status: str = "pending"  # pending | running | completed | failed | low_confidence


class PathoLensPipeline:
    """
    End-to-end inference pipeline.

    Lazy-loads all sub-modules on first run to minimise startup
    when only a subset is needed.

    Parameters
    ----------
    config : Config or None
        If None, loads default config.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self._modules_loaded = False

        # Sub-modules (lazy)
        self._preprocessor = None
        self._segmentor = None
        self._extractor_model = None
        self._batch_processor = None
        self._slide_encoder = None
        self._retriever = None
        self._attention_model = None
        self._heatmap_gen = None
        self._region_mapper = None
        self._evidence_linker = None
        self._report_builder = None

    # ── Lazy loading ─────────────────────────────────────────
    def _load_modules(self):
        """Lazy-import and initialise all submodules."""
        if self._modules_loaded:
            return

        log.info("Loading pipeline modules ...")

        from patholens.preprocessing.tissue_segmentation import TissueSegmentor
        from patholens.preprocessing.patch_extraction import PatchExtractor
        from patholens.preprocessing.patch_store import PatchStore
        from patholens.embedding.uni_extractor import UNIFeatureExtractor
        from patholens.embedding.batch_processor import BatchEmbeddingProcessor
        from patholens.embedding.embedding_store import EmbeddingStore
        from patholens.explainability.attention_mil import GatedAttentionMIL
        from patholens.explainability.heatmap_generator import HeatmapGenerator
        from patholens.explainability.entity_region_mapper import EntityRegionMapper
        from patholens.report_generation.fhir_builder import FHIRReportBuilder
        from patholens.report_generation.evidence_linker import EvidenceLinker

        cfg = self.config

        # Preprocessing
        pp = cfg.preprocessing
        self._segmentor = TissueSegmentor(
            median_blur_ksize=pp.segmentation.median_blur_ksize,
            morph_kernel_size=pp.segmentation.morph_kernel_size,
            min_contour_area=pp.segmentation.min_contour_area,
        )
        self._preprocessor = PatchExtractor(
            patch_size=pp.patch_size,
            magnification=pp.magnification,
            tissue_threshold=pp.tissue_threshold,
            overlap=pp.overlap,
        )
        self._patch_store = PatchStore(cfg.paths.patches_dir)

        # Embedding
        emb = cfg.embedding
        self._extractor_model = UNIFeatureExtractor(
            model_name=emb.model_name,
            device=cfg.system.device,
            normalize=emb.normalize,
        )
        self._embedding_store = EmbeddingStore(cfg.paths.embeddings_dir)
        self._batch_processor = BatchEmbeddingProcessor(
            extractor=self._extractor_model,
            store=self._embedding_store,
            batch_size=emb.batch_size,
        )

        # Slide encoder (hierarchical Mamba/Attention MIL) — optional, loaded
        # from checkpoint if one is present on disk.  Falls back to None so
        # the pipeline degrades gracefully when untrained.
        self._slide_encoder = self._maybe_load_slide_encoder()

        # FAISS retrieval index — optional, loaded from data/faiss_index/
        # if it exists (built by scripts/build_faiss_index.py after training).
        self._retriever = self._maybe_load_retriever()

        # Explainability
        exp = cfg.explainability
        self._attention_model = GatedAttentionMIL(
            d_model=cfg.sequence_model.d_model,
            d_hidden=128,
        )
        self._heatmap_gen = HeatmapGenerator(
            resolution=exp.heatmap_resolution,
            colormap=exp.colormap,
            overlay_alpha=exp.overlay_alpha,
            gaussian_sigma=exp.gaussian_sigma,
        )
        self._region_mapper = EntityRegionMapper(
            patch_size=pp.patch_size,
        )

        # Report
        rg = cfg.report_generation
        self._report_builder = FHIRReportBuilder(
            confidence_threshold=rg.confidence_threshold,
            language=rg.language,
        )
        self._evidence_linker = EvidenceLinker()

        self._modules_loaded = True
        log.info("All pipeline modules loaded.")

    # ── Slide encoder loading ────────────────────────────────
    def _maybe_load_slide_encoder(self):
        """
        Try to load a trained :class:`SlideEncoder` checkpoint.

        Looks for an explicit path on ``config.sequence_model.checkpoint_path``;
        otherwise falls back to ``checkpoints/slide_encoder_final.pt``.
        Returns ``None`` (and logs) if no checkpoint is available — the rest
        of the pipeline degrades gracefully in that case.
        """
        import torch

        from patholens.sequence_model.slide_encoder import SlideEncoder

        ckpt_path = getattr(
            self.config.sequence_model, "checkpoint_path", None
        )
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
        else:
            ckpt_path = Path("checkpoints") / "slide_encoder_final.pt"

        if not ckpt_path.exists():
            log.info(
                "No SlideEncoder checkpoint at %s -- Step 3 will pass raw "
                "UNI features through (untrained mode).",
                ckpt_path,
            )
            return None

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mcfg = ckpt.get("config", {})
            model = SlideEncoder(
                input_dim=mcfg.get("input_dim", 1024),
                d_model=mcfg.get("d_model", self.config.sequence_model.d_model),
                n_layers=mcfg.get("n_layers", 4),
                region_size=mcfg.get("region_size", 64),
                n_classes=mcfg.get("n_classes", 2),
                dropout=mcfg.get("dropout", 0.1),
                backbone=mcfg.get("backbone", "mamba"),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            log.info(
                "Loaded SlideEncoder from %s  (best_val_loss=%.4f, "
                "best_val_acc=%.3f)",
                ckpt_path,
                ckpt.get("best_val_loss", float("nan")),
                ckpt.get("best_val_acc", float("nan")),
            )
            return model
        except Exception as e:
            log.warning(
                "Failed to load SlideEncoder from %s: %s -- continuing untrained",
                ckpt_path,
                e,
            )
            return None

    def _maybe_load_retriever(self):
        """
        Try to load the FAISS slide retrieval index built by
        ``scripts/build_faiss_index.py``.

        Returns a callable ``search(slide_repr, slide_id, top_k)`` that
        returns a list of dicts ``{slide_id, similarity_score, label}``,
        or ``None`` if the index does not exist yet.
        """
        import json as _json

        try:
            import faiss as _faiss
        except ImportError:
            log.info("faiss not available; retrieval step will be skipped.")
            return None

        index_dir = Path(
            getattr(self.config, "faiss_index_dir", "data/faiss_index")
        )
        index_path = index_dir / "slide_index.faiss"
        meta_path = index_dir / "metadata.json"

        if not index_path.exists():
            log.info(
                "No FAISS index at %s -- retrieval step will be skipped. "
                "Run scripts/build_faiss_index.py after training.",
                index_path,
            )
            return None

        try:
            index = _faiss.read_index(str(index_path))
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            id_map = {int(k): v for k, v in meta["id_map"].items()}
            slides_meta = meta["slides"]
            top_k_default = meta.get("top_k", 5)

            def _search(slide_repr: np.ndarray, slide_id: str, top_k: int = top_k_default):
                vec = slide_repr.reshape(1, -1).astype(np.float32)
                _faiss.normalize_L2(vec)
                scores, indices = index.search(vec, top_k + 1)  # +1 to allow self-skip
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0:
                        continue
                    sid = id_map.get(int(idx), f"unknown_{idx}")
                    if sid == slide_id:
                        continue
                    smeta = slides_meta.get(sid, {})
                    results.append({
                        "slide_id": sid,
                        "similarity_score": float(score),
                        "label": smeta.get("label", -1),
                        "label_name": smeta.get("label_name", "unknown"),
                    })
                    if len(results) >= top_k:
                        break
                return results

            log.info(
                "Loaded FAISS index from %s  (%d slides)",
                index_path,
                meta["n_slides"],
            )
            return _search

        except Exception as e:
            log.warning("Failed to load FAISS index: %s -- retrieval skipped", e)
            return None

    # ── Main entry point ────────────────────────────────────
    def run(
        self,
        wsi_path: str | Path,
        slide_id: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
    ) -> PipelineResult:
        """
        Run the full analysis pipeline on a single WSI.

        Parameters
        ----------
        wsi_path : str | Path
        slide_id : str, optional
            Defaults to the file stem.
        output_dir : str | Path, optional
            Where to save results.

        Returns
        -------
        PipelineResult
        """
        wsi_path = Path(wsi_path)
        slide_id = slide_id or wsi_path.stem
        output_dir = Path(output_dir or self.config.paths.results_dir) / slide_id
        output_dir.mkdir(parents=True, exist_ok=True)

        result = PipelineResult(slide_id=slide_id, status="running")
        t0 = time.time()

        try:
            self._load_modules()

            # ─── Step 1: Preprocessing ───────────────────────
            log.info("--- Step 1/7: Preprocessing %s ---", slide_id)
            from patholens.preprocessing.wsi_reader import WSIReader

            with WSIReader(wsi_path) as reader:
                thumbnail = reader.get_thumbnail(
                    (self.config.preprocessing.thumbnail_size,) * 2
                )
                seg_result = self._segmentor.segment(thumbnail)
                extraction = self._preprocessor.extract(reader, seg_result.mask)
                result.num_patches = extraction.num_patches
                wsi_dims = reader.dimensions

                # Optional patch cap (useful for CPU test runs).
                # Random sample (seeded) instead of first-N to avoid top-left bias.
                max_patches = getattr(self.config.preprocessing, "max_patches", None)
                if max_patches and extraction.num_patches > max_patches:
                    log.info(
                        "Capping patches %d -> %d (random sample, seed=0)",
                        extraction.num_patches,
                        max_patches,
                    )
                    rng = np.random.default_rng(0)
                    idx = rng.choice(
                        extraction.num_patches, size=max_patches, replace=False
                    )
                    idx.sort()
                    extraction.coordinates = extraction.coordinates[idx]
                    result.num_patches = max_patches

                # Save patch coords
                self._patch_store.save(
                    slide_id,
                    extraction.coordinates,
                    {
                        "wsi_path": str(wsi_path),
                        "patch_size": extraction.patch_size,
                        "level": extraction.level,
                        "magnification": extraction.magnification,
                    },
                )

                # ─── Step 2: Embedding ───────────────────────
                log.info("--- Step 2/7: Embedding %d patches ---", result.num_patches)
                embeddings = self._batch_processor.process_slide(
                    slide_id=slide_id,
                    wsi_path=wsi_path,
                    coordinates=extraction.coordinates,
                    patch_size=extraction.patch_size,
                    level=extraction.level,
                )

            # ─── Step 3: Sequence encoding ───────────────────
            log.info("--- Step 3/7: Slide encoding ---")
            import torch

            emb_tensor = torch.from_numpy(embeddings).unsqueeze(0).float()  # (1, N, 1024)
            slide_output = None
            slide_attn_per_patch: Optional[np.ndarray] = None
            slide_logits_np: Optional[np.ndarray] = None

            if self._slide_encoder is not None:
                with torch.no_grad():
                    slide_output = self._slide_encoder(emb_tensor)
                # Per-patch importance = region_attn[r] * intra_attn[r, p]
                # then flattened back to length N (trim trailing padding).
                if (
                    slide_output.region_attention is not None
                    and slide_output.intra_region_attention is not None
                ):
                    region_attn = slide_output.region_attention[0].cpu().numpy()  # (R,)
                    intra_attn = slide_output.intra_region_attention[0].cpu().numpy()  # (R, rs)
                    R, rs = intra_attn.shape
                    combined = intra_attn * region_attn[:, None]               # (R, rs)
                    flat = combined.reshape(-1)                                 # (R*rs,)
                    n_real = len(embeddings)
                    slide_attn_per_patch = flat[:n_real].astype(np.float32)
                    # Normalise to a probability distribution over patches
                    s = slide_attn_per_patch.sum()
                    if s > 0:
                        slide_attn_per_patch = slide_attn_per_patch / s
                if slide_output.classification_logits is not None:
                    slide_logits_np = (
                        slide_output.classification_logits[0].cpu().numpy()
                    )
                    log.info(
                        "SlideEncoder logits: %s  (argmax=%d)",
                        np.array2string(slide_logits_np, precision=3),
                        int(np.argmax(slide_logits_np)),
                    )

            # ─── Step 4: Visual similarity retrieval ────────
            log.info("--- Step 4/7: Case retrieval ---")
            if self._retriever is not None and slide_output is not None:
                # Use the trained slide_repr from the encoder — much better
                # than mean-pooling raw UNI features.
                slide_repr_np = slide_output.slide_repr[0].cpu().numpy()
                result.similar_cases = self._retriever(slide_repr_np, slide_id)
                if result.similar_cases:
                    log.info(
                        "Retrieved %d similar cases: %s",
                        len(result.similar_cases),
                        [(c["slide_id"], f"{c['similarity_score']:.3f}")
                         for c in result.similar_cases],
                    )
            else:
                log.info("Retrieval skipped (no index or encoder not loaded).")

            # ─── Step 5: Classifier-based diagnosis ──────────
            # Diagnosis is derived directly from the SlideEncoder classifier
            # output rather than LLM extraction over retrieved reports.
            # This keeps the pipeline deterministic and removes the LLM
            # dependency. Retrieved cases provide supporting evidence only.
            log.info("--- Step 5/7: Diagnosis from classifier ---")
            from patholens.entity_extraction.entity_schema import (
                ClinicalEntity, StructuredDiagnosis
            )
            import torch as _torch

            if slide_logits_np is not None:
                # SlideEncoder trained with n_classes=2 (normal=0, tumor=1)
                probs = _torch.softmax(
                    _torch.from_numpy(slide_logits_np), dim=0
                ).numpy()
                tumor_prob = float(probs[1])
                is_tumor = tumor_prob >= 0.5

                # Corroborate with retrieved cases (majority vote)
                if result.similar_cases:
                    retrieved_labels = [c["label"] for c in result.similar_cases]
                    retrieved_tumor_frac = sum(retrieved_labels) / len(retrieved_labels)
                    # Blend: 70% classifier, 30% retrieval majority
                    tumor_prob = 0.7 * tumor_prob + 0.3 * retrieved_tumor_frac
                    is_tumor = tumor_prob >= 0.5

                diagnosis = StructuredDiagnosis(
                    tumor_type=ClinicalEntity(
                        entity_type="tumor_type",
                        value="Carcinoma" if is_tumor else "Benign",
                        confidence=tumor_prob if is_tumor else (1.0 - tumor_prob),
                        source_report_id=slide_id,
                        source_text_span=(
                            f"SlideEncoder tumor_prob={tumor_prob:.3f}"
                            + (
                                f"  retrieved_support={retrieved_tumor_frac:.2f}"
                                if result.similar_cases else ""
                            )
                        ),
                    )
                )
            else:
                # No trained encoder — empty diagnosis
                diagnosis = StructuredDiagnosis()

            # ─── Step 6: Explainability ──────────────────────
            log.info("--- Step 6/7: Explainability ---")
            # Attention source priority:
            #   1. SlideEncoder (trained Mamba MIL) — best quality
            #   2. GatedAttentionMIL — legacy fallback (random weights until trained)
            #   3. Uniform — final fallback
            attn_weights_np: np.ndarray
            if slide_attn_per_patch is not None:
                attn_weights_np = slide_attn_per_patch
                log.info("Using SlideEncoder attention weights (%d patches)", len(attn_weights_np))
            else:
                attn_weights_np = np.ones(len(embeddings), dtype=np.float32) / len(embeddings)
                if self._attention_model is not None:
                    try:
                        with torch.no_grad():
                            patch_feat = torch.from_numpy(embeddings).unsqueeze(0).float()
                            if patch_feat.shape[-1] != self.config.sequence_model.d_model:
                                proj = torch.nn.Linear(
                                    patch_feat.shape[-1],
                                    self.config.sequence_model.d_model,
                                )
                                patch_feat = proj(patch_feat)
                            attn, _ = self._attention_model(patch_feat)
                            attn_weights_np = attn.squeeze().numpy()
                            if attn_weights_np.ndim > 1:
                                attn_weights_np = attn_weights_np.mean(axis=-1)
                    except Exception as e:
                        log.warning("Attention model failed, using uniform: %s", e)

            heatmap_img, raw_attn = self._heatmap_gen.generate(
                wsi_dimensions=wsi_dims,
                patch_coords=extraction.coordinates,
                attention_weights=attn_weights_np,
                patch_size=extraction.patch_size,
                thumbnail=thumbnail,
            )
            heatmap_path = str(output_dir / "heatmap.png")
            self._heatmap_gen.save(heatmap_img, heatmap_path)
            result.heatmap_path = heatmap_path

            # Persist raw (pre-colormap) attention map so downstream tools
            # (eval harness, threshold sweeps) can work with exact values
            # instead of reverse-engineering the lossy overlay PNG.
            np.save(output_dir / "heatmap_raw.npy", raw_attn.astype(np.float32))

            # Entity-region mapping
            entity_evidence = self._region_mapper.map(
                entities=diagnosis.all_entities,
                patch_coords=extraction.coordinates,
                attention_weights=attn_weights_np,
            )

            # ─── Step 7: Report generation ───────────────────
            log.info("--- Step 7/7: Report generation ---")
            evidence_report = self._evidence_linker.link(
                diagnosis=diagnosis,
                entity_evidence=entity_evidence,
                retrieval_results=result.similar_cases,
            )
            result.evidence_coverage = evidence_report.coverage

            fhir_report = self._report_builder.build(
                diagnosis=diagnosis,
                evidence=entity_evidence,
                slide_id=slide_id,
                heatmap_path=heatmap_path,
                retrieved_cases=result.similar_cases,
            )
            result.fhir_report = fhir_report
            result.fhir_json = self._report_builder.to_json(fhir_report)

            # Save report to disk
            report_path = output_dir / "diagnostic_report.json"
            report_path.write_text(result.fhir_json, encoding="utf-8")

            # Check confidence threshold
            if diagnosis.overall_confidence < self.config.report_generation.confidence_threshold:
                result.status = "low_confidence"
                log.warning(
                    "Low confidence (%.2f) — report marked as preliminary.",
                    diagnosis.overall_confidence,
                )
            else:
                result.status = "completed"

        except Exception as e:
            result.status = "failed"
            log.error("Pipeline failed for %s: %s", slide_id, e, exc_info=True)
            raise

        finally:
            result.elapsed_seconds = time.time() - t0
            log.info(
                "Pipeline %s  |  status=%s  time=%.1fs  patches=%d  "
                "evidence_coverage=%.0f%%",
                slide_id,
                result.status,
                result.elapsed_seconds,
                result.num_patches,
                result.evidence_coverage * 100,
            )

        return result
