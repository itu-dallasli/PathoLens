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
        self._entity_extractor = None
        self._entity_validator = None
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
        from patholens.entity_extraction.entity_extractor import KARGEntityExtractor
        from patholens.entity_extraction.validator import EntityValidator
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

        # Entity extraction
        ee = cfg.entity_extraction
        self._entity_extractor = KARGEntityExtractor(
            llm_backend=ee.llm_backend,
            llm_model=ee.llm_model,
            temperature=ee.temperature,
            max_tokens=ee.max_tokens,
        )
        self._entity_validator = EntityValidator()

        # Report
        rg = cfg.report_generation
        self._report_builder = FHIRReportBuilder(
            confidence_threshold=rg.confidence_threshold,
            language=rg.language,
        )
        self._evidence_linker = EvidenceLinker()

        self._modules_loaded = True
        log.info("All pipeline modules loaded.")

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

                # Optional patch cap (useful for CPU test runs)
                max_patches = getattr(self.config.preprocessing, "max_patches", None)
                if max_patches and extraction.num_patches > max_patches:
                    log.info(
                        "Capping patches %d → %d (max_patches setting)",
                        extraction.num_patches,
                        max_patches,
                    )
                    extraction.coordinates = extraction.coordinates[:max_patches]
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
            # NOTE: SlideEncoder must be loaded separately with trained weights
            # For now, we pass embeddings directly to downstream modules
            import torch
            emb_tensor = torch.from_numpy(embeddings).unsqueeze(0)  # (1, N, 1024)

            # ─── Step 4: Retrieval ───────────────────────────
            log.info("--- Step 4/7: Case retrieval ---")
            # Retriever will be connected once FAISS index is built
            retrieval_output = None
            if self._retriever is not None:
                slide_repr = embeddings.mean(axis=0)  # Simple average as fallback
                retrieval_output = self._retriever.search(
                    query_slide_repr=slide_repr,
                    query_slide_id=slide_id,
                )
                result.similar_cases = [
                    {
                        "slide_id": r.slide_id,
                        "similarity_score": r.similarity_score,
                        "report_text": r.report_text,
                    }
                    for r in retrieval_output.results
                ]

            # ─── Step 5: Entity extraction ───────────────────
            log.info("--- Step 5/7: Entity extraction ---")
            reference_reports = [
                {"slide_id": c["slide_id"], "report_text": c.get("report_text", "")}
                for c in result.similar_cases
                if c.get("report_text")
            ]
            diagnosis = self._entity_extractor.extract(reference_reports)
            validation = self._entity_validator.validate(diagnosis)

            # ─── Step 6: Explainability ──────────────────────
            log.info("--- Step 6/7: Explainability ---")
            # Compute attention weights
            attn_weights_np = np.ones(len(embeddings)) / len(embeddings)  # Uniform fallback
            if self._attention_model is not None:
                try:
                    import torch
                    with torch.no_grad():
                        patch_feat = torch.from_numpy(embeddings).unsqueeze(0).float()
                        # Project to model dim if needed
                        if patch_feat.shape[-1] != self.config.sequence_model.d_model:
                            proj = torch.nn.Linear(patch_feat.shape[-1], self.config.sequence_model.d_model)
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
