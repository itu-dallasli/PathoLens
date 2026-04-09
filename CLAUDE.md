# CLAUDE.md — PathoLens Project Context

> Context file for Claude Code sessions. Keep this short and factual.
> Long-form docs live in `documentation/`. Skills/roadmap: see `SKILLS.md`, `ROADMAP.md`.

## What this project is

**PathoLens** — Traceable retrieval-based clinical decision support for breast cancer histopathology (ITU senior project).

Pipeline (7 steps):
1. **Preprocessing** — WSI -> tissue segmentation -> patch extraction
2. **Embedding** — UNI (frozen ViT-L/16) -> 1024-dim patch embeddings
3. **SlideEncoder** — hierarchical Mamba or Attention MIL -> slide_repr + per-patch attention
4. **CMEA Retrieval** — FAISS cosine search over trained slide_repr vectors -> top-k similar training cases
5. **Diagnosis** — classifier output from SlideEncoder logits (tumor probability); retrieval majority vote as soft corroboration
6. **RAAF Explainability** — per-patch attention weights -> Gaussian-smoothed heatmap overlay
7. **CSAL Report** — FHIR R4 DiagnosticReport: tumor probability + heatmap + retrieved cases as derivedFrom

Internal acronyms: **CMEA** (retrieval), **RAAF** (explainability), **CSAL** (report).
**KARG** (LLM entity extraction) was removed — replaced by deterministic classifier-based diagnosis.

## Current status (2026-04-09)

- Full pipeline wired end-to-end, 105 tests pass on CPU.
- **CPUMamba** implemented (real selective SSM, not a placeholder). `backbone="attention"` also available as drop-in.
- Embedding cache builder running on 121 CAMELYON16 slides (~77/121 done as of 2026-04-09).
- **Not yet built**: FAISS index, trained SlideEncoder checkpoint.
- Pipeline degrades gracefully: no checkpoint -> uniform attention; no index -> retrieval skipped.

## Repo map

```
src/patholens/
  preprocessing/      WSIReader (OpenSlide), TissueSegmentor, PatchExtractor
  embedding/          UNIFeatureExtractor (MahmoodLab/UNI ViT-L/16, frozen)
  sequence_model/     SlideEncoder (Mamba or Attention backbone), MambaEncoder, RegionAggregator, SlideTrainer
  retrieval/          FAISS hierarchical retriever (HierarchicalRetriever + index_builder)
  explainability/     GatedAttentionMIL, HeatmapGenerator, EntityRegionMapper
  report_generation/  FHIRReportBuilder, EvidenceLinker
  pipeline/inference.py  PathoLensPipeline.run() -- main entry point
  api/main.py         FastAPI, lazy model loading
configs/
  default.yaml        Full GPU config
  test.yaml           Mocked synthetic config
  local-run.yaml      CPU + real UNI + max_patches=500
scripts/
  build_embedding_cache.py   Run UNI on all slides, save NPZ
  train_slide_encoder.py     Train SlideEncoder on cached NPZs
  build_faiss_index.py       Build FAISS index from trained encoder
  eval_groundtruth.py        IoU eval vs CAMELYON16 masks
  download_camelyon16.py     AWS S3 downloader
run_real.py                  Convenience CLI for single-WSI runs
```

## Critical environment notes (Windows)

- Shell is **bash** (use `/dev/null`, forward slashes).
- Console codec is **CP1254** — do NOT emit Unicode chars in log format strings (use `->` not arrow). Already fixed across all modules.
- `huggingface_hub.login()` **hangs** in non-interactive shells. `uni_extractor.py` reads token from `~/.cache/huggingface/token` or `$HF_TOKEN` env var.
- `openslide-bin` pip package provides the DLL on Windows (no choco install needed).
- torch/torchvision version pin: `torch==2.10.0` requires `torchvision==0.25.0` (CPU wheel).

## Install

```bash
pip install -e ".[dev]"          # CPU dev (faiss-cpu, no mamba-ssm)
pip install -e ".[dev,gpu]"      # GPU production (+ mamba-ssm, faiss-gpu, causal-conv1d)
```

Tests: `pytest tests/` · Lint: `ruff check src tests` · Run: `python run_real.py X:/Bitirme_Data/tumor/tumor_071.tif`

## Data (user's local)

`X:\Bitirme_Data\`:
- CAMELYON16 training: `tumor/tumor_001-111.tif`, `normal/normal_001-010.tif`
- Masks: `masks/tumor_*_mask.tif` (value 2 = tumor, 1 = normal tissue, 0 = background)
- Annotations: `annotations/tumor_*.xml` (ASAP polygon format, parsed with stdlib `xml.etree.ElementTree`)
- CAMELYON17: `patient_190.zip`, `patient_191.zip` — different task, ignore

## Training order

```bash
# 1. Cache embeddings (once, ~4 h CPU, 121 slides x 500 patches)
PYTHONPATH=src python scripts/build_embedding_cache.py \
    --tumor-dir X:/Bitirme_Data/tumor --normal-dir X:/Bitirme_Data/normal \
    --max-patches 500

# 2. Train SlideEncoder (~1.5 h CPU, 20 epochs)
PYTHONPATH=src python scripts/train_slide_encoder.py --epochs 20

# 3. Build FAISS index (~5 min)
PYTHONPATH=src python scripts/build_faiss_index.py

# 4. Evaluate
PYTHONPATH=src python scripts/eval_groundtruth.py \
    X:/Bitirme_Data/tumor/tumor_071.tif \
    --mask X:/Bitirme_Data/masks/tumor_071_mask.tif
```

## Design invariants — do not violate

- **Lazy loading**: modules load in `PathoLensPipeline._load_modules()`, not in `__init__`. API must start instantly.
- **Graceful degradation**: no checkpoint -> uniform attention; no FAISS index -> retrieval skipped; never crash.
- **UNI is frozen**: `param.requires_grad = False`. Never fine-tune.
- **No LLM dependency**: diagnosis comes from SlideEncoder classifier output only.
- **Evidence traceability**: FHIR report links to patch coordinates (heatmap) + retrieved cases (derivedFrom).

## What's known broken / approximate

- `GatedAttentionMIL` in `_attention_model` is legacy fallback — never used when SlideEncoder is loaded.
- No trained weights yet — heatmaps are noise until `train_slide_encoder.py` completes.
- No FAISS index yet — retrieval skipped until `build_faiss_index.py` runs.
- SlideEncoder `_selective_scan` runs a Python loop over L (slow on CPU, fast on CUDA with `mamba-ssm`).

## User

Emir Arda Eker, co-author, ITU student. Prefers terse responses, direct code edits, no emojis.
