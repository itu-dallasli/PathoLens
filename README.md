# PathoLens

**Traceable Retrieval-Based Clinical Decision Support for Breast Cancer Histopathology**

Istanbul Technical University — Department of AI and Data Engineering

---

## Overview

PathoLens analyses Whole Slide Images (WSIs) of breast cancer tissue and produces traceable, evidence-based diagnostic support in HL7 FHIR R4 format. It combines:

- **Hierarchical Mamba/Attention MIL** for slide-level classification (tumor vs normal)
- **Visual similarity retrieval** (FAISS over trained slide embeddings) for similar-case evidence
- **Spatial attention heatmaps** that highlight diagnostically relevant tissue regions
- **FHIR R4 DiagnosticReport** linking every finding to visual evidence and retrieved cases

## Architecture

```text
WSI
 |
 +-- Tissue Segmentation (Otsu + morphology)
 |
 +-- Patch Extraction (256x256 @ 20x, tissue-masked)
 |
 +-- UNI Embeddings (frozen ViT-L/16, 1024-dim)        <- MahmoodLab/UNI
 |
 +-- SlideEncoder  -------------------------------- trainable
 |    patch Mamba/Attention -> RegionAggregator
 |    -> region Mamba/Attention -> attention pool
 |    |                              |
 |    | slide_repr (256-dim)         | per-patch attention weights
 |    |                              |
 |    +-- CMEA Retrieval             +-- RAAF Heatmap
 |        FAISS top-k similar            Gaussian-smoothed
 |        training cases                 attention overlay
 |                |                           |
 |                +----------+----------------+
 |                           |
 +-- CSAL: FHIR R4 DiagnosticReport
      - tumor probability (classifier)
      - attention heatmap (media attachment)
      - retrieved similar cases (derivedFrom)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| UNI frozen | 303M-param foundation model pre-trained on Mass-100K; fine-tuning is infeasible and unnecessary |
| Mamba O(N) vs Attention O(N^2) | Both supported via `--backbone mamba\|attention`; Mamba scales better to large patch counts |
| No LLM entity extraction | Removes uncontrolled external dependency; classifier output is deterministic and evaluable |
| FAISS over training set | Retrieval is visual similarity in the learned embedding space — no external report corpus needed |
| FHIR R4 | Clinical interoperability standard; every observation is traceable to a patch coordinate |

---

## Quick Start

```bash
# Install (CPU, no GPU needed for dev)
pip install -e ".[dev]"

# Run tests (synthetic, < 30 s, no data needed)
pytest tests/

# Or with make
make install-dev && make test
```

> **Zero Data / Zero GPU:** All tests mock OpenSlide and UNI. Full pipeline including FHIR generation and heatmap rendering is verified with synthetic data in under 30 seconds.

### API Server

```bash
uvicorn patholens.api.main:app --reload --port 8000
curl http://localhost:8000/health   # {"status": "ok"}
```

### GPU / Production Install

```bash
pip install -e ".[dev,gpu]"   # adds mamba-ssm, faiss-gpu, causal-conv1d
```

---

## Training Pipeline

```bash
# 1. Build UNI embedding cache (one-time, ~4 h on CPU for 121 slides)
PYTHONPATH=src python scripts/build_embedding_cache.py \
    --tumor-dir  X:/Bitirme_Data/tumor  \
    --normal-dir X:/Bitirme_Data/normal \
    --max-patches 500

# 2. Train SlideEncoder (~1.5 h on CPU, 20 epochs)
PYTHONPATH=src python scripts/train_slide_encoder.py \
    --epochs 20 --d-model 256 --n-layers 4
    # add --backbone attention to use Transformer blocks instead of Mamba

# 3. Build FAISS retrieval index (~5 min)
PYTHONPATH=src python scripts/build_faiss_index.py

# 4. Run inference on a slide (checkpoint + index auto-loaded)
PYTHONPATH=src python run_real.py X:/Bitirme_Data/tumor/tumor_071.tif

# 5. Evaluate against CAMELYON16 ground truth
PYTHONPATH=src python scripts/eval_groundtruth.py \
    X:/Bitirme_Data/tumor/tumor_071.tif \
    --mask X:/Bitirme_Data/masks/tumor_071_mask.tif
```

See `documentation/PIPELINE_HOWTO.md` for full details.

---

## Project Structure

```
src/patholens/
  preprocessing/       WSI ingestion, tissue segmentation, patch extraction
  embedding/           UNI feature extractor (frozen ViT-L/16)
  sequence_model/      SlideEncoder: Mamba or Attention MIL backbone
  retrieval/           CMEA visual similarity retrieval (FAISS)
  explainability/      RAAF attention heatmaps + entity-region mapping
  report_generation/   CSAL FHIR R4 DiagnosticReport builder
  pipeline/            End-to-end inference orchestrator
  api/                 FastAPI web service

scripts/
  build_embedding_cache.py   Run UNI on all slides, cache to NPZ
  train_slide_encoder.py     Train SlideEncoder on cached embeddings
  build_faiss_index.py       Build FAISS retrieval index from trained encoder
  eval_groundtruth.py        Compute IoU vs CAMELYON16 masks/annotations
  download_camelyon16.py     Download dataset from AWS Open Data

documentation/
  PIPELINE_HOWTO.md          Step-by-step pipeline instructions
```

---

## Evaluation

Metric: **IoU** between thresholded attention heatmap and official CAMELYON16
pixel-level tumor masks (class 2 of `*_mask.tif`).

| Stage | tumor_071 IoU | Notes |
|---|---|---|
| Untrained (uniform attention) | ~0.02 | random baseline |
| Trained SlideEncoder (20 epochs, 121 slides) | ~0.15-0.35 | target |

Comparison baseline: CLAM_SB (`mahmoodlab/CLAM`) trained on the same UNI features.

---

## Dataset

CAMELYON16 (AWS Open Data, `s3://camelyon-dataset/`):
- 111 tumor WSIs + 10 normal WSIs (training set used here)
- Official pixel-level masks in `CAMELYON16/masks/` (value 2 = tumor)
- XML polygon annotations in `CAMELYON16/annotations/`

---

## License

Research use only. UNI model weights are subject to MahmoodLab's license terms.
CLAM (comparison baseline only, not bundled) is GPL v3.
