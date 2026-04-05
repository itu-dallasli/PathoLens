# PathoLens

**A Traceable Retrieval-Based Clinical Decision Support System for Breast Cancer Histopathology**

Istanbul Technical University — Department of AI and Data Engineering

## Overview

PathoLens analyses Whole Slide Images (WSIs) of breast cancer tissue and produces traceable, evidence-based diagnostic reports in HL7 FHIR format. Rather than making autonomous decisions, it supports pathologists by retrieving morphologically similar reference cases and generating structured summaries where every diagnostic statement is linked to visual evidence and source reports.

## Architecture

```text
WSI Input → Tissue Segmentation → Patch Extraction (256×256 @20×)
         → UNI Embedding (1024-dim)
         → Mamba Sequence Encoder (slide + region representations)
         ├→ CMEA Retrieval (FAISS hierarchical search)
         │   └→ KARG Entity Extraction (LLM-based)
         └→ RAAF Explainability (attention heatmaps)
             └→ CSAL Report Generation (FHIR DiagnosticReport)
```

## Quick Start (No GPU Required)

The test suite is fully synthetic — no GPU, no real WSI data, no model downloads needed.

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install system dependency (OpenSlide C library)
#    Windows:  choco install openslide
#    Ubuntu:   sudo apt-get install -y libopenslide-dev
#    macOS:    brew install openslide

# 3. Install in development mode (installs pre-commit hooks)
make install-dev

# 4. Run the synthetic test suite (< 30 seconds, CPU only)
make test

# 5. Format, lint, and test in one shot (CI equivalent)
make check
```

> [!TIP]
> **Zero Data / Zero GPU:** All tests mock OpenSlide, the UNI model, and LLM backends. The full pipeline — including FHIR report generation and heatmap rendering — is verified using synthetic data in < 30 seconds on any laptop.

### Running the API Server

```bash
uvicorn patholens.api.main:app --reload --port 8000
# Server starts immediately; models are loaded lazily only when /api/analyze/ is called.
curl http://localhost:8000/health   # → {"status": "ok"}
```

### GPU / Production Install

To enable Mamba SSM sequence encoding and GPU-accelerated FAISS (requires CUDA):

```bash
pip install -e ".[dev,gpu]"
```

## Managing the Production Pipeline

For deploying the model locally or scaling up to the full TCGA-BRCA dataset, please read the **Production Scaling Blueprint**:
👉 `documentation/production_training_plan.md`

This document covers:
1. Downloading TCGA-BRCA (~1.5 TB) and CAMELYON16.
2. The 4-phase training pipeline (Preprocessing → Mamba → FAISS → LLM).
3. MLOps (DVC + MLflow integration).
4. AIOps Monitoring alerts for data drift and latency.

## Project Structure

```text
src/patholens/
├── preprocessing/     WSI ingestion, tissue segmentation, patch extraction
├── embedding/         UNI feature extractor (frozen ViT-L/16)
├── sequence_model/    Mamba/SAMBA slide encoder
├── retrieval/         CMEA hierarchical case retrieval (FAISS)
├── entity_extraction/ KARG clinical entity extraction (LLM)
├── explainability/    RAAF attention heatmaps & entity-region mapping
├── report_generation/ CSAL FHIR DiagnosticReport builder
├── pipeline/          End-to-end inference orchestrator
└── api/               FastAPI web service
```

## Hardware Requirements (Production)

- NVIDIA RTX 4090 (24GB VRAM) or equivalent
- AMD Ryzen 9 7950X / similar CPU
- 64GB DDR5 RAM
- 2TB NVMe Gen4 SSD (for WSI storage)

## License
Research use only.
