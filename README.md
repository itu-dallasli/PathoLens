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

## Quick Start & Developer Excellence

This repository is optimized for **Sustainable MLOps & AIOps**. You can install, test, format, and lint the entire codebase using simple `make` commands. 

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install in development mode (installs pre-commit hooks)
make install-dev

# 3. Run the synthetic test suite (Zero Data / Zero GPU Required)
make test

# 4. Format and Lint code
make check
```

> [!TIP]
> **Testing Without Data:** The `make test` command runs a fully synthetic test harness. It mocks `OpenSlide`, the HuggingFace `UNI` model, and LLM backends so you can verify pipeline changes in < 5 seconds without downloading the 1.5TB TCGA dataset.

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
