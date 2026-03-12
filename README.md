# PathoLens

**A Traceable Retrieval-Based Clinical Decision Support System for Breast Cancer Histopathology**

Istanbul Technical University — Department of AI and Data Engineering

## Overview

PathoLens analyses Whole Slide Images (WSIs) of breast cancer tissue and produces traceable, evidence-based diagnostic reports in HL7 FHIR format. Rather than making autonomous decisions, it supports pathologists by retrieving morphologically similar reference cases and generating structured summaries where every diagnostic statement is linked to visual evidence and source reports.

## Architecture

```
WSI Input → Tissue Segmentation → Patch Extraction (256×256 @20×)
         → UNI Embedding (1024-dim)
         → Mamba Sequence Encoder (slide + region representations)
         ├→ CMEA Retrieval (FAISS hierarchical search)
         │   └→ KARG Entity Extraction (LLM-based)
         └→ RAAF Explainability (attention heatmaps)
             └→ CSAL Report Generation (FHIR DiagnosticReport)
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "from patholens import __version__; print(__version__)"
```

## Running the API

```bash
uvicorn patholens.api.main:app --reload --port 8000
```

## Project Structure

```
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

## Hardware Requirements

- NVIDIA RTX 4090 (24GB VRAM) or equivalent
- AMD Ryzen 9 7950X / similar CPU
- 64GB DDR5 RAM
- 2TB NVMe Gen4 SSD (for WSI storage)

## License

Research use only.
