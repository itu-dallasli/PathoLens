# CLAUDE.md — PathoLens Project Context

> Context file for Claude Code sessions. Keep this short and factual.
> Long-form docs live in `documentation/`. Skills/roadmap: see `SKILLS.md`, `ROADMAP.md`.

## What this project is

**PathoLens** — Traceable retrieval-based clinical decision support for breast cancer histopathology (ITU senior project).
Pipeline: WSI → Tissue Segmentation → Patch Extraction → UNI Embeddings → Mamba Encoder → FAISS Retrieval → LLM Entity Extraction → Attention Heatmap → FHIR R4 DiagnosticReport.

Internal acronyms (original, not in literature): **CMEA** (retrieval), **KARG** (entity extraction), **RAAF** (explainability), **CSAL** (report).

## Current status (2026-04-07)

- Full pipeline scaffold complete; all 7 stages wired end-to-end.
- **CPU runs work** on real WSI (`normal_136.tif` → 119s, 500 capped patches).
- FAISS index, Mamba weights, trained attention MIL are **not yet built** — pipeline gracefully degrades (uniform attention, empty retrieval → empty report).
- 105 tests pass on CPU. CI configured (`.github/workflows/ci.yml`).
- No GPU in dev environment. Production path uses `[gpu]` extras.

## Repo map

```
src/patholens/
  preprocessing/      # WSIReader (OpenSlide), TissueSegmentor, PatchExtractor
  embedding/          # UNIFeatureExtractor (MahmoodLab/UNI ViT-L/16, frozen)
  sequence_model/     # Mamba encoder (linear fallback on CPU)
  retrieval/          # FAISS hierarchical retriever — NO INDEX BUILT YET
  entity_extraction/  # KARG LLM majority-vote extractor
  explainability/     # GatedAttentionMIL, HeatmapGenerator — UNTRAINED
  report_generation/  # FHIRReportBuilder, EvidenceLinker
  pipeline/inference.py  # PathoLensPipeline.run() — main entry point
  api/main.py         # FastAPI, lazy model loading
configs/
  default.yaml        # Full GPU config
  test.yaml           # Mocked synthetic config
  local-run.yaml      # CPU + real UNI + max_patches=500
run_real.py           # Convenience CLI for single-WSI runs
```

## Critical environment notes (Windows)

- Shell is **bash** (use `/dev/null`, forward slashes).
- Console codec is **CP1254** — do NOT emit Unicode box chars (`━`, `└`, etc.) to logs. Use ASCII (`---`). Already fixed in `logger.py` and `inference.py`.
- `huggingface_hub.login()` **hangs** in non-interactive shells. `uni_extractor.py` reads token from `~/.cache/huggingface/token` or `$HF_TOKEN` env var.
- `openslide-bin` pip package provides the DLL on Windows (no choco install needed).
- torch/torchvision version pin: `torch==2.10.0` requires `torchvision==0.25.0` (CPU wheel).

## Install

```bash
# CPU dev (default)
pip install -e ".[dev]"                    # faiss-cpu, no mamba-ssm
# GPU production
pip install -e ".[dev,gpu]"                # + mamba-ssm, faiss-gpu, causal-conv1d
```

Tests: `pytest tests/` · Lint: `ruff check src tests` · Run: `python run_real.py X:/Bitirme_Data/normal_136.tif`

## Data (user's local)

`X:\Bitirme_Data\`:
- CAMELYON16: `normal_136-139.tif`, `tumor_069-071.tif` (+ XML ground-truth annotations, `<ASAP_Annotations>` format — parse with stdlib `xml.etree.ElementTree`, no ASAP install needed).
- CAMELYON17: `patient_190.zip`, `patient_191.zip` — **different task** (per-patient staging), ignore for now.

## Design invariants — do not violate

- **Lazy loading**: modules load in `PathoLensPipeline._load_modules()`, not in `__init__`. API must start instantly with no models.
- **Graceful degradation**: if `_retriever` / trained weights missing, steps must skip — never crash.
- **UNI is frozen**: `param.requires_grad = False`. Never fine-tune.
- **Evidence traceability**: every FHIR Observation must link to patch coordinates via `EvidenceLinker`.

## What's known broken / approximate

- `max_patches` cap takes **first N sequentially** → top-left corner bias. Should be random sample.
- `GatedAttentionMIL` has random init weights → heatmaps are noise until trained.
- Pipeline's "Step 3 Sequence encoding" currently just `torch.from_numpy(embeddings).unsqueeze(0)` — no real Mamba forward pass.
- No FAISS index → retrieval always empty → entity extraction has no reference reports → empty diagnosis → empty FHIR report.

## User

Emir Arda Eker, co-author, ITU student. Prefers terse responses, direct code edits, no emojis.
