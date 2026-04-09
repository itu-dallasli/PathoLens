# PathoLens — Pipeline How-To

> End-to-end instructions: data prep -> train -> inference -> eval.
> All commands assume the repo root as CWD and `pip install -e ".[dev]"` already done.

---

## 0. Prerequisites

```bash
pip install -e ".[dev]"
# HuggingFace token for UNI (MahmoodLab/UNI is gated)
# Put token in ~/.cache/huggingface/token  OR  set HF_TOKEN env var
```

**Windows note**: set `PYTHONIOENCODING=utf-8` or redirect to a file to avoid
CP1254 encoding errors in the terminal.

---

## 1. Download CAMELYON16 (if needed)

```bash
python scripts/download_camelyon16.py \
    --dest X:/Bitirme_Data \
    --groups annotations masks tumor normal \
    --budget-gb 300
```

Expected layout:

```
X:/Bitirme_Data/
  tumor/          # tumor_001.tif ... tumor_111.tif  (111 slides)
  normal/         # normal_001.tif ... normal_010.tif (10 slides)
  annotations/    # tumor_*.xml  (ASAP polygon format)
  masks/          # tumor_*_mask.tif  (official pixel labels: 2=tumor, 1=tissue, 0=bg)
```

---

## 2. Build UNI embedding cache

Runs UNI (ViT-L/16) on every slide once and saves `(embeddings, coords, label)` as NPZ.
**Required before training.** Resumable — skips already-cached slides.

```bash
PYTHONPATH=src python scripts/build_embedding_cache.py \
    --tumor-dir  X:/Bitirme_Data/tumor  \
    --normal-dir X:/Bitirme_Data/normal \
    --max-patches 500 \
    --out data/processed/slide_cache
```

| Flag | Default | Notes |
|---|---|---|
| `--max-patches` | 4000 | 500 recommended on CPU (~2 min/slide) |
| `--tumor-limit N` | all | cap for quick tests |
| `--overwrite` | off | re-process cached slides |
| `--list` | off | dry-run only |

Estimated time: ~2 min/slide at 500 patches -> ~4 h for 121 slides on CPU.

---

## 3. Train SlideEncoder

Stratified 80/20 train/val split, CE loss, cosine LR + warmup.

```bash
PYTHONPATH=src python scripts/train_slide_encoder.py \
    --cache-dir data/processed/slide_cache \
    --epochs 20 \
    --d-model 256 \
    --n-layers 4

# To use Transformer attention instead of Mamba:
PYTHONPATH=src python scripts/train_slide_encoder.py \
    --epochs 20 --backbone attention
```

| Flag | Default | Notes |
|---|---|---|
| `--backbone` | mamba | `mamba` (O(N) SSM) or `attention` (O(N^2) MHSA) |
| `--epochs` | 20 | ~4 min/epoch on CPU |
| `--d-model` | 256 | internal feature dim |
| `--n-layers` | 4 | 2 patch-level + 2 region-level |
| `--region-size` | 64 | patches per region |
| `--lr` | 2e-4 | |

Outputs (auto-loaded by inference):

```
checkpoints/
  slide_encoder_final.pt        # weights + config + history
  slide_encoder_history.json    # per-epoch loss/acc
```

---

## 4. Build FAISS retrieval index

Runs the trained SlideEncoder on every cached slide to produce 256-dim
`slide_repr` vectors, then builds a flat cosine FAISS index.

```bash
PYTHONPATH=src python scripts/build_faiss_index.py \
    --cache-dir  data/processed/slide_cache \
    --checkpoint checkpoints/slide_encoder_final.pt \
    --out        data/faiss_index
```

Output:

```
data/faiss_index/
  slide_index.faiss     FAISS flat IP index (cosine after L2 normalisation)
  metadata.json         {slide_id -> {label, label_name, int_id, npz_path}}
```

Estimated time: ~5 min for 121 slides.

---

## 5. Run inference on a single WSI

Both the checkpoint and the FAISS index are auto-loaded if present.

```bash
PYTHONPATH=src python run_real.py X:/Bitirme_Data/tumor/tumor_071.tif
```

Or via the API:

```bash
uvicorn patholens.api.main:app --reload
# POST /api/analyze/ with the WSI file
```

Output in `results/<slide_id>/`:

```
results/tumor_071/
  heatmap.png              attention overlay on thumbnail
  heatmap_raw.npy          raw float32 attention weights (for threshold sweeps)
  diagnostic_report.json   FHIR R4 DiagnosticReport containing:
                             - tumor probability (Observation)
                             - heatmap (media attachment)
                             - top-k retrieved similar cases (derivedFrom)
```

---

## 6. Evaluate against CAMELYON16 ground truth

```bash
# Using official mask (preferred)
PYTHONPATH=src python scripts/eval_groundtruth.py \
    X:/Bitirme_Data/tumor/tumor_071.tif \
    --mask X:/Bitirme_Data/masks/tumor_071_mask.tif \
    --threshold 0.5

# Using XML polygon annotations (training set)
PYTHONPATH=src python scripts/eval_groundtruth.py \
    X:/Bitirme_Data/tumor/tumor_069.tif \
    --xml X:/Bitirme_Data/annotations/tumor_069.xml
```

| Flag | Default | Notes |
|---|---|---|
| `--threshold` | 0.5 | binarise attention for IoU |
| `--resolution` | 512 | evaluation grid size |
| `--mask` | — | official `_mask.tif` (value 2 = tumor) |
| `--xml` | — | ASAP XML polygons |

Metrics saved to `results/<slide_id>/eval_metrics.json`.

---

## 7. Expected baselines

| Stage | tumor_071 IoU | Notes |
|---|---|---|
| Untrained (uniform attention) | ~0.02 | random baseline, measured |
| Trained SlideEncoder 20 epochs | ~0.15-0.35 | target |
| CLAM_SB (comparison baseline) | ~0.30-0.50 | literature reference |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: patholens` | run with `PYTHONPATH=src` |
| HuggingFace login hangs | put token in `~/.cache/huggingface/token` |
| UnicodeEncodeError in terminal | set `PYTHONIOENCODING=utf-8` |
| `TIFFRGBAImageGet failed` on mask | handled automatically (pyramid fallback) |
| No checkpoint found | Step 3 uses uniform attention, pipeline continues |
| No FAISS index found | retrieval skipped, pipeline continues |
| Caching job killed / interrupted | re-run same command -- already-cached NPZs are skipped |
