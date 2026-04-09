# PathoLens — Pipeline How-To

> End-to-end instructions: data prep → train → inference → eval.
> All commands assume the repo root as CWD and `pip install -e ".[dev]"` already done.

---

## 0. Prerequisites

```bash
pip install -e ".[dev]"          # installs faiss-cpu, openslide-bin, etc.
# HuggingFace token for UNI (MahmoodLab/UNI is gated)
# Put token in ~/.cache/huggingface/token  OR  set HF_TOKEN env var
```

**Windows note**: the console codec is CP1254. Always redirect output to a file
or set `PYTHONIOENCODING=utf-8` if you see UnicodeEncodeError in the terminal.

---

## 1. Download CAMELYON16 (if needed)

```bash
python scripts/download_camelyon16.py \
    --dest X:/Bitirme_Data \
    --groups annotations masks tumor normal \
    --budget-gb 300
```

Expected layout after download:

```
X:/Bitirme_Data/
  tumor/          # tumor_001.tif ... tumor_111.tif
  normal/         # normal_001.tif ... normal_010.tif
  annotations/    # *.xml  (training set only)
  masks/          # *_mask.tif  (official pixel-level labels)
```

---

## 2. Build embedding cache

Runs UNI (ViT-L/16) on every slide once and saves `(embeddings, coords, label)`
to disk. **Required before training.** Skips slides already cached (resumable).

```bash
PYTHONPATH=src python scripts/build_embedding_cache.py \
    --tumor-dir  X:/Bitirme_Data/tumor  \
    --normal-dir X:/Bitirme_Data/normal \
    --max-patches 500 \
    --out data/processed/slide_cache
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--max-patches` | 4000 | 500 recommended for CPU (~2 min/slide) |
| `--tumor-limit N` | all | cap tumor slides for quick tests |
| `--normal-limit N` | all | |
| `--overwrite` | off | re-process already-cached slides |
| `--list` | off | dry-run: print job list, no processing |

Estimated time on CPU: ~2 min/slide at 500 patches → ~4 h for 121 slides.

Progress log: `data/cache_build.log`

---

## 3. Train the Slide Encoder

Trains the hierarchical Mamba MIL on cached embeddings.
Stratified 80/20 train/val split, CE loss, cosine LR with warmup.

```bash
PYTHONPATH=src python scripts/train_slide_encoder.py \
    --cache-dir data/processed/slide_cache \
    --epochs 20 \
    --d-model 256 \
    --n-layers 4
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 20 | ~4 min/epoch on CPU |
| `--d-model` | 256 | internal feature dim |
| `--n-layers` | 4 | Mamba layers (2 patch + 2 region) |
| `--region-size` | 64 | patches per region |
| `--lr` | 2e-4 | AdamW learning rate |
| `--no-amp` | off | disable mixed precision (auto on CPU) |

Outputs:

```
checkpoints/
  slide_encoder_final.pt        # final weights + config dict + history
  slide_encoder_history.json    # per-epoch loss / acc
  slide_encoder_epoch020.pt     # periodic checkpoints every 10 epochs
```

The inference pipeline auto-loads `checkpoints/slide_encoder_final.pt` if it exists.
To point to a different checkpoint, add to your config YAML:

```yaml
sequence_model:
  checkpoint_path: checkpoints/slide_encoder_final.pt
```

---

## 4. Run inference on a single WSI

```bash
PYTHONPATH=src python run_real.py X:/Bitirme_Data/tumor/tumor_071.tif
```

Or via the API:

```bash
uvicorn patholens.api.main:app --reload   # http://localhost:8000/docs
```

Results written to `results/<slide_id>/`:

```
results/tumor_071/
  heatmap.png              # attention overlay on thumbnail
  heatmap_raw.npy          # raw float32 attention map (for threshold sweeps)
  diagnostic_report.json   # FHIR R4 DiagnosticReport
```

---

## 5. Evaluate against ground truth

Computes IoU between predicted attention heatmap and official CAMELYON16 mask.

```bash
PYTHONPATH=src python scripts/eval_groundtruth.py \
    X:/Bitirme_Data/tumor/tumor_071.tif \
    --mask X:/Bitirme_Data/masks/tumor_071_mask.tif \
    --threshold 0.5
```

Or with XML polygon annotations (training set):

```bash
PYTHONPATH=src python scripts/eval_groundtruth.py \
    X:/Bitirme_Data/tumor/tumor_069.tif \
    --xml X:/Bitirme_Data/annotations/tumor_069.xml
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--threshold` | 0.5 | binarise attention for IoU |
| `--resolution` | 512 | evaluation grid size (pixels) |
| `--mask` | — | official `_mask.tif` (preferred, class 2 = tumor) |
| `--xml` | — | ASAP XML polygon annotations (fallback) |

Metrics printed and saved to `results/<slide_id>/eval_metrics.json`:

```
IoU        : 0.xxx
Precision  : 0.xxx
Recall     : 0.xxx
F1         : 0.xxx
```

---

## 6. Batch evaluation across multiple slides

```bash
for wsi in X:/Bitirme_Data/tumor/tumor_069.tif \
           X:/Bitirme_Data/tumor/tumor_070.tif \
           X:/Bitirme_Data/tumor/tumor_071.tif; do
    mask="${wsi/_069/_069_mask}"; mask="${mask/tumor\//masks\/}"
    # adjust path as needed
    PYTHONPATH=src python scripts/eval_groundtruth.py "$wsi" --mask ...
done
```

---

## 7. Expected baselines

| Stage | tumor_071 IoU | Notes |
|---|---|---|
| Random (uniform attention) | ~0.02 | untrained baseline |
| After 20-epoch Mamba MIL | ~0.15-0.35 | depends on dataset size |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: patholens` | run with `PYTHONPATH=src` |
| `huggingface_hub` login hangs | put HF token in `~/.cache/huggingface/token` |
| UnicodeEncodeError in terminal | `set PYTHONIOENCODING=utf-8` or redirect to file |
| `TIFFRGBAImageGet failed` on mask | handled automatically (pyramid level fallback) |
| FAISS index not found | retrieval step skipped, pipeline continues |
| `checkpoints/slide_encoder_final.pt` missing | Step 3 passes raw UNI features, pipeline continues |
