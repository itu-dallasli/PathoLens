# Production Training Plan — PathoLens

A comprehensive guide for training and deploying PathoLens on the full TCGA-BRCA, CAMELYON16, and Quilt-1M datasets. Covers data acquisition, infrastructure, training phases, MLOps, and AIOps monitoring.

---

## 1. Gradual Scaling Strategy

Do **not** jump straight to 1,100 slides. Scale through three gates:

| Gate | Slides | Purpose | Duration |
|------|--------|---------|----------|
| **DEV** | 5–10 TCGA-BRCA | Validate the pipeline works end-to-end | 1–2 days |
| **STAGING** | 50–100 TCGA-BRCA + 20 CAMELYON16 | Tune hyperparameters, benchmark metrics | 1 week |
| **PROD** | Full TCGA-BRCA (~1,100) + CAMELYON16 (400) + Quilt-1M subset | Final training and evaluation | 2–4 weeks |

> [!TIP]
> The DEV gate uses the same synthetic test harness from this project. Replace mocks with real data only at the STAGING gate.

---

## 2. Data Acquisition

### 2.1 TCGA-BRCA (~1,100 slides, ~1.5 TB)

```bash
# Install GDC Data Transfer Tool
pip install gdc-client

# Download manifest from https://portal.gdc.cancer.gov/
# Filter: Project = TCGA-BRCA, Data Category = Biospecimen, Data Type = Slide Image

# Download slides (parallel)
gdc-client download -m gdc_manifest.txt -d data/raw/tcga_brca/ --n-processes 8

# Download clinical metadata (CSV)
# Portal → Repository → Clinical tab → TSV export
```

**Storage estimate:** ~1.5 TB for SVS files, ~50 GB for extracted patches (compressed), ~20 GB for embeddings (HDF5).

### 2.2 CAMELYON16 (400 slides)

```bash
# Download from https://camelyon16.grand-challenge.org/Data/
# Training set: 270 slides (~700 GB)
# Test set: 130 slides (~350 GB)
# Includes pixel-level tumour annotations (XML)
```

### 2.3 Quilt-1M (Text + Image pairs)

```bash
# HuggingFace dataset
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('wisdomik/Quilt-1M', split='train')
ds.save_to_disk('data/raw/quilt1m/')
"
```

### 2.4 Clinical Reports

TCGA clinical reports are obtained via the GDC API:

```bash
# Download pathology reports for TCGA-BRCA cases
python scripts/download_tcga_reports.py \
    --project TCGA-BRCA \
    --output data/metadata/tcga_brca_reports.csv
```

---

## 3. Infrastructure Requirements

### 3.1 Hardware (Single-Node, as per project constraints)

| Component | Spec | Purpose |
|-----------|------|---------|
| GPU | NVIDIA RTX 4090 (24 GB VRAM) | Embedding extraction, Mamba training, attention model |
| CPU | AMD Ryzen 9 7950X (16C / 32T) | WSI preprocessing, data loading |
| RAM | 64 GB DDR5 | Large WSI handling, FAISS index |
| Storage | 2 TB NVMe Gen4 SSD | WSI files, patches, embeddings |

### 3.2 Software Stack

```bash
# Core
python==3.10+
torch==2.4+  (CUDA 12.1)
torchvision==0.19+
timm==1.0+

# WSI
openslide-python==1.3+
opencv-python-headless==4.9+

# Sequence model
mamba-ssm==2.0+       # CUDA kernel required for production
causal-conv1d==1.2+

# Retrieval
faiss-gpu==1.7.4+     # GPU-accelerated index

# LLM
transformers==4.40+
# BioMistral-7B (~14 GB download)

# MLOps
dvc==3.0+             # Data versioning
mlflow==2.10+         # Experiment tracking
wandb==0.16+          # Alternative tracker
```

---

## 4. Training Phases

### Phase 1: Preprocessing + Embedding Extraction (1–3 days)

```bash
# Process all slides in parallel
python scripts/preprocess_all.py \
    --wsi-dir data/raw/tcga_brca/ \
    --output-dir data/processed/ \
    --patch-size 256 \
    --magnification 20 \
    --workers 8

# Extract UNI embeddings (GPU)
python scripts/extract_embeddings.py \
    --patches-dir data/processed/patches/ \
    --output-dir data/processed/embeddings/ \
    --model MahmoodLab/UNI \
    --batch-size 256 \
    --device cuda:0
```

**Estimated time:** ~8 minutes per slide (patch extraction + embedding) × 1,100 slides ≈ **6 hours** on RTX 4090.

### Phase 2: Mamba Sequence Model Training (3–7 days)

```bash
python scripts/train_mamba.py \
    --embeddings-dir data/processed/embeddings/ \
    --metadata data/metadata/tcga_brca_labels.csv \
    --config configs/default.yaml \
    --epochs 50 \
    --lr 1e-4 \
    --weight-decay 1e-2 \
    --scheduler cosine \
    --warmup 5 \
    --gradient-checkpointing \
    --output checkpoints/mamba_v1/
```

**Key hyperparameters to tune at staging:**
- `d_model`: 256 → 512 → 768
- `n_layers`: 2 → 4 → 6
- `region_size`: 32 → 64 → 128
- Learning rate: 5e-5 → 1e-4 → 3e-4

### Phase 3: FAISS Index Building (1–2 hours)

```bash
python scripts/build_faiss_index.py \
    --embeddings-dir data/processed/embeddings/ \
    --output data/processed/faiss_index/ \
    --index-type IVFFlat \
    --n-list 100 \
    --metric cosine
```

### Phase 4: Evaluation & Validation (1–2 days)

```bash
# Retrieval metrics
python scripts/evaluate_retrieval.py \
    --index data/processed/faiss_index/slide_index.faiss \
    --test-set data/metadata/test_split.csv \
    --top-k 5

# Explainability (CAMELYON16)
python scripts/evaluate_explainability.py \
    --wsi-dir data/raw/camelyon16/test/ \
    --annotations data/raw/camelyon16/annotations/ \
    --checkpoint checkpoints/mamba_v1/best.pt
```

---

## 5. MLOps Pipeline

### 5.1 Data Versioning (DVC)

```bash
dvc init
dvc remote add -d storage s3://patholens-data  # or local NAS

# Track large data files
dvc add data/raw/tcga_brca/
dvc add data/processed/embeddings/
dvc add data/processed/faiss_index/

git add data/*.dvc .dvc/
git commit -m "Track TCGA-BRCA data with DVC"
dvc push
```

### 5.2 Experiment Tracking (MLflow)

```python
import mlflow

mlflow.set_experiment("patholens-mamba-training")

with mlflow.start_run(run_name="mamba_v1_d512_l4"):
    mlflow.log_params({
        "d_model": 512, "n_layers": 4,
        "region_size": 64, "lr": 1e-4,
    })
    # ... training loop ...
    mlflow.log_metrics({
        "recall_at_5": 0.78,
        "mAP": 0.63,
        "iou": 0.42,
    })
    mlflow.pytorch.log_model(model, "mamba_encoder")
```

### 5.3 CI/CD for Model Validation

```yaml
# .github/workflows/model_validation.yml
name: Model Validation
on:
  push:
    paths: ['checkpoints/**', 'configs/**']
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --tb=short
      - run: python scripts/validate_model.py --checkpoint checkpoints/latest/
```

### 5.4 Model Registry

```bash
# After validation passes, register the model
mlflow models serve -m "models:/patholens-mamba/Production" -p 5001
```

---

## 6. AIOps Monitoring

### 6.1 Training Monitoring

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| GPU utilisation | < 50% for 10 min | Check data loading bottleneck |
| Training loss plateau | No improvement in 5 epochs | Reduce LR or stop early |
| VRAM usage | > 22 GB | Reduce batch size or enable gradient checkpointing |
| Disk I/O wait | > 30% | Move data to faster SSD, increase prefetch |

### 6.2 Inference Monitoring (Production)

| Metric | Target | Alert |
|--------|--------|-------|
| WSI inference latency | ≤ 150s | > 200s triggers investigation |
| Evidence coverage | ≥ 95% | < 90% triggers model review |
| Low-confidence rate | < 15% | > 25% triggers retraining |
| Memory usage | < 20 GB | > 22 GB triggers alert |
| API response (P99) | < 5s | > 10s triggers scaling |

### 6.3 Data Drift Detection

```python
# Periodic check: compare embedding distribution of new vs. training data
from scipy.stats import ks_2samp

def check_drift(new_embeddings, reference_embeddings, threshold=0.05):
    for dim in range(new_embeddings.shape[1]):
        stat, pval = ks_2samp(
            new_embeddings[:, dim],
            reference_embeddings[:, dim],
        )
        if pval < threshold:
            alert(f"Distribution drift detected in dimension {dim}")
```

---

## 7. Cost Estimation (Single RTX 4090)

| Phase | GPU Hours | Wall-Clock |
|-------|-----------|------------|
| Preprocessing + Embedding (1,100 slides) | ~6 h | ~6 h |
| Mamba training (50 epochs) | ~72 h | ~3 days |
| FAISS index build | ~1 h | ~1 h |
| Evaluation (retrieval + explainability) | ~4 h | ~4 h |
| **Total** | **~83 h** | **~4-5 days** |

**Storage:** ~2 TB total (WSIs + patches + embeddings + checkpoints).

---

## 8. Checklist Before Production Training

- [ ] DEV gate passed (5–10 slides, all metrics computed)
- [ ] STAGING gate passed (50–100 slides, hyperparameters tuned)
- [ ] DVC remote configured and data versioned
- [ ] MLflow/WandB server running
- [ ] CUDA 12.1 + mamba-ssm installed and tested
- [ ] OpenSlide C library installed on production machine
- [ ] HuggingFace token set (`HF_TOKEN`) for UNI model download
- [ ] Sufficient disk space (≥ 2 TB free)
- [ ] Backup strategy for checkpoints in place
