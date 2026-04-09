# PathoLens — Academic Analysis
## Why Mamba/SSM Over Attention for WSI Classification

Istanbul Technical University — Department of AI and Data Engineering
Emir Arda Eker, 2026

---

## 1. The Core Problem: WSIs Are Extremely Long Sequences

A single Whole Slide Image at 20x magnification produces tens of thousands of patches.
A typical CAMELYON16 tumor slide yields 5,000–50,000 non-background 256x256 patches.

| Model | Complexity per layer | 500 patches | 5,000 patches | 50,000 patches |
|---|---|---|---|---|
| Transformer (CLAM, standard attention) | O(N^2) | 250K ops | 25M ops | 2.5B ops |
| Mamba (SSM) | O(N) | 500 ops | 5K ops | 50K ops |

At 50,000 patches the attention matrix is 50,000 x 50,000 = 2.5 billion entries — impossible
to materialize in GPU memory. This is why CLAM does not model inter-patch dependencies at
all: it reduces each slide to a **bag of independent embeddings** and pools them with a
learned attention weight, skipping any sequence-level interaction between patches.

Mamba handles the full sequence in linear time and constant memory (recurrent state),
making genuine long-range contextual modeling feasible for the first time at WSI scale.

---

## 2. What Is Mamba (Selective State Space Model)?

Mamba (Gu & Dao, 2023) is a sequence model built on **state space models (SSMs)**.
At each position the model maintains a hidden state `h` and updates it recurrently:

```
h_t = A(x_t) * h_{t-1} + B(x_t) * x_t
y_t = C(x_t) * h_t + D * x_t
```

The key innovation is **selectivity**: the transition matrices A, B, C are *functions of the
input* (unlike classical SSMs where they are fixed). This lets the model decide what to
remember and what to forget based on content — similar to an LSTM gating mechanism but
computed in parallel via a selective scan.

In practice this means:
- **Global context in O(N)**: every patch embedding influences the final slide representation
  without the quadratic cost of self-attention.
- **Causal / ordered processing**: Mamba naturally models the order in which patches are
  presented. Sorting patches spatially (row-major) gives the model a left-to-right,
  top-to-bottom scan that approximates how a pathologist systematically examines a slide.
- **Learnable forgetting**: irrelevant background regions are suppressed by the gating;
  diagnostic regions are retained in the hidden state and propagate forward.

---

## 3. PathoLens Architecture vs CLAM

### CLAM (baseline)

```
Patches (N x 1024)
     |
  FC layers: 1024 -> 512 -> 256
     |
  Gated attention: score_i = softmax(W_a * tanh(V * h_i) * sigmoid(U * h_i))
     |
  Weighted sum: z = sum(score_i * h_i)   <- bag-level representation
     |
  Classifier: z -> logit
```

- Patches are **independent** — no information flows between them before pooling.
- Attention scores are computed in isolation per patch.
- Instance-level clustering loss (SmoothSVM) provides implicit supervision signal.
- Proven, fast, well-tuned. The CAMELYON16 published checkpoint achieves strong AUC.

### PathoLens SlideEncoder

```
Patches (N x 1024)
     |
  Input projection: 1024 -> 256
     |
  [Stage 1] Patch-level Mamba encoder (n_layers//2 blocks)
     |     Each patch now contains context from neighboring patches
     |
  RegionAggregator: group into R regions, intra-region attention pooling
     |
  [Stage 2] Region-level Mamba encoder (remaining n_layers//2 blocks)
     |     Each region now contains context from other regions
     |
  Slide attention: softmax-weighted sum over regions -> slide_repr (256-dim)
     |
  Classifier: slide_repr -> 2-class logit
```

Key differences:
1. **Two-stage hierarchy**: patch context is modeled before aggregation, then region
   context is modeled — analogous to how a pathologist first scans at low magnification
   (regions) and then examines suspicious areas at high magnification (patches within regions).
2. **Inter-patch dependencies**: Mamba propagates information across patches before pooling,
   so the pooled representation knows about the spatial distribution of features, not just
   their independent magnitudes.
3. **Swappable backbone**: `--backbone attention` replaces both Mamba stages with
   pre-norm Transformer blocks — same architecture, different sequence mixer.

---

## 4. Computational Comparison

### Memory

| Model | Memory for N patches | N=500 | N=4000 | N=50000 |
|---|---|---|---|---|
| CLAM (flat attention) | O(N) — no sequence mixing | ~2 MB | ~16 MB | ~200 MB |
| Transformer SlideEncoder | O(N^2) per stage | ~1 MB | ~64 MB | OOM |
| Mamba SlideEncoder | O(N) | ~2 MB | ~16 MB | ~200 MB |

CLAM uses O(N) memory because it has no sequence mixing — just element-wise operations.
Mamba matches CLAM's memory profile while adding sequence context.
Transformer at N=4000 is already pushing limits; at N=50000 it cannot run.

### Speed (CPU, d_model=256, 2 Mamba layers)

| Operation | N=500 | N=4000 |
|---|---|---|
| Forward pass | ~0.08 s | ~0.6 s |
| Backward pass | ~1.56 s | ~12 s |

The slow step is the Python-loop selective scan in `CPUMamba`. On GPU with the
fused `mamba-ssm` CUDA kernel the scan runs in microseconds regardless of N.

---

## 5. Why This Might Work Better Than CLAM

### Spatially distributed tumors

In CAMELYON16, metastatic tumor regions can be small and scattered. A pooling-only
model (CLAM) assigns attention scores independently, so a small cluster of 10 high-score
patches competes directly with 490 background patches. The softmax denominator dilutes
their contribution.

Mamba processes the sequence before pooling. If patches 200–210 are tumor, the SSM
state accumulates evidence across this window and the region representation reflects
the cluster coherently — the classifier sees a "tumor cluster signal" rather than 10
individual noisy signals competing against 490 background signals.

### Texture continuity

Carcinoma tissue has characteristic texture that extends across adjacent patches.
CLAM treats each patch as independent and cannot model "this patch looks more like
carcinoma given its neighbors also look carcinomatous." Mamba's recurrent state
carries texture context from previous patches, reinforcing confident predictions.

### Hierarchical structure matches pathology workflow

The two-stage design (patch -> region -> slide) mirrors how pathologists actually work:
1. Scan at low magnification to identify suspicious regions.
2. Zoom into those regions for detailed examination.

CLAM collapses this to a single step.

---

## 6. Honest Limitations of Mamba for WSI

### Patch ordering is arbitrary

Mamba is a causal/ordered model. WSI patches are extracted on a 2D grid; there is
no inherent "correct" order. Row-major sorting is a reasonable approximation but
it introduces an artificial directionality — the model treats the first row of patches
as "earlier" than the last row, which has no biological meaning.

Bidirectional Mamba (Vim, MambaMIL) addresses this by scanning in both directions
and averaging, but that doubles compute.

### Training instability on small datasets

CLAM was designed for small pathology datasets (200–500 slides is typical). It has
a well-studied, stable training regime. Mamba adds more parameters and a more
complex optimization landscape. With only 121 slides our training set is small;
overfitting and high variance across runs is a real risk.

### CPU speed

The selective scan runs in a Python `for` loop on CPU (~6.9 s/batch at N=500).
CLAM's flat MIL head is nearly instantaneous on CPU. Training time is ~1.5 h
vs ~15 min for CLAM on the same hardware. On GPU the gap disappears.

### Unproven at this scale in pathology

CLAM has hundreds of citations, a published CAMELYON16 checkpoint, and validated
performance on TCGA cohorts. Mamba-based WSI models (MambaMIL, ViM-path) are
recent (2024) and fewer in number. The field has not converged on best practices
for Mamba applied to pathology.

---

## 7. Expected Experimental Outcomes

Three plausible scenarios, roughly ordered by probability:

### Scenario A: Mamba >= CLAM (most likely for large N)

With the full embedding cache (4000 patches/slide) and a balanced dataset,
Mamba's sequence context should pay off. Expected: IoU within 5–10% of CLAM,
with Mamba potentially better on slides with spatially small/scattered tumors.

### Scenario B: Mamba ~ CLAM (likely at N=500)

At 500 patches the sequence is short enough that CLAM's independent attention
already captures most of the discriminative signal. Mamba's context advantage
is smaller. Both models will be limited more by dataset size (121 slides) than
by architecture.

### Scenario C: CLAM > Mamba (possible with small dataset)

If the dataset is too small to learn the Mamba SSM parameters (A, B, C, dt_proj)
reliably, CLAM's simpler inductive bias (attend to the highest-scoring patches)
may generalize better. This would be a legitimate negative result and worth
reporting.

---

## 8. What the Paper Should Claim

**Strong claims (supported by existing literature + our architecture):**

1. The hierarchical two-stage design (patch Mamba -> region aggregation -> region
   Mamba) is a novel application of SSMs to WSI analysis with a clear architectural
   motivation in the hierarchical structure of histopathology.

2. The O(N) complexity of Mamba is a fundamental advantage over attention-based
   MIL for WSIs with large patch counts, enabling processing of full slides without
   patch-count caps.

3. The trained slide_repr serves simultaneously as a classifier and a retrieval key —
   a single model supports both tumor detection and visual similarity search (CMEA).

**Modest claims (dependent on experimental results):**

4. On CAMELYON16 with the given training set, Mamba SlideEncoder achieves [X] IoU
   vs CLAM's [Y] IoU on the held-out tumor slides.

5. Retrieval using learned Mamba slide_repr returns visually similar cases more
   reliably than mean-pooled UNI features (if we measure Recall@k).

**What not to claim:**

- Do not claim Mamba is definitively better than CLAM without results on a larger dataset.
- Do not claim clinical utility — this is a research prototype for decision support, not diagnosis.

---

## 9. References

- Gu, A. & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
- Lu, M.Y. et al. (2021). Data-efficient and weakly supervised computational pathology on WSIs. Nature Biomedical Engineering. (CLAM)
- Chen, R.J. et al. (2022). Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning. CVPR. (HIPT)
- Chen, R.J. et al. (2024). Towards a general-purpose foundation model for computational pathology. Nature Medicine. (UNI)
- Yang, Z. et al. (2024). MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology. arXiv:2408.15032.
- Filiot, A. et al. (2024). Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling. Medical Image Analysis. (Phikon)
