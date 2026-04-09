# ROADMAP.md — PathoLens Development Plan

Phased delivery plan. Each phase is a milestone that produces a demonstrable result.
Last updated: 2026-04-09.

---

## Phase 1 — Scaffold & CPU runs [DONE]

- [x] All 7 pipeline stages implemented with lazy loading
- [x] 105 unit/integration tests passing on CPU
- [x] GitHub Actions CI (lint + test, Python 3.10/3.11)
- [x] CPU-safe dependency split (`[dev]` vs `[gpu]` extras)
- [x] Real WSI run on `normal_136.tif` (119s, 500 patches)
- [x] FHIR R4 DiagnosticReport JSON serializes correctly

---

## Phase 2 — Ground-Truth Evaluation Harness [DONE]

- [x] `scripts/eval_groundtruth.py`: parse XML + official mask TIFs
- [x] Official mask decoding (value 2 = tumor, class-aware)
- [x] IoU metric vs CAMELYON16 pixel-level annotations
- [x] Baseline measured: tumor_071 IoU ~0.024 (untrained, expected)
- [x] Random patch sampling (removed top-left bias)

---

## Phase 3 — Real Mamba + Trainable SlideEncoder [DONE]

- [x] Replace `_LinearFallback` with `CPUMamba` (real selective SSM)
- [x] `AttentionBlock` as swappable backbone (`--backbone attention`)
- [x] `scripts/build_embedding_cache.py` — one-time UNI cache
- [x] `scripts/train_slide_encoder.py` — full training CLI
- [x] SlideEncoder wired into inference (Step 3 + attention heatmap)
- [x] `SlideEncoderOutput` exposes per-patch attention for heatmap

---

## Phase 4 — Simplified Retrieval + Diagnosis [DONE]

- [x] Remove LLM entity extraction (KARG) — undeliverable in scope
- [x] Classifier-based diagnosis from SlideEncoder logits (tumor probability)
- [x] `scripts/build_faiss_index.py` — FAISS flat IP index over training slide_repr
- [x] `_maybe_load_retriever()` in inference — auto-loads index if present
- [x] Retrieval majority vote as soft corroboration for tumor probability
- [x] FHIR report: classifier output + heatmap + retrieved cases as `derivedFrom`

---

## Phase 5 — Train & Evaluate [NEXT]

**Goal**: First real trained model; measurable IoU improvement.

- [ ] Embedding cache build completes (121 slides, in progress)
- [ ] Train SlideEncoder 20 epochs (~1.5 h on CPU)
- [ ] Build FAISS retrieval index
- [ ] Re-run eval on tumor_069/070/071 — target IoU > 0.15
- [ ] Compare Mamba backbone vs Attention backbone (same data, same eval)
- [ ] Record results in `documentation/benchmarks.md`

Exit criteria: IoU on held-out tumor slides > 0.15; retrieval returns plausible similar cases.

---

## Phase 6 — CLAM Baseline Comparison

**Goal**: Situate PathoLens results against the field standard.

- [ ] Train CLAM_SB on same UNI features + same CAMELYON16 slides
- [ ] Compute same IoU metric
- [ ] Table: CLAM_SB vs Mamba SlideEncoder vs Attention SlideEncoder
- [ ] Include in paper / final report

---

## Phase 7 — Bump Normals + Data Balance

- [ ] Download 40 more normal slides (currently 10; 111:10 is severe imbalance)
- [ ] Re-train with balanced set (111 tumor : 40-50 normal)
- [ ] Re-evaluate; expect better recall on normal slides

---

## Non-goals (explicitly out of scope)

- Fine-tuning UNI — frozen by design
- LLM entity extraction (KARG) — removed in Phase 4
- TCGA-BRCA reference corpus — retrieval index is over CAMELYON16 training set only
- Real-time inference (< 1s) — batch/async is fine
- Multi-organ pathology — breast only
- Full pathologist replacement — decision support only

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-05 | Use `faiss-cpu` in main deps, `faiss-gpu` in `[gpu]` extra | Can't install `faiss-gpu` on CPU dev machines |
| 2026-04-05 | Move `mamba-ssm` to `[gpu]` extra | CUDA kernel build fails on CPU |
| 2026-04-06 | Add `max_patches: 500` CPU cap | Full 32k patches takes ~2 hours on CPU |
| 2026-04-06 | Read HF token from cached file, not `login()` | `login()` hangs in non-interactive shells |
| 2026-04-07 | Skip ASAP, parse CAMELYON16 XML with stdlib | No C++ dep; XML is simple polygon list |
| 2026-04-07 | Ignore CAMELYON17 ZIPs | Different task (per-patient staging) |
| 2026-04-08 | Replace `_LinearFallback` with `CPUMamba` | Makes ~600 lines of training scaffold actually work |
| 2026-04-09 | Remove KARG LLM extraction | Requires external corpus + LLM backend; not deliverable in scope |
| 2026-04-09 | FAISS index over CAMELYON16 training set | No external corpus needed; retrieval is visual similarity in learned space |
