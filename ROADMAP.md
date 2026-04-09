# ROADMAP.md — PathoLens Development Plan

Phased delivery plan. Each phase is a milestone that produces a demonstrable result.
Last updated: 2026-04-07.

---

## Phase 1 — Scaffold & CPU runs ✅ DONE

**Goal:** Full 7-stage pipeline wired end-to-end, runnable on a dev laptop without GPU.

- [x] All 7 modules implemented with lazy loading
- [x] 105 unit/integration tests passing on CPU
- [x] GitHub Actions CI (lint + test, Python 3.10/3.11)
- [x] CPU-safe dependency split (`[dev]` vs `[gpu]` extras)
- [x] Real WSI run on `normal_136.tif` succeeds (119s, 500 patches)
- [x] FHIR R4 DiagnosticReport JSON serializes correctly
- [x] Windows encoding / HF login / torch version issues resolved

**What's missing at end of phase:** No trained weights, no FAISS index, no ground-truth evaluation. Pipeline produces structurally valid but semantically empty reports.

---

## Phase 2 — Ground-Truth Evaluation Harness 🎯 NEXT

**Goal:** Quantitative + visual comparison of pipeline output against CAMELYON16 tumor annotations. Produces the first objective metric.

### Tasks
- [ ] `scripts/eval_groundtruth.py`: parse `tumor_*.xml` → rasterize polygons → binary mask
- [ ] Downsample GT mask to thumbnail resolution matching `heatmap.png`
- [ ] Compute **IoU** between thresholded attention heatmap and GT tumor region
- [ ] Save side-by-side `comparison.png`: thumbnail | GT overlay | predicted heatmap | diff
- [ ] Fix `max_patches` cap to use **random sampling** (not first-N sequential) — removes top-left corner bias
- [ ] Run on `tumor_069`, `tumor_070`, `tumor_071` — baseline IoU numbers (expected to be ~random, ~0.05–0.15 with untrained attention)

### Exit criteria
- Can run `python scripts/eval_groundtruth.py tumor_070` and get a visual comparison + IoU number
- Baseline recorded in `documentation/benchmarks.md`

---

## Phase 3 — Train Attention MIL

**Goal:** Meaningful heatmaps. The attention head actually highlights tumor regions.

### Tasks
- [ ] Slide-level labels: normal (0) vs tumor (1) from CAMELYON16 file naming
- [ ] Extract UNI embeddings for all training slides **once** (cache to disk via `EmbeddingStore`)
- [ ] Training loop for `GatedAttentionMIL` — binary cross-entropy on bag label, attention supervised implicitly
- [ ] Weights checkpoint → `models/attention_mil.pt`
- [ ] Load in `inference.py` if checkpoint exists; keep random-init fallback
- [ ] Re-run Phase 2 evaluation harness → IoU should improve substantially

### Exit criteria
- IoU on held-out tumor slides > 0.3 (rough target)
- Heatmaps visually align with GT polygons

---

## Phase 4 — Build FAISS Retrieval Index (CMEA)

**Goal:** The "R" in the project title. Similar-case retrieval drives entity extraction.

### Tasks
- [ ] Choose reference corpus: TCGA-BRCA diagnostic reports + slides (public)
- [ ] Compute slide-level representations (mean-pool UNI for now, Mamba later)
- [ ] Build hierarchical FAISS index (`IndexIVFFlat` or `IndexHNSWFlat`)
- [ ] Persist index + metadata (report text per slide_id)
- [ ] Wire `self._retriever` in `inference.py` — currently `None`
- [ ] End-to-end test: KARG entity extractor now receives non-empty `reference_reports` → non-empty diagnosis → non-empty FHIR report

### Exit criteria
- `result.similar_cases` populated with top-k results + similarity scores
- FHIR report contains at least one Observation with evidence traceable to a retrieved case

---

## Phase 5 — Train Mamba SlideEncoder

**Goal:** Replace the mean-pool slide representation with a learned sequence encoding.

### Tasks
- [ ] Set up Mamba on a GPU box (`pip install -e ".[gpu]"`)
- [ ] Contrastive or classification training objective (TBD)
- [ ] Export CPU-compatible state dict for inference (or keep linear fallback for dev)
- [ ] Rebuild FAISS index with learned embeddings
- [ ] Benchmark retrieval quality (Recall@k) vs mean-pool baseline

### Exit criteria
- Mamba encoder weights checkpoint
- Retrieval Recall@5 improves vs mean-pool baseline

---

## Phase 6 — Entity Extraction & Report Quality (KARG + CSAL)

**Goal:** Diagnosis extraction becomes clinically plausible, not a majority-vote toy.

### Tasks
- [ ] Evaluate LLM backends: local (Llama-3-8B, Qwen) vs API (GPT-4, Claude)
- [ ] Improve prompts in `KARGEntityExtractor` — structured output schema (JSON)
- [ ] Add SNOMED CT / ICD-O code mapping in `EntityValidator`
- [ ] FHIR `Observation.code` uses real coded values, not free text
- [ ] Add evidence coverage scoring in `EvidenceLinker` — reject low-support claims

### Exit criteria
- FHIR reports pass HL7 validator
- Manual pathologist review of 5 sample reports → "plausible" rating

---

## Phase 7 — Production Deployment

**Goal:** Ship it.

### Tasks
- [ ] Docker image with GPU extras + OpenSlide + model weights baked in
- [ ] `/analyze` endpoint streaming progress updates (WebSocket)
- [ ] Authentication + rate limiting on API
- [ ] Observability: structured logs, Prometheus metrics, request tracing
- [ ] Load test: target 1 WSI / minute on single GPU
- [ ] Deployment docs + runbook

### Exit criteria
- One-command deployment (`docker compose up`)
- API hitting latency / throughput targets on real slides

---

## Non-goals (explicitly out of scope)

- Fine-tuning UNI — it is frozen by design
- Real-time inference (< 1s) — batch/async is fine
- Multi-organ pathology — breast only for v1
- Full pathologist replacement — this is **decision support**, not diagnosis

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-05 | Use `faiss-cpu` in main deps, `faiss-gpu` in `[gpu]` extra | Can't install `faiss-gpu` on CPU dev machines |
| 2026-04-05 | Move `mamba-ssm` to `[gpu]` extra | CUDA kernel build fails on CPU |
| 2026-04-06 | Add `max_patches: 500` CPU cap | Full 32k patches takes ~2 hours on CPU |
| 2026-04-06 | Read HF token from cached file, not `login()` | `login()` hangs in non-interactive shells |
| 2026-04-07 | Skip ASAP, parse CAMELYON16 XML with stdlib | No C++ dep; XML is simple polygon list |
| 2026-04-07 | Ignore CAMELYON17 ZIPs for now | Different task (per-patient staging) |
