# SKILLS.md — PathoLens Operational Playbook

Recipes for common tasks. Copy-paste ready.

---

## 1. Run the pipeline on a real WSI (CPU)

```bash
# Lightweight: caps at 500 patches, ~2 minutes
python run_real.py X:/Bitirme_Data/normal_136.tif

# With custom slide_id
python run_real.py X:/Bitirme_Data/tumor_070.tif tumor_070
```

Outputs land in `results/<slide_id>/`:
- `diagnostic_report.json` — FHIR R4 DiagnosticReport
- `heatmap.png` — attention overlay on thumbnail

Config: `configs/local-run.yaml` (edit `max_patches` to trade speed vs. coverage).

## 2. Run the full test suite (CPU, no models downloaded)

```bash
pytest tests/ -v                    # ~10s, 105 tests
pytest tests/ --cov=patholens       # with coverage
ruff check src tests                # lint
ruff format --check src tests       # format check
```

Tests use synthetic fixtures from `tests/conftest.py` — **no WSI or HF downloads required**.

## 3. Start the API server

```bash
uvicorn patholens.api.main:app --reload --port 8000
curl http://localhost:8000/health   # {"status": "ok"}
```

Models lazy-load on first `/analyze` call, not at startup.

## 4. Install from scratch on a fresh Windows machine

```bash
# 1. Clone + create venv
python -m venv .venv
source .venv/Scripts/activate   # bash on Windows

# 2. Install CPU torch FIRST (pinned versions — order matters)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install the project
pip install -e ".[dev]"

# 4. OpenSlide DLL (Windows)
pip install openslide-bin

# 5. HF auth for UNI weights (gated model)
# Option A: env var
export HF_TOKEN=hf_xxx
# Option B: cached token
huggingface-cli login    # writes ~/.cache/huggingface/token
```

## 5. Debug pipeline on a specific slide

```python
from pathlib import Path
from patholens.config import Config
from patholens.pipeline.inference import PathoLensPipeline

config = Config.load("configs/local-run.yaml")
pipeline = PathoLensPipeline(config)
result = pipeline.run(Path("X:/Bitirme_Data/tumor_070.tif"))
print(result.status, result.num_patches, result.elapsed_seconds)
```

Step through sub-modules individually by accessing `pipeline._preprocessor`, `pipeline._batch_processor`, etc. — they're populated after first `run()` call due to lazy loading.

## 6. Parse CAMELYON16 XML ground-truth (no ASAP needed)

```python
import xml.etree.ElementTree as ET
tree = ET.parse("X:/Bitirme_Data/tumor_070.xml")
for ann in tree.findall(".//Annotation"):
    coords = [(float(c.get("X")), float(c.get("Y")))
              for c in ann.findall(".//Coordinate")]
    # coords is a polygon in WSI pixel space at level 0
```

Rasterize polygons to a binary mask with `cv2.fillPoly` or `PIL.ImageDraw.polygon`, then downsample to thumbnail resolution for visual comparison with `heatmap.png`.

## 7. Known pitfalls & fixes

| Symptom | Cause | Fix |
|---|---|---|
| `UnicodeEncodeError: 'charmap'` | Windows CP1254 can't print `━` | Use ASCII in logs; `logger.py` forces `force_terminal=False` |
| `huggingface_hub.login()` hangs forever | Non-interactive shell, no stdin | `uni_extractor.py` reads token file directly |
| `RuntimeError: operator torchvision::nms does not exist` | torch/torchvision version mismatch | `pip install torchvision==0.25.0` (match torch 2.10.0) |
| `ModuleNotFoundError: timm` | Optional dep not installed | `pip install timm huggingface-hub` |
| `faiss-gpu` install fails | No CUDA toolkit | Use `[dev]` extra (has `faiss-cpu`), not `[gpu]` |
| `ImportError: _openslide` | Missing OpenSlide DLL | `pip install openslide-bin` |
| Pipeline runs 2+ hours on CPU | All 32k patches embedding | Set `preprocessing.max_patches: 500` in config |
| `mamba-ssm` build fails | CUDA kernel compilation | It's optional; ensure it's NOT in main deps |
| Empty FHIR report | No FAISS index → no reference reports | Expected until Phase 4 roadmap |
| Heatmap is noise / corner-biased | Untrained attention + sequential patch cap | Expected until Phase 3 roadmap |

## 8. Dependency management rules

- **Main deps** (`pyproject.toml [project.dependencies]`): CPU-only, must install on any dev machine. No CUDA builds.
- **`[dev]` extra**: pytest, ruff, coverage.
- **`[gpu]` extra**: `mamba-ssm`, `causal-conv1d`, `faiss-gpu` — production only.
- Never move a CUDA-building package into main deps.

## 9. Git workflow

- Main branch: `main`. Develop on feature branches.
- Claude Code worktrees live in `.claude/worktrees/<name>` — each on its own branch (`claude/<name>`).
- CI runs lint + tests on push/PR to `main` or `develop`.

## 10. Where to look for X

| Need | File |
|---|---|
| Add a pipeline step | `src/patholens/pipeline/inference.py` |
| Change default config | `configs/default.yaml` + `src/patholens/config.py` |
| Mock a module for tests | `tests/conftest.py` |
| Add API endpoint | `src/patholens/api/main.py` |
| Full architecture doc | `documentation/about_project.md` |
| Training/production plan | `documentation/production_training_plan.md` |
