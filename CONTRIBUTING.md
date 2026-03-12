# Contributing to PathoLens

İstanbul Teknik Üniversitesi — AI and Data Engineering Design I

## Team

| Role | Name | GitHub | Responsibility |
|------|------|--------|----------------|
| Language & Integration Lead | Emir Arda Eker | `itu-dallasli` | WP-2 (KARG), WP-4 (CSAL), WP-5 (Web & API) |
| Vision & Architecture Lead | Faruk Rıza Öz | *TBD* | WP-1 (Preprocessing + UNI), WP-3 (CMEA + RAAF) |

---

## Branching Strategy (Git Flow)

```
main ─────────────────────────────────────────── (stable releases)
 │
 └── develop ─────────────────────────────────── (integration branch)
      │
      ├── feature/wsi-preprocessing ──────────── (WP-1)
      ├── feature/patch-embedding ────────────── (WP-1)
      ├── feature/mamba-encoder ──────────────── (WP-3)
      ├── feature/cmea-retrieval ─────────────── (WP-3)
      ├── feature/raaf-explainability ────────── (WP-3)
      ├── feature/karg-entity-extraction ─────── (WP-2)
      ├── feature/csal-report-generation ─────── (WP-4)
      ├── feature/web-api ────────────────────── (WP-5)
      └── fix/issue-description ──────────────── (bug fixes)
```

### Branch Rules

| Branch | Purpose | Who merges | Protection |
|--------|---------|-----------|------------|
| `main` | Stable, tested releases | Both (via PR) | Require PR review + CI pass |
| `develop` | Integration of features | Both (via PR) | Require 1 review |
| `feature/*` | Individual features | Branch owner | None |
| `fix/*` | Bug fixes | Branch owner | None |

### Workflow

1. **Start work:** Create a feature branch from `develop`
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Develop:** Make commits with descriptive messages (see below)

3. **Push:** Push your branch to origin
   ```bash
   git push -u origin feature/your-feature-name
   ```

4. **Pull Request:** Open a PR from your feature branch → `develop`
   - Add the other team member as reviewer
   - Describe what changed and how to test

5. **Review & Merge:** Reviewer approves → merge to `develop`

6. **Release:** When `develop` is stable → PR to `main`

---

## Commit Message Convention

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types
- `feat` — New feature
- `fix` — Bug fix
- `refactor` — Code restructuring (no behavior change)
- `docs` — Documentation only
- `test` — Adding or fixing tests
- `chore` — Build config, dependencies, tooling
- `data` — Data pipeline or dataset changes

### Scopes
`preprocessing`, `embedding`, `sequence-model`, `retrieval`, `entity-extraction`, `explainability`, `report`, `pipeline`, `api`, `config`, `docs`

### Examples
```
feat(preprocessing): add tissue segmentation with Otsu thresholding
feat(embedding): implement UNI feature extractor with HuggingFace download
fix(retrieval): correct FAISS L2 normalization for cosine similarity
docs: update README with architecture diagram
chore: add pytest and ruff to dev dependencies
```

---

## Work Package Ownership

### Faruk (Vision & Architecture Lead)
Work on these directories:
- `src/patholens/preprocessing/`
- `src/patholens/embedding/`  
- `src/patholens/sequence_model/`
- `src/patholens/retrieval/`
- `src/patholens/explainability/`

### Emir (Language & Integration Lead)
Work on these directories:
- `src/patholens/entity_extraction/`
- `src/patholens/report_generation/`
- `src/patholens/api/`
- `src/patholens/pipeline/`
- `configs/`

### Shared
- `src/patholens/config.py`
- `src/patholens/logger.py`
- `tests/`
- `documentation/`

---

## Development Setup

```bash
# 1. Clone (or fork, then clone your fork)
git clone https://github.com/itu-dallasli/PathoLens.git
cd PathoLens

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install in dev mode
pip install -e ".[dev]"

# 4. Verify
python -c "from patholens import __version__; print(__version__)"

# 5. Run tests
pytest tests/ -v
```

## For Faruk (Fork Workflow)

```bash
# 1. Fork the repo on GitHub (click "Fork" button)

# 2. Clone YOUR fork
git clone https://github.com/FARUK_USERNAME/PathoLens.git
cd PathoLens

# 3. Add upstream remote
git remote add upstream https://github.com/itu-dallasli/PathoLens.git

# 4. Keep in sync
git fetch upstream
git checkout develop
git merge upstream/develop

# 5. Work on a feature branch
git checkout -b feature/your-feature
# ... make changes ...
git push origin feature/your-feature

# 6. Open PR from your fork → upstream/develop
```
