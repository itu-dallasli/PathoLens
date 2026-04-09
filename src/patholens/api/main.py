"""
PathoLens FastAPI Application — Web service for WSI analysis.

Endpoints:
  POST /api/upload          Upload a WSI file
  POST /api/analyze/{id}    Start analysis (async)
  GET  /api/status/{id}     Check analysis progress
  GET  /api/results/{id}    Get full results
  GET  /api/heatmap/{id}    Get heatmap image
  GET  /api/report/{id}     Get FHIR DiagnosticReport JSON
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from patholens.config import Config, PROJECT_ROOT
from patholens.logger import get_logger

log = get_logger(__name__)

# ── App setup ────────────────────────────────────────────────
config = Config.load()

app = FastAPI(
    title="PathoLens API",
    description="Traceable Retrieval-Based Clinical Decision Support for Breast Cancer Histopathology",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins if hasattr(config.api, "cors_origins") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ──────────────────────────────────────
_jobs: Dict[str, Dict] = {}

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Pydantic models ─────────────────────────────────────────
class AnalysisStatus(BaseModel):
    job_id: str
    slide_id: str
    status: str  # pending | running | completed | failed | low_confidence
    progress: Optional[str] = None
    elapsed_seconds: Optional[float] = None


class AnalysisResult(BaseModel):
    job_id: str
    slide_id: str
    status: str
    num_patches: int = 0
    elapsed_seconds: float = 0.0
    evidence_coverage: float = 0.0
    similar_cases: list = []
    heatmap_url: Optional[str] = None
    report_url: Optional[str] = None


# ── Background task ─────────────────────────────────────────
def _run_analysis(job_id: str, wsi_path: Path, slide_id: str) -> None:
    """Run the analysis pipeline in the background."""
    from patholens.pipeline.inference import PathoLensPipeline

    _jobs[job_id]["status"] = "running"

    try:
        pipeline = PathoLensPipeline(config)
        result = pipeline.run(
            wsi_path=wsi_path,
            slide_id=slide_id,
            output_dir=RESULTS_DIR / slide_id,
        )

        _jobs[job_id].update({
            "status": result.status,
            "num_patches": result.num_patches,
            "elapsed_seconds": result.elapsed_seconds,
            "evidence_coverage": result.evidence_coverage,
            "similar_cases": result.similar_cases,
            "fhir_report": result.fhir_report,
            "heatmap_path": result.heatmap_path,
        })

    except Exception as e:
        log.error("Analysis failed for job %s: %s", job_id, e, exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# ── Endpoints ────────────────────────────────────────────────
@app.post("/api/upload", response_model=dict)
async def upload_wsi(file: UploadFile):
    """Upload a WSI file for analysis."""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    slide_id = Path(file.filename).stem
    dest = UPLOAD_DIR / file.filename

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    log.info("Uploaded WSI: %s -> %s", file.filename, dest)
    return {"slide_id": slide_id, "path": str(dest), "size_mb": dest.stat().st_size / 1e6}


@app.post("/api/analyze/{slide_id}", response_model=AnalysisStatus)
async def start_analysis(slide_id: str, background_tasks: BackgroundTasks):
    """Start an analysis job for a previously uploaded slide."""
    # Find the WSI file
    matches = list(UPLOAD_DIR.glob(f"{slide_id}.*"))
    if not matches:
        raise HTTPException(404, f"No uploaded file found for slide '{slide_id}'")

    wsi_path = matches[0]
    job_id = str(uuid.uuid4())[:8]

    _jobs[job_id] = {
        "job_id": job_id,
        "slide_id": slide_id,
        "status": "pending",
        "wsi_path": str(wsi_path),
    }

    background_tasks.add_task(_run_analysis, job_id, wsi_path, slide_id)
    log.info("Started analysis job %s for slide %s", job_id, slide_id)

    return AnalysisStatus(job_id=job_id, slide_id=slide_id, status="pending")


@app.get("/api/status/{job_id}", response_model=AnalysisStatus)
async def get_status(job_id: str):
    """Check the status of an analysis job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    job = _jobs[job_id]
    return AnalysisStatus(
        job_id=job_id,
        slide_id=job["slide_id"],
        status=job["status"],
        elapsed_seconds=job.get("elapsed_seconds"),
    )


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get full analysis results."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    job = _jobs[job_id]
    if job["status"] not in ("completed", "low_confidence"):
        raise HTTPException(202, f"Analysis still {job['status']}")

    return JSONResponse({
        "job_id": job_id,
        "slide_id": job["slide_id"],
        "status": job["status"],
        "num_patches": job.get("num_patches", 0),
        "elapsed_seconds": job.get("elapsed_seconds", 0),
        "evidence_coverage": job.get("evidence_coverage", 0),
        "similar_cases": job.get("similar_cases", []),
        "heatmap_url": f"/api/heatmap/{job_id}" if job.get("heatmap_path") else None,
        "report_url": f"/api/report/{job_id}",
    })


@app.get("/api/heatmap/{job_id}")
async def get_heatmap(job_id: str):
    """Download the attention heatmap image."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    path = _jobs[job_id].get("heatmap_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Heatmap not available")

    return FileResponse(path, media_type="image/png")


@app.get("/api/report/{job_id}")
async def get_report(job_id: str):
    """Get the FHIR DiagnosticReport JSON."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job '{job_id}' not found")

    report = _jobs[job_id].get("fhir_report")
    if not report:
        raise HTTPException(404, "Report not available")

    return JSONResponse(report)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
