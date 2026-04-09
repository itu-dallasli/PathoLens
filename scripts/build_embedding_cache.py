"""
Build a UNI embedding cache for slide-level training.

For every WSI in the training set, this script:
  1. Segments tissue, extracts patches, random-samples up to `max_patches`.
  2. Runs the frozen UNI ViT-L/16 feature extractor on the patches.
  3. Saves a single ``.npz`` per slide containing::

        embeddings   (N, 1024)  float32   UNI features
        coords       (N, 2)     int32     level-0 (x, y) of each patch
        label        ()         int32     0 = normal, 1 = tumor
        wsi_dims     (2,)       int32     (width, height) at level 0
        patch_size   ()         int32
        level        ()         int32

Once built, training reads from this cache directly and never touches
the WSIs or UNI again.

Usage
-----
    python scripts/build_embedding_cache.py \\
        --tumor-dir X:/Bitirme_Data/tumor \\
        --normal-dir X:/Bitirme_Data/normal \\
        --out data/processed/slide_cache \\
        --max-patches 4000

    # Dry-run to see which slides would be processed:
    python scripts/build_embedding_cache.py --list
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from patholens.config import Config
from patholens.logger import get_logger
from patholens.preprocessing.wsi_reader import WSIReader

log = get_logger(__name__)


@dataclass
class SlideJob:
    wsi_path: Path
    slide_id: str
    label: int  # 0 normal, 1 tumor


# ── Discovery ────────────────────────────────────────────────

def collect_jobs(
    tumor_dir: Optional[Path],
    normal_dir: Optional[Path],
    tumor_limit: Optional[int],
    normal_limit: Optional[int],
) -> List[SlideJob]:
    jobs: List[SlideJob] = []

    if tumor_dir and tumor_dir.exists():
        tumors = sorted(tumor_dir.glob("tumor_*.tif"))
        if tumor_limit:
            tumors = tumors[:tumor_limit]
        for p in tumors:
            jobs.append(SlideJob(wsi_path=p, slide_id=p.stem, label=1))

    if normal_dir and normal_dir.exists():
        normals = sorted(normal_dir.glob("normal_*.tif"))
        if normal_limit:
            normals = normals[:normal_limit]
        for p in normals:
            jobs.append(SlideJob(wsi_path=p, slide_id=p.stem, label=0))

    return jobs


# ── Per-slide processing ─────────────────────────────────────

def process_slide(
    job: SlideJob,
    out_dir: Path,
    pipeline,
    max_patches: int,
    seed_base: int = 0,
    overwrite: bool = False,
) -> dict:
    """Run preprocessing + UNI on one slide and write ``<slide_id>.npz``."""
    out_path = out_dir / f"{job.slide_id}.npz"
    if out_path.exists() and not overwrite:
        data = np.load(out_path)
        return {
            "status": "skipped",
            "n_patches": int(data["embeddings"].shape[0]),
            "path": str(out_path),
        }

    t0 = time.time()

    with WSIReader(job.wsi_path) as reader:
        wsi_dims = reader.dimensions  # (w, h)
        thumbnail = reader.get_thumbnail(
            (pipeline.config.preprocessing.thumbnail_size,) * 2
        )
        seg_result = pipeline._segmentor.segment(thumbnail)
        extraction = pipeline._preprocessor.extract(reader, seg_result.mask)

        if extraction.num_patches == 0:
            return {"status": "empty_tissue", "n_patches": 0, "path": None}

        # Seeded random sample — per-slide seed for reproducibility
        total = extraction.num_patches
        if max_patches and total > max_patches:
            rng = np.random.default_rng(seed_base + abs(hash(job.slide_id)) % (2**31))
            idx = rng.choice(total, size=max_patches, replace=False)
            idx.sort()
            coords = extraction.coordinates[idx]
        else:
            coords = extraction.coordinates
        n = len(coords)

        # Run UNI on the patches
        embeddings = pipeline._batch_processor.process_slide(
            slide_id=job.slide_id,
            wsi_path=job.wsi_path,
            coordinates=coords,
            patch_size=extraction.patch_size,
            level=extraction.level,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        coords=coords.astype(np.int32),
        label=np.int32(job.label),
        wsi_dims=np.array(wsi_dims, dtype=np.int32),
        patch_size=np.int32(extraction.patch_size),
        level=np.int32(extraction.level),
    )

    elapsed = time.time() - t0
    return {
        "status": "done",
        "n_patches": n,
        "total_patches": total,
        "elapsed": elapsed,
        "path": str(out_path),
        "embeddings_dim": int(embeddings.shape[1]),
    }


# ── Main ─────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tumor-dir", type=Path, default=Path("X:/Bitirme_Data/tumor"),
    )
    parser.add_argument(
        "--normal-dir", type=Path, default=Path("X:/Bitirme_Data/normal"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/slide_cache"),
        help="Directory to write <slide_id>.npz files",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/local-run.yaml"),
    )
    parser.add_argument(
        "--max-patches", type=int, default=4000,
        help="Random sample cap per slide (default 4000)",
    )
    parser.add_argument(
        "--tumor-limit", type=int, default=None,
        help="Cap number of tumor slides processed",
    )
    parser.add_argument(
        "--normal-limit", type=int, default=None,
        help="Cap number of normal slides processed",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process slides even if a .npz already exists",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List jobs and exit without running",
    )
    args = parser.parse_args()

    jobs = collect_jobs(
        args.tumor_dir, args.normal_dir, args.tumor_limit, args.normal_limit
    )
    if not jobs:
        log.error("No slides found in %s or %s", args.tumor_dir, args.normal_dir)
        return 1

    n_tumor = sum(1 for j in jobs if j.label == 1)
    n_normal = sum(1 for j in jobs if j.label == 0)
    print(f"Jobs: {len(jobs)} slides  (tumor={n_tumor}  normal={n_normal})")
    print(f"Cache dir: {args.out}")
    print(f"Max patches per slide: {args.max_patches}")
    print()

    if args.list:
        for j in jobs:
            print(f"  [{'T' if j.label else 'N'}] {j.slide_id}  <- {j.wsi_path}")
        return 0

    # Build a minimal pipeline (loads UNI once, reuse across slides)
    print("Loading pipeline modules (UNI ~303M params) ...")
    from patholens.pipeline.inference import PathoLensPipeline

    cfg = Config.load(args.config)
    pipeline = PathoLensPipeline(cfg)
    pipeline._load_modules()
    print("  ready.\n")

    args.out.mkdir(parents=True, exist_ok=True)

    stats = {"done": 0, "skipped": 0, "failed": 0, "empty_tissue": 0}
    total_patches = 0
    t_start = time.time()

    for i, job in enumerate(jobs, start=1):
        prefix = f"[{i}/{len(jobs)}]"
        try:
            result = process_slide(
                job, args.out, pipeline,
                max_patches=args.max_patches,
                overwrite=args.overwrite,
            )
        except Exception as e:
            log.error("%s %s FAILED: %s", prefix, job.slide_id, e)
            traceback.print_exc()
            stats["failed"] += 1
            continue

        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        n = result.get("n_patches", 0)
        total_patches += n

        if status == "done":
            print(
                f"{prefix} {job.slide_id:20s} [{'T' if job.label else 'N'}]  "
                f"patches={n:>5}/{result.get('total_patches', n):<6}  "
                f"dim={result.get('embeddings_dim', 0)}  "
                f"elapsed={result.get('elapsed', 0):6.1f}s"
            )
        elif status == "skipped":
            print(f"{prefix} {job.slide_id:20s} [skipped, cached] n={n}")
        elif status == "empty_tissue":
            print(f"{prefix} {job.slide_id:20s} [empty, no tissue found]")

    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Done in {elapsed_total/60:.1f} min")
    print(f"  done:         {stats.get('done', 0)}")
    print(f"  skipped:      {stats.get('skipped', 0)}")
    print(f"  empty_tissue: {stats.get('empty_tissue', 0)}")
    print(f"  failed:       {stats.get('failed', 0)}")
    print(f"  total patches embedded: {total_patches}")
    print(f"  cache dir: {args.out}")
    return 0 if stats.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
