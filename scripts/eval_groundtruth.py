"""
Ground-truth evaluation harness for CAMELYON16 tumor slides.

Given a WSI (`tumor_XXX.tif`) and its ASAP XML annotation file
(`tumor_XXX.xml`), this script:

  1. Runs the PathoLens pipeline on the slide (or reuses existing results).
  2. Parses the XML polygons and rasterises them into a binary GT mask
     at the same resolution as the generated heatmap.
  3. Thresholds the predicted attention heatmap.
  4. Computes IoU, Dice, Precision, Recall.
  5. Saves a 4-panel comparison PNG:
       thumbnail | GT overlay | predicted heatmap | diff map.

Usage
-----
    python scripts/eval_groundtruth.py X:/Bitirme_Data/tumor_070.tif
    python scripts/eval_groundtruth.py X:/Bitirme_Data/tumor_070.tif --threshold 0.5
    python scripts/eval_groundtruth.py X:/Bitirme_Data/tumor_070.tif --reuse

Outputs land in `results/<slide_id>/`:
    comparison.png
    groundtruth_mask.png
    metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from patholens.config import Config
from patholens.logger import get_logger
from patholens.pipeline.inference import PathoLensPipeline

log = get_logger(__name__)


# ── Data types ──────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    slide_id: str
    threshold: float
    iou: float
    dice: float
    precision: float
    recall: float
    gt_pixels: int
    pred_pixels: int
    intersection_pixels: int
    resolution: int


# ── XML parsing ─────────────────────────────────────────────────

def load_official_mask(
    mask_path: Path,
    out_resolution: int,
    prefer_level: Optional[int] = None,
) -> np.ndarray:
    """
    Load an official CAMELYON16 mask TIF and return a binary tumor mask
    at `out_resolution` x `out_resolution` (uint8, {0, 255}).

    CAMELYON16 mask encoding (per dataset README):
        0 = background (non-tissue)
        1 = normal tissue
        2 = tumor

    We isolate class 2 (tumor) and discard the rest.

    Parameters
    ----------
    mask_path : Path
        Path to the mask TIF.
    out_resolution : int
        Edge length of the returned mask.
    prefer_level : int, optional
        Pyramid level to read. Defaults to a level whose edge is roughly
        ``2 * out_resolution`` so we don't lose sub-pixel tumors to
        aggressive downsampling (important for micro-metastasis slides
        like ``tumor_070``).
    """
    from patholens.preprocessing.wsi_reader import WSIReader
    from openslide.lowlevel import OpenSlideError

    # Capture pyramid info up-front so a later read error doesn't poison
    # the OpenSlide handle before we know the level dimensions.
    with WSIReader(mask_path) as reader:
        level_count = reader.level_count
        level_dims = list(reader.level_dimensions)

    target = 2 * out_resolution
    if prefer_level is not None:
        candidate_start = max(0, min(prefer_level, level_count - 1))
    else:
        candidate_start = level_count - 1
        for lvl in range(level_count):
            w, h = level_dims[lvl]
            if max(w, h) <= target:
                candidate_start = lvl
                break

    candidates = list(range(candidate_start, level_count))
    arr = None
    chosen_level: Optional[int] = None
    last_err: Optional[Exception] = None
    for lvl in candidates:
        w, h = level_dims[lvl]
        try:
            # Reopen per attempt — OpenSlide caches fatal errors on the handle.
            with WSIReader(mask_path) as reader:
                region = reader.read_region((0, 0), lvl, (w, h))
                arr = np.array(region.convert("L"))
                chosen_level = lvl
            break
        except OpenSlideError as e:
            last_err = e
            log.warning(
                "mask read failed at level %d (%dx%d): %s -- trying coarser",
                lvl, w, h, e,
            )

    if arr is None:
        raise RuntimeError(
            f"Could not read mask {mask_path.name} at any level: {last_err}"
        )
    level = chosen_level
    tumor = (arr == 2).astype(np.uint8) * 255
    tissue_pixels = int((arr == 1).sum())
    tumor_pixels = int((arr == 2).sum())

    resized = cv2.resize(
        tumor, (out_resolution, out_resolution), interpolation=cv2.INTER_NEAREST
    )
    log.info(
        "Loaded official mask %s  level=%d  raw_shape=%s  "
        "tumor_px=%d  tissue_px=%d  tumor_after_resize=%d",
        mask_path.name, level, tuple(arr.shape),
        tumor_pixels, tissue_pixels, int((resized > 0).sum()),
    )
    return resized


def parse_asap_xml(xml_path: Path) -> List[np.ndarray]:
    """
    Parse an ASAP annotation XML into a list of polygons.

    Each polygon is an (N, 2) float array of (x, y) coordinates in
    level-0 WSI pixel space.
    """
    tree = ET.parse(xml_path)
    polygons: List[np.ndarray] = []
    for ann in tree.findall(".//Annotation"):
        coords = []
        for c in ann.findall(".//Coordinate"):
            coords.append((float(c.get("X")), float(c.get("Y"))))
        if len(coords) >= 3:
            polygons.append(np.array(coords, dtype=np.float32))
    log.info("Parsed %d polygons from %s", len(polygons), xml_path.name)
    return polygons


def rasterize_polygons(
    polygons: List[np.ndarray],
    wsi_dimensions: Tuple[int, int],
    out_resolution: int,
) -> np.ndarray:
    """
    Rasterise polygons into a binary mask at `out_resolution`x`out_resolution`.

    Parameters
    ----------
    polygons : list of (N, 2) arrays in level-0 pixel coords
    wsi_dimensions : (width, height) at level 0
    out_resolution : output edge length (matches heatmap resolution)

    Returns
    -------
    uint8 mask, shape (out_resolution, out_resolution), values {0, 255}
    """
    wsi_w, wsi_h = wsi_dimensions
    scale_x = out_resolution / wsi_w
    scale_y = out_resolution / wsi_h

    mask = np.zeros((out_resolution, out_resolution), dtype=np.uint8)
    for poly in polygons:
        scaled = np.stack(
            [poly[:, 0] * scale_x, poly[:, 1] * scale_y], axis=1
        ).astype(np.int32)
        cv2.fillPoly(mask, [scaled], color=255)
    return mask


# ── Metrics ─────────────────────────────────────────────────────

def compute_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Tuple[float, float, float, float, int, int, int]:
    """
    Compute IoU, Dice, precision, recall on two binary masks (bool).
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    iou = inter / union if union > 0 else 0.0
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    precision = inter / pred_sum if pred_sum > 0 else 0.0
    recall = inter / gt_sum if gt_sum > 0 else 0.0

    return (
        float(iou),
        float(dice),
        float(precision),
        float(recall),
        int(gt_sum),
        int(pred_sum),
        int(inter),
    )


# ── Visualisation ───────────────────────────────────────────────

def build_comparison_figure(
    thumbnail: np.ndarray,      # (R, R, 3) uint8
    gt_mask: np.ndarray,        # (R, R) uint8 {0,255}
    heatmap_gray: np.ndarray,   # (R, R) float [0,1]
    pred_mask: np.ndarray,      # (R, R) uint8 {0,255}
    metrics: EvalMetrics,
) -> np.ndarray:
    """
    Build a 4-panel comparison image (stacked horizontally).
    Returns an (R, 4R + 3*gap, 3) uint8 RGB array.
    """
    R = thumbnail.shape[0]
    gap = 10
    gap_col = np.full((R, gap, 3), 255, dtype=np.uint8)

    # Panel 1: thumbnail
    p1 = thumbnail.copy()

    # Panel 2: thumbnail with GT polygon overlay (red)
    p2 = thumbnail.copy()
    gt_overlay = np.zeros_like(p2)
    gt_overlay[gt_mask > 0] = (255, 0, 0)
    p2 = cv2.addWeighted(p2, 0.7, gt_overlay, 0.3, 0)
    # Draw GT contours for a crisper edge
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2, contours, -1, (255, 0, 0), 2)

    # Panel 3: predicted heatmap overlay (jet)
    hm_uint8 = (np.clip(heatmap_gray, 0, 1) * 255).astype(np.uint8)
    hm_bgr = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)
    p3 = cv2.addWeighted(thumbnail, 0.5, hm_rgb, 0.5, 0)

    # Panel 4: diff map — TP green, FP blue, FN red
    p4 = thumbnail.copy()
    diff = np.zeros_like(p4)
    tp = np.logical_and(pred_mask > 0, gt_mask > 0)
    fp = np.logical_and(pred_mask > 0, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask > 0)
    diff[tp] = (0, 255, 0)
    diff[fp] = (0, 128, 255)
    diff[fn] = (255, 0, 0)
    p4 = cv2.addWeighted(p4, 0.55, diff, 0.45, 0)

    # Labels
    def _label(img, text):
        cv2.rectangle(img, (0, 0), (R, 28), (0, 0, 0), -1)
        cv2.putText(
            img, text, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    _label(p1, "Thumbnail")
    _label(p2, "Ground Truth (red)")
    _label(p3, "Predicted Heatmap")
    _label(
        p4,
        f"Diff  IoU={metrics.iou:.3f}  Dice={metrics.dice:.3f}",
    )

    return np.concatenate([p1, gap_col, p2, gap_col, p3, gap_col, p4], axis=1)


# ── Main flow ───────────────────────────────────────────────────

def run_pipeline(wsi_path: Path, slide_id: str) -> Path:
    """Run the pipeline and return the results dir."""
    config = Config.load("configs/local-run.yaml")
    pipeline = PathoLensPipeline(config)
    result = pipeline.run(wsi_path=wsi_path, slide_id=slide_id)
    log.info(
        "Pipeline done: status=%s patches=%d elapsed=%.1fs",
        result.status, result.num_patches, result.elapsed_seconds,
    )
    return Path(config.paths.results_dir) / slide_id


def load_thumbnail_and_dims(
    wsi_path: Path, resolution: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Open the WSI once to pull a thumbnail + level-0 dimensions."""
    from patholens.preprocessing.wsi_reader import WSIReader

    with WSIReader(wsi_path) as reader:
        dims = reader.dimensions  # (w, h) level 0
        thumb = reader.get_thumbnail((resolution, resolution))
    thumb_arr = np.array(thumb.convert("RGB").resize((resolution, resolution)))
    return thumb_arr, dims


def evaluate(
    wsi_path: Path,
    xml_path: Optional[Path],
    results_dir: Path,
    threshold: float,
    resolution: int,
    mask_path: Optional[Path] = None,
) -> EvalMetrics:
    slide_id = results_dir.name

    # Prefer the raw attention map (.npy) saved by the pipeline; fall back
    # to reverse-engineering from the overlay PNG if absent.
    raw_path = results_dir / "heatmap_raw.npy"
    overlay_path = results_dir / "heatmap.png"

    # Load thumbnail + WSI dims (needed regardless)
    thumbnail, wsi_dims = load_thumbnail_and_dims(wsi_path, resolution)

    if raw_path.exists():
        heat_norm = np.load(raw_path).astype(np.float32)
        if heat_norm.shape[0] != resolution:
            heat_norm = cv2.resize(
                heat_norm, (resolution, resolution), interpolation=cv2.INTER_LINEAR
            )
        # Re-normalise to [0, 1] in case of upstream changes
        if heat_norm.max() > heat_norm.min():
            heat_norm = (heat_norm - heat_norm.min()) / (
                heat_norm.max() - heat_norm.min()
            )
    elif overlay_path.exists():
        log.warning(
            "heatmap_raw.npy not found; falling back to overlay PNG (lossy)"
        )
        heatmap_rgb = np.array(Image.open(overlay_path).convert("RGB"))
        if heatmap_rgb.shape[0] != resolution:
            heatmap_rgb = cv2.resize(
                heatmap_rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR
            )
        diff = heatmap_rgb.astype(np.int16) - thumbnail.astype(np.int16)
        heat_signal = np.linalg.norm(diff, axis=-1)
        if heat_signal.max() > heat_signal.min():
            heat_norm = (heat_signal - heat_signal.min()) / (
                heat_signal.max() - heat_signal.min()
            )
        else:
            heat_norm = np.zeros_like(heat_signal, dtype=np.float32)
        heat_norm = heat_norm.astype(np.float32)
    else:
        raise FileNotFoundError(
            f"No heatmap at {raw_path} or {overlay_path}. "
            f"Run the pipeline first (omit --reuse)."
        )

    pred_mask = (heat_norm >= threshold).astype(np.uint8) * 255

    # Ground truth: prefer official binary mask TIF if available, else
    # fall back to rasterising the ASAP XML polygons.
    if mask_path is not None and mask_path.exists():
        gt_mask = load_official_mask(mask_path, resolution)
    elif xml_path is not None and xml_path.exists():
        polygons = parse_asap_xml(xml_path)
        gt_mask = rasterize_polygons(polygons, wsi_dims, resolution)
    else:
        raise FileNotFoundError(
            "Neither official mask TIF nor ASAP XML annotation provided."
        )

    # Metrics
    iou, dice, prec, rec, gt_px, pred_px, inter_px = compute_metrics(
        pred_mask, gt_mask
    )
    metrics = EvalMetrics(
        slide_id=slide_id,
        threshold=threshold,
        iou=iou,
        dice=dice,
        precision=prec,
        recall=rec,
        gt_pixels=gt_px,
        pred_pixels=pred_px,
        intersection_pixels=inter_px,
        resolution=resolution,
    )

    # Persist artefacts
    Image.fromarray(gt_mask).save(results_dir / "groundtruth_mask.png")

    comparison = build_comparison_figure(
        thumbnail, gt_mask, heat_norm, pred_mask, metrics
    )
    Image.fromarray(comparison).save(results_dir / "comparison.png")

    (results_dir / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2), encoding="utf-8"
    )

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wsi_path", type=Path, help="Path to tumor_*.tif")
    parser.add_argument(
        "--xml", type=Path, default=None,
        help="Path to ASAP XML annotation (defaults to sibling .xml). "
             "Only used if --mask is not provided.",
    )
    parser.add_argument(
        "--mask", type=Path, default=None,
        help="Path to official binary mask TIF "
             "(e.g. CAMELYON16/masks/tumor_070_mask.tif). "
             "Preferred over XML when available.",
    )
    parser.add_argument(
        "--slide-id", type=str, default=None,
        help="Slide ID (defaults to file stem)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Heatmap binarisation threshold in [0,1] (default 0.5)",
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Evaluation raster edge length (default 1024)",
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="Skip pipeline run; reuse existing results/<slide_id>/heatmap.png",
    )
    args = parser.parse_args()

    wsi_path: Path = args.wsi_path
    if not wsi_path.exists():
        log.error("WSI not found: %s", wsi_path)
        return 1

    xml_path: Optional[Path] = args.xml or wsi_path.with_suffix(".xml")
    mask_path: Optional[Path] = args.mask

    if (mask_path is None or not mask_path.exists()) and (
        xml_path is None or not xml_path.exists()
    ):
        log.error(
            "No ground truth found. Pass --mask <mask.tif> or --xml <ann.xml>, "
            "or place a sibling .xml next to the WSI."
        )
        return 1

    slide_id = args.slide_id or wsi_path.stem

    if args.reuse:
        config = Config.load("configs/local-run.yaml")
        results_dir = Path(config.paths.results_dir) / slide_id
        if not results_dir.exists():
            log.error(
                "No existing results at %s. Drop --reuse to run the pipeline.",
                results_dir,
            )
            return 1
    else:
        results_dir = run_pipeline(wsi_path, slide_id)

    metrics = evaluate(
        wsi_path=wsi_path,
        xml_path=xml_path,
        results_dir=results_dir,
        threshold=args.threshold,
        resolution=args.resolution,
        mask_path=mask_path,
    )

    print("\n" + "=" * 50)
    print(f"  slide_id    : {metrics.slide_id}")
    print(f"  threshold   : {metrics.threshold:.2f}")
    print(f"  IoU         : {metrics.iou:.4f}")
    print(f"  Dice        : {metrics.dice:.4f}")
    print(f"  Precision   : {metrics.precision:.4f}")
    print(f"  Recall      : {metrics.recall:.4f}")
    print(f"  GT pixels   : {metrics.gt_pixels}")
    print(f"  Pred pixels : {metrics.pred_pixels}")
    print(f"  Comparison  : {results_dir / 'comparison.png'}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
