"""
Patch Extraction — Grid-based patch coordinate extraction from tissue regions.

Generates (x, y) coordinates at level 0 for every patch whose
tissue content exceeds a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from patholens.logger import get_logger
from patholens.preprocessing.wsi_reader import WSIReader

log = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Container for patch extraction output."""

    coordinates: np.ndarray           # (N, 2) int64 — level-0 (x, y)
    patch_size: int                   # Pixel side at target magnification
    level: int                        # Pyramid level used
    magnification: float              # Actual magnification
    num_patches: int = field(init=False)

    def __post_init__(self):
        self.num_patches = len(self.coordinates)


class PatchExtractor:
    """
    Extract patch coordinates over tissue regions of a WSI.

    Parameters
    ----------
    patch_size : int
        Side length of square patches (pixels at target magnification).
    magnification : float
        Target magnification (e.g. 20.0 for 20×).
    tissue_threshold : float
        Minimum fraction of tissue inside a patch (0–1).
    overlap : int
        Pixel overlap between adjacent patches.  0 = no overlap.
    """

    def __init__(
        self,
        patch_size: int = 256,
        magnification: float = 20.0,
        tissue_threshold: float = 0.5,
        overlap: int = 0,
    ):
        self.patch_size = patch_size
        self.magnification = magnification
        self.tissue_threshold = tissue_threshold
        self.overlap = overlap

    def extract(
        self,
        wsi_reader: WSIReader,
        tissue_mask: np.ndarray,
    ) -> ExtractionResult:
        """
        Compute patch coordinates.

        Parameters
        ----------
        wsi_reader : WSIReader
            Opened slide.
        tissue_mask : np.ndarray
            Binary mask (H_thumb × W_thumb, uint8 0/255) from
            :class:`TissueSegmentor`.

        Returns
        -------
        ExtractionResult
        """
        level = wsi_reader.get_best_level_for_magnification(self.magnification)
        downsample = wsi_reader.level_downsamples[level]
        level_w, level_h = wsi_reader.level_dimensions[level]

        # Scale factors between tissue mask and the target level
        mask_h, mask_w = tissue_mask.shape[:2]
        scale_x = mask_w / level_w
        scale_y = mask_h / level_h

        step = self.patch_size - self.overlap
        coords: List[Tuple[int, int]] = []

        for y in range(0, level_h - self.patch_size + 1, step):
            for x in range(0, level_w - self.patch_size + 1, step):
                # Map patch region → mask space
                mx0 = int(x * scale_x)
                mx1 = int((x + self.patch_size) * scale_x)
                my0 = int(y * scale_y)
                my1 = int((y + self.patch_size) * scale_y)

                # Clamp
                mx1 = min(mx1, mask_w)
                my1 = min(my1, mask_h)

                patch_mask = tissue_mask[my0:my1, mx0:mx1]
                if patch_mask.size == 0:
                    continue

                tissue_frac = (patch_mask > 0).sum() / patch_mask.size
                if tissue_frac >= self.tissue_threshold:
                    # Convert level coords → level-0 coords
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    coords.append((x0, y0))

        coordinates = np.array(coords, dtype=np.int64) if coords else np.empty((0, 2), dtype=np.int64)

        log.info(
            "Patch extraction  |  level=%d  downsample=%.1f  patches=%d  "
            "grid_step=%d  tissue_thresh=%.0f%%",
            level,
            downsample,
            len(coordinates),
            step,
            self.tissue_threshold * 100,
        )
        return ExtractionResult(
            coordinates=coordinates,
            patch_size=self.patch_size,
            level=level,
            magnification=self.magnification,
        )
