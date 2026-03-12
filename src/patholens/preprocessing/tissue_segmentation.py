"""
Tissue Segmentation — Separate tissue from background in a WSI thumbnail.

Uses Otsu thresholding on the HSV saturation channel followed by
morphological clean-up to produce a binary tissue mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from patholens.logger import get_logger

log = get_logger(__name__)


@dataclass
class SegmentationResult:
    """Result of tissue segmentation."""

    mask: np.ndarray                     # Binary mask (uint8, 0/255)
    contours: List[np.ndarray]           # Filtered tissue contours
    tissue_ratio: float                  # Fraction of thumbnail that is tissue
    thumbnail_size: Tuple[int, int]      # (H, W) of the mask


class TissueSegmentor:
    """
    HSV saturation + Otsu tissue segmentation.

    Parameters
    ----------
    median_blur_ksize : int
        Median-blur kernel size applied before thresholding.
    morph_kernel_size : int
        Structuring-element size for morphological open/close.
    min_contour_area : int
        Contours smaller than this (in thumbnail pixels²) are discarded.
    """

    def __init__(
        self,
        median_blur_ksize: int = 7,
        morph_kernel_size: int = 5,
        min_contour_area: int = 5000,
    ):
        self.median_blur_ksize = median_blur_ksize
        self.morph_kernel_size = morph_kernel_size
        self.min_contour_area = min_contour_area

    def segment(self, thumbnail: Image.Image | np.ndarray) -> SegmentationResult:
        """
        Produce a binary tissue mask from an RGB thumbnail.

        Steps
        -----
        1. RGB → HSV
        2. Median blur on saturation
        3. Otsu threshold
        4. Morphological close → open (fill holes, remove specks)
        5. Filter small contours

        Returns
        -------
        SegmentationResult
        """
        if isinstance(thumbnail, Image.Image):
            thumbnail = np.array(thumbnail)

        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        # Smooth before threshold
        if self.median_blur_ksize > 0:
            saturation = cv2.medianBlur(saturation, self.median_blur_ksize)

        # Otsu on saturation
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Extract and filter contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept: List[np.ndarray] = []
        clean_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_contour_area:
                kept.append(cnt)
                cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)

        h, w = clean_mask.shape[:2]
        tissue_ratio = float(clean_mask.sum() / 255) / (h * w)

        log.info(
            "Tissue segmentation  |  contours=%d  tissue_ratio=%.2f%%",
            len(kept),
            tissue_ratio * 100,
        )
        return SegmentationResult(
            mask=clean_mask,
            contours=kept,
            tissue_ratio=tissue_ratio,
            thumbnail_size=(h, w),
        )
