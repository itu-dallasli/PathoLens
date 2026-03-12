"""
Heatmap Generator — Create WSI-overlaid attention heatmaps.

Takes patch coordinates and their attention weights, produces
a smooth heatmap image that can be overlaid on the WSI thumbnail.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from patholens.logger import get_logger

log = get_logger(__name__)


class HeatmapGenerator:
    """
    Generate heatmap visualisations from patch attention weights.

    Parameters
    ----------
    resolution : int
        Output heatmap size (edge pixels).
    colormap : str
        OpenCV colormap name: ``"jet"``, ``"viridis"``, ``"inferno"``.
    overlay_alpha : float
        Transparency of the heatmap overlay (0=transparent, 1=opaque).
    gaussian_sigma : float
        Gaussian blur sigma for smoothing.
    """

    _COLORMAPS = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "hot": cv2.COLORMAP_HOT,
        "turbo": cv2.COLORMAP_TURBO,
    }

    def __init__(
        self,
        resolution: int = 1024,
        colormap: str = "jet",
        overlay_alpha: float = 0.4,
        gaussian_sigma: float = 5.0,
    ):
        self.resolution = resolution
        self.colormap = self._COLORMAPS.get(colormap, cv2.COLORMAP_JET)
        self.overlay_alpha = overlay_alpha
        self.gaussian_sigma = gaussian_sigma

    def generate(
        self,
        wsi_dimensions: Tuple[int, int],
        patch_coords: np.ndarray,
        attention_weights: np.ndarray,
        patch_size: int = 256,
        thumbnail: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Create a heatmap from attention weights.

        Parameters
        ----------
        wsi_dimensions : (width, height)
            Level-0 dimensions of the WSI.
        patch_coords : (N, 2) int
            Level-0 (x, y) coordinates.
        attention_weights : (N,) float
            Normalised attention values.
        patch_size : int
            Patch side length at extraction level.
        thumbnail : PIL.Image, optional
            WSI thumbnail for overlay.

        Returns
        -------
        (overlay_image, raw_heatmap)
        """
        wsi_w, wsi_h = wsi_dimensions
        res = self.resolution

        # Scale factors
        scale_x = res / wsi_w
        scale_y = res / wsi_h
        scaled_patch_w = max(1, int(patch_size * scale_x))
        scaled_patch_h = max(1, int(patch_size * scale_y))

        # Build raw attention map
        attn_map = np.zeros((res, res), dtype=np.float32)
        count_map = np.zeros((res, res), dtype=np.float32)

        for (x, y), weight in zip(patch_coords, attention_weights):
            sx = int(x * scale_x)
            sy = int(y * scale_y)
            ex = min(sx + scaled_patch_w, res)
            ey = min(sy + scaled_patch_h, res)
            attn_map[sy:ey, sx:ex] += weight
            count_map[sy:ey, sx:ex] += 1.0

        # Average overlapping regions
        mask = count_map > 0
        attn_map[mask] /= count_map[mask]

        # Gaussian smoothing
        if self.gaussian_sigma > 0:
            ksize = int(self.gaussian_sigma * 6) | 1  # ensure odd
            attn_map = cv2.GaussianBlur(attn_map, (ksize, ksize), self.gaussian_sigma)

        # Normalise to 0-255
        if attn_map.max() > attn_map.min():
            attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        else:
            attn_norm = np.zeros_like(attn_map)
        attn_uint8 = (attn_norm * 255).astype(np.uint8)

        # Apply colormap
        heatmap_bgr = cv2.applyColorMap(attn_uint8, self.colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Overlay on thumbnail if available
        if thumbnail is not None:
            thumb_resized = np.array(thumbnail.resize((res, res)))
            overlay = cv2.addWeighted(
                thumb_resized, 1 - self.overlay_alpha,
                heatmap_rgb, self.overlay_alpha,
                0,
            )
            overlay_img = Image.fromarray(overlay)
        else:
            overlay_img = Image.fromarray(heatmap_rgb)

        log.info(
            "Heatmap generated  |  res=%dx%d  patches=%d  "
            "attn_range=[%.4f, %.4f]",
            res, res, len(patch_coords),
            float(attention_weights.min()), float(attention_weights.max()),
        )
        return overlay_img, attn_norm

    def save(
        self,
        heatmap: Image.Image,
        path: str,
    ) -> None:
        """Save heatmap image to disk."""
        heatmap.save(path)
        log.info("Saved heatmap → %s", path)
