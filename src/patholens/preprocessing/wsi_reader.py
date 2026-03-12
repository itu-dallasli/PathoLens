"""
WSI Reader — OpenSlide wrapper for Whole Slide Image I/O.

Handles SVS / TIF / NDPI / MRXS and other formats supported by
OpenSlide.  Provides helper properties for magnification, mpp,
and safe region reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None  # will raise at instantiation

from patholens.logger import get_logger

log = get_logger(__name__)

_SUPPORTED_EXTENSIONS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".vms", ".vmu", ".scn"}


class WSIReader:
    """
    Thin wrapper around :pymod:`openslide.OpenSlide`.

    Parameters
    ----------
    wsi_path : str | Path
        Path to a whole-slide image file.

    Raises
    ------
    FileNotFoundError
        If *wsi_path* does not exist.
    RuntimeError
        If OpenSlide cannot open the file.
    """

    def __init__(self, wsi_path: str | Path):
        if openslide is None:
            raise ImportError(
                "openslide-python is required. "
                "Install it with `pip install openslide-python` and ensure "
                "the OpenSlide C library is on your system PATH."
            )

        self.path = Path(wsi_path)
        if not self.path.exists():
            raise FileNotFoundError(f"WSI not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            log.warning("Extension '%s' may not be supported by OpenSlide.", suffix)

        try:
            self.slide = openslide.OpenSlide(str(self.path))
        except openslide.OpenSlideError as exc:
            raise RuntimeError(f"OpenSlide cannot open {self.path}: {exc}") from exc

        log.info(
            "Opened WSI  %s  |  dims=%s  levels=%d  mpp=%.4f",
            self.path.name,
            self.dimensions,
            self.level_count,
            self.mpp,
        )

    # ── Core properties ──────────────────────────────────────
    @property
    def dimensions(self) -> Tuple[int, int]:
        """(width, height) at level 0."""
        return self.slide.dimensions

    @property
    def level_count(self) -> int:
        return self.slide.level_count

    @property
    def level_dimensions(self) -> Tuple[Tuple[int, int], ...]:
        return self.slide.level_dimensions

    @property
    def level_downsamples(self) -> Tuple[float, ...]:
        return self.slide.level_downsamples

    @property
    def mpp(self) -> float:
        """Microns per pixel (level 0).  Falls back to 0.5 if missing."""
        val = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        return float(val) if val is not None else 0.5

    @property
    def magnification(self) -> Optional[float]:
        """Objective power reported in slide metadata."""
        val = self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        return float(val) if val is not None else None

    # ── Read helpers ─────────────────────────────────────────
    def read_region(
        self,
        location: Tuple[int, int],
        level: int = 0,
        size: Tuple[int, int] = (256, 256),
    ) -> Image.Image:
        """
        Read an RGBA region and convert to RGB.

        Parameters
        ----------
        location : (x, y)
            Top-left corner in **level-0** coordinates.
        level : int
            Pyramid level to read from.
        size : (w, h)
            Region size in pixels **at the requested level**.
        """
        region = self.slide.read_region(location, level, size)
        return region.convert("RGB")

    def get_thumbnail(self, size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """Return a low-resolution RGB thumbnail."""
        return self.slide.get_thumbnail(size).convert("RGB")

    def get_best_level_for_magnification(self, target_mag: float) -> int:
        """
        Return the pyramid level closest to *target_mag*.

        If the slide does not report objective power, level 0 is returned.
        """
        native = self.magnification
        if native is None:
            return 0
        target_downsample = native / target_mag
        diffs = [abs(d - target_downsample) for d in self.level_downsamples]
        return int(np.argmin(diffs))

    # ── Lifecycle ────────────────────────────────────────────
    def close(self) -> None:
        self.slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"WSIReader({self.path.name}, dims={self.dimensions})"
