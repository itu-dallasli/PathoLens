"""
Patch Store — HDF5-based persistence for patch coordinates & metadata.

Each slide gets one HDF5 file::

    data/processed/patches/<slide_id>.h5
        ├── coordinates   (N, 2) int64
        └── attrs:
              slide_id, wsi_path, patch_size, magnification, level, num_patches
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from patholens.logger import get_logger

log = get_logger(__name__)


class PatchStore:
    """
    Read / write patch coordinate files in HDF5 format.

    Parameters
    ----------
    root_dir : str | Path
        Directory where ``<slide_id>.h5`` files are stored.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, slide_id: str) -> Path:
        return self.root_dir / f"{slide_id}.h5"

    # ── Write ────────────────────────────────────────────────
    def save(
        self,
        slide_id: str,
        coordinates: np.ndarray,
        attrs: Dict[str, Any],
    ) -> Path:
        """
        Save patch coordinates and metadata for one slide.

        Parameters
        ----------
        slide_id : str
            Unique identifier (e.g. TCGA barcode).
        coordinates : np.ndarray
            (N, 2) int64 array of level-0 (x, y) patch locations.
        attrs : dict
            Metadata to store as HDF5 attributes.

        Returns
        -------
        Path to the created file.
        """
        out = self._path(slide_id)
        with h5py.File(out, "w") as f:
            f.create_dataset("coordinates", data=coordinates, compression="gzip")
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["num_patches"] = len(coordinates)

        log.info("Saved %d patch coords -> %s", len(coordinates), out)
        return out

    # ── Read ─────────────────────────────────────────────────
    def load(self, slide_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load coordinates and attrs for *slide_id*.

        Returns
        -------
        (coordinates, attrs_dict)
        """
        path = self._path(slide_id)
        if not path.exists():
            raise FileNotFoundError(f"No patches file for slide '{slide_id}': {path}")

        with h5py.File(path, "r") as f:
            coords = f["coordinates"][:]
            attrs = dict(f.attrs)

        log.debug("Loaded %d patch coords from %s", len(coords), path)
        return coords, attrs

    # ── Query ────────────────────────────────────────────────
    def exists(self, slide_id: str) -> bool:
        return self._path(slide_id).exists()

    def list_slides(self) -> list[str]:
        """Return slide IDs that have stored patches."""
        return [p.stem for p in self.root_dir.glob("*.h5")]
