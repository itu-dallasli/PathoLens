"""
Embedding Store — HDF5 persistence for patch-level UNI embeddings.

Layout per file::

    data/processed/embeddings/<slide_id>.h5
        ├── embeddings    float32 (N, 1024)
        ├── coordinates   int64   (N, 2)
        └── attrs:
              slide_id, wsi_path, patch_size, level, embedding_dim, num_patches
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np

from patholens.logger import get_logger

log = get_logger(__name__)


class EmbeddingStore:
    """
    Read / write embedding arrays in HDF5 format.

    Parameters
    ----------
    root_dir : str | Path
        Directory for ``<slide_id>.h5`` files.
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
        embeddings: np.ndarray,
        coordinates: np.ndarray,
        attrs: Dict[str, Any],
    ) -> Path:
        """
        Persist embeddings to disk.

        Parameters
        ----------
        slide_id : str
        embeddings : np.ndarray, shape (N, D)
        coordinates : np.ndarray, shape (N, 2)
        attrs : dict
        """
        out = self._path(slide_id)
        with h5py.File(out, "w") as f:
            f.create_dataset(
                "embeddings",
                data=embeddings.astype(np.float32),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "coordinates",
                data=coordinates.astype(np.int64),
                compression="gzip",
            )
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["num_patches"] = len(embeddings)

        log.info("Saved %d embeddings -> %s", len(embeddings), out)
        return out

    # ── Read ─────────────────────────────────────────────────
    def load(self, slide_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings and metadata.

        Returns
        -------
        (embeddings, attrs_dict)
        """
        path = self._path(slide_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No embeddings for slide '{slide_id}': {path}"
            )
        with h5py.File(path, "r") as f:
            embeddings = f["embeddings"][:]
            attrs = dict(f.attrs)
        return embeddings, attrs

    def load_with_coordinates(
        self, slide_id: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load embeddings, coordinates, and metadata.

        Returns
        -------
        (embeddings, coordinates, attrs)
        """
        path = self._path(slide_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No embeddings for slide '{slide_id}': {path}"
            )
        with h5py.File(path, "r") as f:
            embeddings = f["embeddings"][:]
            coordinates = f["coordinates"][:]
            attrs = dict(f.attrs)
        return embeddings, coordinates, attrs

    # ── Query ────────────────────────────────────────────────
    def exists(self, slide_id: str) -> bool:
        return self._path(slide_id).exists()

    def list_slides(self) -> list[str]:
        return [p.stem for p in self.root_dir.glob("*.h5")]
