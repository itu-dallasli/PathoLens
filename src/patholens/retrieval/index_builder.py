"""
FAISS Index Builder — Build and persist vector search indices.

Supports IVFFlat (fast approximate) and Flat (exact) index types
for cosine-similarity retrieval of slide/region embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from patholens.logger import get_logger

log = get_logger(__name__)

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False
    log.warning("faiss not installed — retrieval will be unavailable.")


class FAISSIndexBuilder:
    """
    Build, train, and persist a FAISS similarity index.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    index_type : str
        ``"IVFFlat"`` or ``"Flat"``.
    n_list : int
        Number of IVF clusters (only for IVFFlat).
    metric : str
        ``"cosine"`` or ``"l2"``.
    """

    def __init__(
        self,
        dim: int = 512,
        index_type: str = "IVFFlat",
        n_list: int = 100,
        metric: str = "cosine",
    ):
        if not _HAS_FAISS:
            raise ImportError("faiss is required: pip install faiss-gpu")

        self.dim = dim
        self.index_type = index_type
        self.n_list = n_list
        self.metric = metric

    def build(
        self,
        embeddings: np.ndarray,
        ids: Optional[np.ndarray] = None,
    ) -> "faiss.Index":
        """
        Build a FAISS index from an array of embeddings.

        Parameters
        ----------
        embeddings : (N, D) float32
        ids : (N,) int64, optional
            If given, wraps the index with IndexIDMap.

        Returns
        -------
        faiss.Index
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # L2-normalise for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)

        faiss_metric = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2

        if self.index_type == "Flat":
            index = faiss.IndexFlat(self.dim, faiss_metric)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlat(self.dim, faiss_metric)
            index = faiss.IndexIVFFlat(quantizer, self.dim, self.n_list, faiss_metric)
            log.info("Training IVF index with %d vectors ...", len(embeddings))
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Wrap with ID mapping if requested
        if ids is not None:
            id_index = faiss.IndexIDMap(index)
            id_index.add_with_ids(embeddings, ids.astype(np.int64))
            log.info("Built %s index: %d vectors (with IDs)", self.index_type, len(embeddings))
            return id_index

        index.add(embeddings)
        log.info("Built %s index: %d vectors", self.index_type, len(embeddings))
        return index

    @staticmethod
    def save(index: "faiss.Index", path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        log.info("Saved FAISS index → %s", path)

    @staticmethod
    def load(path: str | Path) -> "faiss.Index":
        index = faiss.read_index(str(path))
        log.info("Loaded FAISS index from %s (%d vectors)", path, index.ntotal)
        return index
