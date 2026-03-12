"""
Hierarchical Search — Two-level CMEA retrieval.

Level 1: Region-level search to find similar tissue regions.
Level 2: Aggregate region matches into slide-level ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from patholens.logger import get_logger

log = get_logger(__name__)

try:
    import faiss
except ImportError:
    faiss = None


@dataclass
class RetrievalResult:
    """Single retrieved case."""
    slide_id: str
    similarity_score: float
    matched_regions: List[Tuple[int, int, float]]  # (query_region, ref_region, score)
    report_text: Optional[str] = None


@dataclass
class RetrievalOutput:
    """Container for the full retrieval output."""
    query_slide_id: str
    results: List[RetrievalResult] = field(default_factory=list)
    
    @property
    def top_slide_ids(self) -> List[str]:
        return [r.slide_id for r in self.results]
    
    @property
    def top_scores(self) -> List[float]:
        return [r.similarity_score for r in self.results]


class HierarchicalRetriever:
    """
    Two-stage CMEA case retrieval.

    Stage 1 — **Region matching**: Each region of the query slide is
    searched against a region-level FAISS index.

    Stage 2 — **Slide aggregation**: Region-match scores are aggregated
    per reference slide to produce a slide-level similarity ranking.

    Parameters
    ----------
    slide_index : faiss.Index
        Pre-built slide-level FAISS index.
    region_index : faiss.Index or None
        Pre-built region-level FAISS index (optional).
    slide_id_map : dict
        Mapping from FAISS integer ID → slide_id string.
    region_to_slide : dict or None
        Mapping from FAISS region ID → slide_id string.
    report_db : dict or None
        Mapping from slide_id → report text.
    top_k : int
        Number of results to return.
    n_probe : int
        IVF probes for approximate search.
    """

    def __init__(
        self,
        slide_index,
        slide_id_map: Dict[int, str],
        region_index=None,
        region_to_slide: Optional[Dict[int, str]] = None,
        report_db: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        n_probe: int = 32,
    ):
        self.slide_index = slide_index
        self.slide_id_map = slide_id_map
        self.region_index = region_index
        self.region_to_slide = region_to_slide or {}
        self.report_db = report_db or {}
        self.top_k = top_k
        self.n_probe = n_probe

        # Set nprobe if IVF
        if hasattr(slide_index, "nprobe"):
            slide_index.nprobe = n_probe
        if region_index is not None and hasattr(region_index, "nprobe"):
            region_index.nprobe = n_probe

    def search(
        self,
        query_slide_repr: np.ndarray,
        query_region_reprs: Optional[np.ndarray] = None,
        query_slide_id: str = "query",
    ) -> RetrievalOutput:
        """
        Run hierarchical search.

        Parameters
        ----------
        query_slide_repr : (D,) or (1, D) float32
        query_region_reprs : (R, D) float32, optional
        query_slide_id : str

        Returns
        -------
        RetrievalOutput with top_k results.
        """
        # Ensure 2D
        if query_slide_repr.ndim == 1:
            query_slide_repr = query_slide_repr.reshape(1, -1)
        query_slide_repr = np.ascontiguousarray(query_slide_repr, dtype=np.float32)
        faiss.normalize_L2(query_slide_repr)

        # ── Stage 1: Slide-level search ──────────────────────
        scores, indices = self.slide_index.search(query_slide_repr, self.top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[RetrievalResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            sid = self.slide_id_map.get(int(idx), f"unknown_{idx}")
            
            # Skip self-matches
            if sid == query_slide_id:
                continue

            result = RetrievalResult(
                slide_id=sid,
                similarity_score=float(score),
                matched_regions=[],
                report_text=self.report_db.get(sid),
            )
            results.append(result)

        # ── Stage 2: Region-level refinement (optional) ──────
        if query_region_reprs is not None and self.region_index is not None:
            results = self._refine_with_regions(
                query_region_reprs, results
            )

        # Re-sort by score
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        results = results[: self.top_k]

        log.info(
            "Retrieval for %s  |  top-%d scores: %s",
            query_slide_id,
            self.top_k,
            [f"{r.similarity_score:.3f}" for r in results],
        )
        return RetrievalOutput(query_slide_id=query_slide_id, results=results)

    def _refine_with_regions(
        self,
        query_regions: np.ndarray,
        slide_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Boost slide scores using region-level matches."""
        query_regions = np.ascontiguousarray(query_regions, dtype=np.float32)
        faiss.normalize_L2(query_regions)

        # Search each region
        k_per_region = 10
        scores, indices = self.region_index.search(query_regions, k_per_region)

        # Count region matches per slide
        slide_region_scores: Dict[str, List[float]] = {}
        for q_idx in range(len(query_regions)):
            for score, r_idx in zip(scores[q_idx], indices[q_idx]):
                if r_idx < 0:
                    continue
                sid = self.region_to_slide.get(int(r_idx))
                if sid is None:
                    continue
                slide_region_scores.setdefault(sid, []).append(float(score))

        # Boost slide-level scores
        for result in slide_results:
            region_scores = slide_region_scores.get(result.slide_id, [])
            if region_scores:
                region_boost = np.mean(region_scores)
                result.similarity_score = 0.6 * result.similarity_score + 0.4 * region_boost
                result.matched_regions = [
                    (0, 0, s) for s in sorted(region_scores, reverse=True)[:5]
                ]

        return slide_results
