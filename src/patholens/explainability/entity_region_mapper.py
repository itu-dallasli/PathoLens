"""
Entity-Region Mapper — Link clinical entities to WSI regions.

For each extracted entity, determines which tissue regions are
most relevant by analysing attention distributions, and pairs
them with supporting reference text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from patholens.entity_extraction.entity_schema import ClinicalEntity
from patholens.logger import get_logger

log = get_logger(__name__)


@dataclass
class EvidenceRegion:
    """A single WSI region linked to a clinical entity."""
    bbox: Tuple[int, int, int, int]         # (x, y, w, h) at level 0
    attention_score: float
    patch_indices: List[int] = field(default_factory=list)


@dataclass
class EntityEvidence:
    """Complete evidence package for one entity."""
    entity: ClinicalEntity
    regions: List[EvidenceRegion]
    reference_text_span: str = ""
    overall_attention: float = 0.0


class EntityRegionMapper:
    """
    Map clinical entities to WSI regions via attention analysis.

    Parameters
    ----------
    top_k_regions : int
        Max regions to link per entity.
    attention_threshold : float
        Min attention percentile to consider as "relevant".
    patch_size : int
        Patch side length for bounding-box computation.
    """

    def __init__(
        self,
        top_k_regions: int = 5,
        attention_threshold: float = 0.9,
        patch_size: int = 256,
    ):
        self.top_k_regions = top_k_regions
        self.attention_threshold = attention_threshold
        self.patch_size = patch_size

    def map(
        self,
        entities: List[ClinicalEntity],
        patch_coords: np.ndarray,
        attention_weights: np.ndarray,
    ) -> List[EntityEvidence]:
        """
        Link entities to high-attention WSI regions.

        Parameters
        ----------
        entities : list[ClinicalEntity]
        patch_coords : (N, 2) int — level-0 coords
        attention_weights : (N,) or (N, C) float
            Patch attention weights. If 2D, column ``c`` corresponds
            to the ``c``-th entity.

        Returns
        -------
        list[EntityEvidence]
        """
        results: List[EntityEvidence] = []

        # Ensure 2D
        if attention_weights.ndim == 1:
            attn = np.tile(attention_weights[:, None], (1, max(len(entities), 1)))
        else:
            attn = attention_weights

        for i, entity in enumerate(entities):
            col = min(i, attn.shape[1] - 1)
            weights = attn[:, col]

            # Find high-attention patches
            threshold = np.percentile(weights, self.attention_threshold * 100)
            high_idx = np.where(weights >= threshold)[0]

            if len(high_idx) == 0:
                results.append(EntityEvidence(entity=entity, regions=[]))
                continue

            # Cluster nearby patches into regions
            regions = self._cluster_patches(
                high_idx, patch_coords, weights
            )
            regions = sorted(regions, key=lambda r: r.attention_score, reverse=True)
            regions = regions[: self.top_k_regions]

            evidence = EntityEvidence(
                entity=entity,
                regions=regions,
                reference_text_span=entity.source_text_span,
                overall_attention=float(weights[high_idx].mean()),
            )
            results.append(evidence)

        log.info(
            "Mapped %d entities -> %d total regions",
            len(entities),
            sum(len(e.regions) for e in results),
        )
        return results

    def _cluster_patches(
        self,
        indices: np.ndarray,
        coords: np.ndarray,
        weights: np.ndarray,
    ) -> List[EvidenceRegion]:
        """Group spatially adjacent high-attention patches into regions."""
        ps = self.patch_size
        selected_coords = coords[indices]
        selected_weights = weights[indices]

        # Greedy merge: expand bounding boxes around seed patches
        used = set()
        regions: List[EvidenceRegion] = []

        # Sort by attention (highest first)
        order = np.argsort(-selected_weights)

        for rank in order:
            if rank in used:
                continue

            seed_x, seed_y = selected_coords[rank]
            patch_indices = [int(indices[rank])]
            used.add(rank)

            # Find neighbours within 2× patch size
            for other in range(len(selected_coords)):
                if other in used:
                    continue
                ox, oy = selected_coords[other]
                if abs(ox - seed_x) <= ps * 2 and abs(oy - seed_y) <= ps * 2:
                    patch_indices.append(int(indices[other]))
                    used.add(other)

            # Compute bounding box
            region_coords = coords[patch_indices]
            x_min = int(region_coords[:, 0].min())
            y_min = int(region_coords[:, 1].min())
            x_max = int(region_coords[:, 0].max() + ps)
            y_max = int(region_coords[:, 1].max() + ps)

            region_weights = weights[patch_indices]
            regions.append(EvidenceRegion(
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                attention_score=float(region_weights.mean()),
                patch_indices=patch_indices,
            ))

        return regions
