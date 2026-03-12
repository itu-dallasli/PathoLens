"""
Slide Encoder — Hierarchical WSI representation model.

Two-stage architecture:
  1. **Patch → Region** via :class:`RegionAggregator` (attention pooling)
  2. **Region → Slide** via :class:`MambaEncoder` (SSM over region sequence)

Outputs a slide-level vector, region-level vectors, and per-patch
attention weights (for RAAF explainability downstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from patholens.sequence_model.mamba_encoder import MambaEncoder
from patholens.sequence_model.region_aggregator import RegionAggregator


@dataclass
class SlideEncoderOutput:
    """Container for slide encoder outputs."""

    slide_repr: torch.Tensor          # (B, d_model)
    region_reprs: torch.Tensor        # (B, R, d_model)
    patch_features: torch.Tensor      # (B, L, d_model) — contextualised
    classification_logits: Optional[torch.Tensor] = None  # (B, n_classes)


class SlideEncoder(nn.Module):
    """
    Hierarchical slide-level encoder.

    Parameters
    ----------
    input_dim : int
        Patch embedding dimension (1024 for UNI).
    d_model : int
        Internal feature dimension.
    n_layers : int
        Number of Mamba blocks in the region-level encoder.
    region_size : int
        Patches per region.
    n_classes : int
        Number of classification targets (e.g. tumour subtypes).
        Set to 0 to disable the classification head.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        d_model: int = 512,
        n_layers: int = 4,
        region_size: int = 64,
        n_classes: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Stage 1: project patches and contextualise with a lightweight encoder
        self.patch_encoder = MambaEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=max(1, n_layers // 2),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Region aggregation
        self.region_aggregator = RegionAggregator(
            d_model=d_model,
            region_size=region_size,
        )

        # Stage 2: contextualise region representations
        self.region_encoder = MambaEncoder(
            input_dim=d_model,
            d_model=d_model,
            n_layers=max(1, n_layers - n_layers // 2),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Slide-level attention pooling
        self.slide_attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Optional classifier
        if n_classes > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        else:
            self.classifier = None

    def forward(self, patch_embeddings: torch.Tensor) -> SlideEncoderOutput:
        """
        Parameters
        ----------
        patch_embeddings : (B, N, 1024)
            Raw UNI embeddings for all patches of a slide.

        Returns
        -------
        SlideEncoderOutput
        """
        # Stage 1: patch-level Mamba
        patch_features = self.patch_encoder(patch_embeddings)  # (B, N, d)

        # Aggregate into regions
        region_reprs = self.region_aggregator(patch_features)  # (B, R, d)

        # Stage 2: region-level Mamba
        region_reprs = self.region_encoder(region_reprs)  # (B, R, d)

        # Slide-level pooling via attention over regions
        attn_logits = self.slide_attention(region_reprs)  # (B, R, 1)
        attn_weights = F.softmax(attn_logits, dim=1)
        slide_repr = (attn_weights * region_reprs).sum(dim=1)  # (B, d)

        # Classification (optional)
        logits = None
        if self.classifier is not None:
            logits = self.classifier(slide_repr)

        return SlideEncoderOutput(
            slide_repr=slide_repr,
            region_reprs=region_reprs,
            patch_features=patch_features,
            classification_logits=logits,
        )
