"""
Region Aggregator — Group patches into spatial regions and pool.

Divides a sequence of N patch representations into ⌈N/region_size⌉
regions and applies attention-weighted pooling within each region.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionAggregator(nn.Module):
    """
    Attention-weighted pooling within fixed-size patch regions.

    Parameters
    ----------
    d_model : int
        Feature dimension.
    region_size : int
        Number of patches per region.
    """

    def __init__(self, d_model: int = 512, region_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.region_size = region_size

        # Small attention network for intra-region pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, D)
            Patch-level features.

        Returns
        -------
        (B, R, D)
            Region-level features where ``R = ceil(L / region_size)``.
        """
        B, L, D = x.shape
        rs = self.region_size

        # Pad sequence so it divides evenly into regions
        remainder = L % rs
        if remainder != 0:
            pad_len = rs - remainder
            x = F.pad(x, (0, 0, 0, pad_len))  # pad along L dimension
            L = x.shape[1]

        n_regions = L // rs

        # Reshape: (B, R, rs, D)
        x = x.view(B, n_regions, rs, D)

        # Attention weights within each region: (B, R, rs, 1)
        attn_logits = self.attention(x)  # (B, R, rs, 1)
        attn_weights = F.softmax(attn_logits, dim=2)  # softmax over patches in region

        # Weighted sum: (B, R, D)
        pooled = (attn_weights * x).sum(dim=2)
        return pooled
