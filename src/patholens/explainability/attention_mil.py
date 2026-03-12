"""
Attention MIL — Gated attention mechanism for explainability.

Assigns attention weights to each patch, optionally conditioned
on clinical entity embeddings, enabling entity-guided heatmaps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning mechanism.

    Produces per-patch attention weights that indicate which patches
    contribute most to a given clinical entity or diagnosis.

    Parameters
    ----------
    d_model : int
        Patch feature dimension.
    d_hidden : int
        Hidden dimension for attention network.
    n_classes : int
        Number of entity-types or classification targets.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_hidden: int = 128,
        n_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Gated attention: V (tanh) ⊙ U (sigmoid) → W
        self.attention_V = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(d_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        patch_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        patch_features : (B, N, D)
            Contextualised patch representations.

        Returns
        -------
        attention_weights : (B, N, C)
            Normalised attention for each class/entity.
        weighted_repr : (B, C, D)
            Attention-weighted representations.
        """
        # Gated attention
        v = self.attention_V(patch_features)     # (B, N, H)
        u = self.attention_U(patch_features)     # (B, N, H)
        gated = self.dropout(v * u)              # (B, N, H)
        logits = self.attention_W(gated)         # (B, N, C)

        # Normalise per class
        attention_weights = F.softmax(logits, dim=1)  # (B, N, C)

        # Weighted representation: (B, C, D)
        # attention_weights^T @ patch_features
        weighted_repr = torch.bmm(
            attention_weights.transpose(1, 2),   # (B, C, N)
            patch_features,                       # (B, N, D)
        )

        return attention_weights, weighted_repr


class EntityConditionedAttention(nn.Module):
    """
    Entity-conditioned variant: attention is biased by a clinical
    entity embedding so each entity highlights different regions.

    Parameters
    ----------
    d_model : int
        Patch feature dimension.
    d_entity : int
        Entity embedding dimension.
    d_hidden : int
        Hidden dim.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_entity: int = 256,
        d_hidden: int = 128,
    ):
        super().__init__()
        self.entity_proj = nn.Linear(d_entity, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        entity_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_features : (B, N, D)
        entity_embedding : (B, D_e)

        Returns
        -------
        attention_weights : (B, N)
        """
        # Project entity and expand
        entity_proj = self.entity_proj(entity_embedding)  # (B, D)
        entity_exp = entity_proj.unsqueeze(1).expand_as(patch_features)  # (B, N, D)

        # Concat patch + entity context → gate
        combined = torch.cat([patch_features, entity_exp], dim=-1)  # (B, N, 2D)
        logits = self.gate(combined).squeeze(-1)  # (B, N)

        attention_weights = F.softmax(logits, dim=1)  # (B, N)
        return attention_weights
