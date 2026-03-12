"""
Mamba Encoder — Multi-layer SSM block for patch sequence encoding.

Uses the official ``mamba_ssm.Mamba`` module.  Falls back to a pure-PyTorch
placeholder (linear projection) when the CUDA kernel is unavailable, so
that unit tests and CPU-only development remain possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from patholens.logger import get_logger

log = get_logger(__name__)

# ── Try importing the CUDA-optimised Mamba ───────────────────
try:
    from mamba_ssm import Mamba as _CUDAMamba

    _HAS_MAMBA_CUDA = True
except ImportError:
    _HAS_MAMBA_CUDA = False
    log.warning(
        "mamba-ssm CUDA package not found — using linear fallback. "
        "Install with: pip install mamba-ssm causal-conv1d"
    )


# ── Fallback (CPU-safe) ─────────────────────────────────────
class _LinearFallback(nn.Module):
    """Mimics the Mamba interface using a simple linear layer."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _make_mamba(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    if _HAS_MAMBA_CUDA:
        return _CUDAMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    return _LinearFallback(d_model)


# ── Single Mamba block ───────────────────────────────────────
class MambaBlock(nn.Module):
    """
    Pre-norm residual Mamba block.

    ::

        x → LayerNorm → Mamba SSM → + → out
        └──────────────────────────┘
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = _make_mamba(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        return x + self.dropout(self.mamba(self.norm(x)))


# ── Multi-layer Mamba encoder ────────────────────────────────
class MambaEncoder(nn.Module):
    """
    Stack of :class:`MambaBlock` layers with an input projection.

    Maps ``(B, L, input_dim)`` → ``(B, L, d_model)``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of incoming embeddings (1024 for UNI).
    d_model : int
        Hidden size throughout the Mamba blocks.
    n_layers : int
        Number of stacked Mamba blocks.
    d_state, d_conv, expand, dropout
        Forwarded to :class:`MambaBlock`.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        d_model: int = 512,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, L, input_dim)``
            Patch-level embeddings.

        Returns
        -------
        Tensor of shape ``(B, L, d_model)``
            Contextualised patch representations.
        """
        x = self.projection(x)  # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
