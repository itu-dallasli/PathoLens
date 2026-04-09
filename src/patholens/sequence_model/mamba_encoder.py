"""
Mamba Encoder — Multi-layer SSM block for patch sequence encoding.

Uses the official CUDA ``mamba_ssm.Mamba`` module when it is available
(production / GPU path).  On CPU / dev machines, falls back to
:class:`CPUMamba`, a pure-PyTorch implementation of the Mamba selective
state-space scan — mathematically equivalent to the CUDA version, just
slower because the scan runs in a Python loop.

This is NOT the old ``_LinearFallback`` placeholder: :class:`CPUMamba`
is a real state-space model with learned ``A``/``B``/``C``/``Δ``
parameters, a causal depth-wise conv, and a SiLU gating branch — the
same recipe as the original Mamba paper (Gu & Dao, 2023) and the
reference implementation at github.com/state-spaces/mamba.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from patholens.logger import get_logger

log = get_logger(__name__)

# ── Try importing the CUDA-optimised Mamba ───────────────────
try:
    from mamba_ssm import Mamba as _CUDAMamba

    _HAS_MAMBA_CUDA = True
except ImportError:
    _HAS_MAMBA_CUDA = False
    log.info(
        "mamba-ssm CUDA package not found; using pure-PyTorch CPUMamba "
        "(correct but slower). Install with `pip install -e .[gpu]` for "
        "the fused CUDA kernel."
    )


# ── Pure-PyTorch CPU Mamba ───────────────────────────────────
class CPUMamba(nn.Module):
    """
    Pure-PyTorch Mamba block. Mathematically identical to the reference
    ``mamba_ssm.Mamba`` module, minus the fused CUDA scan kernel.

    Pipeline, per layer::

        x (B, L, D)
          └─ in_proj → (x_inner, z)                 # two branches
             x_inner → causal 1-D conv → SiLU
                     → selective_params(Δ, B, C)    # state-space ingredients
                     → selective_scan (loop over L) # the sequential step
                     → y
             y = y * SiLU(z)                        # gated residual
             return out_proj(y)                     # (B, L, D)

    Parameters
    ----------
    d_model : int
        External hidden size (matches the rest of the network).
    d_state : int
        State dimension N of the SSM (``A ∈ ℝ^{d_inner × d_state}``).
    d_conv : int
        Kernel size of the causal depth-wise conv over the sequence.
    expand : int
        Inner expansion factor: ``d_inner = expand * d_model``.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = (
            math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        )

        # Input projection → (x, z) with x,z ∈ ℝ^{d_inner}
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Causal depth-wise conv over the sequence axis
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # we trim the right side → causal
            groups=self.d_inner,
            bias=conv_bias,
        )

        # x → (Δ_raw, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        # Δ_raw → Δ (per-channel, expanded)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise Δ projection so that softplus(Δ) ≈ small positive at init
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            # bias so softplus(bias) ≈ 1e-3 .. 1e-1
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3))
                + math.log(1e-3)
            )
            inv_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_softplus)

        # Learned state matrix A, parameterised as log for positivity of -A
        # Initialise to the S4D "real" scheme: A_n = -n, n=1..d_state
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )  # (d_inner, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # convention

        # Skip path scaling D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ── Forward ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_model)

        Returns
        -------
        (B, L, d_model)
        """
        B, L, _ = x.shape

        # Input projection + split
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                 # each (B, L, d_inner)

        # Causal depth-wise conv over sequence
        x_in = x_in.transpose(1, 2)                   # (B, d_inner, L)
        x_in = self.conv1d(x_in)[:, :, :L]            # trim right padding
        x_in = x_in.transpose(1, 2)                   # (B, L, d_inner)
        x_in = F.silu(x_in)

        # Selective SSM
        y = self._ssm(x_in)                           # (B, L, d_inner)

        # Gated residual
        y = y * F.silu(z)

        return self.out_proj(y)                       # (B, L, d_model)

    # ── Selective SSM ────────────────────────────────────────
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_inner)

        Returns
        -------
        (B, L, d_inner)
        """
        A = -torch.exp(self.A_log.float())            # (d_inner, d_state)
        D = self.D.float()                            # (d_inner,)

        x_dbl = self.x_proj(x)                        # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt))             # (B, L, d_inner)

        return self._selective_scan(x, dt, A, B, C, D)

    # ── The sequential scan (the only non-parallel part) ─────
    @staticmethod
    def _selective_scan(
        u: torch.Tensor,       # (B, L, d_inner)
        delta: torch.Tensor,   # (B, L, d_inner)
        A: torch.Tensor,       # (d_inner, d_state)
        B: torch.Tensor,       # (B, L, d_state)
        C: torch.Tensor,       # (B, L, d_state)
        D: torch.Tensor,       # (d_inner,)
    ) -> torch.Tensor:
        """Discretised selective state-space scan.

        Follows eq. (2a/b) of Gu & Dao 2023. Δ comes from the input and
        is used to discretise A and B at every step, making the recursion
        "selective" (content-dependent).
        """
        batch, L, d_inner = u.shape
        d_state = A.shape[1]

        # Discretise: Ā_t = exp(Δ_t · A), B̄_t · u_t = Δ_t · B_t · u_t
        deltaA = torch.exp(delta.unsqueeze(-1) * A)                       # (B, L, d_inner, d_state)
        deltaB_u = (
            delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        )                                                                 # (B, L, d_inner, d_state)

        # Scan
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]                         # (B, d_inner, d_state)
            y_t = torch.einsum("bds,bs->bd", h, C[:, t])                  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                                        # (B, L, d_inner)

        return y + u * D


def _make_mamba(d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    """Factory: real CUDA Mamba when available, CPUMamba otherwise."""
    if _HAS_MAMBA_CUDA:
        return _CUDAMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    return CPUMamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )


# ── Single Mamba block ───────────────────────────────────────
class MambaBlock(nn.Module):
    """
    Pre-norm residual Mamba block::

        x -> LayerNorm -> Mamba SSM -> + -> out
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
        """x: (B, L, D) -> (B, L, D)"""
        return x + self.dropout(self.mamba(self.norm(x)))


# ── Single Attention block ───────────────────────────────────
class AttentionBlock(nn.Module):
    """
    Pre-norm residual Transformer block (drop-in for :class:`MambaBlock`).

    Architecture::

        x -> LayerNorm -> MultiheadSelfAttention -> + -> LayerNorm -> FFN -> + -> out

    n_heads is inferred as ``max(1, d_model // 64)`` so the head dim stays
    near 64 regardless of d_model.
    """

    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        # mamba compat params — accepted but unused so callers can pass them
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        n_heads = max(1, d_model // 64)
        # Make sure d_model is divisible by n_heads
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        ffn_dim = d_model * 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ── Block factory ────────────────────────────────────────────
_BLOCK_REGISTRY: dict[str, type] = {
    "mamba":     MambaBlock,
    "attention": AttentionBlock,
}


def _make_block(
    backbone: str,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    dropout: float,
) -> nn.Module:
    cls = _BLOCK_REGISTRY.get(backbone.lower())
    if cls is None:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from: {list(_BLOCK_REGISTRY)}"
        )
    return cls(d_model=d_model, d_state=d_state, d_conv=d_conv,
               expand=expand, dropout=dropout)


# ── Multi-layer sequence encoder ─────────────────────────────
class MambaEncoder(nn.Module):
    """
    Stack of sequence-mixing blocks with an input projection.

    Maps ``(B, L, input_dim)`` -> ``(B, L, d_model)``.

    Parameters
    ----------
    backbone : {"mamba", "attention"}
        Which block type to use.  ``"mamba"`` uses :class:`CPUMamba` /
        CUDA Mamba (O(N), causal).  ``"attention"`` uses a standard
        pre-norm Transformer block (O(N^2), non-causal).
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
        backbone: str = "mamba",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.backbone = backbone

        self.projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            _make_block(backbone, d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, input_dim) -> (B, L, d_model)
        """
        x = self.projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
