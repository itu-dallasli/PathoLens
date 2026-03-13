"""
Tests for patholens.sequence_model — Mamba encoder, region aggregator, slide encoder.

Uses CPU linear fallback (no mamba-ssm CUDA required).
All tests use small dimensions for speed.
"""

from __future__ import annotations

import torch
import pytest

from patholens.sequence_model.mamba_encoder import MambaBlock, MambaEncoder
from patholens.sequence_model.region_aggregator import RegionAggregator
from patholens.sequence_model.slide_encoder import SlideEncoder, SlideEncoderOutput


class TestMambaEncoder:
    """Test MambaEncoder with CPU linear fallback."""

    def test_output_shape(self):
        """MambaEncoder maps (B, L, input_dim) → (B, L, d_model)."""
        encoder = MambaEncoder(input_dim=1024, d_model=64, n_layers=2)
        x = torch.randn(1, 100, 1024)
        out = encoder(x)
        assert out.shape == (1, 100, 64)

    def test_batch_dimension(self):
        """Works with batch size > 1."""
        encoder = MambaEncoder(input_dim=512, d_model=64, n_layers=2)
        x = torch.randn(4, 50, 512)
        out = encoder(x)
        assert out.shape == (4, 50, 64)

    def test_single_layer(self):
        """Works with a single layer."""
        encoder = MambaEncoder(input_dim=256, d_model=64, n_layers=1)
        x = torch.randn(1, 20, 256)
        out = encoder(x)
        assert out.shape == (1, 20, 64)

    def test_gradient_flow(self):
        """Gradients flow through the encoder."""
        encoder = MambaEncoder(input_dim=128, d_model=64, n_layers=2)
        x = torch.randn(1, 10, 128, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestMambaBlock:
    """Test individual MambaBlock."""

    def test_residual_connection(self):
        """Output is NOT identical to input (transformation applied)."""
        block = MambaBlock(d_model=64, dropout=0.0)
        x = torch.randn(1, 10, 64)
        out = block(x)
        assert out.shape == x.shape
        # Should not be identical due to the Mamba/Linear transformation
        assert not torch.allclose(out, x, atol=1e-5)

    def test_output_shape(self):
        """Preserves input shape."""
        block = MambaBlock(d_model=128)
        x = torch.randn(2, 20, 128)
        out = block(x)
        assert out.shape == (2, 20, 128)


class TestRegionAggregator:
    """Test attention-weighted region pooling."""

    def test_exact_division(self):
        """128 patches / 64 region_size = 2 regions."""
        agg = RegionAggregator(d_model=64, region_size=64)
        x = torch.randn(1, 128, 64)
        out = agg(x)
        assert out.shape == (1, 2, 64)

    def test_padding_for_non_divisible(self):
        """Non-divisible sequence gets padded: 100 / 64 → ceil = 2 regions."""
        agg = RegionAggregator(d_model=64, region_size=64)
        x = torch.randn(1, 100, 64)
        out = agg(x)
        assert out.shape == (1, 2, 64)

    def test_single_region(self):
        """Fewer patches than region_size → 1 region."""
        agg = RegionAggregator(d_model=64, region_size=64)
        x = torch.randn(1, 30, 64)
        out = agg(x)
        assert out.shape == (1, 1, 64)

    def test_small_region_size(self):
        """Small region_size = 8, 32 patches → 4 regions."""
        agg = RegionAggregator(d_model=32, region_size=8)
        x = torch.randn(2, 32, 32)
        out = agg(x)
        assert out.shape == (2, 4, 32)


class TestSlideEncoder:
    """Test full hierarchical slide encoder."""

    def test_output_types(self):
        """SlideEncoder produces SlideEncoderOutput with correct shapes."""
        encoder = SlideEncoder(
            input_dim=128,
            d_model=64,
            n_layers=2,
            region_size=8,
            n_classes=0,
        )
        x = torch.randn(1, 32, 128)
        out = encoder(x)

        assert isinstance(out, SlideEncoderOutput)
        assert out.slide_repr.shape == (1, 64)
        assert out.patch_features.shape[0] == 1
        assert out.patch_features.shape[2] == 64
        assert out.classification_logits is None

    def test_with_classifier(self):
        """n_classes > 0 produces classification logits."""
        encoder = SlideEncoder(
            input_dim=128,
            d_model=64,
            n_layers=2,
            region_size=8,
            n_classes=3,
        )
        x = torch.randn(1, 32, 128)
        out = encoder(x)
        assert out.classification_logits is not None
        assert out.classification_logits.shape == (1, 3)

    def test_region_representations(self):
        """Region representations have expected shape: (B, R, d_model)."""
        encoder = SlideEncoder(
            input_dim=128,
            d_model=64,
            n_layers=2,
            region_size=16,
        )
        x = torch.randn(1, 64, 128)
        out = encoder(x)
        # 64 / 16 = 4 regions
        assert out.region_reprs.shape == (1, 4, 64)

    def test_gradient_flow(self):
        """Gradients flow through the full slide encoder."""
        encoder = SlideEncoder(
            input_dim=128,
            d_model=64,
            n_layers=2,
            region_size=8,
            n_classes=2,
        )
        x = torch.randn(1, 16, 128, requires_grad=True)
        out = encoder(x)
        loss = out.classification_logits.sum()
        loss.backward()
        assert x.grad is not None
