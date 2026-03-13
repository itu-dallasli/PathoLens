"""
Tests for patholens.explainability — Attention MIL, heatmap generator, entity-region mapper.

Uses synthetic embeddings and coordinates (no real WSI or model needed).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from patholens.entity_extraction.entity_schema import ClinicalEntity
from patholens.explainability.attention_mil import (
    EntityConditionedAttention,
    GatedAttentionMIL,
)
from patholens.explainability.entity_region_mapper import (
    EntityEvidence,
    EntityRegionMapper,
    EvidenceRegion,
)
from patholens.explainability.heatmap_generator import HeatmapGenerator


# ── Helpers ─────────────────────────────────────────────────
def _make_entity(entity_type="tumor_type", value="IDC"):
    return ClinicalEntity(
        entity_type=entity_type,
        value=value,
        confidence=0.9,
        source_report_id="test",
        source_text_span="some text",
    )


class TestGatedAttentionMIL:
    """Test gated attention mechanism."""

    def test_output_shapes(self):
        """Attention weights: (B, N, C), weighted repr: (B, C, D)."""
        model = GatedAttentionMIL(d_model=64, d_hidden=32, n_classes=2)
        x = torch.randn(1, 50, 64)
        attn, weighted = model(x)

        assert attn.shape == (1, 50, 2)
        assert weighted.shape == (1, 2, 64)

    def test_attention_sums_to_one(self):
        """Attention weights per class sum to ~1.0 over patches."""
        model = GatedAttentionMIL(d_model=64, d_hidden=32, n_classes=3)
        x = torch.randn(1, 20, 64)
        attn, _ = model(x)

        # Sum over patches (dim=1) for each class
        sums = attn.sum(dim=1)  # (1, 3)
        np.testing.assert_allclose(sums.detach().numpy(), 1.0, atol=1e-5)

    def test_single_class(self):
        """Default n_classes=1 works."""
        model = GatedAttentionMIL(d_model=128, d_hidden=64, n_classes=1)
        x = torch.randn(2, 30, 128)
        attn, weighted = model(x)
        assert attn.shape == (2, 30, 1)
        assert weighted.shape == (2, 1, 128)

    def test_gradient_flow(self):
        """Gradients flow through attention."""
        model = GatedAttentionMIL(d_model=64, d_hidden=32)
        x = torch.randn(1, 10, 64, requires_grad=True)
        _, weighted = model(x)
        loss = weighted.sum()
        loss.backward()
        assert x.grad is not None


class TestEntityConditionedAttention:
    """Test entity-conditioned attention variant."""

    def test_output_shape(self):
        model = EntityConditionedAttention(d_model=64, d_entity=32, d_hidden=16)
        patches = torch.randn(1, 20, 64)
        entity_emb = torch.randn(1, 32)
        attn = model(patches, entity_emb)
        assert attn.shape == (1, 20)

    def test_attention_sums_to_one(self):
        model = EntityConditionedAttention(d_model=64, d_entity=32)
        patches = torch.randn(1, 15, 64)
        entity_emb = torch.randn(1, 32)
        attn = model(patches, entity_emb)
        assert attn.sum().item() == pytest.approx(1.0, abs=1e-5)


class TestHeatmapGenerator:
    """Test heatmap generation from attention weights."""

    def test_output_types(self, synthetic_patch_coords, synthetic_attention_weights):
        """Returns (PIL.Image, np.ndarray) pair."""
        gen = HeatmapGenerator(resolution=256, colormap="jet")
        heatmap, raw = gen.generate(
            wsi_dimensions=(40000, 30000),
            patch_coords=synthetic_patch_coords,
            attention_weights=synthetic_attention_weights,
            patch_size=256,
        )
        assert isinstance(heatmap, Image.Image)
        assert isinstance(raw, np.ndarray)
        assert heatmap.size == (256, 256)
        assert raw.shape == (256, 256)

    def test_with_thumbnail_overlay(
        self, synthetic_thumbnail, synthetic_patch_coords, synthetic_attention_weights
    ):
        """Overlay on thumbnail produces an image."""
        gen = HeatmapGenerator(resolution=256)
        heatmap, raw = gen.generate(
            wsi_dimensions=(40000, 30000),
            patch_coords=synthetic_patch_coords,
            attention_weights=synthetic_attention_weights,
            thumbnail=synthetic_thumbnail,
        )
        assert isinstance(heatmap, Image.Image)
        assert heatmap.size == (256, 256)

    def test_save_file(
        self, tmp_path, synthetic_patch_coords, synthetic_attention_weights
    ):
        """Saves a valid PNG file."""
        gen = HeatmapGenerator(resolution=256)
        heatmap, _ = gen.generate(
            wsi_dimensions=(40000, 30000),
            patch_coords=synthetic_patch_coords,
            attention_weights=synthetic_attention_weights,
        )
        out_path = str(tmp_path / "heatmap.png")
        gen.save(heatmap, out_path)
        assert (tmp_path / "heatmap.png").exists()
        # Verify it's a valid image
        loaded = Image.open(out_path)
        assert loaded.size == (256, 256)

    def test_raw_heatmap_normalised(
        self, synthetic_patch_coords, synthetic_attention_weights
    ):
        """Raw heatmap values are in [0, 1]."""
        gen = HeatmapGenerator(resolution=256)
        _, raw = gen.generate(
            wsi_dimensions=(40000, 30000),
            patch_coords=synthetic_patch_coords,
            attention_weights=synthetic_attention_weights,
        )
        assert raw.min() >= 0.0
        assert raw.max() <= 1.0


class TestEntityRegionMapper:
    """Test entity → WSI region mapping."""

    def test_map_produces_evidence(
        self, synthetic_patch_coords, synthetic_attention_weights
    ):
        """Maps entities to evidence regions."""
        mapper = EntityRegionMapper(
            top_k_regions=3,
            attention_threshold=0.9,
            patch_size=256,
        )
        entities = [_make_entity(), _make_entity("histological_grade", "Grade 2")]
        results = mapper.map(entities, synthetic_patch_coords, synthetic_attention_weights)

        assert len(results) == 2
        for ev in results:
            assert isinstance(ev, EntityEvidence)
            assert ev.entity is not None

    def test_evidence_regions_have_bbox(
        self, synthetic_patch_coords, synthetic_attention_weights
    ):
        """Evidence regions have valid bounding boxes."""
        mapper = EntityRegionMapper(top_k_regions=5, patch_size=256)
        entities = [_make_entity()]
        results = mapper.map(entities, synthetic_patch_coords, synthetic_attention_weights)

        for ev in results:
            for region in ev.regions:
                assert isinstance(region, EvidenceRegion)
                assert len(region.bbox) == 4
                x, y, w, h = region.bbox
                assert w > 0
                assert h > 0

    def test_empty_entities(self, synthetic_patch_coords, synthetic_attention_weights):
        """Empty entity list returns empty results."""
        mapper = EntityRegionMapper(patch_size=256)
        results = mapper.map([], synthetic_patch_coords, synthetic_attention_weights)
        assert len(results) == 0

    def test_top_k_limits_regions(
        self, synthetic_patch_coords, synthetic_attention_weights
    ):
        """top_k_regions limits the number of regions per entity."""
        mapper = EntityRegionMapper(top_k_regions=2, patch_size=256)
        entities = [_make_entity()]
        results = mapper.map(entities, synthetic_patch_coords, synthetic_attention_weights)

        for ev in results:
            assert len(ev.regions) <= 2
