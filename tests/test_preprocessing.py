"""
Tests for patholens.preprocessing — Tissue segmentation & patch extraction.

Uses synthetic images (no OpenSlide dependency).
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from patholens.preprocessing.tissue_segmentation import (
    SegmentationResult,
    TissueSegmentor,
)
from patholens.preprocessing.patch_extraction import (
    ExtractionResult,
    PatchExtractor,
)


class TestTissueSegmentor:
    """Test tissue segmentation on synthetic thumbnails."""

    def test_segmentor_on_pink_tissue(self, synthetic_thumbnail):
        """Pink blobs on white bg should produce non-zero tissue ratio."""
        seg = TissueSegmentor(
            median_blur_ksize=7,
            morph_kernel_size=5,
            min_contour_area=100,
        )
        result = seg.segment(synthetic_thumbnail)
        assert isinstance(result, SegmentationResult)
        assert result.tissue_ratio > 0.0
        assert result.mask.shape == (512, 512)
        assert result.mask.dtype == np.uint8
        assert len(result.contours) > 0

    def test_segmentor_numpy_input(self, synthetic_thumbnail_np):
        """Accepts numpy arrays as well as PIL images."""
        seg = TissueSegmentor(min_contour_area=100)
        result = seg.segment(synthetic_thumbnail_np)
        assert result.tissue_ratio > 0.0

    def test_segmentor_all_background(self, white_thumbnail):
        """All-white image should produce ~0 tissue ratio."""
        seg = TissueSegmentor(min_contour_area=100)
        result = seg.segment(white_thumbnail)
        assert result.tissue_ratio < 0.01
        assert len(result.contours) == 0

    def test_segmentor_thumbnail_size(self, synthetic_thumbnail):
        """thumbnail_size in result matches mask shape."""
        seg = TissueSegmentor(min_contour_area=100)
        result = seg.segment(synthetic_thumbnail)
        assert result.thumbnail_size == result.mask.shape[:2]

    def test_segmentor_mask_binary(self, synthetic_thumbnail):
        """Mask should only contain 0 and 255."""
        seg = TissueSegmentor(min_contour_area=100)
        result = seg.segment(synthetic_thumbnail)
        unique = set(np.unique(result.mask))
        assert unique.issubset({0, 255})


class TestPatchExtractor:
    """Test patch coordinate extraction using mock WSI reader."""

    def test_produces_coordinates(self, mock_wsi_reader, synthetic_tissue_mask):
        """Extraction produces valid (N, 2) coordinates."""
        extractor = PatchExtractor(
            patch_size=64,
            magnification=20.0,
            tissue_threshold=0.3,
            overlap=0,
        )
        result = extractor.extract(mock_wsi_reader, synthetic_tissue_mask)
        assert isinstance(result, ExtractionResult)
        assert result.coordinates.ndim == 2
        assert result.coordinates.shape[1] == 2
        assert result.num_patches == len(result.coordinates)

    def test_produces_patches_on_tissue(self, mock_wsi_reader, synthetic_tissue_mask):
        """With tissue present, should produce at least some patches."""
        extractor = PatchExtractor(
            patch_size=64,
            magnification=20.0,
            tissue_threshold=0.3,
            overlap=0,
        )
        result = extractor.extract(mock_wsi_reader, synthetic_tissue_mask)
        assert result.num_patches > 0

    def test_empty_mask_no_patches(self, mock_wsi_reader):
        """All-black mask should produce zero patches."""
        empty_mask = np.zeros((256, 256), dtype=np.uint8)
        extractor = PatchExtractor(
            patch_size=64,
            magnification=20.0,
            tissue_threshold=0.5,
            overlap=0,
        )
        result = extractor.extract(mock_wsi_reader, empty_mask)
        assert result.num_patches == 0
        assert result.coordinates.shape == (0, 2)

    def test_extraction_result_level(self, mock_wsi_reader, synthetic_tissue_mask):
        """Should select the correct pyramid level for target magnification."""
        extractor = PatchExtractor(
            patch_size=64,
            magnification=20.0,
        )
        result = extractor.extract(mock_wsi_reader, synthetic_tissue_mask)
        # Mock magnification is 40x, so 20x should give level 1
        assert result.level == 1
        assert result.magnification == 20.0

    def test_extraction_result_dataclass(self):
        """ExtractionResult.num_patches computed in __post_init__."""
        coords = np.array([[0, 0], [100, 100], [200, 200]], dtype=np.int64)
        result = ExtractionResult(
            coordinates=coords,
            patch_size=256,
            level=0,
            magnification=20.0,
        )
        assert result.num_patches == 3
