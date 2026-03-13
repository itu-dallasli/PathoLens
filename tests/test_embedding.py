"""
Tests for patholens.embedding — EmbeddingStore HDF5 persistence.

Does NOT test UNIFeatureExtractor (requires model download).
Tests HDF5 read/write and batch processor logic with mock extractor.
"""

from __future__ import annotations

import numpy as np
import pytest

from patholens.embedding.embedding_store import EmbeddingStore


class TestEmbeddingStore:
    """Test HDF5 embedding persistence."""

    def test_save_and_load(self, tmp_path, synthetic_embeddings, synthetic_patch_coords):
        """Save → load round-trip preserves data."""
        store = EmbeddingStore(tmp_path / "embeddings")

        store.save(
            slide_id="test_slide",
            embeddings=synthetic_embeddings,
            coordinates=synthetic_patch_coords,
            attrs={"slide_id": "test_slide", "patch_size": 256},
        )

        loaded_emb, attrs = store.load("test_slide")

        assert loaded_emb.shape == synthetic_embeddings.shape
        np.testing.assert_allclose(loaded_emb, synthetic_embeddings, atol=1e-6)
        assert attrs["slide_id"] == "test_slide"
        assert attrs["num_patches"] == len(synthetic_embeddings)

    def test_exists(self, tmp_path, synthetic_embeddings, synthetic_patch_coords):
        """exists() returns True after save, False before."""
        store = EmbeddingStore(tmp_path / "embeddings")

        assert not store.exists("test_slide")

        store.save(
            slide_id="test_slide",
            embeddings=synthetic_embeddings,
            coordinates=synthetic_patch_coords,
            attrs={"slide_id": "test_slide"},
        )

        assert store.exists("test_slide")
        assert not store.exists("nonexistent")

    def test_list_slides(self, tmp_path, synthetic_embeddings, synthetic_patch_coords):
        """list_slides() returns saved slide IDs."""
        store = EmbeddingStore(tmp_path / "embeddings")

        for sid in ["slide_a", "slide_b", "slide_c"]:
            store.save(
                slide_id=sid,
                embeddings=synthetic_embeddings,
                coordinates=synthetic_patch_coords,
                attrs={"slide_id": sid},
            )

        slides = store.list_slides()
        assert set(slides) == {"slide_a", "slide_b", "slide_c"}

    def test_load_with_coordinates(
        self, tmp_path, synthetic_embeddings, synthetic_patch_coords
    ):
        """load_with_coordinates returns (embeddings, coordinates, attrs)."""
        store = EmbeddingStore(tmp_path / "embeddings")

        store.save(
            slide_id="test",
            embeddings=synthetic_embeddings,
            coordinates=synthetic_patch_coords,
            attrs={"slide_id": "test"},
        )

        emb, coords, attrs = store.load_with_coordinates("test")
        assert emb.shape == synthetic_embeddings.shape
        assert coords.shape == synthetic_patch_coords.shape
        np.testing.assert_array_equal(coords, synthetic_patch_coords)

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading a nonexistent slide raises FileNotFoundError."""
        store = EmbeddingStore(tmp_path / "embeddings")
        with pytest.raises(FileNotFoundError):
            store.load("doesnt_exist")

    def test_creates_directory(self, tmp_path):
        """Store creates root directory if it doesn't exist."""
        new_dir = tmp_path / "deeply" / "nested" / "embeddings"
        store = EmbeddingStore(new_dir)
        assert new_dir.exists()
