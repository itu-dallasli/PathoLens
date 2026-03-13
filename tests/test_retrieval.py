"""
Tests for patholens.retrieval — FAISS index building, search, and reference DB.

Uses faiss-cpu (pip install faiss-cpu). Skips if faiss is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Check for faiss availability ────────────────────────────
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

from patholens.retrieval.reference_db import ReferenceDatabase

# Conditionally import faiss-dependent modules
if _HAS_FAISS:
    from patholens.retrieval.index_builder import FAISSIndexBuilder
    from patholens.retrieval.hierarchical_search import (
        HierarchicalRetriever,
        RetrievalOutput,
        RetrievalResult,
    )

pytestmark = pytest.mark.skipif(not _HAS_FAISS, reason="faiss not installed")


# ── Helpers ─────────────────────────────────────────────────
def _random_embeddings(n: int, dim: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


class TestFAISSIndexBuilder:
    """Test FAISS index construction."""

    def test_build_flat_index(self):
        """Build a Flat index and search it."""
        builder = FAISSIndexBuilder(dim=512, index_type="Flat", metric="cosine")
        emb = _random_embeddings(100)
        index = builder.build(emb)

        assert index.ntotal == 100

        # Search
        query = _random_embeddings(1, seed=99)
        faiss.normalize_L2(query)
        scores, indices = index.search(query, 5)
        assert scores.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert all(idx >= 0 for idx in indices[0])

    def test_build_with_ids(self):
        """Build with custom IDs via IndexIDMap."""
        builder = FAISSIndexBuilder(dim=128, index_type="Flat", metric="cosine")
        emb = _random_embeddings(50, dim=128)
        ids = np.arange(1000, 1050, dtype=np.int64)
        index = builder.build(emb, ids=ids)

        assert index.ntotal == 50

    def test_save_and_load(self, tmp_path):
        """Save/load round-trip preserves the index."""
        builder = FAISSIndexBuilder(dim=64, index_type="Flat", metric="cosine")
        emb = _random_embeddings(30, dim=64)
        index = builder.build(emb)

        save_path = tmp_path / "test.index"
        builder.save(index, save_path)
        loaded = builder.load(save_path)

        assert loaded.ntotal == 30

        # Same search results
        query = _random_embeddings(1, dim=64, seed=99)
        faiss.normalize_L2(query)
        s1, i1 = index.search(query, 3)
        s2, i2 = loaded.search(query, 3)
        np.testing.assert_array_equal(i1, i2)


class TestHierarchicalRetriever:
    """Test hierarchical case retrieval."""

    def _build_retriever(self, n_slides=20, dim=128):
        """Build a retriever with synthetic data."""
        builder = FAISSIndexBuilder(dim=dim, index_type="Flat", metric="cosine")
        emb = _random_embeddings(n_slides, dim=dim)
        index = builder.build(emb)

        slide_id_map = {i: f"slide_{i:03d}" for i in range(n_slides)}
        report_db = {f"slide_{i:03d}": f"Report for slide {i}" for i in range(n_slides)}

        return HierarchicalRetriever(
            slide_index=index,
            slide_id_map=slide_id_map,
            report_db=report_db,
            top_k=5,
        ), emb

    def test_search_returns_results(self):
        """Search returns RetrievalOutput with results."""
        retriever, emb = self._build_retriever()
        query = _random_embeddings(1, dim=128, seed=99).squeeze()
        output = retriever.search(query, query_slide_id="query")

        assert isinstance(output, RetrievalOutput)
        assert output.query_slide_id == "query"
        assert len(output.results) > 0
        assert len(output.results) <= 5

    def test_results_have_reports(self):
        """Retrieved results include report text."""
        retriever, emb = self._build_retriever()
        query = emb[0]  # Use first slide as query
        output = retriever.search(query, query_slide_id="query")

        for result in output.results:
            assert isinstance(result, RetrievalResult)
            assert result.slide_id.startswith("slide_")
            assert result.report_text is not None
            assert result.similarity_score > 0

    def test_self_exclusion(self):
        """Query slide ID is excluded from results."""
        retriever, emb = self._build_retriever()
        query = emb[5]  # slide_005's embedding
        output = retriever.search(query, query_slide_id="slide_005")

        result_ids = [r.slide_id for r in output.results]
        assert "slide_005" not in result_ids

    def test_scores_sorted_descending(self):
        """Results are sorted by similarity (descending)."""
        retriever, emb = self._build_retriever()
        query = _random_embeddings(1, dim=128, seed=77).squeeze()
        output = retriever.search(query)

        scores = [r.similarity_score for r in output.results]
        assert scores == sorted(scores, reverse=True)


class TestReferenceDatabase:
    """Test reference database for slide metadata."""

    def test_from_dict(self):
        """Build from a plain dict."""
        records = {
            "slide_a": {"slide_id": "slide_a", "report_text": "Report A", "subtype": "IDC"},
            "slide_b": {"slide_id": "slide_b", "report_text": "Report B", "subtype": "ILC"},
        }
        db = ReferenceDatabase.from_dict(records)
        assert len(db) == 2
        assert "slide_a" in db
        assert db.get_report("slide_a") == "Report A"
        assert db.get_subtype("slide_b") == "ILC"

    def test_from_csv(self, tmp_path):
        """Read from a small synthetic CSV."""
        csv_path = tmp_path / "metadata.csv"
        df = pd.DataFrame({
            "slide_id": ["s1", "s2", "s3"],
            "subtype": ["IDC", "ILC", "IDC"],
            "grade": ["2", "1", "3"],
            "report_text": ["Report 1", "Report 2", "Report 3"],
        })
        df.to_csv(csv_path, index=False)

        db = ReferenceDatabase.from_csv(csv_path)
        assert len(db) == 3
        assert db.get_report("s1") == "Report 1"
        assert db.get_subtype("s2") == "ILC"

    def test_get_all_reports(self):
        records = {
            "a": {"report_text": "R1"},
            "b": {"report_text": "R2"},
            "c": {},  # No report
        }
        db = ReferenceDatabase.from_dict(records)
        reports = db.get_all_reports()
        assert len(reports) == 2
        assert "a" in reports
        assert "b" in reports
        assert "c" not in reports

    def test_slide_ids(self):
        db = ReferenceDatabase.from_dict({"x": {}, "y": {}, "z": {}})
        assert set(db.slide_ids) == {"x", "y", "z"}

    def test_nonexistent_returns_none(self):
        db = ReferenceDatabase()
        assert db.get_report("doesnt_exist") is None
        assert db.get_metadata("doesnt_exist") is None
