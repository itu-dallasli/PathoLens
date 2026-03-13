"""
Integration test — Full end-to-end pipeline with all mocks.

Patches WSIReader, UNIFeatureExtractor, and LLM calls with synthetic
data. Verifies the complete PathoLens pipeline runs to completion and produces
valid output files (FHIR report JSON + heatmap PNG).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from patholens.config import Config
from patholens.pipeline.inference import PathoLensPipeline, PipelineResult
from tests.conftest import MOCK_LLM_RESPONSE

# ── Mocking Heavy Modules Before Import ──────────────────────
# We mock these out in sys.modules so PathoLensPipeline._load_modules()
# doesn't even try to import torch/torchvision which can fail in some envs.

class FakeUNIExtractor:
    def __init__(self, *args, **kwargs):
        self._rng = np.random.RandomState(42)

    def extract_batch(self, images):
        import torch
        n = len(images)
        emb = torch.randn(n, 1024)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    def extract_single(self, image):
        emb = self._rng.randn(1024).astype(np.float32)
        return emb / np.linalg.norm(emb)

class FakeBatchProcessor:
    def __init__(self, *args, **kwargs):
        pass

    def process_slide(self, slide_id, wsi_path, coordinates, patch_size, level):
        # Return random embeddings
        rng = np.random.RandomState(42)
        n = len(coordinates)
        emb = rng.randn(n, 1024).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

mock_uni_module = MagicMock()
mock_uni_module.UNIFeatureExtractor = FakeUNIExtractor

mock_batch_module = MagicMock()
mock_batch_module.BatchEmbeddingProcessor = FakeBatchProcessor

# Apply the mocks to sys.modules
sys.modules["patholens.embedding.uni_extractor"] = mock_uni_module
sys.modules["patholens.embedding.batch_processor"] = mock_batch_module

# ── Synthetic WSI Reader ────────────────────────────────────
class FakeWSIReader:
    """Drop-in replacement for WSIReader that needs no OpenSlide."""

    def __init__(self, wsi_path):
        self.path = Path(wsi_path)
        self._rng = np.random.RandomState(42)

    @property
    def dimensions(self): return (4000, 3000)

    @property
    def level_count(self): return 2

    @property
    def level_dimensions(self): return ((4000, 3000), (2000, 1500))

    @property
    def level_downsamples(self): return (1.0, 2.0)

    @property
    def mpp(self): return 0.5

    @property
    def magnification(self): return 40.0

    def read_region(self, location, level=0, size=(256, 256)):
        import cv2
        patch_img = self._rng.randint(140, 230, (size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(patch_img, "RGB")

    def get_thumbnail(self, size=(1024, 1024)):
        import cv2
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240
        cv2.circle(img, (size[0] // 3, size[1] // 3), size[0] // 5, (220, 150, 160), -1)
        cv2.circle(img, (size[0] // 2, size[1] // 2), size[0] // 6, (210, 140, 170), -1)
        return Image.fromarray(img, "RGB")

    def get_best_level_for_magnification(self, target_mag):
        return 1 if target_mag <= 20 else 0

    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

# ── Test config ─────────────────────────────────────────────
def _build_test_config(tmp_dir: Path) -> Config:
    return Config.from_dict({
        "system": {"device": "cpu", "seed": 42, "num_workers": 0, "mixed_precision": False, "log_level": "WARNING"},
        "paths": {
            "data_root": str(tmp_dir / "data"),
            "raw_wsi_dir": str(tmp_dir / "data/raw"),
            "processed_dir": str(tmp_dir / "data/processed"),
            "patches_dir": str(tmp_dir / "data/processed/patches"),
            "embeddings_dir": str(tmp_dir / "data/processed/embeddings"),
            "features_dir": str(tmp_dir / "data/processed/features"),
            "faiss_index_dir": str(tmp_dir / "data/processed/faiss_index"),
            "metadata_dir": str(tmp_dir / "data/metadata"),
            "checkpoints_dir": str(tmp_dir / "checkpoints"),
            "results_dir": str(tmp_dir / "results"),
            "logs_dir": str(tmp_dir / "logs"),
        },
        "preprocessing": {
            "patch_size": 64, "magnification": 20, "tissue_threshold": 0.3, "overlap": 0, "thumbnail_size": 256,
            "segmentation": {"method": "otsu", "median_blur_ksize": 7, "morph_kernel_size": 5, "min_contour_area": 100},
        },
        "embedding": {"model_name": "mock", "architecture": "vit_large_patch16_224", "embedding_dim": 1024, "batch_size": 8, "normalize": True, "input_size": 224},
        "sequence_model": {"d_model": 64, "d_state": 4, "d_conv": 2, "expand_factor": 2, "n_layers": 2, "dropout": 0.0, "region_size": 8},
        "retrieval": {"index_type": "Flat", "n_list": 10, "n_probe": 4, "top_k": 3, "similarity_metric": "cosine", "rerank": False},
        "entity_extraction": {"llm_backend": "local", "llm_model": "mock", "max_tokens": 512, "temperature": 0.0},
        "explainability": {"attention_heads": 2, "mil_pooling": "gated_attention", "heatmap_resolution": 256, "overlay_alpha": 0.4, "colormap": "jet", "gaussian_sigma": 3.0},
        "report_generation": {"fhir_version": "R4", "confidence_threshold": 0.7, "language": "en", "include_evidence_links": True, "style_divergence_target": 0.20},
        "api": {"host": "0.0.0.0", "port": 8000, "max_upload_size_mb": 100, "cors_origins": ["http://localhost:3000"]},
    })

def _run_pipeline_with_mocks(config, wsi_path, tmp_dir, slide_id="test"):
    # Patch WSIReader during the run
    with patch("patholens.preprocessing.wsi_reader.WSIReader", side_effect=lambda p: FakeWSIReader(p)):
        pipeline = PathoLensPipeline(config=config)
        pipeline._load_modules()
        pipeline._entity_extractor._call_llm = MagicMock(return_value=MOCK_LLM_RESPONSE)
        result = pipeline.run(wsi_path=wsi_path, slide_id=slide_id, output_dir=tmp_dir / "results")
    return result

class TestPipelineIntegration:
    @pytest.fixture
    def pipeline_setup(self, tmp_path):
        config = _build_test_config(tmp_path)
        wsi_dir = tmp_path / "data" / "raw"
        wsi_dir.mkdir(parents=True)
        fake_wsi = wsi_dir / "test_slide.svs"
        fake_wsi.write_text("fake WSI content")
        return config, fake_wsi, tmp_path

    def test_full_pipeline_with_mocks(self, pipeline_setup):
        """Pipeline runs to completion."""
        config, wsi_path, tmp_dir = pipeline_setup
        result = _run_pipeline_with_mocks(config, wsi_path, tmp_dir, "integration_test")

        assert isinstance(result, PipelineResult)
        assert result.slide_id == "integration_test"
        assert result.status in ("completed", "low_confidence")
        assert result.num_patches > 0
        assert result.elapsed_seconds > 0

    def test_pipeline_creates_output_files(self, pipeline_setup):
        """Creates heatmap and JSON report."""
        config, wsi_path, tmp_dir = pipeline_setup
        result = _run_pipeline_with_mocks(config, wsi_path, tmp_dir, "file_test")

        assert result.heatmap_path is not None
        heatmap_file = Path(result.heatmap_path)
        assert heatmap_file.exists()
        
        report_path = tmp_dir / "results" / "file_test" / "diagnostic_report.json"
        assert report_path.exists()
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        assert report_data["resourceType"] == "DiagnosticReport"

    def test_pipeline_result_has_fhir_json(self, pipeline_setup):
        """PipelineResult contains valid FHIR string."""
        config, wsi_path, tmp_dir = pipeline_setup
        result = _run_pipeline_with_mocks(config, wsi_path, tmp_dir, "json_test")

        assert result.fhir_json != ""
        parsed = json.loads(result.fhir_json)
        assert parsed["resourceType"] == "DiagnosticReport"
