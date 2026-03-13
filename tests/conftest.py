"""
PathoLens — Shared pytest fixtures for synthetic testing.

Provides mock WSI readers, synthetic images/embeddings, fake LLM
backends, and test configs so every module can be tested without
real data, GPU, or model downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, PropertyMock

import cv2
import numpy as np
import pytest
from PIL import Image

from patholens.config import Config


# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CONFIG_PATH = PROJECT_ROOT / "configs" / "test.yaml"


# ── Config ───────────────────────────────────────────────────
@pytest.fixture
def test_config() -> Config:
    """Load the test-specific configuration."""
    return Config.load(TEST_CONFIG_PATH)


@pytest.fixture
def test_config_dict() -> Dict[str, Any]:
    """Minimal dict-based config for isolated tests."""
    return {
        "system": {"device": "cpu", "seed": 42, "num_workers": 0,
                   "mixed_precision": False, "log_level": "WARNING"},
        "paths": {
            "data_root": "data", "raw_wsi_dir": "data/raw",
            "processed_dir": "data/processed",
            "patches_dir": "data/processed/patches",
            "embeddings_dir": "data/processed/embeddings",
            "features_dir": "data/processed/features",
            "faiss_index_dir": "data/processed/faiss_index",
            "metadata_dir": "data/metadata",
            "checkpoints_dir": "checkpoints",
            "results_dir": "results", "logs_dir": "logs",
        },
        "preprocessing": {
            "patch_size": 64, "magnification": 20,
            "tissue_threshold": 0.3, "overlap": 0,
            "thumbnail_size": 256,
            "segmentation": {
                "method": "otsu", "median_blur_ksize": 7,
                "morph_kernel_size": 5, "min_contour_area": 100,
            },
        },
        "embedding": {
            "model_name": "mock", "architecture": "vit_large_patch16_224",
            "embedding_dim": 1024, "batch_size": 8,
            "normalize": True, "input_size": 224,
        },
        "sequence_model": {
            "d_model": 64, "d_state": 4, "d_conv": 2,
            "expand_factor": 2, "n_layers": 2, "dropout": 0.0,
            "region_size": 8,
            "training": {
                "learning_rate": 1e-4, "weight_decay": 1e-2,
                "epochs": 2, "batch_size": 1,
                "gradient_checkpointing": False,
                "scheduler": "cosine", "warmup_epochs": 1,
            },
        },
        "retrieval": {
            "index_type": "Flat", "n_list": 10, "n_probe": 4,
            "top_k": 3, "similarity_metric": "cosine", "rerank": False,
        },
        "entity_extraction": {
            "llm_backend": "mock", "llm_model": "mock",
            "max_tokens": 512, "temperature": 0.0,
            "entity_types": ["tumor_type", "histological_grade"],
            "cross_validation": False,
        },
        "explainability": {
            "attention_heads": 2, "mil_pooling": "gated_attention",
            "heatmap_resolution": 256, "overlay_alpha": 0.4,
            "colormap": "jet", "gaussian_sigma": 3.0,
        },
        "report_generation": {
            "fhir_version": "R4", "confidence_threshold": 0.7,
            "language": "en", "include_evidence_links": True,
            "style_divergence_target": 0.20,
        },
        "api": {
            "host": "0.0.0.0", "port": 8000,
            "max_upload_size_mb": 100,
            "cors_origins": ["http://localhost:3000"],
        },
    }


# ── Synthetic images ────────────────────────────────────────
@pytest.fixture
def synthetic_thumbnail() -> Image.Image:
    """
    512×512 RGB image with white background and pink 'tissue' blobs.
    Simulates a WSI thumbnail without OpenSlide.
    """
    img = np.ones((512, 512, 3), dtype=np.uint8) * 240  # near-white bg

    # Draw pink tissue blobs (high saturation in HSV → triggers Otsu)
    cv2.circle(img, (200, 200), 100, (220, 150, 160), -1)
    cv2.circle(img, (350, 300), 80, (210, 140, 170), -1)
    cv2.ellipse(img, (150, 400), (120, 60), 30, 0, 360, (200, 130, 150), -1)

    return Image.fromarray(img, "RGB")


@pytest.fixture
def synthetic_thumbnail_np(synthetic_thumbnail) -> np.ndarray:
    """Numpy version of the synthetic thumbnail."""
    return np.array(synthetic_thumbnail)


@pytest.fixture
def white_thumbnail() -> Image.Image:
    """All-white image (no tissue)."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    return Image.fromarray(img, "RGB")


@pytest.fixture
def synthetic_tissue_mask() -> np.ndarray:
    """Binary mask with some tissue regions. Shape (256, 256), uint8 0/255."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)
    cv2.circle(mask, (180, 150), 40, 255, -1)
    return mask


# ── Synthetic coordinates & embeddings ──────────────────────
@pytest.fixture
def synthetic_patch_coords() -> np.ndarray:
    """(N, 2) int64 array of 50 fake level-0 patch coordinates."""
    rng = np.random.RandomState(42)
    coords = rng.randint(0, 10000, size=(50, 2)).astype(np.int64)
    return coords


@pytest.fixture
def synthetic_embeddings() -> np.ndarray:
    """(50, 1024) float32 L2-normalised random embeddings."""
    rng = np.random.RandomState(42)
    emb = rng.randn(50, 1024).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


@pytest.fixture
def synthetic_attention_weights() -> np.ndarray:
    """(50,) float attention weights from softmax of random logits."""
    rng = np.random.RandomState(42)
    logits = rng.randn(50).astype(np.float32)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


# ── Mock WSI Reader ─────────────────────────────────────────
@pytest.fixture
def mock_wsi_reader(synthetic_thumbnail):
    """
    MagicMock that mimics WSIReader interface.

    Properties: dimensions, level_count, level_dimensions,
                level_downsamples, mpp, magnification
    Methods:    read_region(), get_thumbnail(), close()
    Context:    __enter__, __exit__
    """
    reader = MagicMock()

    # Properties
    type(reader).dimensions = PropertyMock(return_value=(40000, 30000))
    type(reader).level_count = PropertyMock(return_value=3)
    type(reader).level_dimensions = PropertyMock(
        return_value=((40000, 30000), (10000, 7500), (2500, 1875))
    )
    type(reader).level_downsamples = PropertyMock(
        return_value=(1.0, 4.0, 16.0)
    )
    type(reader).mpp = PropertyMock(return_value=0.5)
    type(reader).magnification = PropertyMock(return_value=40.0)

    # Methods
    def _read_region(location, level=0, size=(256, 256)):
        """Return a synthetic RGB patch image."""
        rng = np.random.RandomState(hash(location) % 2**31)
        patch = rng.randint(150, 240, size=(size[1], size[0], 3), dtype=np.uint8)
        return Image.fromarray(patch, "RGB")

    reader.read_region = _read_region
    reader.get_thumbnail.return_value = synthetic_thumbnail

    def _best_level(target_mag):
        # 40x native → 20x = level 1
        if target_mag <= 5:
            return 2
        elif target_mag <= 20:
            return 1
        return 0

    reader.get_best_level_for_magnification = _best_level

    # Context manager
    reader.__enter__ = MagicMock(return_value=reader)
    reader.__exit__ = MagicMock(return_value=False)

    return reader


# ── Fake reference reports ──────────────────────────────────
@pytest.fixture
def fake_reference_reports() -> List[Dict[str, str]]:
    """Realistic-ish breast cancer pathology report snippets."""
    return [
        {
            "slide_id": "TCGA-A2-0001",
            "report_text": (
                "Invasive ductal carcinoma, grade 2. ER positive, PR positive, "
                "HER2 negative. Ki-67 index 15%. No lymph node metastasis. "
                "Surgical margins negative. Tumor stage IIA. No necrosis observed."
            ),
        },
        {
            "slide_id": "TCGA-A2-0002",
            "report_text": (
                "Invasive ductal carcinoma, histological grade 2 "
                "(moderately differentiated). Estrogen receptor (ER): positive. "
                "Progesterone receptor (PR): positive. HER2: negative (IHC 1+). "
                "Ki-67 proliferation index: 18%. Sentinel lymph node: negative. "
                "Margins: free of tumor. Stage IIA."
            ),
        },
        {
            "slide_id": "TCGA-A2-0003",
            "report_text": (
                "Invasive ductal carcinoma, grade 3. ER positive, PR negative, "
                "HER2 positive (3+). Ki-67 index 40%. Lymph node positive "
                "(2 of 12). Surgical margins: close (1mm). Stage IIB. "
                "Focal necrosis present."
            ),
        },
    ]


# ── Mock LLM response ──────────────────────────────────────
MOCK_LLM_RESPONSE = """\
```json
{
  "tumor_type": "Invasive Ductal Carcinoma",
  "histological_grade": "Grade 2",
  "tumor_stage": "IIA",
  "receptor_status": {
    "ER": "positive",
    "PR": "positive",
    "HER2": "negative"
  },
  "ki67_index": "15%",
  "lymph_node_status": "negative",
  "surgical_margins": "negative",
  "necrosis": null,
  "additional_findings": []
}
```
"""


@pytest.fixture
def mock_llm_response() -> str:
    return MOCK_LLM_RESPONSE
