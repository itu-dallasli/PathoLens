"""
Tests for patholens.config — Config loading, dot-access, env overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from patholens.config import Config, DEFAULT_CONFIG_PATH, _cast_value


class TestConfigLoading:
    """Test config file loading and merging."""

    def test_load_default(self):
        """Default config loads and has expected top-level sections."""
        cfg = Config.load()
        assert hasattr(cfg, "system")
        assert hasattr(cfg, "paths")
        assert hasattr(cfg, "preprocessing")
        assert hasattr(cfg, "embedding")
        assert hasattr(cfg, "sequence_model")
        assert hasattr(cfg, "retrieval")
        assert hasattr(cfg, "entity_extraction")
        assert hasattr(cfg, "explainability")
        assert hasattr(cfg, "report_generation")
        assert hasattr(cfg, "api")

    def test_dot_access(self):
        """Dot notation works for nested access."""
        cfg = Config.load()
        assert isinstance(cfg.system.device, str)
        assert isinstance(cfg.embedding.batch_size, int)
        assert isinstance(cfg.preprocessing.patch_size, int)
        assert cfg.system.seed == 42

    def test_load_with_override(self):
        """Loading with test.yaml override changes values."""
        test_yaml = Path(__file__).parent.parent / "configs" / "test.yaml"
        if test_yaml.exists():
            cfg = Config.load(test_yaml)
            assert cfg.system.device == "cpu"
            assert cfg.system.num_workers == 0

    def test_default_config_path_exists(self):
        """The default config file should exist."""
        assert DEFAULT_CONFIG_PATH.exists()


class TestConfigFromDict:
    """Test creating Config from a plain dict."""

    def test_from_dict(self, test_config_dict):
        cfg = Config.from_dict(test_config_dict)
        assert cfg.system.device == "cpu"
        assert cfg.embedding.batch_size == 8
        assert cfg.preprocessing.patch_size == 64

    def test_to_dict_roundtrip(self, test_config_dict):
        cfg = Config.from_dict(test_config_dict)
        result = cfg.to_dict()
        assert isinstance(result, dict)
        assert result["system"]["device"] == "cpu"
        assert result["embedding"]["batch_size"] == 8

    def test_contains(self, test_config_dict):
        cfg = Config.from_dict(test_config_dict)
        assert "system" in cfg
        assert "nonexistent_key" not in cfg

    def test_get_with_default(self, test_config_dict):
        cfg = Config.from_dict(test_config_dict)
        assert cfg.get("system") is not None
        assert cfg.get("nonexistent", "fallback") == "fallback"


class TestEnvOverrides:
    """Test environment variable overrides."""

    def test_env_override_device(self, monkeypatch):
        monkeypatch.setenv("PATHOLENS_SYSTEM_DEVICE", "cpu")
        cfg = Config.load()
        assert cfg.system.device == "cpu"

    def test_cast_value_bool(self):
        assert _cast_value("true") is True
        assert _cast_value("false") is False
        assert _cast_value("yes") is True
        assert _cast_value("no") is False

    def test_cast_value_int(self):
        assert _cast_value("42") == 42
        assert isinstance(_cast_value("42"), int)

    def test_cast_value_float(self):
        assert _cast_value("3.14") == pytest.approx(3.14)
        assert isinstance(_cast_value("3.14"), float)

    def test_cast_value_string(self):
        assert _cast_value("hello") == "hello"
