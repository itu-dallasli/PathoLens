"""
Configuration management for PathoLens.

Loads YAML configs with hierarchical merging: default.yaml ← overrides.
Provides dot-notation access via a frozen dataclass-like object.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ── Project root detection ───────────────────────────────────
def _find_project_root() -> Path:
    """Walk upward from this file until we find pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# ── Deep merge utility ──────────────────────────────────────
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ── Config wrapper with dot-access ──────────────────────────
class Config:
    """
    Immutable, dot-accessible configuration object.

    Usage::

        cfg = Config.load()          # loads configs/default.yaml
        cfg = Config.load("custom.yaml")  # merges custom on top
        print(cfg.system.device)     # "cuda:0"
        print(cfg.embedding.batch_size)  # 256
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, Config(value))
            else:
                object.__setattr__(self, key, value)

    # ── Convenience constructors ─────────────────────────────
    @classmethod
    def load(cls, override_path: Optional[str | Path] = None) -> "Config":
        """Load default config, optionally merging an override file."""
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}

        if override_path is not None:
            override_path = Path(override_path)
            with open(override_path, "r", encoding="utf-8") as fh:
                overrides = yaml.safe_load(fh) or {}
            base = _deep_merge(base, overrides)

        # Also allow env-var overrides (e.g. PATHOLENS_SYSTEM_DEVICE=cpu)
        base = cls._apply_env_overrides(base)
        return cls(base)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from a plain dictionary."""
        return cls(data)

    # ── Env-var overrides ────────────────────────────────────
    @staticmethod
    def _apply_env_overrides(data: dict, prefix: str = "PATHOLENS") -> dict:
        """
        Allow overrides via env vars.
        PATHOLENS_SYSTEM_DEVICE=cpu  →  data["system"]["device"] = "cpu"
        """
        result = copy.deepcopy(data)
        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix + "_"):
                continue
            parts = env_key[len(prefix) + 1 :].lower().split("_")
            node = result
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            # Try to cast to int/float/bool
            node[parts[-1]] = _cast_value(env_val)
        return result

    # ── Dict-like helpers ────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to a plain dict."""
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = value.to_dict() if isinstance(value, Config) else value
        return out

    def get(self, key: str, default: Any = None) -> Any:
        """Safe attribute access with a default."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


# ── Casting helper ───────────────────────────────────────────
def _cast_value(val: str) -> Any:
    """Best-effort cast of an env-var string to a Python type."""
    if val.lower() in ("true", "yes", "1"):
        return True
    if val.lower() in ("false", "no", "0"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
