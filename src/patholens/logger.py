"""
Centralised logging for PathoLens.

Provides a ``get_logger(name)`` factory that returns a ``logging.Logger``
pre-configured with:
  * Rich console handler (coloured, readable)
  * Rotating file handler (``logs/patholens.log``)

Usage::

    from patholens.logger import get_logger
    log = get_logger(__name__)
    log.info("Processing slide %s", slide_id)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

from patholens.config import PROJECT_ROOT

# ── Defaults ─────────────────────────────────────────────────
_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "patholens.log"
_DEFAULT_LEVEL = logging.INFO
_FMT = "%(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_initialised = False


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int | str = _DEFAULT_LEVEL) -> None:
    """
    One-time global logging configuration.

    Called automatically by ``get_logger`` on first use, but may be
    invoked explicitly if you need an early override.
    """
    global _initialised
    if _initialised:
        return
    _initialised = True

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    _ensure_log_dir()

    # Root logger
    root = logging.getLogger("patholens")
    root.setLevel(level)

    # ── Rich console handler ─────────────────────────────────
    console_handler = RichHandler(
        level=level,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
    root.addHandler(console_handler)

    # ── File handler ─────────────────────────────────────────
    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # file always captures DEBUG+
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt=_DATE_FMT,
        )
    )
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``patholens`` namespace.

    Automatically initialises the logging subsystem on first call.
    """
    setup_logging()
    return logging.getLogger(f"patholens.{name}")
