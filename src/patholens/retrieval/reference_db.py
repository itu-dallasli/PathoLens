"""
Reference Database — Slide metadata & report storage for CMEA.

Provides a unified interface for looking up clinical reports,
labels, and metadata by slide ID.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from patholens.logger import get_logger

log = get_logger(__name__)


class ReferenceDatabase:
    """
    In-memory store for reference slide metadata and reports.

    Can be loaded from a CSV or built programmatically.

    Expected CSV columns:
        slide_id, subtype, grade, stage, report_text, [additional...]
    """

    def __init__(self):
        self._records: Dict[str, dict] = {}

    # ── Loading ──────────────────────────────────────────────
    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "ReferenceDatabase":
        """Load from a metadata CSV."""
        db = cls()
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sid = str(row["slide_id"])
            db._records[sid] = row.to_dict()
        log.info("Loaded %d records from %s", len(db._records), csv_path)
        return db

    @classmethod
    def from_dict(cls, records: Dict[str, dict]) -> "ReferenceDatabase":
        db = cls()
        db._records = records
        return db

    # ── Accessors ────────────────────────────────────────────
    def get_report(self, slide_id: str) -> Optional[str]:
        record = self._records.get(slide_id)
        return record.get("report_text") if record else None

    def get_subtype(self, slide_id: str) -> Optional[str]:
        record = self._records.get(slide_id)
        return record.get("subtype") if record else None

    def get_metadata(self, slide_id: str) -> Optional[dict]:
        return self._records.get(slide_id)

    def get_all_reports(self) -> Dict[str, str]:
        """Return {slide_id: report_text} for all records with reports."""
        return {
            sid: rec["report_text"]
            for sid, rec in self._records.items()
            if rec.get("report_text")
        }

    @property
    def slide_ids(self) -> List[str]:
        return list(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, slide_id: str) -> bool:
        return slide_id in self._records
