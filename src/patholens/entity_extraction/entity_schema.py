"""
Entity Schema — Structured clinical entity definitions for KARG.

Provides dataclasses that represent the output of clinical entity
extraction from pathology reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClinicalEntity:
    """A single extracted clinical entity."""

    entity_type: str          # e.g. "tumor_type", "histological_grade"
    value: str                # e.g. "Invasive Ductal Carcinoma"
    confidence: float         # 0.0–1.0
    source_report_id: str     # Which reference report it came from
    source_text_span: str     # Relevant text snippet from the source

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "confidence": self.confidence,
            "source_report_id": self.source_report_id,
            "source_text_span": self.source_text_span,
        }


@dataclass
class StructuredDiagnosis:
    """
    Complete structured diagnosis assembled from extracted entities.

    Mirrors the key fields expected in a breast cancer pathology
    report as defined in the KARG entity schema.
    """

    tumor_type: Optional[ClinicalEntity] = None
    histological_grade: Optional[ClinicalEntity] = None
    tumor_stage: Optional[ClinicalEntity] = None
    receptor_status: List[ClinicalEntity] = field(default_factory=list)
    ki67_index: Optional[ClinicalEntity] = None
    lymph_node_status: Optional[ClinicalEntity] = None
    surgical_margins: Optional[ClinicalEntity] = None
    necrosis: Optional[ClinicalEntity] = None
    additional_findings: List[ClinicalEntity] = field(default_factory=list)

    # ── Aggregate confidence ─────────────────────────────────
    @property
    def overall_confidence(self) -> float:
        """Weighted average confidence of all present entities."""
        entities = self.all_entities
        if not entities:
            return 0.0
        return sum(e.confidence for e in entities) / len(entities)

    @property
    def all_entities(self) -> List[ClinicalEntity]:
        """Flat list of every non-None entity."""
        out: List[ClinicalEntity] = []
        for fld in [
            self.tumor_type,
            self.histological_grade,
            self.tumor_stage,
            self.ki67_index,
            self.lymph_node_status,
            self.surgical_margins,
            self.necrosis,
        ]:
            if fld is not None:
                out.append(fld)
        out.extend(self.receptor_status)
        out.extend(self.additional_findings)
        return out

    @property
    def evidence_coverage(self) -> float:
        """Fraction of entities that have a non-empty source text span."""
        entities = self.all_entities
        if not entities:
            return 0.0
        linked = sum(1 for e in entities if e.source_text_span)
        return linked / len(entities)

    def to_dict(self) -> dict:
        return {
            "tumor_type": self.tumor_type.to_dict() if self.tumor_type else None,
            "histological_grade": (
                self.histological_grade.to_dict()
                if self.histological_grade
                else None
            ),
            "tumor_stage": self.tumor_stage.to_dict() if self.tumor_stage else None,
            "receptor_status": [e.to_dict() for e in self.receptor_status],
            "ki67_index": self.ki67_index.to_dict() if self.ki67_index else None,
            "lymph_node_status": (
                self.lymph_node_status.to_dict()
                if self.lymph_node_status
                else None
            ),
            "surgical_margins": (
                self.surgical_margins.to_dict()
                if self.surgical_margins
                else None
            ),
            "necrosis": self.necrosis.to_dict() if self.necrosis else None,
            "additional_findings": [e.to_dict() for e in self.additional_findings],
            "overall_confidence": self.overall_confidence,
            "evidence_coverage": self.evidence_coverage,
        }
