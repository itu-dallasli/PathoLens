"""
FHIR Report Builder — HL7 FHIR R4 DiagnosticReport generation.

Assembles a FHIR-compliant DiagnosticReport from structured clinical
entities and evidence links produced by KARG and RAAF.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from patholens.entity_extraction.entity_schema import ClinicalEntity, StructuredDiagnosis
from patholens.explainability.entity_region_mapper import EntityEvidence
from patholens.logger import get_logger

log = get_logger(__name__)


class FHIRReportBuilder:
    """
    Build HL7 FHIR R4 DiagnosticReport resources.

    Parameters
    ----------
    confidence_threshold : float
        Below this → status is ``"preliminary"``; above → ``"final"``.
    language : str
        Report language tag.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        language: str = "en",
    ):
        self.confidence_threshold = confidence_threshold
        self.language = language

    def build(
        self,
        diagnosis: StructuredDiagnosis,
        evidence: List[EntityEvidence],
        slide_id: str = "",
        heatmap_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete FHIR R4 DiagnosticReport.

        Parameters
        ----------
        diagnosis : StructuredDiagnosis
        evidence : list[EntityEvidence]
        slide_id : str
        heatmap_path : str, optional

        Returns
        -------
        dict — FHIR JSON-serialisable resource.
        """
        confidence = diagnosis.overall_confidence
        status = "final" if confidence >= self.confidence_threshold else "preliminary"

        report_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Build Observations (one per entity)
        observations = []
        for entity in diagnosis.all_entities:
            obs = self._build_observation(entity)
            observations.append(obs)

        # Build evidence references
        evidence_refs = self._build_evidence_references(evidence)

        # Conclusion text
        conclusion = self._generate_conclusion(diagnosis)

        # Assemble DiagnosticReport
        report = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "meta": {
                "profile": [
                    "http://hl7.org/fhir/StructureDefinition/DiagnosticReport"
                ],
                "lastUpdated": now,
            },
            "language": self.language,
            "status": status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "PAT",
                            "display": "Pathology",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "22034-3",
                        "display": "Pathology study",
                    }
                ]
            },
            "subject": {
                "reference": f"Specimen/{slide_id}",
                "display": f"WSI Slide {slide_id}",
            },
            "effectiveDateTime": now,
            "issued": now,
            "result": [
                {"reference": f"#obs-{obs['id']}"} for obs in observations
            ],
            "conclusion": conclusion,
            "conclusionCode": self._conclusion_codes(diagnosis),
            "contained": observations,
            "extension": [
                {
                    "url": "http://patholens.itu.edu.tr/fhir/StructureDefinition/confidence-score",
                    "valueDecimal": round(confidence, 4),
                },
                {
                    "url": "http://patholens.itu.edu.tr/fhir/StructureDefinition/evidence-coverage",
                    "valueDecimal": round(diagnosis.evidence_coverage, 4),
                },
            ],
        }

        # Attach heatmap as media reference
        if heatmap_path:
            report["media"] = [
                {
                    "comment": "Attention heatmap overlay",
                    "link": {"reference": f"Media/{heatmap_path}"},
                }
            ]

        # Attach evidence details
        if evidence_refs:
            report["extension"].append({
                "url": "http://patholens.itu.edu.tr/fhir/StructureDefinition/evidence-details",
                "valueString": json.dumps(evidence_refs),
            })

        log.info(
            "Built FHIR DiagnosticReport  |  id=%s  status=%s  "
            "entities=%d  confidence=%.2f",
            report_id,
            status,
            len(observations),
            confidence,
        )
        return report

    def to_json(self, report: Dict[str, Any], indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(report, indent=indent, ensure_ascii=False)

    # ── Observation builder ──────────────────────────────────
    @staticmethod
    def _build_observation(entity: ClinicalEntity) -> Dict[str, Any]:
        obs_id = str(uuid.uuid4())[:8]
        return {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://patholens.itu.edu.tr/fhir/CodeSystem/entity-type",
                        "code": entity.entity_type,
                        "display": entity.entity_type.replace("_", " ").title(),
                    }
                ]
            },
            "valueString": entity.value,
            "extension": [
                {
                    "url": "http://patholens.itu.edu.tr/fhir/StructureDefinition/confidence",
                    "valueDecimal": round(entity.confidence, 4),
                },
                {
                    "url": "http://patholens.itu.edu.tr/fhir/StructureDefinition/source-report",
                    "valueString": entity.source_report_id,
                },
                {
                    "url": "http://patholens.itu.edu.tr/fhir/StructureDefinition/source-text",
                    "valueString": entity.source_text_span,
                },
            ],
        }

    # ── Evidence references ──────────────────────────────────
    @staticmethod
    def _build_evidence_references(
        evidence_list: List[EntityEvidence],
    ) -> List[dict]:
        refs = []
        for ev in evidence_list:
            entry = {
                "entity_type": ev.entity.entity_type,
                "entity_value": ev.entity.value,
                "overall_attention": round(ev.overall_attention, 4),
                "reference_text": ev.reference_text_span,
                "regions": [
                    {
                        "bbox": list(r.bbox),
                        "attention_score": round(r.attention_score, 4),
                    }
                    for r in ev.regions
                ],
            }
            refs.append(entry)
        return refs

    # ── Conclusion text generation ───────────────────────────
    @staticmethod
    def _generate_conclusion(diagnosis: StructuredDiagnosis) -> str:
        parts = []

        if diagnosis.tumor_type:
            parts.append(f"Tumor type: {diagnosis.tumor_type.value}.")

        if diagnosis.histological_grade:
            parts.append(f"Histological grade: {diagnosis.histological_grade.value}.")

        if diagnosis.tumor_stage:
            parts.append(f"Stage: {diagnosis.tumor_stage.value}.")

        for receptor in diagnosis.receptor_status:
            name = receptor.entity_type.replace("receptor_", "").upper()
            parts.append(f"{name}: {receptor.value}.")

        if diagnosis.ki67_index:
            parts.append(f"Ki-67 index: {diagnosis.ki67_index.value}.")

        if diagnosis.lymph_node_status:
            parts.append(f"Lymph node status: {diagnosis.lymph_node_status.value}.")

        if diagnosis.surgical_margins:
            parts.append(f"Surgical margins: {diagnosis.surgical_margins.value}.")

        if diagnosis.necrosis:
            parts.append(f"Necrosis: {diagnosis.necrosis.value}.")

        for finding in diagnosis.additional_findings:
            parts.append(f"Additional: {finding.value}.")

        if not parts:
            return "Insufficient data for diagnostic conclusion."

        return " ".join(parts)

    # ── Conclusion codes ─────────────────────────────────────
    @staticmethod
    def _conclusion_codes(diagnosis: StructuredDiagnosis) -> List[dict]:
        codes = []
        if diagnosis.tumor_type:
            codes.append({
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "display": diagnosis.tumor_type.value,
                }]
            })
        return codes
