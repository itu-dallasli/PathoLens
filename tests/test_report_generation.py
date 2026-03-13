"""
Tests for patholens.report_generation — FHIR report builder, evidence linker.
"""

from __future__ import annotations

import json

import pytest

from patholens.entity_extraction.entity_schema import (
    ClinicalEntity,
    StructuredDiagnosis,
)
from patholens.explainability.entity_region_mapper import (
    EntityEvidence,
    EvidenceRegion,
)
from patholens.report_generation.evidence_linker import (
    EvidenceLinker,
    EvidenceReport,
)
from patholens.report_generation.fhir_builder import FHIRReportBuilder


# ── Helpers ─────────────────────────────────────────────────
def _make_entity(
    entity_type="tumor_type",
    value="Invasive Ductal Carcinoma",
    confidence=0.9,
    source_report_id="TCGA-001",
    source_text_span="invasive ductal carcinoma, grade 2",
):
    return ClinicalEntity(
        entity_type=entity_type,
        value=value,
        confidence=confidence,
        source_report_id=source_report_id,
        source_text_span=source_text_span,
    )


def _make_diagnosis():
    return StructuredDiagnosis(
        tumor_type=_make_entity(),
        histological_grade=_make_entity(
            entity_type="histological_grade",
            value="Grade 2",
            confidence=0.85,
        ),
        receptor_status=[
            _make_entity(entity_type="receptor_ER", value="positive", confidence=0.95),
            _make_entity(entity_type="receptor_PR", value="positive", confidence=0.90),
            _make_entity(entity_type="receptor_HER2", value="negative", confidence=0.80),
        ],
    )


def _make_entity_evidence(entity: ClinicalEntity):
    return EntityEvidence(
        entity=entity,
        regions=[
            EvidenceRegion(
                bbox=(100, 200, 256, 256),
                attention_score=0.85,
                patch_indices=[0, 1, 2],
            ),
        ],
        reference_text_span=entity.source_text_span,
        overall_attention=0.85,
    )


class TestFHIRReportBuilder:
    """Test FHIR DiagnosticReport generation."""

    def test_basic_report(self):
        """Produces a valid FHIR DiagnosticReport dict."""
        builder = FHIRReportBuilder(confidence_threshold=0.7, language="en")
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]

        report = builder.build(diag, evidence, slide_id="test_001")

        assert isinstance(report, dict)
        assert report["resourceType"] == "DiagnosticReport"

    def test_required_fields(self):
        """Report has all required FHIR fields."""
        builder = FHIRReportBuilder()
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence, slide_id="test")

        assert "resourceType" in report
        assert "id" in report
        assert "status" in report
        assert "category" in report
        assert "code" in report
        assert "conclusion" in report
        assert "contained" in report  # Observations

    def test_status_final_high_confidence(self):
        """High confidence → 'final' status."""
        builder = FHIRReportBuilder(confidence_threshold=0.5)
        diag = _make_diagnosis()  # All entities have confidence >= 0.8
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence)
        assert report["status"] == "final"

    def test_status_preliminary_low_confidence(self):
        """Low confidence → 'preliminary' status."""
        builder = FHIRReportBuilder(confidence_threshold=0.99)
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence)
        assert report["status"] == "preliminary"

    def test_observations_count(self):
        """Number of contained Observations matches entity count."""
        builder = FHIRReportBuilder()
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence)

        observations = [r for r in report["contained"] if r["resourceType"] == "Observation"]
        assert len(observations) == len(diag.all_entities)

    def test_conclusion_text(self):
        """Conclusion includes entity values."""
        builder = FHIRReportBuilder()
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence)

        conclusion = report["conclusion"]
        assert "Invasive Ductal Carcinoma" in conclusion
        assert "Grade 2" in conclusion

    def test_heatmap_media_reference(self):
        """Heatmap path produces a media entry."""
        builder = FHIRReportBuilder()
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence, heatmap_path="/path/heatmap.png")

        assert "media" in report
        assert len(report["media"]) == 1

    def test_to_json(self):
        """to_json produces valid JSON string."""
        builder = FHIRReportBuilder()
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence)

        json_str = builder.to_json(report)
        parsed = json.loads(json_str)
        assert parsed["resourceType"] == "DiagnosticReport"

    def test_empty_diagnosis(self):
        """Empty diagnosis still produces a valid report."""
        builder = FHIRReportBuilder()
        diag = StructuredDiagnosis()
        report = builder.build(diag, evidence=[], slide_id="empty")

        assert report["resourceType"] == "DiagnosticReport"
        assert "Insufficient data" in report["conclusion"]

    def test_extensions_present(self):
        """Extensions include confidence and evidence coverage."""
        builder = FHIRReportBuilder()
        diag = _make_diagnosis()
        evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        report = builder.build(diag, evidence)

        ext_urls = [e["url"] for e in report["extension"]]
        assert any("confidence-score" in u for u in ext_urls)
        assert any("evidence-coverage" in u for u in ext_urls)


class TestEvidenceLinker:
    """Test evidence linking across visual, textual, and retrieval sources."""

    def test_link_produces_report(self):
        """link() returns EvidenceReport with coverage."""
        linker = EvidenceLinker()
        diag = _make_diagnosis()
        visual_evidence = [_make_entity_evidence(e) for e in diag.all_entities]

        retrieval_results = [
            {"slide_id": "TCGA-001", "similarity_score": 0.92},
            {"slide_id": "TCGA-002", "similarity_score": 0.85},
        ]

        report = linker.link(diag, visual_evidence, retrieval_results)
        assert isinstance(report, EvidenceReport)
        assert report.coverage > 0.0

    def test_all_entities_linked(self):
        """When all sources available, coverage should be 1.0."""
        linker = EvidenceLinker()
        diag = _make_diagnosis()
        visual_evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        retrieval_results = [{"slide_id": "TCGA-001", "similarity_score": 0.9}]

        report = linker.link(diag, visual_evidence, retrieval_results)
        assert report.coverage == 1.0

    def test_empty_diagnosis(self):
        """Empty diagnosis produces empty evidence report."""
        linker = EvidenceLinker()
        diag = StructuredDiagnosis()
        report = linker.link(diag, [], [])
        assert isinstance(report, EvidenceReport)
        assert report.coverage == 0.0

    def test_linked_evidence_has_sources(self):
        """Each linked evidence references visual, textual, and retrieval sources."""
        linker = EvidenceLinker()
        diag = _make_diagnosis()
        visual_evidence = [_make_entity_evidence(e) for e in diag.all_entities]
        retrieval_results = [{"slide_id": "TCGA-001", "similarity_score": 0.9}]

        report = linker.link(diag, visual_evidence, retrieval_results)

        for linked in report.linked:
            # At minimum, textual evidence should be present (from source_text_span)
            assert linked.is_linked
            assert linked.textual_evidence != "" or linked.retrieval_source != ""
