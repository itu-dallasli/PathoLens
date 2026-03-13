"""
Tests for patholens.entity_extraction — Entity extraction, validation, schema.

Uses a mock LLM backend (no model download required).
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from patholens.entity_extraction.entity_schema import (
    ClinicalEntity,
    StructuredDiagnosis,
)
from patholens.entity_extraction.entity_extractor import KARGEntityExtractor
from patholens.entity_extraction.validator import (
    EntityValidator,
    ValidationResult,
)
from tests.conftest import MOCK_LLM_RESPONSE


# ── Helpers ─────────────────────────────────────────────────
def _make_entity(
    entity_type: str = "tumor_type",
    value: str = "Invasive Ductal Carcinoma",
    confidence: float = 0.9,
    source_report_id: str = "test",
    source_text_span: str = "invasive ductal carcinoma",
) -> ClinicalEntity:
    return ClinicalEntity(
        entity_type=entity_type,
        value=value,
        confidence=confidence,
        source_report_id=source_report_id,
        source_text_span=source_text_span,
    )


class TestEntitySchema:
    """Test ClinicalEntity and StructuredDiagnosis dataclasses."""

    def test_entity_to_dict(self):
        entity = _make_entity()
        d = entity.to_dict()
        assert d["entity_type"] == "tumor_type"
        assert d["value"] == "Invasive Ductal Carcinoma"
        assert d["confidence"] == 0.9

    def test_diagnosis_overall_confidence(self):
        diag = StructuredDiagnosis(
            tumor_type=_make_entity(confidence=0.8),
            histological_grade=_make_entity(
                entity_type="histological_grade", value="Grade 2", confidence=0.6
            ),
        )
        assert diag.overall_confidence == pytest.approx(0.7, abs=0.01)

    def test_diagnosis_empty(self):
        diag = StructuredDiagnosis()
        assert diag.overall_confidence == 0.0
        assert diag.evidence_coverage == 0.0
        assert len(diag.all_entities) == 0

    def test_all_entities(self):
        diag = StructuredDiagnosis(
            tumor_type=_make_entity(),
            receptor_status=[
                _make_entity(entity_type="receptor_ER", value="positive"),
                _make_entity(entity_type="receptor_PR", value="negative"),
            ],
        )
        entities = diag.all_entities
        assert len(entities) == 3

    def test_evidence_coverage(self):
        diag = StructuredDiagnosis(
            tumor_type=_make_entity(source_text_span="some text"),
            histological_grade=_make_entity(
                entity_type="histological_grade",
                value="Grade 2",
                source_text_span="",    # No source text
            ),
        )
        assert diag.evidence_coverage == pytest.approx(0.5)

    def test_to_dict(self):
        diag = StructuredDiagnosis(tumor_type=_make_entity())
        d = diag.to_dict()
        assert d["tumor_type"]["value"] == "Invasive Ductal Carcinoma"
        assert "overall_confidence" in d
        assert "evidence_coverage" in d


class TestKARGEntityExtractor:
    """Test entity extraction with mock LLM."""

    def _make_extractor(self):
        return KARGEntityExtractor(
            llm_backend="local",
            llm_model="mock",
            temperature=0.0,
            max_tokens=512,
        )

    def test_extract_single_report(self, fake_reference_reports, mock_llm_response):
        """Extract entities from one report using mock LLM."""
        extractor = self._make_extractor()

        with patch.object(extractor, "_call_llm", return_value=mock_llm_response):
            result = extractor.extract(fake_reference_reports[:1])

        assert isinstance(result, StructuredDiagnosis)
        assert result.tumor_type is not None
        assert result.tumor_type.value == "Invasive Ductal Carcinoma"

    def test_majority_vote_merge(self, fake_reference_reports, mock_llm_response):
        """Multiple extractions merge with majority vote."""
        extractor = self._make_extractor()

        with patch.object(extractor, "_call_llm", return_value=mock_llm_response):
            result = extractor.extract(fake_reference_reports)

        assert result.tumor_type is not None
        # All 3 reports produce same value → confidence = 1.0
        assert result.tumor_type.confidence == pytest.approx(1.0)

    def test_empty_reports(self):
        """Empty report list returns empty diagnosis."""
        extractor = self._make_extractor()
        result = extractor.extract([])
        assert isinstance(result, StructuredDiagnosis)
        assert len(result.all_entities) == 0

    def test_json_extraction_from_markdown(self):
        """_extract_json handles code-fenced JSON."""
        text = '```json\n{"tumor_type": "IDC"}\n```'
        result = KARGEntityExtractor._extract_json(text)
        assert '"tumor_type"' in result

    def test_json_extraction_bare(self):
        """_extract_json handles bare JSON."""
        text = 'Some text before {"tumor_type": "IDC"} and after'
        result = KARGEntityExtractor._extract_json(text)
        assert '"tumor_type"' in result

    def test_json_fix_trailing_comma(self):
        """_attempt_json_fix repairs trailing commas."""
        broken = '{"key": "value",}'
        result = KARGEntityExtractor._attempt_json_fix(broken)
        assert result == {"key": "value"}

    def test_extraction_failure_graceful(self, fake_reference_reports):
        """If LLM returns garbage, extraction fails gracefully."""
        extractor = self._make_extractor()

        with patch.object(extractor, "_call_llm", return_value="not json at all!!!"):
            result = extractor.extract(fake_reference_reports[:1])

        # Should still return a StructuredDiagnosis, possibly empty
        assert isinstance(result, StructuredDiagnosis)


class TestEntityValidator:
    """Test rule-based entity validation."""

    def test_valid_entities(self):
        """Known valid entities pass validation."""
        diag = StructuredDiagnosis(
            tumor_type=_make_entity(value="Invasive Ductal Carcinoma"),
            histological_grade=_make_entity(
                entity_type="histological_grade", value="Grade 2"
            ),
        )
        validator = EntityValidator()
        result = validator.validate(diag)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_unknown_tumor_type(self):
        """Unrecognised tumor type produces a warning."""
        diag = StructuredDiagnosis(
            tumor_type=_make_entity(value="Unknown Alien Cancer"),
        )
        validator = EntityValidator()
        result = validator.validate(diag)
        assert len(result.warnings) > 0
        assert any("tumor_type" in w.entity_type for w in result.warnings)

    def test_consistency_check_benign_advanced(self):
        """Benign tumor + advanced stage produces an error."""
        diag = StructuredDiagnosis(
            tumor_type=_make_entity(value="Phyllodes Tumor"),
            tumor_stage=_make_entity(entity_type="tumor_stage", value="IV"),
        )
        validator = EntityValidator()
        result = validator.validate(diag)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_low_confidence_warning(self):
        """Low overall confidence produces a warning."""
        diag = StructuredDiagnosis(
            tumor_type=_make_entity(confidence=0.3),
        )
        validator = EntityValidator()
        result = validator.validate(diag)
        confidence_warnings = [w for w in result.warnings if w.entity_type == "confidence"]
        assert len(confidence_warnings) > 0

    def test_empty_diagnosis(self):
        """Empty diagnosis passes validation."""
        diag = StructuredDiagnosis()
        validator = EntityValidator()
        result = validator.validate(diag)
        assert result.is_valid

    def test_invalid_receptor_value(self):
        """Unrecognised receptor value produces a warning."""
        diag = StructuredDiagnosis(
            receptor_status=[
                _make_entity(entity_type="receptor_ER", value="strongly positive"),
            ],
        )
        validator = EntityValidator()
        result = validator.validate(diag)
        assert len(result.warnings) > 0
