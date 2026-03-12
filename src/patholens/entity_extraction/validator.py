"""
Entity Validator — Rule-based sanity checks for extracted clinical entities.

Validates entity values against known ontologies, ranges, and
cross-entity consistency rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from patholens.entity_extraction.entity_schema import StructuredDiagnosis
from patholens.logger import get_logger

log = get_logger(__name__)

# ── Known valid values ───────────────────────────────────────
VALID_TUMOR_TYPES = {
    "invasive ductal carcinoma",
    "invasive lobular carcinoma",
    "ductal carcinoma in situ",
    "lobular carcinoma in situ",
    "mucinous carcinoma",
    "medullary carcinoma",
    "tubular carcinoma",
    "papillary carcinoma",
    "metaplastic carcinoma",
    "inflammatory breast cancer",
    "phyllodes tumor",
    "paget disease",
    "idc",
    "ilc",
    "dcis",
    "lcis",
}

VALID_GRADES = {"1", "2", "3", "grade 1", "grade 2", "grade 3", "i", "ii", "iii",
                "low", "intermediate", "high", "well differentiated",
                "moderately differentiated", "poorly differentiated"}

VALID_STAGES = {"i", "ii", "iii", "iv", "ia", "ib", "iia", "iib", "iiia", "iiib",
                "iiic", "1", "2", "3", "4", "t1", "t2", "t3", "t4"}

VALID_RECEPTOR_VALUES = {"positive", "negative", "equivocal", "indeterminate",
                          "+", "-", "++", "+++", "0", "1+", "2+", "3+"}


@dataclass
class ValidationIssue:
    """A single validation warning or error."""
    entity_type: str
    severity: str       # "warning" or "error"
    message: str


@dataclass
class ValidationResult:
    """Aggregate validation outcome."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]


class EntityValidator:
    """Rule-based validator for StructuredDiagnosis."""

    def validate(self, diagnosis: StructuredDiagnosis) -> ValidationResult:
        issues: List[ValidationIssue] = []

        # Tumor type
        if diagnosis.tumor_type:
            val = diagnosis.tumor_type.value.lower().strip()
            if val not in VALID_TUMOR_TYPES:
                issues.append(ValidationIssue(
                    "tumor_type", "warning",
                    f"Unrecognised tumor type: '{diagnosis.tumor_type.value}'",
                ))

        # Grade
        if diagnosis.histological_grade:
            val = diagnosis.histological_grade.value.lower().strip()
            if val not in VALID_GRADES:
                issues.append(ValidationIssue(
                    "histological_grade", "warning",
                    f"Unrecognised grade: '{diagnosis.histological_grade.value}'",
                ))

        # Stage
        if diagnosis.tumor_stage:
            val = diagnosis.tumor_stage.value.lower().strip()
            if val not in VALID_STAGES:
                issues.append(ValidationIssue(
                    "tumor_stage", "warning",
                    f"Unrecognised stage: '{diagnosis.tumor_stage.value}'",
                ))

        # Receptor status
        for entity in diagnosis.receptor_status:
            val = entity.value.lower().strip()
            if val not in VALID_RECEPTOR_VALUES:
                issues.append(ValidationIssue(
                    entity.entity_type, "warning",
                    f"Unrecognised receptor value: '{entity.value}'",
                ))

        # Cross-entity consistency checks
        issues.extend(self._check_consistency(diagnosis))

        is_valid = len([i for i in issues if i.severity == "error"]) == 0

        if issues:
            log.warning("Validation found %d issues (%d errors)", len(issues),
                        len([i for i in issues if i.severity == "error"]))
        else:
            log.info("Validation passed — no issues.")

        return ValidationResult(is_valid=is_valid, issues=issues)

    def _check_consistency(self, diagnosis: StructuredDiagnosis) -> List[ValidationIssue]:
        """Detect logical inconsistencies between entities."""
        issues: List[ValidationIssue] = []

        # If tumor type is "benign" but stage is advanced → contradiction
        if diagnosis.tumor_type and diagnosis.tumor_stage:
            tt = diagnosis.tumor_type.value.lower()
            ts = diagnosis.tumor_stage.value.lower()
            benign_terms = {"benign", "fibroadenoma", "phyllodes"}
            advanced_stages = {"iii", "iv", "iiia", "iiib", "iiic", "3", "4"}
            if any(b in tt for b in benign_terms) and ts in advanced_stages:
                issues.append(ValidationIssue(
                    "consistency", "error",
                    f"Benign tumor type '{tt}' contradicts advanced stage '{ts}'",
                ))

        # Low confidence warning
        if diagnosis.overall_confidence < 0.5:
            issues.append(ValidationIssue(
                "confidence", "warning",
                f"Overall confidence is low ({diagnosis.overall_confidence:.2f}). "
                "Manual review recommended.",
            ))

        return issues
