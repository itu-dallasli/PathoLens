"""
Evidence Linker — Connect diagnostic statements to their sources.

Ensures every extracted entity is traceable to:
  1. A visual evidence region (RAAF attention + bounding box)
  2. A textual evidence span (from the reference report)
  3. A retrieval source (similar case ID + similarity score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from patholens.entity_extraction.entity_schema import ClinicalEntity, StructuredDiagnosis
from patholens.explainability.entity_region_mapper import EntityEvidence
from patholens.logger import get_logger

log = get_logger(__name__)


@dataclass
class LinkedEvidence:
    """Fully linked evidence for one entity."""
    entity: ClinicalEntity
    visual_evidence: Optional[EntityEvidence] = None  # RAAF region
    textual_evidence: str = ""                        # Source report text span
    retrieval_source: str = ""                        # Slide ID of matched case
    retrieval_score: float = 0.0                      # Similarity score
    is_linked: bool = False                           # Has at least one source


@dataclass
class EvidenceReport:
    """Aggregate evidence report."""
    linked: List[LinkedEvidence] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of entities with at least one evidence link."""
        if not self.linked:
            return 0.0
        return sum(1 for e in self.linked if e.is_linked) / len(self.linked)


class EvidenceLinker:
    """
    Link each diagnostic entity to its evidence sources.

    Merges outputs from RAAF (visual), KARG (textual), and
    CMEA (retrieval) to create a comprehensive evidence chain.
    """

    def link(
        self,
        diagnosis: StructuredDiagnosis,
        entity_evidence: List[EntityEvidence],
        retrieval_results: Optional[List[Dict]] = None,
    ) -> EvidenceReport:
        """
        Parameters
        ----------
        diagnosis : StructuredDiagnosis
        entity_evidence : list[EntityEvidence]
            Visual evidence from RAAF.
        retrieval_results : list[dict], optional
            Retrieval results with ``slide_id`` and ``similarity_score``.

        Returns
        -------
        EvidenceReport
        """
        # Map entity type → visual evidence
        visual_map: Dict[str, EntityEvidence] = {}
        for ev in entity_evidence:
            visual_map[ev.entity.entity_type] = ev

        linked: List[LinkedEvidence] = []
        for entity in diagnosis.all_entities:
            evidence = LinkedEvidence(entity=entity)

            # Visual evidence
            if entity.entity_type in visual_map:
                evidence.visual_evidence = visual_map[entity.entity_type]

            # Textual evidence
            if entity.source_text_span:
                evidence.textual_evidence = entity.source_text_span

            # Retrieval source
            if entity.source_report_id and entity.source_report_id != "unknown":
                evidence.retrieval_source = entity.source_report_id
                # Find score from retrieval results
                if retrieval_results:
                    for rr in retrieval_results:
                        if rr.get("slide_id") == entity.source_report_id:
                            evidence.retrieval_score = rr.get("similarity_score", 0.0)
                            break

            # Mark as linked if any source is present
            evidence.is_linked = bool(
                evidence.visual_evidence
                or evidence.textual_evidence
                or evidence.retrieval_source
            )

            linked.append(evidence)

        report = EvidenceReport(linked=linked)

        log.info(
            "Evidence linking  |  entities=%d  linked=%d  coverage=%.0f%%",
            len(linked),
            sum(1 for e in linked if e.is_linked),
            report.coverage * 100,
        )
        return report
