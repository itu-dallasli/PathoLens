"""
Entity Extractor — KARG clinical entity extraction from pathology reports.

Uses an LLM (local or API-based) to extract structured clinical entities
from one or more retrieved reference reports, then cross-validates
across reports via majority voting.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from patholens.entity_extraction.entity_schema import ClinicalEntity, StructuredDiagnosis
from patholens.logger import get_logger

log = get_logger(__name__)

# ── Extraction prompt ────────────────────────────────────────
_EXTRACTION_PROMPT = """\
You are a clinical pathology expert. Extract structured diagnostic \
entities from the following pathology report. Return ONLY valid JSON.

Report:
{report_text}

Extract the following fields (use null if not found):
{{
  "tumor_type": "string or null",
  "histological_grade": "string or null",
  "tumor_stage": "string or null",
  "receptor_status": {{
    "ER": "positive/negative/null",
    "PR": "positive/negative/null",
    "HER2": "positive/negative/null"
  }},
  "ki67_index": "string or null",
  "lymph_node_status": "string or null",
  "surgical_margins": "string or null",
  "necrosis": "string or null",
  "additional_findings": ["string"]
}}
"""


class KARGEntityExtractor:
    """
    Retrieval-augmented clinical entity extractor.

    Processes multiple reference reports, extracts entities from each,
    then merges via majority vote with confidence scoring.

    Parameters
    ----------
    llm_backend : str
        ``"local"``, ``"openai"``, or ``"anthropic"``.
    llm_model : str
        Model identifier.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Max response tokens.
    """

    def __init__(
        self,
        llm_backend: str = "local",
        llm_model: str = "BioMistral-7B",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None  # lazy-loaded

    # ── Public API ───────────────────────────────────────────
    def extract(
        self,
        reference_reports: List[Dict[str, str]],
    ) -> StructuredDiagnosis:
        """
        Extract and merge entities from multiple reference reports.

        Parameters
        ----------
        reference_reports : list of dicts
            Each dict has ``slide_id`` and ``report_text``.

        Returns
        -------
        StructuredDiagnosis
        """
        if not reference_reports:
            log.warning("No reference reports provided — returning empty diagnosis.")
            return StructuredDiagnosis()

        # Extract from each report independently
        extractions: List[Dict[str, Any]] = []
        for report in reference_reports:
            try:
                raw = self._extract_single(report["report_text"])
                raw["_source_id"] = report["slide_id"]
                raw["_source_text"] = report["report_text"]
                extractions.append(raw)
            except Exception as e:
                log.warning("Extraction failed for %s: %s", report.get("slide_id"), e)

        if not extractions:
            return StructuredDiagnosis()

        # Merge via majority vote
        return self._merge_extractions(extractions)

    # ── Single-report extraction ─────────────────────────────
    def _extract_single(self, report_text: str) -> Dict[str, Any]:
        """Send report to LLM and parse JSON response."""
        prompt = _EXTRACTION_PROMPT.format(report_text=report_text)
        response = self._call_llm(prompt)

        # Parse JSON from response
        json_str = self._extract_json(response)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            log.warning("Failed to parse LLM JSON, attempting fix...")
            parsed = self._attempt_json_fix(json_str)
        return parsed

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM backend.
        
        This is a dispatcher that routes to the appropriate backend.
        Actual LLM integration will be configured based on user's
        choice of local vs API model.
        """
        if self.llm_backend == "local":
            return self._call_local(prompt)
        elif self.llm_backend == "openai":
            return self._call_openai(prompt)
        elif self.llm_backend == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {self.llm_backend}")

    def _call_local(self, prompt: str) -> str:
        """Call a local HuggingFace model."""
        if self._llm is None:
            from transformers import pipeline
            self._llm = pipeline(
                "text-generation",
                model=self.llm_model,
                torch_dtype="auto",
                device_map="auto",
            )
        result = self._llm(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        return result[0]["generated_text"]

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.llm_model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # ── JSON helpers ─────────────────────────────────────────
    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object from a text response."""
        # Try to find JSON block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # Try bare JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    @staticmethod
    def _attempt_json_fix(text: str) -> dict:
        """Best-effort repair of malformed JSON."""
        # Remove trailing commas
        text = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            log.error("Could not parse JSON even after fix attempt.")
            return {}

    # ── Merge / majority vote ────────────────────────────────
    def _merge_extractions(
        self,
        extractions: List[Dict[str, Any]],
    ) -> StructuredDiagnosis:
        """Merge multiple extractions via majority vote."""
        diagnosis = StructuredDiagnosis()
        n = len(extractions)

        # Simple fields
        for field_name in [
            "tumor_type", "histological_grade", "tumor_stage",
            "ki67_index", "lymph_node_status", "surgical_margins", "necrosis",
        ]:
            values = []
            sources = []
            for ext in extractions:
                val = ext.get(field_name)
                if val and val != "null" and str(val).lower() != "none":
                    values.append(str(val))
                    sources.append(ext)

            if values:
                # Majority vote
                counter = Counter(values)
                best_value, count = counter.most_common(1)[0]
                confidence = count / n

                # Find a source for this value
                source_ext = next(e for e, v in zip(sources, values) if v == best_value)
                entity = ClinicalEntity(
                    entity_type=field_name,
                    value=best_value,
                    confidence=confidence,
                    source_report_id=source_ext.get("_source_id", "unknown"),
                    source_text_span=self._find_span(
                        source_ext.get("_source_text", ""), best_value
                    ),
                )
                setattr(diagnosis, field_name, entity)

        # Receptor status (compound field)
        for receptor in ["ER", "PR", "HER2"]:
            values = []
            sources = []
            for ext in extractions:
                rs = ext.get("receptor_status", {})
                if isinstance(rs, dict):
                    val = rs.get(receptor)
                    if val and val != "null":
                        values.append(str(val))
                        sources.append(ext)

            if values:
                counter = Counter(values)
                best_value, count = counter.most_common(1)[0]
                source_ext = next(e for e, v in zip(sources, values) if v == best_value)
                diagnosis.receptor_status.append(
                    ClinicalEntity(
                        entity_type=f"receptor_{receptor}",
                        value=best_value,
                        confidence=count / n,
                        source_report_id=source_ext.get("_source_id", "unknown"),
                        source_text_span=self._find_span(
                            source_ext.get("_source_text", ""), receptor
                        ),
                    )
                )

        log.info(
            "Merged %d extractions → %d entities  |  overall_conf=%.2f",
            n,
            len(diagnosis.all_entities),
            diagnosis.overall_confidence,
        )
        return diagnosis

    @staticmethod
    def _find_span(text: str, value: str, context: int = 80) -> str:
        """Find and return a text span containing *value*."""
        idx = text.lower().find(value.lower())
        if idx < 0:
            return ""
        start = max(0, idx - context)
        end = min(len(text), idx + len(value) + context)
        return text[start:end]
