"""In-Context Learning (ICL) Engine — Dynamic Few-Shot Example Selection.

Instead of hardcoded IF/THEN rules, this engine selects relevant past cases
from memory and formats them as few-shot examples for the LLM.

The LLM learns from EXAMPLES, not from rules — this is how LLMs naturally work.
Each successful case grows the example pool → the system gets smarter over time.

Architecture:
  1. New patient arrives
  2. ICL Engine queries the memory DB for similar past cases (FTS5 search)
  3. Filters for SUCCESSFUL cases only (IE decision = FINALIZE, confidence > 0.5)
  4. Formats top matches as few-shot examples for R1 prompt injection
  5. After pipeline completes, successful cases auto-enter the pool

This is equivalent to self-training without fine-tuning:
  - No GPU/VRAM needed
  - No catastrophic forgetting
  - Grows organically with each case
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("rrrie-cdss.icl")


@dataclass
class ICLExample:
    """A single In-Context Learning example from a past case."""
    case_id: int
    patient_summary: str    # Abbreviated patient description
    diagnosis: str          # The correct diagnosis
    clinical_pearl: str     # Key lesson learned (if available)
    confidence: float       # IE confidence score
    similarity_rank: float  # FTS5 rank (lower = more similar)


class ICLEngine:
    """Dynamic Few-Shot Example Selection Engine.
    
    Uses the existing CaseStore's FTS5 search to find similar past cases,
    then formats them as ICL examples for prompt injection.
    
    No external dependencies — uses existing memory infrastructure.
    """

    def __init__(self, case_store):
        """
        Args:
            case_store: CaseStore instance (from src/memory/case_store.py)
        """
        self.store = case_store

    def select_examples(
        self,
        patient_text: str,
        max_examples: int = 3,
        min_confidence: float = 0.5,
    ) -> list[ICLExample]:
        """Select the most relevant past cases as few-shot examples.
        
        Criteria:
          - FTS5 similarity match on patient text
          - IE decision = FINALIZE (case was successfully resolved)
          - IE confidence >= min_confidence (case was resolved with confidence)
          - Prefers cases WITH clinical pearls (more instructive)
        
        Args:
            patient_text: The new patient's clinical text
            max_examples: Maximum number of examples to return
            min_confidence: Minimum IE confidence to consider
            
        Returns:
            List of ICLExample sorted by relevance
        """
        # Use the store's keyword extraction for FTS5 search
        keywords = self.store._extract_keywords(patient_text)
        if not keywords:
            return []

        # FTS5 search — find similar cases
        import re
        import sqlite3

        search_terms = keywords[:12]  # Use more keywords for broader matching
        safe_terms = []
        for t in search_terms:
            t_clean = re.sub(r'[^\w]', '', t)
            if t_clean and len(t_clean) >= 3:
                safe_terms.append(f'"{t_clean}"')
        if not safe_terms:
            return []

        fts_query = " OR ".join(safe_terms)

        try:
            rows = self.store.conn.execute(
                """SELECT t.*, rank FROM tier1_fts f
                   JOIN tier1_cases t ON t.id = f.rowid
                   WHERE tier1_fts MATCH ?
                   AND t.ie_decision = 'FINALIZE'
                   AND t.ie_confidence >= ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, min_confidence, max_examples * 2),  # Fetch extra for filtering
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("[ICL] FTS5 search failed: %s", exc)
            return []

        if not rows:
            return []

        examples = []
        for row in rows:
            row_dict = dict(row)
            pearl = row_dict.get("clinical_pearl", "")
            
            # Abbreviate patient text to ~200 chars for prompt efficiency
            pt = row_dict.get("patient_text", "")
            patient_summary = self._abbreviate_patient(pt)

            examples.append(ICLExample(
                case_id=row_dict["id"],
                patient_summary=patient_summary,
                diagnosis=row_dict.get("primary_diagnosis", "Unknown"),
                clinical_pearl=pearl,
                confidence=row_dict.get("ie_confidence", 0.0),
                similarity_rank=row_dict.get("rank", 0.0),
            ))

        # Sort: prefer examples WITH pearls first, then by similarity
        examples.sort(key=lambda e: (
            0 if e.clinical_pearl else 1,  # Pearl examples first
            e.similarity_rank,              # Then by FTS5 rank
        ))

        result = examples[:max_examples]
        if result:
            logger.info(
                "[ICL] Selected %d examples (pearls: %d, avg_conf: %.2f)",
                len(result),
                sum(1 for e in result if e.clinical_pearl),
                sum(e.confidence for e in result) / len(result),
            )
        return result

    def format_for_prompt(self, examples: list[ICLExample]) -> str:
        """Format ICL examples into a prompt-ready string.
        
        Uses a clear, structured format that LLMs can easily learn from.
        Each example shows: patient summary → correct diagnosis → lesson.
        
        Target: ~200-400 tokens total.
        """
        if not examples:
            return ""

        parts = [
            "🧠 SIMILAR PAST CASES (use these as diagnostic references):\n"
        ]

        for i, ex in enumerate(examples, 1):
            block = f"--- CASE {i} ---\n"
            block += f"Patient: {ex.patient_summary}\n"
            block += f"Correct Diagnosis: {ex.diagnosis}\n"
            block += f"Diagnostic Confidence: {ex.confidence:.0%}\n"
            if ex.clinical_pearl:
                block += f"🎓 Key Lesson: {ex.clinical_pearl}\n"
            block += "---"
            parts.append(block)

        return "\n".join(parts)

    def select_failure_examples(
        self,
        patient_text: str,
        max_examples: int = 2,
    ) -> list[ICLExample]:
        """Select past FAILED cases that have clinical pearls (lessons from mistakes).
        
        This is the FAILURE CHANNEL of the dual-channel ICL system.
        Only retrieves cases where:
          - The system misdiagnosed (ie_decision != 'FINALIZE')
          - A clinical pearl was generated (lesson learned)
        
        These are formatted as anti-patterns so the model avoids repeating
        the same mistakes.
        """
        keywords = self.store._extract_keywords(patient_text)
        if not keywords:
            return []

        import re
        import sqlite3

        search_terms = keywords[:12]
        safe_terms = []
        for t in search_terms:
            t_clean = re.sub(r'[^\w]', '', t)
            if t_clean and len(t_clean) >= 3:
                safe_terms.append(f'"{t_clean}"')
        if not safe_terms:
            return []

        fts_query = " OR ".join(safe_terms)

        try:
            rows = self.store.conn.execute(
                """SELECT t.*, rank FROM tier1_fts f
                   JOIN tier1_cases t ON t.id = f.rowid
                   WHERE tier1_fts MATCH ?
                   AND t.ie_decision != 'FINALIZE'
                   AND t.clinical_pearl IS NOT NULL
                   AND t.clinical_pearl != ''
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, max_examples),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("[ICL] Failure examples FTS5 search failed: %s", exc)
            return []

        if not rows:
            return []

        examples = []
        for row in rows:
            row_dict = dict(row)
            pt = row_dict.get("patient_text", "")
            patient_summary = self._abbreviate_patient(pt)

            examples.append(ICLExample(
                case_id=row_dict["id"],
                patient_summary=patient_summary,
                diagnosis=row_dict.get("primary_diagnosis", "Unknown"),
                clinical_pearl=row_dict.get("clinical_pearl", ""),
                confidence=row_dict.get("ie_confidence", 0.0),
                similarity_rank=row_dict.get("rank", 0.0),
            ))

        if examples:
            logger.info(
                "[ICL] Found %d FAILURE examples with pearls for anti-pattern learning",
                len(examples),
            )
        return examples

    def format_failures_for_prompt(self, examples: list[ICLExample]) -> str:
        """Format failure examples as anti-pattern warnings.
        
        These are styled differently from success examples to clearly
        signal: "the system made THIS mistake before — don't repeat it."
        """
        if not examples:
            return ""

        parts = [
            "⚠️ PAST MISTAKES — LEARN FROM THESE ERRORS (do NOT repeat):\n"
        ]

        for i, ex in enumerate(examples, 1):
            block = f"--- MISTAKE {i} ---\n"
            block += f"Patient: {ex.patient_summary}\n"
            block += f"System Misdiagnosed As: {ex.diagnosis}\n"
            if ex.clinical_pearl:
                block += f"🎓 Lesson Learned: {ex.clinical_pearl}\n"
            block += f"⛔ Do NOT repeat this diagnostic error for similar presentations.\n"
            block += "---"
            parts.append(block)

        return "\n".join(parts)

    @staticmethod
    def _abbreviate_patient(text: str, max_chars: int = 250) -> str:
        """Abbreviate patient text for prompt-efficient display.
        
        Extracts: age, sex, chief complaint, key vitals, key labs.
        """
        if len(text) <= max_chars:
            return text.strip()

        # Take first 250 chars and truncate at last sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        cut_point = max(last_period, last_newline)
        if cut_point > max_chars // 2:
            truncated = truncated[:cut_point + 1]
        
        return truncated.strip() + " [...]"
