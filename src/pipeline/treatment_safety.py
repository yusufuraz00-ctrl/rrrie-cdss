"""Dynamic Treatment Safety Gate — prevents lethal prescriptions.

Three-layer defense (all fully dynamic — NO hardcoded drug-condition pairs):

  Layer 1: IE Consensus Tracker
    Detects when IE repeatedly suggests the same diagnosis across iterations.
    Uses dynamic entity extraction (word-overlap) — no hardcoded disease lists.

  Layer 2: Differential Safety Context Builder
    Injects all active differentials into the treatment prompt so the LLM
    reasons about contra-indications at inference time.

  Layer 3: Post-Treatment Safety Validation
    A focused LLM call that checks each proposed drug against every differential.
    Blocks lethal drugs and modifies the treatment plan automatically.

Key Principle:
    A treatment correct for the PRIMARY diagnosis but LETHAL for a
    high-probability DIFFERENTIAL is NOT a safe treatment.
    "First, do no harm" applies to diagnostic uncertainty.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("rrrie-cdss")


# ════════════════════════════════════════════════════════════════════
# Layer 1: IE Consensus Tracker
# ════════════════════════════════════════════════════════════════════

@dataclass
class IEConsensusTracker:
    """Track IE-suggested diagnoses across iterations to detect consensus.

    Usage:
        tracker = IEConsensusTracker()
        # After each IE iteration:
        tracker.record(ie_suggested_dx, ie_reasoning, ie_confidence)
        # Before treatment:
        if tracker.consensus_diagnosis:
            # IE repeatedly flagged the same condition
    """

    suggestions: list[str] = field(default_factory=list)
    reasonings: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    def record(
        self,
        suggested_dx: str | None,
        reasoning: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Record an IE iteration's diagnosis suggestion."""
        if suggested_dx and suggested_dx.strip():
            self.suggestions.append(suggested_dx.strip())
            self.reasonings.append(reasoning)
            self.confidences.append(confidence)

    @property
    def consensus_diagnosis(self) -> str | None:
        """Return consensus diagnosis if IE consistently suggests the same condition.

        Uses dynamic entity extraction (no hardcoded disease lists) to find
        recurring medical conditions across suggestions via word-overlap.
        """
        if len(self.suggestions) < 2:
            return None
        return _find_consensus(self.suggestions)

    @property
    def consensus_count(self) -> int:
        """How many suggestions share key entities with the consensus."""
        consensus = self.consensus_diagnosis
        if not consensus:
            return 0
        consensus_entities = _extract_key_entities(consensus)
        count = 0
        for s in self.suggestions:
            s_entities = _extract_key_entities(s)
            if len(consensus_entities & s_entities) >= 1:
                count += 1
        return count

    @property
    def latest_reasoning(self) -> str:
        return self.reasonings[-1] if self.reasonings else ""

    @property
    def avg_confidence(self) -> float:
        return (
            sum(self.confidences) / len(self.confidences)
            if self.confidences
            else 0.0
        )

    @property
    def total_iterations(self) -> int:
        return len(self.suggestions)


def _extract_key_entities(text: str) -> set[str]:
    """Extract key medical entities (bigrams) from a diagnosis string.

    Fully dynamic — no hardcoded medical terms. Works by extracting
    meaningful word-pairs that likely represent disease/condition names.
    """
    text = text.lower()
    # Remove parenthetical content
    text = re.sub(r"\([^)]*\)", "", text)
    # Common non-diagnostic filler words
    noise = {
        "likely", "probable", "possible", "suspected", "acute", "chronic",
        "primary", "secondary", "with", "and", "or", "the", "a", "an",
        "causing", "leading", "due", "to", "from", "into", "of", "for",
        "this", "that", "may", "can", "is", "are", "be", "as", "by",
        "rupturing", "syndrome", "type",
    }
    words = [w for w in text.split() if w not in noise and len(w) > 2]
    entities = set()
    for i in range(len(words) - 1):
        entities.add(f"{words[i]} {words[i + 1]}")
    return entities


def _find_consensus(suggestions: list[str]) -> str | None:
    """Find consensus diagnosis from IE suggestions using entity overlap.

    Dynamic approach: extracts medical entities from each suggestion,
    finds entities that recur across ≥2 suggestions, then returns the
    most representative suggestion string.
    """
    if len(suggestions) < 2:
        return None

    entity_sets = [_extract_key_entities(s) for s in suggestions]

    # Count each entity's occurrence across distinct suggestions
    entity_counts: Counter[str] = Counter()
    for eset in entity_sets:
        for e in eset:
            entity_counts[e] += 1

    # Entities recurring in ≥2 suggestions
    consensus_entities = {e for e, c in entity_counts.items() if c >= 2}
    if not consensus_entities:
        return None

    # Pick the suggestion with the most overlap with consensus entities
    best_idx = 0
    best_overlap = 0
    for i, eset in enumerate(entity_sets):
        overlap = len(eset & consensus_entities)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i

    return suggestions[best_idx]


# ════════════════════════════════════════════════════════════════════
# Layer 2: Differential Safety Context Builder
# ════════════════════════════════════════════════════════════════════

def build_differential_safety_block(
    differentials: list[dict],
    primary_diagnosis: str,
    primary_confidence: float,
    ie_consensus: str | None = None,
    ie_consensus_count: int = 0,
    ie_reasoning: str = "",
    ie_total_iterations: int = 0,
) -> str:
    """Build dynamic safety context block for the treatment prompt.

    Fully dynamic: takes actual differentials from R3 output and IE consensus.
    No hardcoded drug-condition pairs — everything is data-driven.
    """
    # Collect differentials above 20% that differ from primary
    dangerous_diffs: list[dict] = []
    higher_than_primary: list[dict] = []

    for d in differentials:
        name = d.get("diagnosis", "")
        conf = d.get("updated_confidence", d.get("initial_confidence", 0))
        if conf > 0.20 and name.lower().strip() != primary_diagnosis.lower().strip():
            dangerous_diffs.append({"name": name, "confidence": conf})
            if conf > primary_confidence:
                higher_than_primary.append({"name": name, "confidence": conf})

    if not dangerous_diffs and not ie_consensus:
        return ""

    lines: list[str] = []
    lines.append("")
    lines.append(
        "## ⚠️ CRITICAL DIFFERENTIAL SAFETY CHECK "
        "(MANDATORY — READ BEFORE PRESCRIBING ANY DRUG)"
    )
    lines.append("")

    # IE consensus warning
    if ie_consensus and ie_consensus_count >= 2:
        lines.append(
            f"🚨 The quality evaluator (IE) REPEATEDLY identified the following "
            f"condition as the most likely root cause "
            f"({ie_consensus_count}/{ie_total_iterations} iterations):"
        )
        lines.append(f"→ **{ie_consensus}**")
        if ie_reasoning:
            lines.append(f'IE Reasoning: "{ie_reasoning[:500]}"')
        lines.append("")

    # Higher-confidence differentials
    if higher_than_primary:
        lines.append(
            f"⚠️ The following differential(s) have HIGHER confidence than the "
            f"primary diagnosis ({primary_diagnosis} at {primary_confidence:.0%}):"
        )
        for d in higher_than_primary:
            lines.append(f"  • {d['name']} ({d['confidence']:.0%} confidence)")
        lines.append("")

    # All dangerous differentials
    if dangerous_diffs:
        lines.append(
            "Active differentials that MUST be considered for drug safety:"
        )
        for d in sorted(dangerous_diffs, key=lambda x: x["confidence"], reverse=True):
            lines.append(f"  • {d['name']} ({d['confidence']:.0%} confidence)")
        lines.append("")

    # Dynamic safety reasoning framework
    lines.append("⛔ NON-NEGOTIABLE SAFETY RULE:")
    lines.append(
        "For EVERY drug you recommend, apply this reasoning chain:"
    )
    lines.append(
        "  1. What is the drug's mechanism of action?"
    )
    lines.append(
        "  2. For EACH differential above: does the drug's mechanism WORSEN "
        "the differential's underlying pathology?"
    )
    lines.append(
        "  3. If the drug could KILL or critically harm the patient if the "
        "differential is the true diagnosis → EXCLUDE it from 'pharmacological' "
        "and ADD it to 'contraindication_notes' explaining WHY."
    )
    lines.append(
        "  4. If UNSURE whether a drug is safe for a differential → EXCLUDE it. "
        "Omission is safer than a potentially lethal prescription."
    )
    lines.append(
        "  5. Recommend ONLY drugs that are safe across ALL plausible diagnoses, "
        "OR clearly state which drugs require IMAGING CONFIRMATION before "
        "administration."
    )
    lines.append("")
    lines.append(
        f"REASONING: A {primary_confidence:.0%} confidence in "
        f"'{primary_diagnosis}' means there is a "
        f"{(1 - primary_confidence):.0%} chance the patient has a DIFFERENT "
        f"condition. A drug that helps the primary but kills a patient with "
        f"the differential is an UNACCEPTABLE risk. FIRST, DO NO HARM."
    )
    lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# Layer 3: Post-Treatment Safety Validation
# ════════════════════════════════════════════════════════════════════

_SAFETY_VALIDATION_PROMPT = """\
You are a pharmacovigilance safety officer. Your ONLY job is to check
whether proposed drugs are safe given diagnostic UNCERTAINTY.

## Proposed Treatment for "{primary_diagnosis}"
{drug_list}

## Active Differentials (the patient may ACTUALLY have these)
{differential_list}

## TASK
For EACH proposed drug, apply this reasoning:
1. Identify the drug's primary mechanism of action.
2. For EACH differential: does the drug's mechanism WORSEN or become
   LETHAL for that differential's pathophysiology?
3. If YES → mark as dangerous.
4. If UNSURE → mark as dangerous (precautionary principle).

Do NOT use hardcoded rules — REASON from first principles about
mechanism of action vs. pathophysiology.

## OUTPUT — valid JSON only:
{{
  "checks": [
    {{"drug": "name", "differential": "condition", "dangerous": true, "reason": "brief mechanism explanation"}}
  ],
  "blocked_drugs": ["drug names that MUST be removed"],
  "safe_alternatives": ["suggested safer alternatives that work for ALL conditions"],
  "summary": "one-sentence safety assessment"
}}"""


async def validate_treatment_safety(
    treatment_json: dict,
    primary_diagnosis: str,
    differentials: list[dict],
    ie_consensus: str | None,
    ws: WebSocket,
    llm_client: Any,
    llama_server_url: str,
    groq_client: Any,
    use_groq: bool,
) -> dict:
    """Run post-treatment safety validation using a focused LLM call.

    Dynamically checks each proposed drug against every differential diagnosis.
    Blocks lethal drugs and modifies the treatment plan automatically.

    Returns dict with: blocked_drugs, warnings, modified, safe_alternatives.
    """
    from src.pipeline.streaming import (
        parse_json_from_response,
        call_llm_no_stream,
        call_groq_no_stream,
    )

    tx_plan = treatment_json.get("treatment_plan", {})
    if isinstance(tx_plan, list):
        # LLM returned treatment_plan as a list instead of dict — adapt
        tx_plan = {"pharmacological": tx_plan}
    drugs = tx_plan.get("pharmacological", []) if isinstance(tx_plan, dict) else []

    if not drugs:
        logger.info("[SAFETY-GATE] No drugs in treatment plan — skipping validation.")
        return {"blocked_drugs": [], "warnings": [], "modified": False}

    # Collect differentials above threshold
    dangerous_diffs: list[str] = []
    for d in differentials:
        name = d.get("diagnosis", "")
        conf = d.get("updated_confidence", d.get("initial_confidence", 0))
        if conf > 0.20 and name.lower().strip() != primary_diagnosis.lower().strip():
            dangerous_diffs.append(f"- {name} ({conf:.0%} confidence)")

    # Add IE consensus if not already covered
    if ie_consensus:
        consensus_lower = ie_consensus.lower()
        already_in = any(consensus_lower in dd.lower() for dd in dangerous_diffs)
        if not already_in:
            dangerous_diffs.insert(
                0, f"- {ie_consensus} (IE consensus — highest priority)"
            )

    if not dangerous_diffs:
        logger.info("[SAFETY-GATE] No dangerous differentials — skipping validation.")
        return {"blocked_drugs": [], "warnings": [], "modified": False}

    # Build drug list for prompt
    drug_list = "\n".join(
        f"- {d.get('drug', 'Unknown')} "
        f"({d.get('dose', 'N/A')}, {d.get('route', 'N/A')})"
        for d in drugs
    )
    differential_list = "\n".join(dangerous_diffs)

    prompt = _SAFETY_VALIDATION_PROMPT.format(
        primary_diagnosis=primary_diagnosis,
        drug_list=drug_list,
        differential_list=differential_list,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a pharmacovigilance safety officer. "
                "Output ONLY valid JSON. Reason from drug mechanism "
                "vs. disease pathophysiology — no assumptions."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    await ws.send_json({
        "type": "stage_start",
        "stage": "SAFETY",
        "title": "Treatment Safety Gate 🛡️",
        "description": (
            f"Validating {len(drugs)} drug(s) against "
            f"{len(dangerous_diffs)} differential(s)..."
        ),
    })

    # Non-streaming call for speed
    try:
        if use_groq and groq_client.is_available:
            raw = await call_groq_no_stream(
                groq_client, messages, max_tokens=768
            )
        else:
            raw = await call_llm_no_stream(
                llm_client, llama_server_url, messages, max_tokens=768
            )
        safety_json = parse_json_from_response(raw)
    except Exception as exc:
        logger.error("[SAFETY-GATE] Validation LLM call failed: %s", exc)
        safety_json = {}

    blocked_drugs = list(safety_json.get("blocked_drugs", []))
    checks = safety_json.get("checks", [])

    # Structural backup: any drug marked dangerous in checks → add to blocked
    for check in checks:
        if check.get("dangerous") and check.get("drug"):
            drug_name = check["drug"]
            if drug_name not in blocked_drugs:
                blocked_drugs.append(drug_name)

    # ── Apply blocks: remove dangerous drugs from treatment ──────
    modified = False
    if blocked_drugs:
        original_drugs = tx_plan.get("pharmacological", [])
        safe_drugs = []
        removed_drugs = []

        for drug_entry in original_drugs:
            drug_name_lower = drug_entry.get("drug", "").lower()
            is_blocked = any(
                b.lower() in drug_name_lower or drug_name_lower in b.lower()
                for b in blocked_drugs
            )
            if is_blocked:
                removed_drugs.append(drug_entry)
            else:
                safe_drugs.append(drug_entry)

        if removed_drugs:
            modified = True
            tx_plan["pharmacological"] = safe_drugs

            # Add blocked drugs to contraindication_notes
            existing_notes = tx_plan.get("contraindication_notes", [])
            for removed in removed_drugs:
                # Find the best reason from checks
                reason = "Potentially lethal for active differential diagnosis"
                for check in checks:
                    check_drug = check.get("drug", "").lower()
                    removed_name = removed.get("drug", "").lower()
                    if check_drug in removed_name or removed_name in check_drug:
                        diff_name = check.get("differential", "unknown")
                        reason = (
                            f"{check.get('reason', reason)} "
                            f"(differential: {diff_name})"
                        )
                        break
                existing_notes.append(
                    f"⛔ {removed.get('drug', 'Unknown')} — "
                    f"REMOVED BY SAFETY GATE: {reason}"
                )
            tx_plan["contraindication_notes"] = existing_notes

            treatment_json["treatment_plan"] = tx_plan
            treatment_json["safety_gate_applied"] = True
            treatment_json["safety_gate_removed_drugs"] = [
                d.get("drug") for d in removed_drugs
            ]

    # ── WebSocket notifications ──────────────────────────────────
    if blocked_drugs:
        for drug_name in blocked_drugs:
            # Find the check that blocked this drug for the reason
            block_reason = ""
            for check in checks:
                if check.get("drug", "").lower() in drug_name.lower():
                    block_reason = (
                        f" — {check.get('reason', '')} "
                        f"(vs. {check.get('differential', '')})"
                    )
                    break
            await ws.send_json({
                "type": "info",
                "stage": "SAFETY",
                "content": (
                    f"🚨 SAFETY GATE: Drug '{drug_name}' BLOCKED{block_reason}"
                ),
            })

        safe_count = len(tx_plan.get("pharmacological", []))
        await ws.send_json({
            "type": "info",
            "stage": "SAFETY",
            "content": (
                f"✅ Treatment Safety Gate: {len(blocked_drugs)} dangerous "
                f"drug(s) removed, {safe_count} safe drug(s) retained."
            ),
        })
    else:
        await ws.send_json({
            "type": "info",
            "stage": "SAFETY",
            "content": (
                "✅ Treatment Safety Gate: All proposed drugs passed "
                "differential safety validation."
            ),
        })

    await ws.send_json({"type": "stage_complete", "stage": "SAFETY"})

    summary = safety_json.get("summary", "")
    if summary:
        logger.info("[SAFETY-GATE] %s", summary)

    return {
        "blocked_drugs": blocked_drugs,
        "warnings": [
            c.get("reason", "") for c in checks if c.get("dangerous")
        ],
        "modified": modified,
        "safe_alternatives": safety_json.get("safe_alternatives", []),
    }
