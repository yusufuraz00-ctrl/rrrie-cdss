"""Post-Hoc Confidence Calibration Engine — Parallel Overlay.

SAFETY PRINCIPLE: This engine does NOT modify the existing LLM confidence score.
It adds a PARALLEL 'calibrated' score alongside it. The existing system flow
is NEVER broken.

Frontend display: "LLM: 88% | Calibrated: 42% ⚠️"

Formula:
  C_cal = (C_llm × 0.30) + (E_ground × 0.35) + (S_coverage × 0.35) - P_conflict

Components:
  - C_llm (weight: 0.30): LLM's raw confidence — low weight because LLMs overconfident
  - E_ground (weight: 0.35): Evidence grounding ratio from R2 sources
  - S_coverage (weight: 0.35): Symptom coverage ratio (explained/total)
  - P_conflict (0.0-0.5): Penalty for unresolved red flags or rule violations

Research basis: DACA (arXiv 2026), SteeringConf (arXiv 2026), AU-Probe (2025-2026)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("rrrie-cdss.confidence")


@dataclass
class CalibrationResult:
    """Result from the confidence calibration engine."""
    raw: float                # Original LLM confidence — UNTOUCHED
    calibrated: float         # New calibrated confidence
    zone: str                 # 'safe' | 'caution' | 'critical'
    breakdown: dict           # Individual component scores
    warning_message: str      # Human-readable warning (empty if safe)


def calculate_calibrated_confidence(
    llm_raw_confidence: float,
    r2_evidence_count: int,
    r2_supporting_count: int,
    r1_symptom_count: int,
    r3_explained_symptom_count: int,
    unresolved_red_flags: int = 0,
    logic_violations: int = 0,
) -> CalibrationResult:
    """Calculate a calibrated confidence score as a parallel overlay.
    
    This does NOT replace the existing system — it runs alongside it.
    
    Args:
        llm_raw_confidence: The LLM's original confidence (0.0-1.0)
        r2_evidence_count: Total number of R2 evidence sources found
        r2_supporting_count: How many of those support the R3 diagnosis
        r1_symptom_count: Total symptoms identified by R1
        r3_explained_symptom_count: How many symptoms R3 diagnosis explains
        unresolved_red_flags: Count of unresolved red flags
        logic_violations: Count of logic rule violations (from safety net)
    
    Returns:
        CalibrationResult with raw, calibrated, zone, and breakdown
    """
    # Clamp raw confidence to [0, 1]
    c_llm = max(0.0, min(1.0, llm_raw_confidence))
    
    # Evidence grounding ratio
    if r2_evidence_count > 0:
        e_ground = r2_supporting_count / r2_evidence_count
    else:
        # No evidence at all → heavily penalize
        e_ground = 0.2  # Baseline without any evidence
    
    # Symptom coverage ratio
    if r1_symptom_count > 0:
        s_coverage = r3_explained_symptom_count / r1_symptom_count
    else:
        s_coverage = 0.5  # Neutral if no symptoms extracted
    
    # Conflict penalty
    conflict_items = unresolved_red_flags + logic_violations
    if conflict_items == 0:
        p_conflict = 0.0
    elif conflict_items <= 2:
        p_conflict = 0.15 * conflict_items  # Moderate penalty
    else:
        p_conflict = 0.5  # Heavy penalty for 3+ conflicts
    
    # Composite formula
    c_calibrated = (
        (c_llm * 0.30) +
        (e_ground * 0.35) +
        (s_coverage * 0.35) -
        p_conflict
    )
    
    # Clamp to [0, 1]
    c_calibrated = max(0.0, min(1.0, c_calibrated))
    
    # Determine zone
    if c_calibrated >= 0.70:
        zone = "safe"
        warning = ""
    elif c_calibrated >= 0.40:
        zone = "caution"
        warning = "⚠️ Calibrated confidence is low — additional review recommended."
    else:
        zone = "critical"
        warning = "🔴 Calibrated system does NOT find this diagnosis reliable."
    
    breakdown = {
        "c_llm": round(c_llm, 3),
        "c_llm_weighted": round(c_llm * 0.30, 3),
        "e_ground": round(e_ground, 3),
        "e_ground_weighted": round(e_ground * 0.35, 3),
        "s_coverage": round(s_coverage, 3),
        "s_coverage_weighted": round(s_coverage * 0.35, 3),
        "p_conflict": round(p_conflict, 3),
    }
    
    result = CalibrationResult(
        raw=round(c_llm, 3),
        calibrated=round(c_calibrated, 3),
        zone=zone,
        breakdown=breakdown,
        warning_message=warning,
    )
    
    logger.info(
        "[CONFIDENCE] Raw: %.2f → Calibrated: %.2f (%s) | "
        "Evidence: %d/%d, Symptoms: %d/%d, Conflicts: %d",
        c_llm, c_calibrated, zone,
        r2_supporting_count, r2_evidence_count,
        r3_explained_symptom_count, r1_symptom_count,
        conflict_items,
    )
    
    return result
