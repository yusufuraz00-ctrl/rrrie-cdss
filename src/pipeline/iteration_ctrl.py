"""Smart Iteration Controller — decides when to stop R3↔IE loop.

Replaces the static min/max iteration settings with a dynamic controller
that tracks:
  - Confidence progression per iteration
  - Diagnosis stability (stagnation detection)
  - Evidence diversity
  - IE-reported issue severity decay

Decides should_continue() based on plateau detection + target confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _IterationSnapshot:
    """What happened in one R3↔IE iteration."""
    iteration: int
    primary_diagnosis: str = ""
    confidence: float = 0.0
    ie_verdict: str = ""
    ie_issue_count: int = 0
    ie_max_severity: str = "low"


class IterationController:
    """Dynamic controller for the R3↔IE iteration loop.

    Args:
        max_iterations: Hard ceiling from settings / router config.
        min_iterations: Minimum before early-exit is allowed.
        confidence_target: Stop signal when primary dx reaches this.
        stagnation_limit: How many consecutive stale iterations before forcing stop.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        min_iterations: int = 2,
        confidence_target: float = 0.85,
        stagnation_limit: int = 2,
    ) -> None:
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.confidence_target = confidence_target
        self.stagnation_limit = stagnation_limit
        self._history: list[_IterationSnapshot] = []

    # ── Public API ──────────────────────────────────────────────

    def record(
        self,
        iteration: int,
        primary_diagnosis: str,
        confidence: float,
        ie_verdict: str,
        ie_issue_count: int = 0,
        ie_max_severity: str = "low",
    ) -> None:
        """Record one complete R3+IE iteration result."""
        self._history.append(_IterationSnapshot(
            iteration=iteration,
            primary_diagnosis=primary_diagnosis.strip().lower(),
            confidence=confidence,
            ie_verdict=ie_verdict.strip().upper(),
            ie_issue_count=ie_issue_count,
            ie_max_severity=ie_max_severity.lower(),
        ))

    def should_continue(self, current_iteration: int) -> bool:
        """Decide whether to run another R3↔IE iteration.

        Returns True if the loop should continue, False to stop.
        """
        # Hard ceiling
        if current_iteration >= self.max_iterations:
            logger.info("[IterCtrl] Max iterations (%d) reached → STOP", self.max_iterations)
            return False

        # Minimum floor
        if current_iteration < self.min_iterations:
            return True

        if not self._history:
            return True

        latest = self._history[-1]

        # IE says ACCEPT → stop
        if latest.ie_verdict == "ACCEPT":
            logger.info("[IterCtrl] IE ACCEPT at iter %d → STOP", current_iteration)
            return False

        # Confidence target met with no critical issues
        if (
            latest.confidence >= self.confidence_target
            and latest.ie_max_severity not in ("critical", "high")
        ):
            logger.info(
                "[IterCtrl] Confidence %.2f >= %.2f target, no critical issues → STOP",
                latest.confidence, self.confidence_target,
            )
            return False

        # Stagnation detection
        if self._is_plateauing():
            logger.info("[IterCtrl] Plateau detected at iter %d → STOP", current_iteration)
            return False

        return True

    @property
    def stagnation_count(self) -> int:
        """How many consecutive iterations had the same primary dx."""
        if len(self._history) < 2:
            return 0
        count = 0
        latest_dx = self._history[-1].primary_diagnosis
        for snap in reversed(self._history[:-1]):
            if snap.primary_diagnosis == latest_dx:
                count += 1
            else:
                break
        return count

    @property
    def is_improving(self) -> bool:
        """Whether confidence is improving between iterations."""
        if len(self._history) < 2:
            return True
        return self._history[-1].confidence > self._history[-2].confidence

    # ── Private ─────────────────────────────────────────────────

    def _is_plateauing(self) -> bool:
        """Detect if the loop is stuck: same dx repeated with no confidence gain."""
        if len(self._history) < 2:
            return False

        # Check consecutive same-diagnosis iterations
        consecutive_same = 0
        latest_dx = self._history[-1].primary_diagnosis
        for snap in reversed(self._history[:-1]):
            if snap.primary_diagnosis == latest_dx:
                consecutive_same += 1
            else:
                break

        if consecutive_same >= self.stagnation_limit:
            return True

        # Check confidence flatline (delta < 1% for 2+ iterations)
        if len(self._history) >= 3:
            recent = self._history[-3:]
            deltas = [
                abs(recent[i + 1].confidence - recent[i].confidence)
                for i in range(len(recent) - 1)
            ]
            if all(d < 0.01 for d in deltas):
                return True

        return False
