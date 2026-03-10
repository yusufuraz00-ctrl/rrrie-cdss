"""Dynamic Token Budget Manager — pool-based allocation with efficiency tracking.

Replaces static per-stage token budgets with a shared pool that adapts
based on actual token usage (EMA-tracked efficiency) and reserves tokens
for future pipeline stages.

Architecture:
    shared_pool = ctx_window × POOL_RATIO
    allocate(stage) → deducts from pool, returns budget  
    report(stage, used) → updates EMA, returns unused to pool
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("rrrie-cdss")

# ── Pool configuration ──────────────────────────────────────────
POOL_RATIO = 0.65          # fraction of ctx_window usable for generation
PROMPT_HEADROOM = 0.15     # reserve 15% of ctx for prompt growth across iterations
EMA_ALPHA = 0.4            # weight of new observation in efficiency EMA

# Stage execution order (for future-reservation calculations)
STAGE_ORDER = ["R1", "R3", "IE", "TX"]

# Minimum guaranteed tokens per stage (hard floor — never go below)
STAGE_FLOORS: dict[str, dict[str, int]] = {
    #              thinking   fast
    "R1":  {"thinking": 2048, "fast": 1024},
    "R3":  {"thinking": 2048, "fast": 1024},
    "IE":  {"thinking": 1536, "fast": 768},
    "TX":  {"thinking": 1024, "fast": 768},
}

# Base allocation per stage (what the static system used to give)
STAGE_BASE_TOKENS: dict[str, dict[str, int]] = {
    "R1":  {"thinking": 3072, "fast": 2048},
    "R3":  {"thinking": 3072, "fast": 2048},
    "IE":  {"thinking": 2560, "fast": 1280},
    "TX":  {"thinking": 1536, "fast": 1024},
}

# Groq (cloud) base tokens — separate limits, not pool-managed
GROQ_BASE_TOKENS: dict[str, dict[str, int]] = {
    "R1":  {"thinking": 4096, "fast": 2048},
    "R3":  {"thinking": 4096, "fast": 3072},
    "IE":  {"thinking": 1536, "fast": 1024},
    "TX":  {"thinking": 1536, "fast": 1024},
}


@dataclass
class StageStats:
    """Efficiency tracking for a single stage."""
    allocations: int = 0             # how many times allocated
    total_allocated: int = 0         # sum of tokens allocated
    total_used: int = 0              # sum of tokens actually used
    efficiency_ema: float = 0.70     # running efficiency (starts at 70%)
    last_allocated: int = 0          # most recent allocation
    last_used: int = 0               # most recent usage


@dataclass
class TokenBudgetManager:
    """Pool-based dynamic token allocator.

    Usage:
        budget = TokenBudgetManager(ctx_window=16384, is_fast=True)
        tokens = budget.allocate("R3", iteration=1, prompt_tokens=800)
        # ... run LLM with max_tokens=tokens ...
        budget.report("R3", allocated=tokens, used=actual_completion_tokens, prompt_tokens=800)
    """
    ctx_window: int = 16384
    is_fast: bool = False
    max_iterations: int = 5

    # Internal state (initialized in __post_init__)
    _pool: int = field(init=False)
    _initial_pool: int = field(init=False)
    _stage_stats: dict[str, StageStats] = field(init=False, default_factory=dict)
    _history: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._pool = int(self.ctx_window * POOL_RATIO)
        self._initial_pool = self._pool
        self._stage_stats = {s: StageStats() for s in STAGE_ORDER}
        self._history = []
        mode = "fast" if self.is_fast else "thinking"
        logger.info(
            "[TOKEN-BUDGET] Initialized: ctx=%d, pool=%d (%s mode)",
            self.ctx_window, self._pool, mode,
        )

    @property
    def pool_remaining(self) -> int:
        return max(0, self._pool)

    @property
    def pool_utilization(self) -> float:
        """Fraction of pool that has been consumed."""
        if self._initial_pool == 0:
            return 0.0
        return 1.0 - (self._pool / self._initial_pool)

    # ── Core: allocate tokens for a stage ───────────────────────
    def allocate(
        self,
        stage: str,
        iteration: int = 1,
        prompt_tokens: int = 0,
        is_groq: bool = False,
    ) -> int:
        """Allocate tokens for a stage.

        Args:
            stage: "R1", "R3", "IE", or "TX"
            iteration: Current RRRIE iteration (1-based)
            prompt_tokens: Estimated prompt token count (for context ceiling)
            is_groq: If True, use Groq limits (not pool-managed)

        Returns:
            Number of max_tokens to pass to the LLM.
        """
        mode = "fast" if self.is_fast else "thinking"

        # ── Groq: fixed limits, no pool management ──
        if is_groq:
            groq_tokens = GROQ_BASE_TOKENS.get(stage, {}).get(mode, 2048)
            # Iteration escalation for Groq too (+10% per iter)
            groq_tokens = int(groq_tokens * (1.0 + (iteration - 1) * 0.10))
            logger.info(
                "[TOKEN-BUDGET] %s (Groq, iter %d): %d tokens (no pool deduction)",
                stage, iteration, groq_tokens,
            )
            return groq_tokens

        # ── Local model: pool-based allocation ──
        base = STAGE_BASE_TOKENS.get(stage, {}).get(mode, 1024)
        floor = STAGE_FLOORS.get(stage, {}).get(mode, 384)
        stats = self._stage_stats.get(stage, StageStats())

        # 1) Efficiency-adjusted base
        if stats.allocations >= 1:
            # If the model consistently uses less, shrink allocation
            eff = stats.efficiency_ema
            adjusted = int(base * max(eff + 0.15, 0.50))  # at least 50% of base
        else:
            adjusted = base  # first call: use full base

        # 2) Iteration escalation (+12% per iteration beyond first)
        iter_mult = 1.0 + (iteration - 1) * 0.12
        adjusted = int(adjusted * iter_mult)

        # 3) Reserve tokens for future stages
        future_reserve = self._future_reserve(stage, iteration)

        # 4) Pool-aware ceiling: don't exceed available pool minus reserves
        available = max(0, self._pool - future_reserve)
        budget = min(adjusted, available)

        # 5) Context-aware ceiling: prompt + generation ≤ ctx_window
        if prompt_tokens > 0:
            ctx_ceiling = self.ctx_window - prompt_tokens - 64  # 64-token safety margin
            budget = min(budget, ctx_ceiling)

        # 6) Hard floor
        budget = max(budget, floor)

        # 7) Deduct from pool (allow negative but cap debt to prevent runaway)
        self._pool -= budget
        if self._pool < 0:
            logger.warning(
                "[TOKEN-BUDGET] Pool debt: %d (floor allocation exceeded available pool)",
                self._pool,
            )
            # Cap debt at -50% of initial pool to prevent infinite spiraling
            self._pool = max(self._pool, -self._initial_pool // 2)

        stats.allocations += 1
        stats.total_allocated += budget
        stats.last_allocated = budget

        self._history.append({
            "action": "allocate",
            "stage": stage,
            "iteration": iteration,
            "base": base,
            "adjusted_pre_cap": adjusted,
            "future_reserve": future_reserve,
            "available_pool": available + budget,  # before deduction
            "final_budget": budget,
            "pool_after": self._pool,
            "prompt_tokens": prompt_tokens,
        })

        logger.info(
            "[TOKEN-BUDGET] %s alloc (iter %d): base=%d → adj=%d → final=%d | pool: %d→%d | eff=%.0f%%",
            stage, iteration, base, adjusted, budget,
            available + budget, self._pool,
            stats.efficiency_ema * 100,
        )

        return budget

    # ── Core: report actual usage ───────────────────────────────
    def report(
        self,
        stage: str,
        allocated: int,
        used: int,
        prompt_tokens: int = 0,
    ) -> None:
        """Report how many tokens were actually used after generation.

        Unused tokens are returned to the pool. Efficiency EMA is updated.
        """
        stats = self._stage_stats.get(stage)
        if stats is None:
            return

        # Update usage stats
        stats.total_used += used
        stats.last_used = used

        # Update efficiency EMA
        if allocated > 0:
            current_eff = min(used / allocated, 1.0)
            stats.efficiency_ema = (
                (1 - EMA_ALPHA) * stats.efficiency_ema + EMA_ALPHA * current_eff
            )

        # Return unused tokens to pool
        unused = max(0, allocated - used)
        self._pool += unused

        self._history.append({
            "action": "report",
            "stage": stage,
            "allocated": allocated,
            "used": used,
            "unused_returned": unused,
            "efficiency": round(used / allocated, 3) if allocated > 0 else 0,
            "ema": round(stats.efficiency_ema, 3),
            "pool_after": self._pool,
        })

        logger.info(
            "[TOKEN-BUDGET] %s report: used %d/%d (%.0f%%) | returned %d → pool=%d | EMA=%.0f%%",
            stage, used, allocated,
            (used / allocated * 100) if allocated > 0 else 0,
            unused, self._pool,
            stats.efficiency_ema * 100,
        )

    # ── Status for WebSocket / UI ───────────────────────────────
    def get_status(self) -> dict:
        """Return a dict suitable for WebSocket broadcast."""
        stage_info = {}
        for name, stats in self._stage_stats.items():
            stage_info[name] = {
                "allocations": stats.allocations,
                "total_allocated": stats.total_allocated,
                "total_used": stats.total_used,
                "efficiency_ema": round(stats.efficiency_ema, 3),
                "last_allocated": stats.last_allocated,
                "last_used": stats.last_used,
            }

        return {
            "pool_total": self._initial_pool,
            "pool_remaining": max(0, self._pool),
            "pool_utilization": round(self.pool_utilization, 3),
            "ctx_window": self.ctx_window,
            "mode": "fast" if self.is_fast else "thinking",
            "stages": stage_info,
        }

    # ── Summary for final_result ────────────────────────────────
    def get_summary(self) -> dict:
        """Return efficiency summary for inclusion in pipeline final_result."""
        total_alloc = sum(s.total_allocated for s in self._stage_stats.values())
        total_used = sum(s.total_used for s in self._stage_stats.values())
        overall_eff = (total_used / total_alloc) if total_alloc > 0 else 0

        return {
            "pool_total": self._initial_pool,
            "pool_remaining": max(0, self._pool),
            "total_allocated": total_alloc,
            "total_used": total_used,
            "overall_efficiency": round(overall_eff, 3),
            "tokens_saved": total_alloc - total_used,
            "stages": {
                name: {
                    "allocated": s.total_allocated,
                    "used": s.total_used,
                    "efficiency": round(s.efficiency_ema, 3),
                }
                for name, s in self._stage_stats.items()
                if s.allocations > 0
            },
        }

    # ── Internal helpers ────────────────────────────────────────
    def _future_reserve(self, current_stage: str, iteration: int) -> int:
        """Calculate minimum tokens to reserve for stages after current_stage."""
        mode = "fast" if self.is_fast else "thinking"
        try:
            idx = STAGE_ORDER.index(current_stage)
        except ValueError:
            return 0

        reserve = 0
        remaining_stages = STAGE_ORDER[idx + 1:]

        for future_stage in remaining_stages:
            floor = STAGE_FLOORS.get(future_stage, {}).get(mode, 384)
            # Reserve floor + small buffer for each remaining stage
            reserve += int(floor * 1.2)

        # For iterations beyond the first, also reserve for the next R3+IE cycle
        remaining_iters = max(0, self.max_iterations - iteration)
        if remaining_iters > 0 and current_stage in ("R3", "IE"):
            r3_floor = STAGE_FLOORS["R3"][mode]
            ie_floor = STAGE_FLOORS["IE"][mode]
            # Reserve a minimum cycle budget for at least 1 more iteration
            reserve += r3_floor + ie_floor

        return reserve

    def reset_pool(self) -> None:
        """Reset pool to initial size (e.g., between pipeline runs)."""
        self._pool = self._initial_pool
        self._stage_stats = {s: StageStats() for s in STAGE_ORDER}
        self._history = []
        logger.info("[TOKEN-BUDGET] Pool reset to %d", self._pool)
