"""Adaptive Pipeline Router — maps R0 complexity to pipeline configuration.

Uses R0Result to determine:
  - How many R3↔IE iterations to run
  - Which model to prefer per stage
  - Which R2 tools to activate
  - Whether to enable speculative execution
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run, derived from R0 analysis."""

    # Iteration control
    max_iterations: int = 3
    min_iterations: int = 2
    confidence_target: float = 0.85

    # Model routing hints
    prefer_cloud_r1: bool = True
    prefer_cloud_r3: bool = True

    # R2 tool selection
    r2_tools: list[str] = field(default_factory=lambda: [
        "pubmed", "openfda", "medlineplus", "wikipedia",
    ])

    # Complexity metadata
    complexity: str = "moderate"
    urgency: str = "moderate"

    # Speculative R1↔R2 overlap (run R2 while R1 is still finishing)
    speculative_r2: bool = False


# ── Route Table ─────────────────────────────────────────────────

_ROUTE_TABLE: dict[str, PipelineConfig] = {
    "simple": PipelineConfig(
        max_iterations=2,
        min_iterations=1,
        confidence_target=0.80,
        prefer_cloud_r1=False,
        prefer_cloud_r3=False,
        r2_tools=["pubmed", "wikipedia"],
        complexity="simple",
        speculative_r2=False,
    ),
    "moderate": PipelineConfig(
        max_iterations=3,
        min_iterations=2,
        confidence_target=0.85,
        prefer_cloud_r1=True,
        prefer_cloud_r3=True,
        r2_tools=["pubmed", "openfda", "medlineplus", "wikipedia"],
        complexity="moderate",
        speculative_r2=False,
    ),
    "complex": PipelineConfig(
        max_iterations=4,
        min_iterations=2,
        confidence_target=0.90,
        prefer_cloud_r1=True,
        prefer_cloud_r3=True,
        r2_tools=[
            "pubmed", "openfda", "medlineplus", "wikipedia",
            "clinical_trials", "europe_pmc", "semantic_scholar",
        ],
        complexity="complex",
        speculative_r2=True,
    ),
    "critical": PipelineConfig(
        max_iterations=3,
        min_iterations=1,
        confidence_target=0.80,
        prefer_cloud_r1=True,
        prefer_cloud_r3=True,
        r2_tools=["pubmed", "openfda", "medlineplus"],
        complexity="critical",
        urgency="critical",
        speculative_r2=False,
    ),
}


def route(complexity: str, urgency: str = "moderate") -> PipelineConfig:
    """Map R0 result to a pipeline configuration.

    Args:
        complexity: R0 assessed complexity (simple/moderate/complex/critical)
        urgency: R0 assessed urgency (low/moderate/high/critical)

    Returns:
        PipelineConfig tuned for this case.
    """
    # Critical urgency forces critical path regardless of complexity
    if urgency == "critical":
        cfg = PipelineConfig(**{
            k: getattr(_ROUTE_TABLE["critical"], k)
            for k in PipelineConfig.__dataclass_fields__
        })
        cfg.urgency = "critical"
        return cfg

    key = complexity if complexity in _ROUTE_TABLE else "moderate"
    cfg = PipelineConfig(**{
        k: getattr(_ROUTE_TABLE[key], k)
        for k in PipelineConfig.__dataclass_fields__
    })
    cfg.urgency = urgency
    return cfg
