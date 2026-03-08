"""OpenFDA Drug API tool — drug interactions, adverse events, label information."""

from __future__ import annotations

from typing import Dict, List

import httpx

from config.api_config import OpenFDAConfig
from src.utils.logger import get_logger
from src.utils.rate_limiter import openfda_limiter, tool_cache

logger = get_logger(__name__)

_config = OpenFDAConfig()


async def search_drug_interactions(
    drug_name: str,
    interaction_with: str | None = None,
) -> Dict:
    """Check OpenFDA for drug label data including interactions and warnings.

    Args:
        drug_name: Primary drug name to look up.
        interaction_with: Optional second drug to check specific interactions.

    Returns:
        Dict with drug warnings, interactions, contraindications, adverse reactions.
    """
    logger.info("openfda_drug_search", drug=drug_name)

    # TTL cache check
    cache_key = tool_cache.make_key("openfda_label", drug=drug_name, interaction_with=interaction_with)
    cached = tool_cache.get(cache_key)
    if cached is not None:
        logger.info("openfda_cache_hit", drug=drug_name)
        return cached

    async with httpx.AsyncClient(timeout=_config.timeout) as client:
        # ── Drug Label lookup ────────────────────────────────────────
        try:
            async with openfda_limiter:
                resp = await client.get(
                f"{_config.base_url}/label.json",
                params={
                    "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                    "limit": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("openfda_label_error", error=str(exc))
            return {"drug_name": drug_name, "error": str(exc)}

        results = data.get("results", [])
        if not results:
            return {"drug_name": drug_name, "message": "No FDA label data found"}

        label = results[0]

        output: Dict = {
            "drug_name": drug_name,
            "warnings": _truncate_list(label.get("warnings", [])),
            "drug_interactions": _truncate_list(label.get("drug_interactions", [])),
            "contraindications": _truncate_list(label.get("contraindications", [])),
            "adverse_reactions": _truncate_list(label.get("adverse_reactions", [])),
            "dosage_and_administration": _truncate_list(label.get("dosage_and_administration", [])),
        }

        # If checking interaction with a specific second drug, note it
        if interaction_with:
            interactions_text = " ".join(output.get("drug_interactions", []))
            if interaction_with.lower() in interactions_text.lower():
                output["specific_interaction_found"] = True
                output["interaction_note"] = (
                    f"Potential interaction between {drug_name} and {interaction_with} "
                    f"mentioned in FDA labeling."
                )
            else:
                output["specific_interaction_found"] = False
                output["interaction_note"] = (
                    f"No specific mention of {interaction_with} in {drug_name} FDA label. "
                    f"This does NOT rule out an interaction."
                )

        logger.info("openfda_drug_result", drug=drug_name, has_data=True)
        tool_cache.set(cache_key, output)
        return output


async def get_adverse_events(
    drug_name: str,
    limit: int = 10,
) -> Dict:
    """Get reported adverse event counts for a drug.

    Args:
        drug_name: Drug name to look up.
        limit: Number of top adverse events to return.

    Returns:
        Dict with adverse event frequency counts.
    """
    logger.info("openfda_adverse_events", drug=drug_name)

    async with httpx.AsyncClient(timeout=_config.timeout) as client:
        try:
            async with openfda_limiter:
                resp = await client.get(
                    f"{_config.base_url}/event.json",
                params={
                    "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                    "count": "patient.reaction.reactionmeddrapt.exact",
                    "limit": limit,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("openfda_adverse_error", error=str(exc))
            return {"drug_name": drug_name, "adverse_events": [], "error": str(exc)}

        events = data.get("results", [])
        return {
            "drug_name": drug_name,
            "adverse_events": [
                {"reaction": e.get("term", ""), "count": e.get("count", 0)}
                for e in events
            ],
        }


def _truncate_list(items: list, max_chars: int = 1500) -> list:
    """Truncate list items to keep total text under max_chars."""
    result = []
    total = 0
    for item in items:
        text = str(item)
        if total + len(text) > max_chars:
            result.append(text[: max_chars - total] + "... [truncated]")
            break
        result.append(text)
        total += len(text)
    return result
