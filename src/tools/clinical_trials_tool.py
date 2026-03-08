"""ClinicalTrials.gov API v2 tool — searches clinical trial studies.

Note: ClinicalTrials.gov blocks httpx via TLS fingerprinting / WAF.
We use urllib.request (which works) inside asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
import urllib.parse
from typing import Dict, List, Optional

from config.api_config import ClinicalTrialsConfig
from src.utils.logger import get_logger
from src.utils.rate_limiter import clinicaltrials_limiter, tool_cache

logger = get_logger(__name__)

_config = ClinicalTrialsConfig()

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RRRIE-CDSS/1.0",
    "Accept": "application/json",
}


def _fetch_sync(url: str) -> dict:
    """Synchronous fetch using urllib (bypasses httpx TLS fingerprinting)."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=int(_config.timeout)) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def search_clinical_trials(
    condition: str,
    intervention: Optional[str] = None,
    status: Optional[str] = None,
    max_results: int = 5,
) -> Dict:
    """Search ClinicalTrials.gov for relevant studies.

    Args:
        condition: Medical condition to search for.
        intervention: Specific intervention/drug (optional).
        status: Trial status filter: RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING.
        max_results: Maximum studies to return.

    Returns:
        Dict with "studies" list and "total_found" count.
    """
    params: Dict = {
        "query.cond": condition,
        "pageSize": max_results,
        "format": "json",
    }

    if intervention:
        params["query.intr"] = intervention
    if status:
        params["filter.overallStatus"] = status

    logger.info("clinical_trials_search", condition=condition, intervention=intervention)

    # TTL cache check
    cache_key = tool_cache.make_key(
        "clinical_trials", condition=condition,
        intervention=intervention, status=status, max_results=max_results,
    )
    cached = tool_cache.get(cache_key)
    if cached is not None:
        logger.info("clinical_trials_cache_hit", condition=condition)
        return cached

    # Build URL with query params
    query_string = urllib.parse.urlencode(params)
    url = f"{_config.base_url}/studies?{query_string}"

    try:
        async with clinicaltrials_limiter:
            data = await asyncio.to_thread(_fetch_sync, url)
    except Exception as exc:
        logger.error("clinical_trials_error", error=str(exc))
        return {"studies": [], "total_found": 0, "error": str(exc)}

    studies = []
    for study in data.get("studies", []):
        protocol = study.get("protocolSection", {})
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        arms_module = protocol.get("armsInterventionsModule", {})

        interventions = []
        for arm in arms_module.get("interventions", []):
            interventions.append(arm.get("name", ""))

        studies.append({
            "nct_id": id_module.get("nctId", ""),
            "title": id_module.get("briefTitle", ""),
            "status": status_module.get("overallStatus", ""),
            "phase": ", ".join(design_module.get("phases", [])),
            "interventions": interventions,
            "start_date": status_module.get("startDateStruct", {}).get("date", ""),
            "completion_date": status_module.get("completionDateStruct", {}).get("date", ""),
        })

    total = data.get("totalCount", len(studies))
    logger.info("clinical_trials_results", count=len(studies), total=total)

    result = {"studies": studies, "total_found": total}
    tool_cache.set(cache_key, result)
    return result
