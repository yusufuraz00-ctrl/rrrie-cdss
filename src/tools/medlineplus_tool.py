"""MedlinePlus Connect API tool — patient-oriented health information."""

from __future__ import annotations

from typing import Dict

import httpx

from config.api_config import MedlinePlusConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_config = MedlinePlusConfig()

# Code system OIDs — MedlinePlus Connect supports ICD-10-CM and SNOMED-CT natively.
# For ICD-11 codes we first try SNOMED-CT cross-map; if not available we pass the
# ICD-11 stem directly (best-effort).
_CODE_SYSTEMS = {
    "ICD-11": "2.16.840.1.113883.6.90",   # Uses cross-mapped ICD-10-CM OID as fallback
    "ICD-10-CM": "2.16.840.1.113883.6.90",
    "SNOMED-CT": "2.16.840.1.113883.6.96",
}

# ICD-11 → ICD-10-CM cross-map for MedlinePlus compatibility (common codes)
_ICD11_TO_ICD10_MAP: dict[str, str] = {
    "CA40": "J18.9",
    "CA40.0": "J18.9",
    "CA07": "J06.9",
    "CA22": "J44.1",
    "BA41": "I21.9",
    "BA41.0": "I21.0",
    "BD10": "I50.9",
    "5A10": "E10.9",
    "5A11": "E11.9",
    "GC08": "N39.0",
    "CA23": "J45.9",
    "CA23.0": "J45.41",
    "DB10": "K35.9",
    "8A80": "G43.9",
    "8A80.1": "G43.1",
    # New cases
    "8B11": "I63.9",
    "8B11.0": "I63.9",
    "BB00": "I26.99",
    "BB00.0": "I26.99",
    "DC31": "K85.9",
    "1G40": "A41.9",
    "GB60": "N17.9",
    "6A70": "F32.9",
    "BD40": "I82.409",
    "4A84": "T78.2",
    "JA01": "O00.9",
    "1D01": "G00.9",
    "5A00": "E05.0",
    "5A00.0": "E05.0",
    "3A51": "D57.0",
    "3A51.0": "D57.0",
    "1F40": "B50.9",
    "1F40.0": "B50.0",
}


def _resolve_code_for_medlineplus(condition_code: str, code_system: str) -> tuple[str, str]:
    """Resolve an ICD-11 code to a MedlinePlus-compatible system.

    Returns (resolved_code, oid) tuple.
    """
    if code_system == "ICD-11":
        mapped = _ICD11_TO_ICD10_MAP.get(condition_code.upper())
        if mapped:
            return mapped, _CODE_SYSTEMS["ICD-10-CM"]
        # Fallback: pass as-is (some ICD-11 alpha stems work)
        return condition_code, _CODE_SYSTEMS["ICD-10-CM"]
    return condition_code, _CODE_SYSTEMS.get(code_system, _CODE_SYSTEMS["ICD-11"])


async def get_treatment_guidelines(
    condition_code: str,
    code_system: str = "ICD-11",
    language: str = "en",
) -> Dict:
    """Get health information from MedlinePlus Connect for a condition code.

    Args:
        condition_code: ICD-11, ICD-10-CM, or SNOMED-CT code.
        code_system: Code system — "ICD-11" (default), "ICD-10-CM", or "SNOMED-CT".
        language: Language code (default "en").

    Returns:
        Dict with health topic title, summary, and URL.
    """
    resolved_code, oid = _resolve_code_for_medlineplus(condition_code, code_system)

    params = {
        "mainSearchCriteria.v.cs": oid,
        "mainSearchCriteria.v.c": resolved_code,
        "informationRecipient.languageCode.c": language,
        "knowledgeResponseType": "application/json",
    }

    logger.info("medlineplus_search", code=condition_code, system=code_system)

    async with httpx.AsyncClient(timeout=_config.timeout) as client:
        try:
            resp = await client.get(_config.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("medlineplus_error", error=str(exc))
            return {
                "condition_code": condition_code,
                "topics": [],
                "error": str(exc),
            }

    # Parse feed entries
    feed = data.get("feed", {})
    entries = feed.get("entry", [])

    topics = []
    for entry in entries[:5]:
        title_val = entry.get("title", {}).get("_value", "")
        summary_val = entry.get("summary", {}).get("_value", "")

        # Extract URL from link
        links = entry.get("link", [])
        url = ""
        if links:
            url = links[0].get("href", "") if isinstance(links[0], dict) else ""

        topics.append({
            "title": title_val,
            "summary": summary_val[:500] if summary_val else "",
            "url": url,
        })

    logger.info("medlineplus_results", code=condition_code, topics=len(topics))

    return {
        "condition_code": condition_code,
        "code_system": code_system,
        "topics": topics,
    }
