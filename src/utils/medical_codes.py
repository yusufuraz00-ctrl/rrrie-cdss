"""Medical code helpers — ICD-11, SNOMED-CT mappings and validators.

ICD-11 stem-code format:
  - 1–2 uppercase letters followed by digits, optional dot + digits
  - Extension codes may follow separated by '&'
  - Examples: CA40.0, BA41, 1A00, 8B11.0, CA40.0&XY9Z
  
WHO ICD-11 Coding Tool API: https://icd.who.int/icdapi
"""

from __future__ import annotations

import re
from typing import Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── ICD-11 stem code pattern ────────────────────────────────────────────────────
# ICD-11 MMS codes follow the pattern: [block][category][optional extension]
#   Block:     1-2 chars (letter or digit+letter) e.g. C, CA, 1A, 8B
#   Category:  1-4 alphanumeric chars e.g. 40, 40.0, 1C (in 1C1C), 84
# Examples: CA40.0  BA41  1A00  8B11.0  MG30.00  1C1C  4A84  JA01  5A10
ICD11_PATTERN = re.compile(
    r"^(?:[A-Z]\d|[A-Z]{2}|\d[A-Z])[A-Z0-9]{1,4}(\.\d{1,3})?(\/[A-Z0-9.]+)?(&[A-Z0-9.]+)*$",
    re.IGNORECASE,
)


def is_valid_icd11(code: str) -> bool:
    """Check if a string looks like a valid ICD-11 stem code."""
    return bool(ICD11_PATTERN.match(code.strip()))


def normalize_icd11(code: str) -> Optional[str]:
    """Normalize an ICD-11 code to uppercase, stripped.

    Returns None if the code doesn't match the expected pattern.
    """
    cleaned = code.strip().upper()
    if is_valid_icd11(cleaned):
        return cleaned
    return None


# ── Commonly referenced ICD-11 codes ────────────────────────────────────────────
# Mapping of the most frequent codes in differential diagnosis
COMMON_ICD11: dict[str, str] = {
    # Respiratory
    "CA40": "Pneumonia, organism unspecified",
    "CA40.0": "Community-acquired pneumonia",
    "CA07": "Acute upper respiratory infections",
    "CA22": "COPD with acute exacerbation",
    # Cardiovascular
    "BA41": "Acute myocardial infarction",
    "BA41.0": "ST-elevation myocardial infarction",
    "BD10": "Heart failure",
    # Metabolic
    "5A10": "Type 1 diabetes mellitus",
    "5A11": "Type 2 diabetes mellitus",
    # Genitourinary
    "GC08": "Urinary tract infection, site not specified",
    # Respiratory – Asthma
    "CA23": "Asthma",
    # Gastrointestinal
    "DB10": "Acute appendicitis",
    # Neurological
    "8A80": "Migraine",
    # Symptoms & Signs
    "MG26": "Fever, unspecified",
    "MD11": "Cough",
    "MD12": "Dyspnoea",
    "8A84": "Headache disorders",
    "MD81": "Abdominal pain, unspecified",
    # Infections
    "1C12": "COVID-19",
    "1A00": "Cholera",
    "1F40": "Malaria due to Plasmodium falciparum",
    "1F40.0": "Severe falciparum malaria",
    "1G40": "Sepsis",
    "1D01": "Bacterial meningitis",
    # Circulatory – additional
    "BB00": "Pulmonary embolism",
    "BB00.0": "Pulmonary embolism without acute cor pulmonale",
    "BD40": "Deep vein thrombosis of lower extremity",
    # Digestive – additional
    "DC31": "Acute pancreatitis",
    # Genitourinary – additional
    "GB60": "Acute kidney injury",
    # Mental / Behavioural
    "6A70": "Single episode depressive disorder",
    # Immune system
    "4A84": "Anaphylaxis",
    # Obstetric
    "JA01": "Ectopic pregnancy",
    # Endocrine – additional
    "5A00": "Hyperthyroidism",
    "5A00.0": "Thyrotoxicosis with diffuse goitre (Graves disease)",
    # Haematologic
    "3A51": "Sickle cell disease",
    "3A51.0": "Sickle cell disease with crisis",
}


def get_icd11_description(code: str) -> str | None:
    """Get description for a common ICD-11 code."""
    normalized = normalize_icd11(code)
    if normalized:
        return COMMON_ICD11.get(normalized)
    return None


# ── WHO ICD-11 API integration ──────────────────────────────────────────────────
# Uses OAuth2 Client Credentials grant to obtain a bearer token.
# Register at https://icd.who.int/icdapi to get client_id and client_secret.

_WHO_TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
_WHO_MMS_SEARCH = "https://id.who.int/icd/release/11/2024-01/mms/search"
_WHO_CODEINFO = "https://id.who.int/icd/release/11/2024-01/mms/codeinfo"
_WHO_ENTITY_SEARCH = "https://id.who.int/icd/entity/search"


def _strip_html(text: str) -> str:
    """Strip HTML tags (e.g. <em class='found'>) from WHO API results."""
    import re
    return re.sub(r"<[^>]+>", "", text).strip()

# Module-level cache for the OAuth2 token
_cached_token: str | None = None
_token_expiry: float = 0.0


async def _get_who_bearer_token(
    client_id: str | None = None,
    client_secret: str | None = None,
) -> str | None:
    """Obtain a bearer token via WHO ICD API OAuth2 client credentials flow.

    Caches the token until it expires (typically 1 hour).
    Falls back to settings if client_id/client_secret not provided.
    """
    import time

    global _cached_token, _token_expiry

    # Return cached token if still valid (with 60s buffer)
    if _cached_token and time.time() < (_token_expiry - 60):
        return _cached_token

    # Resolve credentials
    if not client_id or not client_secret:
        try:
            from config.settings import get_settings
            settings = get_settings()
            client_id = client_id or settings.icd11_client_id
            client_secret = client_secret or settings.icd11_client_secret
        except Exception:
            pass

    if not client_id or not client_secret:
        logger.debug("who_icd_no_credentials", msg="No ICD-11 client credentials configured")
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _WHO_TOKEN_URL,
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials",
                    "scope": "icdapi_access",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                data = resp.json()
                _cached_token = data["access_token"]
                _token_expiry = time.time() + data.get("expires_in", 3600)
                logger.info("who_icd_token_obtained", expires_in=data.get("expires_in"))
                return _cached_token
            else:
                logger.warning(
                    "who_icd_token_failed",
                    status=resp.status_code,
                    body=resp.text[:200],
                )
                return None
    except Exception as exc:
        logger.error("who_icd_token_error", error=str(exc))
        return None


async def lookup_icd11_who(
    code: str,
    api_token: str | None = None,
) -> dict:
    """Look up an ICD-11 code via the WHO MMS codeinfo endpoint.

    Uses the path-based codeinfo endpoint:
        /icd/release/11/2024-01/mms/codeinfo/{code}

    Args:
        code: ICD-11 stem code (e.g. "CA40.0", "BA41").
        api_token: Optional pre-supplied bearer token. If not provided,
                   automatically obtains one via OAuth2 client credentials.

    Returns:
        Dict with code, title, found (bool), definition, and browserUrl.
    """
    if not api_token:
        api_token = await _get_who_bearer_token()

    headers = {
        "Accept": "application/json",
        "Accept-Language": "en",
        "API-Version": "v2",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    # Path-based codeinfo — code goes in the URL path, NOT query params
    url = f"{_WHO_CODEINFO}/{code.strip()}"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                stem_id = data.get("stemId", "")
                result = {
                    "code": data.get("code", code),
                    "found": True,
                    "stemId": stem_id,
                    "title": "",
                    "definition": "",
                    "browserUrl": "",
                }
                # Follow stemId to get title, definition, browserUrl
                if stem_id:
                    try:
                        entity_resp = await client.get(stem_id, headers=headers)
                        if entity_resp.status_code == 200:
                            entity = entity_resp.json()
                            title_obj = entity.get("title", {})
                            defn_obj = entity.get("definition", {})
                            result["title"] = title_obj.get("@value", "") if isinstance(title_obj, dict) else str(title_obj)
                            result["definition"] = defn_obj.get("@value", "") if isinstance(defn_obj, dict) else str(defn_obj)
                            result["browserUrl"] = entity.get("browserUrl", "")
                            result["classKind"] = entity.get("classKind", "")
                    except Exception as e:
                        logger.debug("icd11_stemid_follow_error", stem_id=stem_id, error=str(e))
                return result
            logger.warning("icd11_who_lookup_failed", code=code, status=resp.status_code)
            return {"code": code, "found": False, "status_code": resp.status_code}
    except Exception as exc:
        logger.error("icd11_who_lookup_error", code=code, error=str(exc))
        return {"code": code, "found": False, "error": str(exc)}


async def search_icd11_who(
    query: str,
    api_token: str | None = None,
    max_results: int = 5,
) -> list[dict]:
    """Search ICD-11 by diagnosis text via the WHO MMS search endpoint.

    Uses the linearization (MMS) search which returns proper ICD-11 codes.

    Args:
        query: Free-text search (e.g. "community acquired pneumonia").
        api_token: Optional pre-supplied bearer token. If not provided,
                   automatically obtains one via OAuth2 client credentials.
        max_results: Maximum results to return.

    Returns:
        List of dicts with id, title, theCode, score.
    """
    if not api_token:
        api_token = await _get_who_bearer_token()

    headers = {
        "Accept": "application/json",
        "Accept-Language": "en",
        "API-Version": "v2",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    params = {
        "q": query,
        "useFlexisearch": "true",
        "flatResults": "true",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_WHO_MMS_SEARCH, headers=headers, params=params)
            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("destinationEntities", [])[:max_results]:
                    results.append({
                        "id": item.get("id", ""),
                        "title": _strip_html(item.get("title", "")),
                        "theCode": item.get("theCode", ""),
                        "score": item.get("score", 0),
                    })
                return results
            return []
    except Exception as exc:
        logger.error("icd11_who_search_error", query=query, error=str(exc))
        return []
