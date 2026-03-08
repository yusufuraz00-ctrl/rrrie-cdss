"""Pharmacology Lookup Tool — Dynamic drug resolution via external APIs.

Design Philosophy:
  ZERO hardcoded drug data. All pharmacological knowledge is retrieved
  dynamically from external APIs at runtime. Results are cached locally
  so the system "learns" drug names over time (like clinical experience).

Resolution Chain:
  1. Local cache check (instant — previously resolved drugs)
  2. RxNorm API (NLM) — drug name → generic name + pharmacological class
  3. OpenFDA API — drug label → warnings, contraindications, mechanism
  4. Web search — fallback for regional brand names (e.g. Turkish "Dideral" → Propranolol)

Cache:
  JSON file at config/drug_cache.json — grows with each new drug encounter.
  Acts as "clinical experience memory" — the system never looks up the
  same drug twice. This is the RAG-like retrieval the architecture needs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Cache path ──────────────────────────────────────────────────────
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_CACHE_FILE = _CACHE_DIR / "drug_cache.json"

# In-memory cache (loaded from disk on first use)
_drug_cache: dict[str, dict] = {}
_cache_loaded = False


# ═════════════════════════════════════════════════════════════════════
# DrugInfo dataclass
# ═════════════════════════════════════════════════════════════════════

@dataclass
class DrugInfo:
    """Resolved pharmacological profile for a drug."""
    original_name: str           # As found in patient text (e.g. "Dideral")
    generic_name: str            # International generic (e.g. "Propranolol")
    drug_class: list[str]        # Pharmacological classes
    mechanism: list[str]         # Mechanism of action
    effects: list[str]           # Pharmacological effects
    warnings: list[str]          # Key warnings from FDA label
    contraindications: list[str] # Key contraindications
    interactions_note: str = ""  # Notable drug interaction text
    source: str = ""             # Where data came from
    rxcui: str = ""              # RxNorm Concept ID

    def to_dict(self) -> dict:
        return {
            "original_name": self.original_name,
            "generic_name": self.generic_name,
            "drug_class": self.drug_class,
            "mechanism": self.mechanism,
            "effects": self.effects,
            "warnings": self.warnings[:3],
            "contraindications": self.contraindications[:3],
            "interactions_note": self.interactions_note[:500],
            "source": self.source,
            "rxcui": self.rxcui,
        }

    def format_for_prompt(self) -> str:
        """Format as a concise fact block for LLM prompt injection."""
        lines = [f"💊 {self.original_name} → {self.generic_name}"]
        if self.drug_class:
            lines.append(f"   Class: {', '.join(self.drug_class[:3])}")
        if self.mechanism:
            lines.append(f"   Mechanism: {', '.join(self.mechanism[:3])}")
        if self.effects:
            lines.append(f"   Effects: {', '.join(self.effects[:5])}")
        if self.warnings:
            # Truncate each warning to keep prompt compact
            short_warnings = [w[:150] + "..." if len(w) > 150 else w for w in self.warnings[:2]]
            lines.append(f"   ⚠️ Warnings: {'; '.join(short_warnings)}")
        if self.contraindications:
            short_contra = [c[:150] + "..." if len(c) > 150 else c for c in self.contraindications[:2]]
            lines.append(f"   🚫 Contraindications: {'; '.join(short_contra)}")
        lines.append(f"   Source: {self.source}")
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════
# Cache Management — "Clinical Experience Memory"
# ═════════════════════════════════════════════════════════════════════

def _load_cache() -> None:
    """Load drug cache from disk (once per session)."""
    global _drug_cache, _cache_loaded
    if _cache_loaded:
        return
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _drug_cache = json.load(f)
            logger.info("[PHARMA-CACHE] Loaded %d cached drug profiles", len(_drug_cache))
    except Exception as e:
        logger.warning("[PHARMA-CACHE] Load failed: %s", e)
        _drug_cache = {}
    _cache_loaded = True


def _save_cache() -> None:
    """Persist drug cache to disk."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_drug_cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("[PHARMA-CACHE] Save failed: %s", e)


def _get_cached(name: str) -> Optional[DrugInfo]:
    """Check cache for a previously resolved drug."""
    _load_cache()
    key = name.lower().strip()
    if key in _drug_cache:
        d = _drug_cache[key]
        logger.info("[PHARMA-CACHE] Hit: '%s' → '%s'", name, d.get("generic_name", "?"))
        return DrugInfo(
            original_name=name,
            generic_name=d.get("generic_name", name),
            drug_class=d.get("drug_class", []),
            mechanism=d.get("mechanism", []),
            effects=d.get("effects", []),
            warnings=d.get("warnings", []),
            contraindications=d.get("contraindications", []),
            interactions_note=d.get("interactions_note", ""),
            source="Cache (previously resolved)",
            rxcui=d.get("rxcui", ""),
        )
    return None


def _cache_drug(name: str, info: DrugInfo) -> None:
    """Cache a resolved drug profile (also by generic name for future lookups)."""
    _load_cache()
    data = info.to_dict()
    key = name.lower().strip()
    _drug_cache[key] = data
    # Also cache by generic name so "Propranolol" is instant next time
    generic_key = info.generic_name.lower().strip()
    if generic_key and generic_key != key:
        _drug_cache[generic_key] = data
    _save_cache()
    logger.info("[PHARMA-CACHE] Cached: '%s' → '%s' (total: %d entries)", name, info.generic_name, len(_drug_cache))


# ═════════════════════════════════════════════════════════════════════
# RxNorm API (NLM — free, no API key)
# ═════════════════════════════════════════════════════════════════════

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
_RXNORM_TIMEOUT = 12.0


async def _rxnorm_resolve_name(drug_name: str) -> tuple[str, str]:
    """Resolve drug name → (rxcui, generic_name) via RxNorm.

    Tries exact match first, then approximate match.
    Returns ("", "") if not found.
    """
    async with httpx.AsyncClient(timeout=_RXNORM_TIMEOUT) as client:
        # ── Try exact/normalized match ──
        try:
            resp = await client.get(
                f"{RXNORM_BASE}/rxcui.json",
                params={"name": drug_name, "search": 2},  # 2 = exact + normalized
            )
            resp.raise_for_status()
            data = resp.json()
            id_group = data.get("idGroup", {})
            rxnorm_ids = id_group.get("rxnormId", [])
            if rxnorm_ids:
                rxcui = rxnorm_ids[0]
                resolved_name = id_group.get("name", drug_name)
                logger.info("[RXNORM] Resolved '%s' → rxcui=%s, name='%s'", drug_name, rxcui, resolved_name)
                return rxcui, resolved_name
        except Exception as e:
            logger.debug("[RXNORM] Exact match failed for '%s': %s", drug_name, e)

        # ── Try approximate match (catches similar brand names: Dideral→Inderal) ──
        try:
            resp = await client.get(
                f"{RXNORM_BASE}/approximateTerm.json",
                params={"term": drug_name, "maxEntries": 3},
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("approximateGroup", {}).get("candidate", [])
            if candidates:
                best = candidates[0]
                rxcui = best.get("rxcui", "")
                score = best.get("score", "0")
                if rxcui and int(score) >= 60:  # Only accept reasonable matches
                    # Get proper name for this rxcui
                    prop_resp = await client.get(f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json")
                    prop_data = prop_resp.json()
                    props = prop_data.get("properties", {})
                    resolved_name = props.get("name", drug_name)
                    logger.info(
                        "[RXNORM] Approximate: '%s' → rxcui=%s, name='%s' (score=%s)",
                        drug_name, rxcui, resolved_name, score,
                    )
                    return rxcui, resolved_name
        except Exception as e:
            logger.debug("[RXNORM] Approximate match failed for '%s': %s", drug_name, e)

    return "", ""


async def _rxnorm_get_classes(drug_name: str) -> tuple[list[str], list[str], list[str]]:
    """Get drug classes, mechanism of action, and pharmacological effects from RxNorm.

    Uses byDrugName endpoint (simpler, doesn't need rxcui).
    Returns (classes, mechanisms, effects).
    """
    classes: list[str] = []
    mechanisms: list[str] = []
    effects: list[str] = []

    async with httpx.AsyncClient(timeout=_RXNORM_TIMEOUT) as client:
        # ── Mechanism of Action ──
        try:
            resp = await client.get(
                f"{RXNORM_BASE}/rxclass/class/byDrugName.json",
                params={"drugName": drug_name, "relaSource": "MEDRT", "rela": "has_MoA"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
                    name = item.get("rxclassMinConceptItem", {}).get("className", "")
                    if name and name not in mechanisms:
                        mechanisms.append(name)
        except Exception as e:
            logger.debug("[RXNORM] MoA lookup failed for '%s': %s", drug_name, e)

        # ── Pharmacological Effect ──
        try:
            resp = await client.get(
                f"{RXNORM_BASE}/rxclass/class/byDrugName.json",
                params={"drugName": drug_name, "relaSource": "MEDRT", "rela": "has_PE"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
                    name = item.get("rxclassMinConceptItem", {}).get("className", "")
                    if name and name not in effects:
                        effects.append(name)
        except Exception as e:
            logger.debug("[RXNORM] PE lookup failed for '%s': %s", drug_name, e)

        # ── ATC Classification (drug class hierarchy) ──
        try:
            resp = await client.get(
                f"{RXNORM_BASE}/rxclass/class/byDrugName.json",
                params={"drugName": drug_name, "relaSource": "ATC"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", []):
                    name = item.get("rxclassMinConceptItem", {}).get("className", "")
                    if name and name not in classes:
                        classes.append(name)
        except Exception as e:
            logger.debug("[RXNORM] ATC lookup failed for '%s': %s", drug_name, e)

    return classes, mechanisms, effects


# ═════════════════════════════════════════════════════════════════════
# OpenFDA Enrichment
# ═════════════════════════════════════════════════════════════════════

async def _openfda_get_label(drug_name: str) -> dict:
    """Get drug label data from OpenFDA (pharm class, warnings, contraindications)."""
    from config.api_config import OpenFDAConfig
    config = OpenFDAConfig()

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        try:
            resp = await client.get(
                f"{config.base_url}/label.json",
                params={
                    "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
                    "limit": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return {}

            label = results[0]
            openfda = label.get("openfda", {})

            return {
                "generic_name": (openfda.get("generic_name") or [""])[0],
                "brand_name": (openfda.get("brand_name") or [""])[0],
                "pharm_class_epc": openfda.get("pharm_class_epc", []),
                "pharm_class_moa": openfda.get("pharm_class_moa", []),
                "pharm_class_pe": openfda.get("pharm_class_pe", []),
                "warnings": _truncate_list(label.get("warnings", []), 600),
                "contraindications": _truncate_list(label.get("contraindications", []), 600),
                "drug_interactions": _truncate_list(label.get("drug_interactions", []), 400),
            }
        except Exception as e:
            logger.debug("[OPENFDA] Label lookup failed for '%s': %s", drug_name, e)
            return {}


def _truncate_list(items: list, max_chars: int = 600) -> list:
    """Truncate list items to stay under max_chars total."""
    result = []
    total = 0
    for item in items:
        text = str(item)
        if total + len(text) > max_chars:
            result.append(text[:max_chars - total] + "... [truncated]")
            break
        result.append(text)
        total += len(text)
    return result


# ═════════════════════════════════════════════════════════════════════
# Web Search Fallback — for regional/unknown brand names
# ═════════════════════════════════════════════════════════════════════

async def _web_resolve_drug(drug_name: str) -> str:
    """Last resort: web search for drug's active ingredient. Returns generic name or ""."""
    try:
        from src.tools.web_search_tool import web_search
        result = await web_search(
            query=f'"{drug_name}" active ingredient generic name pharmacological class',
            search_depth="basic",
        )
        answer = result.get("answer", "")
        if answer:
            import re
            # Extract drug names from the answer text
            # Common pharmaceutical suffixes: -ol, -ine, -ide, -mab, -nib, -pril, -olol, -sartan
            patterns = [
                r'(?:active ingredient|generic name|contains|known as|same as)\s+(?:is\s+)?([A-Za-z]+(?:ol|ine|ide|mab|nib|pril|olol|sartan|pine|done|pam|lam|zole|mycin|cillin|cycline))\b',
                r'\b([A-Z][a-z]+(?:ol|ine|ide|mab|nib|pril|olol|sartan|pine|done|pam|lam|zole|mycin|cillin|cycline))\b',
            ]
            for pat in patterns:
                m = re.search(pat, answer, re.IGNORECASE)
                if m:
                    candidate = m.group(1).strip()
                    if candidate.lower() != drug_name.lower() and len(candidate) > 4:
                        logger.info("[WEB-RESOLVE] '%s' → candidate: '%s' (from answer)", drug_name, candidate)
                        return candidate

        # Also check search result snippets
        for res in result.get("results", [])[:3]:
            content = res.get("content", "") + " " + res.get("title", "")
            if content:
                import re
                for pat in [
                    r'\b([A-Z][a-z]+(?:ol|ine|ide|olol|pril|sartan|pine|done|pam|zole))\b',
                ]:
                    for m in re.finditer(pat, content):
                        candidate = m.group(1)
                        if candidate.lower() != drug_name.lower() and len(candidate) > 5:
                            logger.info("[WEB-RESOLVE] '%s' → candidate: '%s' (from snippet)", drug_name, candidate)
                            return candidate

    except ImportError:
        logger.debug("[WEB-RESOLVE] web_search not available")
    except Exception as e:
        logger.debug("[WEB-RESOLVE] Failed for '%s': %s", drug_name, e)

    return ""


# ═════════════════════════════════════════════════════════════════════
# Main Resolution Function
# ═════════════════════════════════════════════════════════════════════

async def resolve_drug(drug_name: str) -> Optional[DrugInfo]:
    """Fully resolve a drug name to its pharmacological profile.

    Resolution chain (stops at first sufficient data):
      1. Local cache (instant — "clinical memory")
      2. RxNorm exact match → RxNorm classes + OpenFDA enrichment
      3. RxNorm approximate match → same enrichment
      4. OpenFDA direct search
      5. Web search → re-try RxNorm + OpenFDA with resolved generic name

    All successful resolutions are cached permanently.
    """
    if not drug_name or len(drug_name) < 2:
        return None

    clean_name = drug_name.strip()

    # ── 1. Cache check (instant) ──
    cached = _get_cached(clean_name)
    if cached:
        return cached

    logger.info("[PHARMA] Resolving '%s' via external APIs...", clean_name)

    rxcui = ""
    generic_name = ""
    classes: list[str] = []
    mechanisms: list[str] = []
    effects: list[str] = []
    fda_data: dict = {}
    source_parts: list[str] = []

    # ── 2+3. RxNorm (exact + approximate) ──
    rxcui, generic_name = await _rxnorm_resolve_name(clean_name)

    if rxcui and generic_name:
        # Got RxNorm match → get classification
        classes, mechanisms, effects = await _rxnorm_get_classes(generic_name)
        source_parts.append("RxNorm")

        # Enrich with OpenFDA
        fda_data = await _openfda_get_label(generic_name)
        if fda_data:
            source_parts.append("OpenFDA")
            if not classes and fda_data.get("pharm_class_epc"):
                classes = fda_data["pharm_class_epc"]
            if not mechanisms and fda_data.get("pharm_class_moa"):
                mechanisms = fda_data["pharm_class_moa"]
            if not effects and fda_data.get("pharm_class_pe"):
                effects = fda_data["pharm_class_pe"]
    else:
        # ── 4. Try OpenFDA directly ──
        fda_data = await _openfda_get_label(clean_name)
        if fda_data and fda_data.get("generic_name"):
            generic_name = fda_data["generic_name"]
            classes = fda_data.get("pharm_class_epc", [])
            mechanisms = fda_data.get("pharm_class_moa", [])
            effects = fda_data.get("pharm_class_pe", [])
            source_parts.append("OpenFDA")

            # Now try RxNorm with resolved generic for more class data
            rxcui2, _ = await _rxnorm_resolve_name(generic_name)
            if rxcui2:
                c2, m2, e2 = await _rxnorm_get_classes(generic_name)
                classes = classes or c2
                mechanisms = mechanisms or m2
                effects = effects or e2
                rxcui = rxcui2
                source_parts.append("RxNorm")
        else:
            # ── 5. Web search fallback ──
            web_generic = await _web_resolve_drug(clean_name)
            if web_generic:
                source_parts.append("Web Search")
                generic_name = web_generic

                # Re-try RxNorm with web-resolved name
                rxcui, resolved = await _rxnorm_resolve_name(web_generic)
                if rxcui:
                    generic_name = resolved or web_generic
                    classes, mechanisms, effects = await _rxnorm_get_classes(generic_name)
                    source_parts.append("RxNorm")

                # Re-try OpenFDA
                fda_data = await _openfda_get_label(web_generic)
                if fda_data:
                    source_parts.append("OpenFDA")
                    if not classes and fda_data.get("pharm_class_epc"):
                        classes = fda_data["pharm_class_epc"]
                    if not mechanisms and fda_data.get("pharm_class_moa"):
                        mechanisms = fda_data["pharm_class_moa"]

    # ── If we have zero useful data, return None ──
    if not generic_name and not classes and not mechanisms:
        logger.info("[PHARMA] Could not resolve '%s' — not a recognized drug", clean_name)
        return None

    if not generic_name:
        generic_name = clean_name

    info = DrugInfo(
        original_name=clean_name,
        generic_name=generic_name,
        drug_class=classes,
        mechanism=mechanisms,
        effects=effects,
        warnings=fda_data.get("warnings", []),
        contraindications=fda_data.get("contraindications", []),
        interactions_note="; ".join(fda_data.get("drug_interactions", []))[:500],
        source=" + ".join(source_parts) if source_parts else "Unknown",
        rxcui=rxcui,
    )

    # Cache for future lookups ("learning from experience")
    _cache_drug(clean_name, info)

    logger.info(
        "[PHARMA] ✓ Resolved '%s' → %s (%s) [%s]",
        clean_name, info.generic_name,
        ", ".join(info.drug_class[:2]) or "class unknown",
        info.source,
    )
    return info
