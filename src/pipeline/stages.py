"""RRRIE pipeline stage functions — Safety, R1, R2.

Each stage is a standalone async function that:
  - Receives WebSocket + dependencies as explicit parameters
  - Sends stage_start / stage_result / stage_complete events
  - Returns structured data for downstream stages
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from typing import Any

from fastapi import WebSocket

from src.llm.prompt_templates import (
    R1_SYSTEM_PROMPT,
    FAST_MODE_INSTRUCTION,
    R2_QUERY_GENERATION_PROMPT,
    R2_QUERY_USER_TEMPLATE,
    adapt_prompt_for_complexity,
)
from src.llm.stage_adapter import resolve_icd_codes
from src.utils.safety_checks import detect_red_flags, detect_demographic_severity
from src.core.zebra_detector import detect_zebras, format_zebra_alerts

# Conditional imports — same pattern as original server.py
try:
    from src.tools.pubmed_tool import search_pubmed
    HAS_PUBMED = True
except ImportError:
    HAS_PUBMED = False

try:
    from src.utils.medical_codes import lookup_icd11_who, search_icd11_who
    HAS_ICD11 = True
except ImportError:
    HAS_ICD11 = False

# Dynamic tool imports (all optional — gracefully degrade if unavailable)
try:
    from src.tools.web_search_tool import web_search
    HAS_WEB_SEARCH = True
except ImportError:
    HAS_WEB_SEARCH = False

try:
    from src.tools.clinical_trials_tool import search_clinical_trials
    HAS_CLINICAL_TRIALS = True
except ImportError:
    HAS_CLINICAL_TRIALS = False

try:
    from src.tools.openfda_tool import search_drug_interactions, get_adverse_events
    HAS_FDA = True
except ImportError:
    HAS_FDA = False

try:
    from src.tools.medlineplus_tool import get_treatment_guidelines
    HAS_GUIDELINES = True
except ImportError:
    HAS_GUIDELINES = False

try:
    from src.tools.europe_pmc_tool import search_europe_pmc
    HAS_EUROPE_PMC = True
except ImportError:
    HAS_EUROPE_PMC = False

try:
    from src.tools.wikipedia_tool import search_wikipedia_medical
    HAS_WIKIPEDIA = True
except ImportError:
    HAS_WIKIPEDIA = False

try:
    from src.tools.semantic_scholar_tool import search_semantic_scholar
    HAS_SEMANTIC_SCHOLAR = True
except ImportError:
    HAS_SEMANTIC_SCHOLAR = False

from src.pipeline.streaming import (
    stream_groq_completion,
    stream_gemini_completion,
    stream_llm_completion,
    call_llm_no_stream,
    call_groq_no_stream,
    call_gemini_no_stream,
    parse_json_from_response,
)
from src.pipeline.medical_knowledge import (
    MEDICAL_SYNONYMS,
    CLINICAL_LINKS,
    PUBMED_MESH,
    BROAD_CATEGORIES,
    QUERY_FILLER_WORDS,
    EPMC_FILLER_WORDS,
    MESH_SUBHEADINGS,
    CASE_CATEGORY_KEYWORDS,
)

logger = logging.getLogger("rrrie-cdss")


# ═══════════════════════════════════════════════════════════════════
# UTILITIES: Language Detection + Turkish Transliteration
# ═══════════════════════════════════════════════════════════════════

_TR_CHAR_MAP = str.maketrans({
    'ş': 's', 'Ş': 'S', 'ğ': 'g', 'Ğ': 'G',
    'ü': 'u', 'Ü': 'U', 'ö': 'o', 'Ö': 'O',
    'ç': 'c', 'Ç': 'C', 'ı': 'i', 'İ': 'I',
})


def _transliterate_turkish(text: str) -> str:
    """Transliterate Turkish special characters to ASCII equivalents.

    Ş→S, ğ→g, ü→u, ö→o, ç→c, ı→i, İ→I, etc.
    Prevents encoding disasters like 'şurup' → 'urup' in search APIs.
    """
    return text.translate(_TR_CHAR_MAP)


def _has_non_latin_medical_text(text: str) -> bool:
    """Detect if text contains non-English medical terms (Turkish etc.).

    Uses two strategies:
    1. Presence of ANY Turkish-specific character (ş, ğ, ı, İ) → immediate True
    2. ≥3 non-ASCII alphabetic characters → likely non-English

    Returns True when R2 should apply clinical term translation.
    """
    # Strategy 1: Turkish-specific characters (not shared with French/German)
    _TR_UNIQUE = set("şŞğĞıİ")
    if any(c in _TR_UNIQUE for c in text):
        return True

    # Strategy 2: General non-ASCII alpha count (catches other languages too)
    non_ascii_alpha = sum(
        1 for c in text if ord(c) > 127 and c.isalpha()
    )
    return non_ascii_alpha >= 3


# ═══════════════════════════════════════════════════════════════════
# PRE-R1 CLINICAL TEXT ANNOTATION (adaptive language bridge)
# ═══════════════════════════════════════════════════════════════════

async def annotate_clinical_text_pre_r1(
    patient_text: str,
    r0_language: str,
    r0_entities: dict,
    groq_client: Any,
    llm_client: Any,
    llama_server_url: str,
    local_only: bool,
) -> str:
    """Annotate non-English patient text with English medical terms BEFORE R1.

    This is the adaptive language bridge: instead of relying on R1 to understand
    every language's medical jargon, we pre-annotate with English equivalents.
    R0 L1 already detects language — when non-English, this runs.

    Approach: Ask LLM to extract and translate key medical terms inline,
    returning annotated text that preserves the original but adds English context.

    Returns:
        Annotation block to APPEND to patient text before R1, or "" if not needed.
    """
    if r0_language == "en":
        return ""

    if not _has_non_latin_medical_text(patient_text[:800]):
        return ""

    # Build a focused annotation request from R0 entities + raw text
    symptoms = r0_entities.get("symptoms", [])
    vitals = r0_entities.get("vitals", {})
    history = r0_entities.get("history", [])
    medications = r0_entities.get("medications", [])

    prompt_parts = [f"PATIENT TEXT (first 600 chars):\n{patient_text[:600]}"]
    if symptoms:
        prompt_parts.append(f"EXTRACTED SYMPTOMS: {', '.join(symptoms[:10])}")
    if history:
        prompt_parts.append(f"HISTORY ITEMS: {', '.join(history[:5])}")
    if medications:
        prompt_parts.append(f"MEDICATIONS: {', '.join(medications[:5])}")

    system_msg = (
        "You are a medical translator. The patient text below is in a non-English language.\n"
        "Extract EVERY medical/clinical term and translate to standard English medical terminology.\n"
        "For colloquial expressions, provide the correct medical term.\n"
        "Include demographic information (age, sex, reproductive status) in English.\n\n"
        "Output valid JSON ONLY:\n"
        "{\n"
        '  "demographic_summary": "e.g., 24-year-old female, reproductive age",\n'
        '  "clinical_terms": [\n'
        '    {"original": "term in original language", "english": "English medical term"}\n'
        "  ],\n"
        '  "key_findings_english": ["finding1 in English", "finding2 in English"],\n'
        '  "symptom_pattern_english": "Brief English description of the clinical picture"\n'
        "}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "\n\n".join(prompt_parts)},
    ]

    raw = ""
    try:
        if groq_client and getattr(groq_client, "is_available", False) and not local_only:
            raw = await call_groq_no_stream(groq_client, messages, max_tokens=512)
        elif llm_client and llama_server_url:
            raw = await call_llm_no_stream(llm_client, llama_server_url, messages, max_tokens=512)
    except Exception as exc:
        logger.warning("[PRE-R1-ANNOTATE] Translation failed: %s", exc)
        return ""

    if not raw:
        return ""

    result = parse_json_from_response(raw)
    if not result:
        return ""

    # Build annotation block
    parts = ["\n\n🌐 PRE-R1 CLINICAL ANNOTATION (translated from non-English text):"]

    demo = result.get("demographic_summary", "")
    if demo:
        parts.append(f"DEMOGRAPHICS: {demo}")

    terms = result.get("clinical_terms", [])
    if terms:
        parts.append("TERM TRANSLATIONS:")
        for t in terms[:12]:
            orig = t.get("original", "")
            eng = t.get("english", "")
            if orig and eng:
                parts.append(f"  • '{orig}' = {eng}")

    findings = result.get("key_findings_english", [])
    if findings:
        parts.append("KEY FINDINGS (English): " + "; ".join(findings[:8]))

    pattern = result.get("symptom_pattern_english", "")
    if pattern:
        parts.append(f"CLINICAL PATTERN: {pattern}")

    annotation = "\n".join(parts)
    logger.info(
        "[PRE-R1-ANNOTATE] Translated %d terms, %d findings for R1. Annotation: %d chars",
        len(terms), len(findings), len(annotation),
    )
    return annotation


async def _translate_clinical_context_for_r2(
    diagnoses: list[dict],
    patient_text: str,
    groq_client: Any,
    llm_client: Any,
    llama_server_url: str,
    local_only: bool,
) -> dict:
    """Translate non-English R1 diagnoses + patient findings to English medical jargon.

    Uses Groq 70B (preferred) or local LLM to perform a QUICK translation
    of clinical terms before R2 query generation. This ensures:
    - Turkish/regional terms → standard English medical terminology
    - Colloquial descriptions → MeSH-compatible jargon
    - e.g., 'Şurup İntoksikasyonu' → 'Toxic Ingestion / Poisoning'
    - e.g., 'Akut Böbrek Hasarı' → 'Acute Kidney Injury'
    - e.g., 'Hızlı derin solunum' → 'Kussmaul breathing'

    Dynamic: uses the LLM's world knowledge, not hardcoded dictionaries.

    Returns:
        dict with:
        - 'translated_diagnoses': list of {original, english, mesh_term}
        - 'clinical_brief': Formatted text for R2 context injection
        - 'english_dx_names': list of English diagnosis names for programmatic queries
    """
    # Collect non-English diagnosis names
    terms_to_translate = []
    for dx in diagnoses[:5]:
        name = dx.get("diagnosis", "")
        if name and _has_non_latin_medical_text(name):
            terms_to_translate.append(name)

    # Also check patient text for overall language
    patient_needs_translation = _has_non_latin_medical_text(patient_text[:800])

    if not terms_to_translate and not patient_needs_translation:
        return {"translated_diagnoses": [], "clinical_brief": "", "english_dx_names": []}

    # Build a compact translation prompt
    prompt_parts = []
    if terms_to_translate:
        prompt_parts.append(
            "DIAGNOSES TO TRANSLATE:\n"
            + "\n".join(f"  {i+1}. {t}" for i, t in enumerate(terms_to_translate))
        )
    if patient_needs_translation:
        # Extract key clinical sentences (max 500 chars)
        prompt_parts.append(
            f"PATIENT TEXT (extract key clinical findings in English):\n{patient_text[:500]}"
        )

    system_msg = (
        "You are a medical term translator. Given clinical terms and/or patient text "
        "in ANY language, output their standard English medical equivalents.\n"
        "For colloquial terms (e.g., 'syrup poisoning'), ALSO provide the proper "
        "toxicological/medical term (e.g., 'Toxic ingestion', 'Diethylene glycol poisoning').\n"
        "For symptom descriptions, provide the medical sign name "
        "(e.g., 'rapid deep breathing' → 'Kussmaul respiration').\n"
        "Think about what underlying medical CONDITIONS could cause these presentations.\n\n"
        "Output valid JSON ONLY:\n"
        '{\n'
        '  "translations": [\n'
        '    {"original": "...", "english": "...", "mesh_terms": ["MeSH1", "MeSH2"]}\n'
        '  ],\n'
        '  "clinical_findings_english": ["finding in English medical terms", ...],\n'
        '  "differential_considerations": ["possible diagnosis in English based on the pattern", ...]\n'
        '}'
    )
    user_msg = "\n\n".join(prompt_parts)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    raw = ""
    try:
        # Prefer Groq 70B for translation (much better at multilingual understanding)
        if groq_client and getattr(groq_client, 'is_available', False) and not local_only:
            raw = await call_groq_no_stream(groq_client, messages, max_tokens=512)
        elif llm_client and llama_server_url:
            raw = await call_llm_no_stream(llm_client, llama_server_url, messages, max_tokens=512)
    except Exception as exc:
        logger.warning("[R2-TRANSLATE] Clinical translation failed: %s", exc)
        # Fallback: just transliterate Turkish chars
        fallback_names = [_transliterate_turkish(t) for t in terms_to_translate]
        return {
            "translated_diagnoses": [],
            "clinical_brief": "",
            "english_dx_names": fallback_names,
        }

    if not raw:
        # Fallback: transliterate
        fallback_names = [_transliterate_turkish(t) for t in terms_to_translate]
        return {
            "translated_diagnoses": [],
            "clinical_brief": "",
            "english_dx_names": fallback_names,
        }

    # Parse response
    result = parse_json_from_response(raw)
    translations = result.get("translations", [])
    findings = result.get("clinical_findings_english", [])
    diff_considerations = result.get("differential_considerations", [])

    # Build clinical brief for injection into R2 context
    brief_parts = []
    english_dx_names = []

    if translations:
        brief_parts.append("CLINICAL TERM TRANSLATION (use these English terms for searches):")
        for t in translations:
            orig = t.get("original", "")
            eng = t.get("english", "")
            mesh = t.get("mesh_terms", [])
            if orig and eng:
                line = f"  • '{orig}' = {eng}"
                if mesh:
                    line += f" [MeSH: {', '.join(mesh[:3])}]"
                brief_parts.append(line)
                english_dx_names.append(eng)

    if findings:
        brief_parts.append("KEY CLINICAL FINDINGS (English medical terms):")
        for f in findings[:6]:
            brief_parts.append(f"  • {f}")

    if diff_considerations:
        brief_parts.append("DIFFERENTIAL CONSIDERATIONS from translation context:")
        for dc in diff_considerations[:4]:
            brief_parts.append(f"  • {dc}")
            english_dx_names.append(dc)

    clinical_brief = "\n".join(brief_parts)

    logger.info(
        "[R2-TRANSLATE] Translated %d terms, %d findings, %d differentials. Brief: %d chars",
        len(translations), len(findings), len(diff_considerations), len(clinical_brief),
    )

    return {
        "translated_diagnoses": translations,
        "clinical_brief": clinical_brief,
        "english_dx_names": english_dx_names,
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE 0: Safety + Zebra Detection
# ═══════════════════════════════════════════════════════════════════

async def run_safety(ws: WebSocket, patient_text: str, r0_result=None) -> dict:
    """Run safety pre-checks: red flags + zebra detection + DLLM R0 merge.

    Args:
        ws: WebSocket for streaming events.
        patient_text: Raw patient text.
        r0_result: Optional R0Result from DLLM — provides context-aware red flags.

    Returns:
        dict with keys: red_flags (list[str]), zebra_matches (list),
        zebra_alert_text (str).
    """
    await ws.send_json({
        "type": "stage_start",
        "stage": "SAFETY",
        "title": "Safety Pre-Check",
        "description": "Scanning for red flags, emergency indicators and rare disease patterns...",
    })

    # Red flags — keyword-based
    red_flags = detect_red_flags(patient_text, patient_text.split())

    # Demographic-aware severity escalation (adaptive: demographics × vitals → alerts)
    # Pass red_flags as proxy for vitals context; also scan raw text for vital signs
    demographic_alerts = detect_demographic_severity(patient_text, red_flags)
    red_flags.extend(demographic_alerts)

    # Merge DLLM R0 context-aware red flags (if available)
    if r0_result is not None:
        dllm_flags = getattr(r0_result, "red_flags", [])
        for rf in dllm_flags:
            if isinstance(rf, dict):
                label = rf.get("flag", "")
                severity = rf.get("severity", "moderate")
                context = rf.get("context", "")
                text = f"[DLLM/{severity}] {label}"
                if context:
                    text += f" — {context}"
                if text not in red_flags:
                    red_flags.append(text)

    if red_flags:
        await ws.send_json({
            "type": "red_flags",
            "stage": "SAFETY",
            "flags": red_flags,
        })
    else:
        await ws.send_json({
            "type": "info",
            "stage": "SAFETY",
            "content": "No immediate red flags detected.",
        })

    # Zebra detection
    zebra_matches = detect_zebras(patient_text)
    zebra_alert_text = ""
    if zebra_matches:
        zebra_alert_text = format_zebra_alerts(zebra_matches)
        zebra_data = []
        for z in zebra_matches:
            zebra_data.append({
                "disease": z.disease,
                "icd11": z.icd11,
                "confidence": z.confidence,
                "matched": len(z.matched_criteria) if isinstance(z.matched_criteria, list) else z.matched_criteria,
                "total": z.total_criteria,
                "key_question": z.key_question,
            })
        await ws.send_json({
            "type": "zebra_flags",
            "stage": "SAFETY",
            "zebras": zebra_data,
        })
    else:
        await ws.send_json({
            "type": "info",
            "stage": "SAFETY",
            "content": "No rare disease patterns (zebra flags) detected.",
        })

    await ws.send_json({"type": "stage_complete", "stage": "SAFETY"})

    return {
        "red_flags": red_flags,
        "zebra_matches": zebra_matches,
        "zebra_alert_text": zebra_alert_text,
    }


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: R1 — Reasoned Analysis
# ═══════════════════════════════════════════════════════════════════

async def run_r1(
    ws: WebSocket,
    patient_text: str,
    *,
    red_flags: list[str],
    zebra_alert_text: str,
    drug_facts: str = "",
    is_fast: bool,
    groq_client: Any,
    gemini_client: Any = None,
    gemini_use_pro: bool = False,
    llm_client: Any,
    llama_server_url: str,
    thinking_enabled: bool = True,
    local_only: bool = False,
    budget: Any = None,
    complexity: str = "moderate",
) -> tuple[dict, dict, str]:
    """Run R1 stage: reasoned differential analysis.

    Returns:
        (r1_json, r1_result, r1_model_label)
    """
    use_gemini_r1 = gemini_client is not None and getattr(gemini_client, "is_available", False)
    use_groq_r1 = (not use_gemini_r1) and groq_client.is_available and not is_fast and not local_only

    if use_gemini_r1:
        r1_model_label = f"☁️ Gemini {'Pro' if gemini_use_pro else 'Flash'}"
    elif use_groq_r1:
        r1_model_label = f"☁️ {groq_client.model}"
    else:
        r1_model_label = "💻 Local 4B"

    await ws.send_json({
        "type": "stage_start",
        "stage": "R1",
        "title": "R1 — Reasoned Analysis" + (" ⚡" if is_fast else " 🧠"),
        "description": (
            f"{'Quick differential...' if is_fast else 'Deep analysis — differential diagnoses + knowledge gaps...'}"
            f" [{r1_model_label}]"
        ),
    })

    # Build user prompt
    patient_text_r1 = patient_text
    if red_flags:
        patient_text_r1 += "\n\n⚠️ DETECTED RED FLAGS:\n" + "\n".join(f"- {f}" for f in red_flags)
    if zebra_alert_text:
        patient_text_r1 += "\n\n" + zebra_alert_text
    if drug_facts:
        patient_text_r1 += "\n\n" + drug_facts
    if is_fast:
        patient_text_r1 = FAST_MODE_INSTRUCTION + "\n\n" + patient_text_r1

    r1_messages = [
        {"role": "system", "content": adapt_prompt_for_complexity(R1_SYSTEM_PROMPT, complexity) if not (use_gemini_r1 or use_groq_r1) else R1_SYSTEM_PROMPT},
        {"role": "user", "content": patient_text_r1},
    ]

    r1_prompt_est = len(patient_text_r1) // 3

    if use_gemini_r1:
        r1_max_tokens = budget.allocate("R1", iteration=1, prompt_tokens=r1_prompt_est, is_groq=True) if budget else 8192
        try:
            r1_result = await stream_gemini_completion(
                ws, gemini_client, r1_messages, stage="R1",
                max_tokens=r1_max_tokens, use_pro=gemini_use_pro,
            )
        except Exception as exc:
            logger.warning("[R1] Gemini failed (%s), falling back to Groq/local", exc)
            await ws.send_json({
                "type": "info", "stage": "R1",
                "content": f"⚠ Gemini unavailable ({exc}), falling back...",
            })
            if groq_client.is_available:
                r1_result = await stream_groq_completion(
                    ws, groq_client, r1_messages, stage="R1", max_tokens=r1_max_tokens,
                )
            else:
                r1_max_tokens = budget.allocate("R1", iteration=1, prompt_tokens=r1_prompt_est, is_groq=False) if budget else 2048
                r1_result = await stream_llm_completion(
                    ws, llm_client, llama_server_url, r1_messages,
                    stage="R1", max_tokens=r1_max_tokens, thinking_enabled=thinking_enabled,
                    budget_managed=bool(budget),
                )
    elif use_groq_r1:
        r1_max_tokens = budget.allocate("R1", iteration=1, prompt_tokens=r1_prompt_est, is_groq=True) if budget else (2048 if is_fast else 4096)
        try:
            r1_result = await stream_groq_completion(
                ws, groq_client, r1_messages, stage="R1", max_tokens=r1_max_tokens,
            )
        except Exception as exc:
            logger.warning("[R1] Groq failed (%s), falling back to local model", exc)
            await ws.send_json({
                "type": "info", "stage": "R1",
                "content": f"⚠ Groq unavailable ({exc}), falling back to local model...",
            })
            r1_max_tokens = budget.allocate("R1", iteration=1, prompt_tokens=r1_prompt_est, is_groq=False) if budget else (1536 if is_fast else 2048)
            r1_result = await stream_llm_completion(
                ws, llm_client, llama_server_url, r1_messages,
                stage="R1", max_tokens=r1_max_tokens, thinking_enabled=thinking_enabled,
                budget_managed=bool(budget),
            )
    else:
        r1_max_tokens = budget.allocate("R1", iteration=1, prompt_tokens=r1_prompt_est, is_groq=False) if budget else (1536 if is_fast else 2048)
        r1_result = await stream_llm_completion(
            ws, llm_client, llama_server_url, r1_messages,
            stage="R1", max_tokens=r1_max_tokens, thinking_enabled=thinking_enabled,
            budget_managed=bool(budget),
        )

    # Report R1 usage to budget manager
    if budget:
        budget.report("R1", allocated=r1_max_tokens, used=r1_result.get("completion_tokens", 0))

    # Parse R1 JSON
    r1_json = parse_json_from_response(r1_result["clean_content"])
    if not r1_json or not r1_json.get("differential_diagnoses"):
        logger.warning(
            "[R1] JSON parse produced empty/incomplete result. clean_content length=%d",
            len(r1_result["clean_content"]),
        )
        await ws.send_json({
            "type": "info",
            "stage": "R1",
            "content": (
                "⚠ R1 produced incomplete JSON — model output may be truncated. "
                f"Raw length: {len(r1_result['clean_content'])} chars."
            ),
        })

    await ws.send_json({
        "type": "stage_result",
        "stage": "R1",
        "data": r1_json,
        "stats": {
            "tokens": r1_result["completion_tokens"],
            "time": r1_result["elapsed"],
            "tok_per_sec": r1_result["tok_per_sec"],
        },
    })
    await ws.send_json({"type": "stage_complete", "stage": "R1"})

    return r1_json, r1_result, r1_model_label


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: R2 — Reinforced Fact-Finding (LLM-Driven Dynamic Research)
# ═══════════════════════════════════════════════════════════════════

# Tool dispatcher: maps tool name → (available_flag, async_callable)
_TOOL_DISPATCH = {}


def _init_tool_dispatch():
    """Lazy-init the tool dispatch table after imports resolve."""
    global _TOOL_DISPATCH
    if _TOOL_DISPATCH:
        return
    if HAS_PUBMED:
        _TOOL_DISPATCH["search_pubmed"] = search_pubmed
    if HAS_WEB_SEARCH:
        _TOOL_DISPATCH["web_search"] = web_search
    if HAS_CLINICAL_TRIALS:
        _TOOL_DISPATCH["search_clinical_trials"] = search_clinical_trials
    if HAS_FDA:
        _TOOL_DISPATCH["search_drug_interactions"] = search_drug_interactions
        _TOOL_DISPATCH["get_adverse_events"] = get_adverse_events
    if HAS_GUIDELINES:
        _TOOL_DISPATCH["get_treatment_guidelines"] = get_treatment_guidelines
    if HAS_EUROPE_PMC:
        _TOOL_DISPATCH["search_europe_pmc"] = search_europe_pmc
    if HAS_WIKIPEDIA:
        _TOOL_DISPATCH["search_wikipedia_medical"] = search_wikipedia_medical
    if HAS_SEMANTIC_SCHOLAR:
        _TOOL_DISPATCH["search_semantic_scholar"] = search_semantic_scholar


# ── Query Optimization ──────────────────────────────────────────────
# MEDICAL_SYNONYMS, CLINICAL_LINKS, PUBMED_MESH, BROAD_CATEGORIES,
# QUERY_FILLER_WORDS, EPMC_FILLER_WORDS — all imported from
# src.pipeline.medical_knowledge (comprehensive, all-specialty coverage).


def _optimize_query_for_tool(tool_name: str, params: dict) -> dict:
    """Optimize search parameters for any tool based on medical keyword extraction.

    Applies synonym expansion, clinical linking, and query cleanup.
    Returns modified params dict (does not mutate the original).
    """
    params = dict(params)  # shallow copy

    # Only optimize tools that have a text query
    query_key = None
    if tool_name in ("search_pubmed", "search_europe_pmc", "search_semantic_scholar", "web_search"):
        query_key = "query"
    elif tool_name == "search_wikipedia_medical":
        query_key = "disease_name"

    if not query_key or query_key not in params:
        return params

    raw_query = str(params[query_key]).strip()
    query_lower = raw_query.lower()

    # ── Step 1: Synonym expansion — replace known terms with preferred forms ──
    #   Uses word-boundary regex to avoid substring false-positives
    expanded_terms: list[str] = []
    remaining = query_lower
    for term in sorted(MEDICAL_SYNONYMS.keys(), key=len, reverse=True):
        pattern = r'(?<!\w)' + re.escape(term) + r'(?!\w)'
        if re.search(pattern, remaining):
            preferred = MEDICAL_SYNONYMS[term]
            if preferred.lower() not in [t.lower() for t in expanded_terms]:
                expanded_terms.append(preferred)
            remaining = re.sub(pattern, " ", remaining)

    # ── Step 2: Clinical linking — inject linked terms when pairs match ──
    all_words = set(query_lower.split())
    # Also add the expanded terms as words for linking
    for et in expanded_terms:
        all_words.update(et.lower().split())

    linked_additions: list[str] = []
    seen_links: set[str] = set()
    for trigger_set, addition in CLINICAL_LINKS:
        if trigger_set.issubset(all_words):
            add_lower = addition.lower()
            if add_lower not in query_lower and add_lower not in " ".join(expanded_terms).lower() and add_lower not in seen_links:
                linked_additions.append(addition)
                seen_links.add(add_lower)

    # ── Tool-specific optimization ──
    if tool_name == "search_pubmed":
        # Use existing PubMed MeSH optimizer + inject only 1 linked term
        # PubMed AND-chains with 4+ terms return 0 results, so be conservative
        optimized = _optimize_pubmed_query(raw_query)
        if linked_additions:
            # Only append 1 linked term (the most specific one, usually longest)
            best_link = max(linked_additions, key=len)
            optimized += f' AND "{best_link}"'
            logger.info("[R2-LINK] PubMed enriched with clinical link: %s", best_link)
        params["query"] = optimized

    elif tool_name == "search_europe_pmc":
        # Europe PMC works well with natural language but benefits from focused queries.
        # Combine expanded terms + linked additions + important raw words.
        # Use WORD-LEVEL dedup to avoid "Wernicke encephalopathy" appearing twice.
        import re as _re
        # Collect all words from expanded terms, linked additions, and raw query
        seen_words: set[str] = set()
        final_words: list[str] = []

        def _add_words(phrase: str) -> None:
            for w in phrase.split():
                wl = w.lower().strip(".,;:()")
                if len(wl) > 2 and wl not in EPMC_FILLER_WORDS and wl not in seen_words:
                    seen_words.add(wl)
                    final_words.append(w)

        # Priority: expanded terms > linked additions > raw keywords
        for t in expanded_terms:
            _add_words(t)
        for la in linked_additions:
            _add_words(la)
        # Also keep important raw words not covered
        raw_words = _re.findall(r'[a-zA-Z]{4,}', raw_query)
        for w in raw_words:
            wl = w.lower()
            if wl not in EPMC_FILLER_WORDS and wl not in seen_words:
                seen_words.add(wl)
                final_words.append(w)

        params["query"] = " ".join(final_words[:8])
        logger.info("[R2-EPMC] Optimized: '%s' → '%s'", raw_query[:60], params["query"])

    elif tool_name == "search_semantic_scholar":
        # Semantic Scholar handles natural language well; inject linked terms
        if linked_additions:
            params["query"] = raw_query + " " + " ".join(linked_additions[:2])
            logger.info("[R2-S2] Enriched: '%s' → '%s'", raw_query[:60], params["query"])

    elif tool_name == "web_search":
        # Web search — inject linked terms for better discovery
        if linked_additions:
            params["query"] = raw_query + " " + " ".join(linked_additions[:1])
            logger.info("[R2-WEB] Enriched: '%s' → '%s'", raw_query[:60], params["query"])

    elif tool_name == "search_wikipedia_medical":
        # Wikipedia needs a clean disease name, not a sentence
        if expanded_terms:
            # Pick the most specific (longest) expanded term — it's usually the disease name
            best = max(expanded_terms, key=len)
            params["disease_name"] = best
            logger.info("[R2-WIKI] Optimized: '%s' → '%s'", raw_query, params["disease_name"])
        elif raw_query:
            # Fallback: take first 3 meaningful words as disease name
            import re as _re
            words = _re.findall(r'[a-zA-Z]{3,}', raw_query)
            _wiki_filler = {"and", "the", "with", "for", "from", "diagnosis", "treatment"}
            words = [w for w in words if w.lower() not in _wiki_filler][:3]
            if words:
                params["disease_name"] = " ".join(words)
                logger.info("[R2-WIKI] Fallback: '%s' → '%s'", raw_query, params["disease_name"])

    return params

def _optimize_pubmed_query(raw_query: str) -> str:
    """Transform a verbose LLM query into PubMed-compatible search syntax.

    Strategy:
    1. Remove filler/conversational words
    2. Detect quoted phrases that should stay intact
    3. Split remaining into MeSH-friendly terms
    4. Join with AND, cap at 3 terms max
    5. Add [MeSH] hints for known medical terms

    Uses PUBMED_MESH from medical_knowledge.py (200+ terms, all specialties).
    """
    import re as _re

    query_lower = raw_query.lower().strip()

    # Step 1: Check for MeSH term matches (longest first)
    mesh_hits = []
    remaining = query_lower
    for term in sorted(PUBMED_MESH.keys(), key=len, reverse=True):
        if term in remaining:
            mesh_hits.append(PUBMED_MESH[term])
            remaining = remaining.replace(term, " ")

    # Step 2: Extract useful remaining words
    remaining_words = _re.findall(r'[a-z]{3,}', remaining)
    useful_words = [w for w in remaining_words if w not in QUERY_FILLER_WORDS and len(w) > 3]

    # Step 3: Build query — MAX 3 AND terms for PubMed (2 MeSH + 1 keyword)
    # PubMed AND-chains with 4+ terms almost always return 0 results.
    parts = []

    # Deduplicate MeSH hits (e.g., "Wernicke Encephalopathy" captured twice)
    seen_mesh = set()
    unique_mesh = []
    for mh in mesh_hits:
        if mh not in seen_mesh:
            seen_mesh.add(mh)
            unique_mesh.append(mh)

    # Add MeSH terms with [MeSH] qualifier (max 2)
    for mh in unique_mesh[:2]:
        parts.append(f'"{mh}"[MeSH]')

    # Add remaining useful keywords (max 1)
    for w in useful_words[:1]:
        if not any(w.lower() in p.lower() for p in parts):
            parts.append(w)

    if not parts:
        # Nothing extracted — return first 4 words of original
        words = raw_query.split()[:4]
        return ' '.join(words)

    optimized = ' AND '.join(parts)

    logger.info("[R2-QUERY] Optimized: '%s' → '%s'", raw_query[:60], optimized)
    return optimized


def _relax_pubmed_query(mesh_query: str) -> str:
    """Relax a MeSH-optimized PubMed query that returned 0 results.

    Strategy:
    1. Strip [MeSH] qualifiers (use free-text matching instead)
    2. Remove quotes
    3. Keep only the first 2 terms joined by AND
    """
    import re as _re

    # Strip [MeSH] tags and quotes
    relaxed = _re.sub(r'\[MeSH\]', '', mesh_query)
    relaxed = relaxed.replace('"', '').strip()

    # Split by AND and keep first 2 terms
    parts = [p.strip() for p in relaxed.split('AND') if p.strip()]
    if len(parts) > 2:
        parts = parts[:2]

    return ' AND '.join(parts) if parts else mesh_query


def _get_broad_query(original_query: str) -> str | None:
    """Get a broader search query using BROAD_CATEGORIES.

    When a specific query returns 0 results, this provides a semantically
    broader alternative instead of just removing words. Works for ANY
    medical condition that's in our knowledge base.

    Returns broader query string or None if no match found.
    """
    q_lower = original_query.lower().strip()
    # Try longest match first — "wernicke encephalopathy" before "wernicke"
    for term in sorted(BROAD_CATEGORIES.keys(), key=len, reverse=True):
        if term in q_lower:
            broad = BROAD_CATEGORIES[term]
            logger.info("[R2-BROAD] '%s' → broad category: '%s'", term, broad)
            return broad
    # Also try matching against MEDICAL_SYNONYMS values → BROAD_CATEGORIES
    for syn_key in sorted(MEDICAL_SYNONYMS.keys(), key=len, reverse=True):
        if syn_key in q_lower:
            preferred = MEDICAL_SYNONYMS[syn_key].lower()
            if preferred in BROAD_CATEGORIES:
                broad = BROAD_CATEGORIES[preferred]
                logger.info("[R2-BROAD] '%s' → synonym '%s' → broad: '%s'", syn_key, preferred, broad)
                return broad
    return None


# ── Dynamic Case Category Detection ─────────────────────────────────

def _detect_case_categories(diagnosis_text: str) -> list[str]:
    """Dynamically detect medical case categories from diagnosis text.

    Uses CASE_CATEGORY_KEYWORDS from medical_knowledge.py to detect categories
    like 'toxicology', 'infectious', 'autoimmune', etc. Returns all matching
    categories sorted by match strength (number of keyword hits).
    """
    text_lower = diagnosis_text.lower()
    scores: dict[str, int] = {}
    for category, keywords in CASE_CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scores[category] = hits
    return sorted(scores, key=scores.get, reverse=True)


def _get_mesh_subheadings_for_categories(categories: list[str]) -> list[str]:
    """Return appropriate MeSH subheadings for detected case categories.

    Uses MESH_SUBHEADINGS from medical_knowledge.py. E.g. toxicology → /po (poisoning).
    """
    subheadings: list[str] = []
    seen: set[str] = set()
    for cat in categories:
        for sh in MESH_SUBHEADINGS.get(cat, []):
            if sh not in seen:
                subheadings.append(sh)
                seen.add(sh)
    return subheadings


def _extract_core_terms(diagnosis_text: str) -> list[str]:
    """Extract meaningful medical terms from a diagnosis string.

    Strips filler words, severity adjectives, and returns significant
    medical terms in order of significance (longest multi-word matches first).
    """
    import re as _re
    text_lower = diagnosis_text.lower().strip()
    _severity_words = {
        "severe", "acute", "chronic", "mild", "moderate", "suspected",
        "probable", "possible", "likely", "unlikely", "early", "late",
        "advanced", "progressive", "recurrent", "secondary", "primary",
    }
    # First: find multi-word terms that match MEDICAL_SYNONYMS or PUBMED_MESH
    matched_terms: list[str] = []
    remaining = text_lower
    for term in sorted(
        list(MEDICAL_SYNONYMS.keys()) + list(PUBMED_MESH.keys()),
        key=len, reverse=True,
    ):
        if term in remaining and len(term) > 3:
            preferred = MEDICAL_SYNONYMS.get(term, term)
            if preferred.lower() not in [m.lower() for m in matched_terms]:
                matched_terms.append(preferred)
            remaining = remaining.replace(term, " ")

    # Then: collect remaining meaningful words
    words = _re.findall(r'[a-z]{4,}', remaining)
    extra = [w for w in words if w not in QUERY_FILLER_WORDS and w not in _severity_words]

    return matched_terms + extra


# ── Query Quality Gate ───────────────────────────────────────────────

def _quality_gate(queries: list[dict], diagnoses: list[dict]) -> list[dict]:
    """Filter and fix low-quality LLM-generated queries before execution.

    Dynamic rules (not disease-specific):
    1. Reject single-word queries for literature tools (PubMed, EPMC, S2)
    2. Fix compound-name fragments using MEDICAL_SYNONYMS context
    3. Enforce PubMed AND-chain limit (max 4)
    4. Reject filler-only queries
    5. Fix queries that are a substring of a known diagnosis term
    """
    if not queries:
        return queries

    # Build context from R1 diagnoses for fragment detection
    dx_context = " ".join(
        d.get("diagnosis", "") for d in (diagnoses or [])
    ).lower()

    filtered: list[dict] = []
    literature_tools = {
        "search_pubmed", "search_europe_pmc", "search_semantic_scholar",
    }

    for q in queries:
        tool = q.get("tool", "")
        params = q.get("params", {})
        query_text = str(params.get("query", params.get("disease_name", ""))).strip()

        if not query_text:
            logger.warning("[QUALITY-GATE] Empty query for %s → SKIP", tool)
            continue

        words = query_text.split()

        # ── RULE 1: Minimum word count for literature searches ──
        if tool in literature_tools and len(words) < 2:
            # Try to fix: check if this single word is a fragment of a diagnosis term
            fixed = _fix_fragment_query(query_text, dx_context)
            if fixed and fixed != query_text:
                logger.info(
                    "[QUALITY-GATE] Fixed single-word '%s' → '%s'",
                    query_text, fixed,
                )
                params = dict(params)
                params["query"] = fixed
                q = {**q, "params": params}
            else:
                logger.warning(
                    "[QUALITY-GATE] Rejected single-word %s query: '%s'",
                    tool, query_text,
                )
                continue

        # ── RULE 2: Fix compound-name fragments ──
        if tool in literature_tools and len(words) <= 2:
            fixed = _fix_fragment_query(query_text, dx_context)
            if fixed and fixed != query_text:
                logger.info(
                    "[QUALITY-GATE] Fixed fragment '%s' → '%s'",
                    query_text, fixed,
                )
                params = dict(params)
                params["query"] = fixed
                q = {**q, "params": params}

        # ── RULE 3: PubMed AND-chain limit ──
        if tool == "search_pubmed":
            query_val = params.get("query", "")
            and_count = query_val.upper().count(" AND ")
            if and_count > 3:
                parts = query_val.split(" AND ")[:3]
                params = dict(params)
                params["query"] = " AND ".join(parts)
                q = {**q, "params": params}
                logger.info("[QUALITY-GATE] Trimmed AND-chain to 3 terms")

        # ── RULE 4: Reject filler-only queries ──
        query_val = str(params.get("query", params.get("disease_name", "")))
        meaningful = [
            w for w in query_val.lower().split()
            if w not in QUERY_FILLER_WORDS and len(w) > 3
        ]
        if not meaningful:
            logger.warning(
                "[QUALITY-GATE] Rejected all-filler query: '%s'", query_val,
            )
            continue

        # ── RULE 5: Sanitize non-ASCII characters in queries ──
        # Prevents encoding disasters (e.g., 'şurup' → 'urup' on PubMed/EPMC)
        query_val = str(params.get("query", params.get("disease_name", "")))
        if any(ord(c) > 127 for c in query_val):
            sanitized = _transliterate_turkish(query_val)
            if sanitized != query_val:
                logger.info(
                    "[QUALITY-GATE] Transliterated non-ASCII: '%s' → '%s'",
                    query_val[:40], sanitized[:40],
                )
                params = dict(params)
                if "query" in params:
                    params["query"] = sanitized
                elif "disease_name" in params:
                    params["disease_name"] = sanitized
                q = {**q, "params": params}

        filtered.append(q)

    rejected = len(queries) - len(filtered)
    if rejected:
        logger.info(
            "[QUALITY-GATE] Passed %d/%d queries (%d rejected/fixed)",
            len(filtered), len(queries), rejected,
        )
    return filtered


def _fix_fragment_query(fragment: str, dx_context: str) -> str | None:
    """Try to complete a query fragment using diagnosis context + knowledge base.

    Dynamic approach: find the longest known multi-word term from
    MEDICAL_SYNONYMS or PUBMED_MESH that contains the fragment AND
    appears in the diagnosis context.
    """
    frag_lower = fragment.lower().strip()
    if not frag_lower:
        return None

    # Strategy 1: Find the fragment inside a known compound term from context
    # E.g. "methyl" in context "methyl salicylate toxicity"
    # → look for "methyl salicylate" in MEDICAL_SYNONYMS keys
    best_match = None
    best_len = 0
    all_known_terms = set(MEDICAL_SYNONYMS.keys()) | set(PUBMED_MESH.keys())

    for term in all_known_terms:
        if frag_lower in term and len(term) > best_len:
            # Verify this term is relevant to the diagnosis context
            if term in dx_context or any(
                w in dx_context for w in term.split() if len(w) > 3
            ):
                best_match = term
                best_len = len(term)

    if best_match:
        # Return the preferred form from MEDICAL_SYNONYMS if available
        return MEDICAL_SYNONYMS.get(best_match, best_match)

    # Strategy 2: Find the fragment as a word-start in the context
    # E.g. "methyl" appears in context "methyl salicylate toxicity"
    import re as _re
    pattern = rf'\b{_re.escape(frag_lower)}\s+\w+'
    match = _re.search(pattern, dx_context)
    if match:
        expansion = match.group().strip()
        if len(expansion.split()) >= 2:
            return expansion

    return None


# ── Programmatic Query Generation ────────────────────────────────────

def _generate_programmatic_queries(
    diagnoses: list[dict],
    patient_text: str,
) -> list[dict]:
    """Generate guaranteed-quality queries directly from R1 diagnoses.

    This bypasses the LLM entirely — uses knowledge base lookups to create
    proper MeSH-aware queries for each diagnosis. Dynamic: works for ANY
    condition in the knowledge base; uses smart extraction for unknown ones.
    """
    queries: list[dict] = []
    seen_queries: set[str] = set()  # dedup key: (tool, query_text_lower)

    for dx in diagnoses[:5]:
        name = dx.get("diagnosis", "")
        if not name:
            continue

        # Skip non-Latin diagnosis names — translated English versions are
        # appended separately by run_r2() as synthetic diagnosis entries.
        # Querying PubMed/EPMC with Turkish names yields zero results.
        if _has_non_latin_medical_text(name):
            continue

        core_terms = _extract_core_terms(name)
        categories = _detect_case_categories(name)
        subheadings = _get_mesh_subheadings_for_categories(categories)

        # ── 1. PubMed: Smart MeSH query ──
        pubmed_q = _build_dynamic_pubmed_query(name, core_terms, subheadings)
        dedup_key = f"search_pubmed|{pubmed_q.lower()}"
        if dedup_key not in seen_queries:
            seen_queries.add(dedup_key)
            queries.append({
                "tool": "search_pubmed",
                "params": {"query": pubmed_q},
                "rationale": f"[PROGRAMMATIC] MeSH-optimized search for: {name}",
            })

        # ── 2. Europe PMC: Natural language query ──
        epmc_terms = core_terms[:3] if core_terms else name.split()[:3]
        epmc_q = " ".join(epmc_terms)
        dedup_key = f"search_europe_pmc|{epmc_q.lower()}"
        if dedup_key not in seen_queries:
            seen_queries.add(dedup_key)
            queries.append({
                "tool": "search_europe_pmc",
                "params": {"query": epmc_q, "max_results": 5},
                "rationale": f"[PROGRAMMATIC] Full-text evidence for: {name}",
            })

        # ── 3. Wikipedia: Disease overview (always works) ──
        wiki_name = core_terms[0] if core_terms else name
        # Clean for wiki: pick the most specific disease-like term
        dedup_key = f"search_wikipedia_medical|{wiki_name.lower()}"
        if dedup_key not in seen_queries:
            seen_queries.add(dedup_key)
            queries.append({
                "tool": "search_wikipedia_medical",
                "params": {"disease_name": wiki_name},
                "rationale": f"[PROGRAMMATIC] Overview for: {name}",
            })

    logger.info(
        "[R2-PROGRAMMATIC] Generated %d queries for %d diagnoses (categories: %s)",
        len(queries),
        min(len(diagnoses), 3),
        ", ".join(_detect_case_categories(
            " ".join(d.get("diagnosis", "") for d in diagnoses[:3])
        )[:3]) or "general",
    )
    return queries


def _build_dynamic_pubmed_query(
    diagnosis_name: str,
    core_terms: list[str],
    subheadings: list[str],
) -> str:
    """Build an optimized PubMed query dynamically from extracted terms.

    Strategy:
    1. Look up MeSH terms from PUBMED_MESH for known terms
    2. Apply MeSH subheadings based on case category (e.g. /po for toxicology)
    3. If no MeSH match, use ATM-friendly untagged terms (let PubMed auto-map)
    4. Cap at 3 AND terms maximum
    """
    name_lower = diagnosis_name.lower()

    # Step 1: Find MeSH matches from core terms
    mesh_parts: list[str] = []
    non_mesh_parts: list[str] = []
    used_terms: set[str] = set()

    for term in core_terms:
        term_lower = term.lower()
        if term_lower in used_terms:
            continue

        mesh_heading = PUBMED_MESH.get(term_lower)
        if not mesh_heading:
            # Try the raw term from diagnosis
            mesh_heading = PUBMED_MESH.get(term_lower.replace("-", " "))

        if mesh_heading and mesh_heading not in used_terms:
            used_terms.add(mesh_heading)
            used_terms.add(term_lower)

            # Step 2: Apply subheading if relevant
            if subheadings and len(mesh_parts) == 0:
                # Apply first relevant subheading to first MeSH term
                mesh_parts.append(f'"{mesh_heading}/{subheadings[0]}"[MeSH]')
            else:
                mesh_parts.append(f'"{mesh_heading}"[MeSH]')
        elif len(term_lower) > 3 and term_lower not in QUERY_FILLER_WORDS:
            if term_lower not in used_terms:
                non_mesh_parts.append(term)
                used_terms.add(term_lower)

    # Step 3: Build final query — max 3 AND terms
    parts: list[str] = []

    # MeSH terms first (max 2)
    for mp in mesh_parts[:2]:
        parts.append(mp)

    # Then non-MeSH terms (up to total 3)
    for nmp in non_mesh_parts:
        if len(parts) >= 3:
            break
        parts.append(nmp)

    if not parts:
        # Absolute fallback: use first 3 words of diagnosis name
        words = [w for w in name_lower.split() if len(w) > 3 and w not in QUERY_FILLER_WORDS]
        return " ".join(words[:3]) if words else diagnosis_name

    return " AND ".join(parts)


# ── Safety Valve — Zero-Evidence Fallback ────────────────────────────

async def _safety_valve_search(
    ws: WebSocket,
    diagnoses: list[dict],
    r1_confidence: float,
    existing_evidence: list[dict],
) -> list[dict]:
    """Last-resort search when R2 found insufficient evidence + R1 confidence is high.

    Dynamically generates simple, broad queries that are almost guaranteed
    to return results. Only activates when:
    - R1 primary confidence >= 0.75
    - Total article count < 3

    Returns augmented evidence list.
    """
    # Count existing articles
    total_articles = 0
    for ev in existing_evidence:
        total_articles += len(ev.get("articles", []))
        total_articles += len(ev.get("papers", []))
        total_articles += len(ev.get("results", []))
        if ev.get("summary"):
            total_articles += 1

    # Check activation conditions
    if total_articles >= 3 or r1_confidence < 0.75:
        return existing_evidence

    if not diagnoses:
        return existing_evidence

    primary_dx = diagnoses[0].get("diagnosis", "")
    if not primary_dx:
        return existing_evidence

    await ws.send_json({
        "type": "info",
        "stage": "R2",
        "content": (
            f"🛡️ Safety Valve: Only {total_articles} articles found with "
            f"R1 confidence {r1_confidence:.0%}. Running backup searches..."
        ),
    })

    # Generate simple backup queries dynamically
    core_terms = _extract_core_terms(primary_dx)
    simple_name = core_terms[0] if core_terms else primary_dx

    backup_queries: list[dict] = []

    # 1. Europe PMC with just the disease name (most forgiving search engine)
    if HAS_EUROPE_PMC:
        backup_queries.append({
            "tool": "search_europe_pmc",
            "params": {"query": simple_name, "max_results": 5},
            "rationale": f"[SAFETY-VALVE] Broad EPMC search: {simple_name}",
        })

    # 2. Wikipedia — almost always returns something for known conditions
    if HAS_WIKIPEDIA:
        backup_queries.append({
            "tool": "search_wikipedia_medical",
            "params": {"disease_name": simple_name},
            "rationale": f"[SAFETY-VALVE] Wikipedia fallback: {simple_name}",
        })

    # 3. PubMed with untagged terms (let ATM auto-map to MeSH)
    if HAS_PUBMED:
        # Use plain terms WITHOUT [MeSH] tags — PubMed ATM will auto-expand
        atm_query = " ".join(core_terms[:2]) if len(core_terms) >= 2 else simple_name
        backup_queries.append({
            "tool": "search_pubmed",
            "params": {"query": atm_query},
            "rationale": f"[SAFETY-VALVE] ATM-friendly PubMed: {atm_query}",
        })

    # 4. Web search for treatment guidelines
    if HAS_WEB_SEARCH:
        backup_queries.append({
            "tool": "web_search",
            "params": {"query": f"{simple_name} treatment guidelines"},
            "rationale": f"[SAFETY-VALVE] Web guidelines: {simple_name}",
        })

    # Execute backup queries
    total = len(backup_queries)
    tasks = [
        _execute_tool_query(ws, q["tool"], q["params"], q["rationale"], i + 1, total)
        for i, q in enumerate(backup_queries)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    new_evidence: list[dict] = []
    for res in results:
        if isinstance(res, dict):
            new_evidence.append(res)

    if new_evidence:
        logger.info(
            "[SAFETY-VALVE] Recovered %d additional evidence pieces", len(new_evidence),
        )
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": f"🛡️ Safety Valve recovered {len(new_evidence)} additional sources",
        })
    else:
        logger.warning("[SAFETY-VALVE] Still no evidence — R2 search failure confirmed")
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": "⚠️ Safety Valve: No additional evidence found. R3 will be informed of search limitations.",
        })

    return existing_evidence + new_evidence


async def _execute_tool_query(
    ws: WebSocket,
    tool_name: str,
    params: dict,
    rationale: str,
    index: int,
    total: int,
) -> dict | None:
    """Execute a single tool query and send WS events. Returns evidence dict or None."""
    fn = _TOOL_DISPATCH.get(tool_name)
    if fn is None:
        logger.warning("[R2] Tool %s not available — skipping", tool_name)
        return None

    # ── Strip params the tool function doesn't accept ──
    # The LLM often injects extra keys (rationale, filters, specific_condition, etc.)
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        extra_keys = set(params.keys()) - accepted
        if extra_keys:
            logger.info("[R2] Stripping unexpected params for %s: %s", tool_name, extra_keys)
            params = {k: v for k, v in params.items() if k in accepted}
    except (ValueError, TypeError):
        pass  # fallback: pass params as-is

    # ── Universal query optimization (synonym expansion + clinical linking) ──
    original_query = params.get("query", params.get("condition", params.get("drug_name", "")))
    params = _optimize_query_for_tool(tool_name, params)
    optimized_query = params.get("query", params.get("condition", params.get("drug_name", "")))

    display_query = optimized_query
    if original_query and optimized_query != original_query:
        display_query = f"{optimized_query}  (← {original_query[:50]})"

    await ws.send_json({
        "type": "api_call",
        "stage": "R2",
        "api": tool_name,
        "query": display_query,
        "index": index,
        "total": total,
    })

    try:
        result = await fn(**params)

        # ── Auto-retry: if a tool returned 0 results, use SMART RETRY ──
        # Strategy: first try BROAD_CATEGORIES (semantic broadening), then
        # fallback to simple word truncation. This is generic — works for ANY condition.

        # PubMed: relax MeSH query, then try broad category
        if tool_name == "search_pubmed" and result.get("total_found", 0) == 0:
            relaxed = _relax_pubmed_query(params["query"])
            if relaxed and relaxed != params["query"]:
                logger.info("[R2-RETRY] PubMed 0 results → relaxing: '%s' → '%s'", params["query"], relaxed)
                params["query"] = relaxed
                display_query = f"{relaxed}  (← relaxed)"
                result = await fn(**params)
            # If still 0, try broad category
            if result.get("total_found", 0) == 0:
                broad = _get_broad_query(original_query)
                if broad and broad != relaxed:
                    logger.info("[R2-RETRY] PubMed still 0 → broad category: '%s'", broad)
                    params["query"] = broad
                    display_query = f"{broad}  (← broad category)"
                    result = await fn(**params)

        # Europe PMC: try broad category first, then simplify
        if tool_name == "search_europe_pmc" and len(result.get("articles", [])) == 0:
            broad = _get_broad_query(original_query)
            if broad:
                logger.info("[R2-RETRY] Europe PMC 0 results → broad: '%s'", broad)
                params["query"] = broad
                display_query = f"{broad}  (← broad category)"
                result = await fn(**params)
            if len(result.get("articles", [])) == 0:
                q = params.get("query", "")
                words = q.split()
                if len(words) > 2:
                    simplified = " ".join(words[:3])
                    logger.info("[R2-RETRY] Europe PMC still 0 → simplifying: '%s' → '%s'", q, simplified)
                    params["query"] = simplified
                    display_query = f"{simplified}  (← simplified)"
                    result = await fn(**params)

        # Semantic Scholar: try broad category first, then simplify
        if tool_name == "search_semantic_scholar" and len(result.get("papers", [])) == 0:
            broad = _get_broad_query(original_query)
            if broad:
                logger.info("[R2-RETRY] S2 0 results → broad: '%s'", broad)
                params["query"] = broad
                display_query = f"{broad}  (← broad category)"
                result = await fn(**params)
            if len(result.get("papers", [])) == 0:
                q = params.get("query", "")
                words = q.split()
                if len(words) > 3:
                    simplified = " ".join(words[:4])
                    logger.info("[R2-RETRY] S2 still 0 → simplifying: '%s' → '%s'", q, simplified)
                    params["query"] = simplified
                    display_query = f"{simplified}  (← simplified)"
                    result = await fn(**params)

        # Wikipedia: try with just the first 2 words if no summary
        if tool_name == "search_wikipedia_medical" and not result.get("summary"):
            dn = params.get("disease_name", "")
            words = dn.split()
            if len(words) > 2:
                simplified = " ".join(words[:2])
                logger.info("[R2-RETRY] Wikipedia no summary → simplifying: '%s' → '%s'", dn, simplified)
                params["disease_name"] = simplified
                display_query = f"{simplified}  (← simplified)"
                result = await fn(**params)

        # Normalise result shape for evidence list
        evidence_entry = {
            "source": tool_name,
            "query": display_query,
            "rationale": rationale,
            "raw_result": result,
        }

        # Build a nice WS summary depending on tool type
        summary: dict = {"type": "api_result", "stage": "R2", "api": tool_name}
        if tool_name == "search_pubmed":
            articles = result.get("articles", [])
            evidence_entry["articles"] = articles
            evidence_entry["count"] = len(articles)
            summary["count"] = len(articles)
            summary["articles"] = [
                {"title": a.get("title", ""), "pmid": a.get("pmid", ""),
                 "has_abstract": bool(a.get("abstract"))}
                for a in articles[:3]
            ]
        elif tool_name == "web_search":
            results_list = result.get("results", [])
            evidence_entry["results"] = results_list
            evidence_entry["count"] = len(results_list)
            evidence_entry["answer"] = result.get("answer", "")
            summary["count"] = len(results_list)
            summary["answer"] = (result.get("answer", "") or "")[:200]
        elif tool_name == "search_clinical_trials":
            studies = result.get("studies", [])
            evidence_entry["studies"] = studies
            evidence_entry["count"] = len(studies)
            summary["count"] = len(studies)
        elif tool_name in ("search_drug_interactions", "get_adverse_events"):
            evidence_entry["count"] = 1
            summary["found"] = bool(result)
        elif tool_name == "get_treatment_guidelines":
            topics = result.get("topics", [])
            evidence_entry["topics"] = topics
            evidence_entry["count"] = len(topics)
            summary["count"] = len(topics)
        elif tool_name == "search_europe_pmc":
            articles = result.get("articles", [])
            evidence_entry["articles"] = articles
            evidence_entry["count"] = len(articles)
            summary["count"] = len(articles)
            summary["articles"] = [
                {"title": a.get("title", ""), "pmid": a.get("pmid", "")}
                for a in articles[:3]
            ]
        elif tool_name == "search_wikipedia_medical":
            evidence_entry["summary"] = result.get("summary", "")
            evidence_entry["url"] = result.get("url", "")
            evidence_entry["count"] = 1 if result.get("summary") else 0
            summary["found"] = bool(result.get("summary"))
            summary["title"] = result.get("title", "")
        elif tool_name == "search_semantic_scholar":
            papers = result.get("papers", [])
            evidence_entry["papers"] = papers
            evidence_entry["count"] = len(papers)
            summary["count"] = len(papers)
            summary["papers"] = [
                {"title": p.get("title", ""), "citations": p.get("citation_count", 0)}
                for p in papers[:3]
            ]
        else:
            evidence_entry["count"] = 1

        summary["query"] = display_query
        await ws.send_json(summary)
        return evidence_entry

    except Exception as exc:
        logger.error("[R2] Tool %s failed: %s", tool_name, exc)
        await ws.send_json({
            "type": "api_error",
            "stage": "R2",
            "api": tool_name,
            "error": str(exc),
        })
        return None


async def _generate_research_plan(
    llm_client: Any,
    llama_server_url: str,
    groq_client: Any,
    patient_text: str,
    r1_json: dict,
    is_fast: bool,
    local_only: bool,
    clinical_translation_brief: str = "",
    super_thinking: bool = False,
    gemini_client: Any = None,
    gemini_use_pro: bool = False,
) -> list[dict]:
    """Call the LLM to produce a dynamic research plan (list of tool queries)."""
    max_queries = 4 if is_fast else 8

    # Build context strings for the prompt
    diagnoses = r1_json.get("differential_diagnoses", [])
    gaps = r1_json.get("knowledge_gaps", [])

    # ── Rich R1 summary: include confidence, supporting factors, key symptoms
    # This gives R2 enough context to generate ANCHORED queries, not random ones.
    r1_summary_parts = []
    for dx in diagnoses[:5]:
        name = dx.get("diagnosis", "Unknown")
        conf = dx.get("confidence", dx.get("probability", "?"))
        rank = dx.get("rank", "?")
        supporting = dx.get("supporting_factors", [])
        supporting_str = "; ".join(supporting[:3]) if supporting else "not specified"
        r1_summary_parts.append(
            f"- [Rank {rank}, confidence: {conf}] {name}\n"
            f"  Supporting: {supporting_str}"
        )
    r1_summary = "\n".join(r1_summary_parts) if r1_summary_parts else "R1 did not produce structured diagnoses."

    # Inject key positives from patient summary — gives R2 the actual symptoms to search for
    patient_summary = r1_json.get("patient_summary", {})
    key_positives = patient_summary.get("key_positives", [])
    if key_positives:
        r1_summary += "\n\nKey clinical findings from R1: " + "; ".join(key_positives[:8])

    gaps_parts = []
    for g in gaps:
        gaps_parts.append(f"- {g.get('gap', g.get('search_query', str(g)))}")
    gaps_text = "\n".join(gaps_parts) if gaps_parts else "No explicit gaps identified by R1."

    # Use .replace() instead of .format() — the prompts contain literal JSON
    # braces that .format() misinterprets as placeholders.
    system_msg = R2_QUERY_GENERATION_PROMPT.replace("{max_queries}", str(max_queries))
    user_msg = (
        R2_QUERY_USER_TEMPLATE
        .replace("{patient_text}", patient_text[:2000])
        .replace("{r1_summary}", r1_summary)
        .replace("{gaps_text}", gaps_text)
        .replace("{max_queries}", str(max_queries))
    )

    # Inject clinical translation brief (from non-English text translation step)
    if clinical_translation_brief:
        user_msg += (
            "\n\n## ⚠️ CLINICAL TRANSLATION (patient text was in non-English language)\n"
            "The following medical terms were translated from the original language.\n"
            "You MUST use the ENGLISH equivalents in your search queries, NOT the original terms.\n\n"
            + clinical_translation_brief
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Choose LLM for R2 query generation
    # - Gemini mode: use Gemini Flash/Pro (highest priority when available)
    # - Super Thinking (no Gemini): use Groq 70B
    # - Non-English: prefer Groq 70B (better at translation + medical jargon)
    # - Default: prefer local (fast & cheap) for English-only cases
    raw_response = ""
    needs_translation = bool(clinical_translation_brief)
    r2_gen_tokens = 768 if (llm_client and llama_server_url) else 1024

    if gemini_client and getattr(gemini_client, 'is_available', False) and super_thinking:
        model_label = "Gemini Pro" if gemini_use_pro else "Gemini Flash"
        logger.info("[R2] Using %s for query generation", model_label)
        raw_response = await call_gemini_no_stream(gemini_client, messages, max_tokens=1024, use_pro=gemini_use_pro)
    elif super_thinking and groq_client and getattr(groq_client, 'is_available', False):
        # Super Thinking: always use 70B for highest-quality research plans
        logger.info("[R2] Using Groq 70B for query generation (Super Thinking mode)")
        raw_response = await call_groq_no_stream(groq_client, messages, max_tokens=1024)
    elif needs_translation and groq_client and getattr(groq_client, 'is_available', False) and not local_only:
        # Non-English input → use 70B (better at multilingual + medical jargon)
        logger.info("[R2] Using Groq 70B for query generation (non-English input detected)")
        raw_response = await call_groq_no_stream(groq_client, messages, max_tokens=1024)
    elif llm_client and llama_server_url:
        raw_response = await call_llm_no_stream(llm_client, llama_server_url, messages, max_tokens=r2_gen_tokens)
    elif groq_client and not local_only:
        raw_response = await call_groq_no_stream(groq_client, messages, max_tokens=r2_gen_tokens)

    if not raw_response:
        logger.warning("[R2] LLM query generation returned empty")
        return []

    logger.info("[R2] Raw LLM response (%d chars): %.300s", len(raw_response), raw_response)

    # Parse the JSON array from response
    plan = _parse_research_plan(raw_response)
    logger.info("[R2] Generated research plan with %d queries (requested %d)", len(plan), max_queries)

    # ── R2 QUERY VALIDATION & ANCHORING ──
    # Ensure queries are grounded in R1's top diagnoses, not hallucinated topics.
    plan = _validate_and_anchor_queries(plan, diagnoses, gaps, max_queries)

    return plan[:max_queries]


def _validate_and_anchor_queries(
    plan: list[dict],
    diagnoses: list[dict],
    gaps: list[dict],
    max_queries: int,
) -> list[dict]:
    """Validate R2 queries are anchored to R1's diagnoses. Inject missing queries.

    Self-correction mechanism:
    1. Extract R1's top diagnosis names
    2. Check how many queries mention at least one R1 diagnosis
    3. If < 50% anchored → force-inject queries for top 2 diagnoses
    4. Log warnings for off-topic queries

    This prevents the R2 LLM from going rogue (e.g., inventing "cyanide poisoning"
    when R1 said "methemoglobinemia").
    """
    if not diagnoses:
        return plan

    # Extract top diagnosis names (lowercase for matching)
    top_dx_names: list[str] = []
    for dx in diagnoses[:3]:
        name = dx.get("diagnosis", "").lower().strip()
        if name:
            top_dx_names.append(name)

    if not top_dx_names:
        return plan

    # Build keyword sets from diagnosis names (split multi-word names)
    dx_keywords: set[str] = set()
    for name in top_dx_names:
        # Add full name and individual significant words
        dx_keywords.add(name)
        for word in name.split():
            if len(word) > 3:  # skip "and", "of", etc.
                dx_keywords.add(word)

    # Also add synonyms from MEDICAL_SYNONYMS values that match
    for syn_key, syn_val in MEDICAL_SYNONYMS.items():
        if syn_val.lower() in top_dx_names or syn_key in dx_keywords:
            dx_keywords.add(syn_key)
            for word in syn_val.lower().split():
                if len(word) > 3:
                    dx_keywords.add(word)

    # Check each query for anchoring
    anchored_count = 0
    for q in plan:
        params = q.get("params", {})
        query_text = " ".join(str(v) for v in params.values()).lower()
        rationale = q.get("rationale", "").lower()
        combined = query_text + " " + rationale

        is_anchored = any(kw in combined for kw in dx_keywords)
        if is_anchored:
            anchored_count += 1
        else:
            logger.warning(
                "[R2-VALIDATE] Off-topic query detected: tool=%s query='%s' — "
                "not related to R1 diagnoses: %s",
                q.get("tool", "?"),
                query_text[:80],
                ", ".join(top_dx_names[:3]),
            )

    anchor_ratio = anchored_count / len(plan) if plan else 0
    logger.info(
        "[R2-VALIDATE] Anchoring: %d/%d queries (%.0f%%) mention R1 diagnoses",
        anchored_count, len(plan), anchor_ratio * 100,
    )

    # If less than 50% anchored, force-inject queries for top diagnoses
    if anchor_ratio < 0.5 and top_dx_names:
        injected = []
        for dx_name in top_dx_names[:2]:
            # Check if we already have a PubMed query for this diagnosis
            has_pubmed = any(
                q.get("tool") == "search_pubmed" and dx_name in
                " ".join(str(v) for v in q.get("params", {}).values()).lower()
                for q in plan
            )
            if not has_pubmed:
                injected.append({
                    "tool": "search_pubmed",
                    "params": {"query": dx_name},
                    "rationale": f"[AUTO-INJECTED] R2 queries were off-topic. Searching for R1's diagnosis: {dx_name}",
                })
            # Also inject Wikipedia for the top diagnosis
            has_wiki = any(q.get("tool") == "search_wikipedia_medical" for q in plan)
            if not has_wiki:
                injected.append({
                    "tool": "search_wikipedia_medical",
                    "params": {"disease_name": dx_name},
                    "rationale": f"[AUTO-INJECTED] Wikipedia overview for R1's top diagnosis",
                })
            # Add Europe PMC for broader search
            has_epmc = any(
                q.get("tool") == "search_europe_pmc" and dx_name in
                " ".join(str(v) for v in q.get("params", {}).values()).lower()
                for q in plan
            )
            if not has_epmc:
                injected.append({
                    "tool": "search_europe_pmc",
                    "params": {"query": dx_name, "max_results": 5},
                    "rationale": f"[AUTO-INJECTED] Europe PMC evidence for R1's diagnosis: {dx_name}",
                })

        if injected:
            logger.warning(
                "[R2-VALIDATE] Anchor ratio too low (%.0f%%). Injecting %d forced queries "
                "for R1 diagnoses: %s",
                anchor_ratio * 100, len(injected), ", ".join(top_dx_names[:2]),
            )
            # Place injected queries at the BEGINNING (highest priority)
            plan = injected + plan

    return plan


def _parse_research_plan(text: str) -> list[dict]:
    """Parse the LLM's research plan JSON array from response text."""
    import re as _re
    text = text.strip()

    # Remove thinking blocks that may have leaked through
    text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()

    # Remove markdown fences
    if "```" in text:
        lines = text.split("\n")
        filtered = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(filtered)

    # Try to find a JSON array
    # Strategy 1: direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [q for q in parsed if isinstance(q, dict) and "tool" in q]
        if isinstance(parsed, dict) and "research_plan" in parsed:
            return [q for q in parsed["research_plan"] if isinstance(q, dict) and "tool" in q]
    except json.JSONDecodeError:
        pass

    # Strategy 2: find [ ... ] block
    bracket_match = _re.search(r'\[.*\]', text, _re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group())
            if isinstance(parsed, list):
                return [q for q in parsed if isinstance(q, dict) and "tool" in q]
        except json.JSONDecodeError:
            pass

    # Strategy 3: find individual { ... } blocks (handles nested params)
    # Use a simple brace-depth counter to extract top-level objects
    result = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                obj_str = text[start:i+1]
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and "tool" in obj:
                        result.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    if result:
        logger.info("[R2] Parsed %d queries via Strategy 3 (brace matching)", len(result))
    return result


async def run_r2(
    ws: WebSocket,
    r1_json: dict,
    *,
    is_fast: bool,
    patient_text: str = "",
    drug_facts: str = "",
    llm_client: Any = None,
    llama_server_url: str = "",
    groq_client: Any = None,
    gemini_client: Any = None,
    gemini_use_pro: bool = False,
    local_only: bool = False,
    super_thinking: bool = False,
) -> tuple[list[dict], dict]:
    """Run R2 stage: LLM-driven dynamic multi-tool evidence gathering.

    Flow:
    1. Call Qwen 3.5 (or Groq) to generate a research plan
    2. Execute all queries in parallel across available tools
    3. ICD-11 code resolution
    4. Fallback to static PubMed search if LLM fails

    Returns:
        (r2_evidence, icd_resolved)
    """
    _init_tool_dispatch()

    await ws.send_json({
        "type": "stage_start",
        "stage": "R2",
        "title": "R2 — Reinforced Fact-Finding",
        "description": "LLM generating dynamic research queries across multiple medical APIs...",
    })

    r2_evidence: list[dict] = []
    diagnoses = r1_json.get("differential_diagnoses", [])

    # ── Step 0: Extract R1 primary confidence for downstream decisions ──
    r1_primary_confidence = 0.0
    if diagnoses:
        r1_primary_confidence = float(diagnoses[0].get("confidence", 0))

    # ══════════════════════════════════════════════════════════════
    # Step 0.5: CLINICAL TERM TRANSLATION (non-English → English MeSH)
    # ══════════════════════════════════════════════════════════════
    # If patient text or R1 diagnoses are in Turkish/non-English,
    # translate clinical terms to English medical jargon BEFORE generating
    # search queries. This prevents "şurup" → "urup" query disasters.
    clinical_translation = {"translated_diagnoses": [], "clinical_brief": "", "english_dx_names": []}
    if _has_non_latin_medical_text(patient_text) or any(
        _has_non_latin_medical_text(d.get("diagnosis", "")) for d in diagnoses[:5]
    ):
        try:
            clinical_translation = await _translate_clinical_context_for_r2(
                diagnoses=diagnoses,
                patient_text=patient_text,
                groq_client=groq_client,
                llm_client=llm_client,
                llama_server_url=llama_server_url,
                local_only=local_only,
            )
            if clinical_translation.get("clinical_brief"):
                await ws.send_json({
                    "type": "info",
                    "stage": "R2",
                    "content": (
                        f"🌐 Clinical term translation: "
                        f"{len(clinical_translation.get('translated_diagnoses', []))} terms translated, "
                        f"{len(clinical_translation.get('english_dx_names', []))} English diagnoses extracted"
                    ),
                })
                logger.info(
                    "[R2-TRANSLATE] Translation results: %s",
                    clinical_translation.get("clinical_brief", "")[:200],
                )
        except Exception as exc:
            logger.warning("[R2-TRANSLATE] Translation step failed (non-fatal): %s", exc)

    # ── Step 1A: Programmatic queries — LLM-independent, guaranteed quality ──
    programmatic_plan: list[dict] = []
    if diagnoses:
        # For non-English diagnoses, also generate programmatic queries for translated names
        english_dx_names = clinical_translation.get("english_dx_names", [])
        if english_dx_names:
            # Create synthetic diagnosis entries with translated names for programmatic generation
            translated_diagnoses = []
            for eng_name in english_dx_names[:3]:
                translated_diagnoses.append({
                    "diagnosis": eng_name,
                    "confidence": 0.8,
                    "rank": len(translated_diagnoses) + 1,
                })
            programmatic_plan = _generate_programmatic_queries(
                diagnoses + translated_diagnoses, patient_text,
            )
        else:
            programmatic_plan = _generate_programmatic_queries(diagnoses, patient_text)
        if programmatic_plan:
            await ws.send_json({
                "type": "info",
                "stage": "R2",
                "content": f"🧬 Generated {len(programmatic_plan)} programmatic queries from R1 diagnoses",
            })

    # ── Step 1A-bis: Suspicion-Driven Queries (from R1 clinical_suspicion) ──
    suspicion = r1_json.get("clinical_suspicion", {})
    suspicion_queries: list[dict] = []
    if suspicion:
        env_triggers = suspicion.get("environmental_triggers", [])
        for trigger in env_triggers[:2]:  # max 2 environmental queries
            if isinstance(trigger, str) and len(trigger) > 5:
                suspicion_queries.append({
                    "tool": "web_search",
                    "params": {"query": f"{trigger} medical condition pathophysiology"},
                    "rationale": f"[SUSPICION] Environmental trigger investigation: {trigger}",
                })

        dangerous_alts = suspicion.get("dangerous_alternatives", [])
        for alt in dangerous_alts[:2]:  # max 2 dangerous alternative queries
            alt_dx = alt.get("diagnosis", "") if isinstance(alt, dict) else str(alt)
            if alt_dx and len(alt_dx) > 3:
                suspicion_queries.append({
                    "tool": "search_pubmed",
                    "params": {"query": _optimize_pubmed_query(alt_dx)},
                    "rationale": f"[SUSPICION] Dangerous alternative: {alt_dx}",
                })
                suspicion_queries.append({
                    "tool": "search_wikipedia_medical",
                    "params": {"disease_name": alt_dx.split("(")[0].strip()},
                    "rationale": f"[SUSPICION] Wikipedia overview: {alt_dx}",
                })

        if suspicion_queries:
            await ws.send_json({
                "type": "info",
                "stage": "R2",
                "content": f"🔍 Generated {len(suspicion_queries)} suspicion-driven queries from R1 self-critique",
            })
            logger.info("[R2-SUSPICION] Generated %d queries from clinical_suspicion field", len(suspicion_queries))

    # Enrich patient_text with drug facts for R2 query generation context
    r2_patient_text = patient_text
    if drug_facts:
        r2_patient_text = patient_text + "\n\n" + drug_facts

    # ── Step 1B: LLM generates research plan ──
    research_plan: list[dict] = []
    if llm_client or (groq_client and not local_only):
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": "🔬 Generating dynamic research queries with AI...",
        })
        research_plan = await _generate_research_plan(
            llm_client=llm_client,
            llama_server_url=llama_server_url,
            groq_client=groq_client,
            patient_text=r2_patient_text,
            r1_json=r1_json,
            is_fast=is_fast,
            local_only=local_only,
            clinical_translation_brief=clinical_translation.get("clinical_brief", ""),
            super_thinking=super_thinking,
            gemini_client=gemini_client,
            gemini_use_pro=gemini_use_pro,
        )
        if research_plan:
            tool_names = set(q.get('tool', 'unknown') for q in research_plan)
            # ── Detailed logging: show R1 anchoring and query targets ──
            auto_injected = sum(1 for q in research_plan if "[AUTO-INJECTED]" in q.get("rationale", ""))
            dx_names = [d.get("diagnosis", "?") for d in diagnoses[:3]]
            plan_summary = f"✓ LLM plan: {len(research_plan)} queries across {len(tool_names)} tools"
            if auto_injected:
                plan_summary += f" ({auto_injected} auto-injected for R1 diagnoses)"
            await ws.send_json({
                "type": "info",
                "stage": "R2",
                "content": plan_summary,
            })

            # ── Quality Gate: filter/fix bad LLM queries ──
            pre_gate = len(research_plan)
            research_plan = _quality_gate(research_plan, diagnoses)
            if len(research_plan) < pre_gate:
                await ws.send_json({
                    "type": "info",
                    "stage": "R2",
                    "content": f"🔍 Quality Gate: {pre_gate} → {len(research_plan)} queries (filtered {pre_gate - len(research_plan)} bad queries)",
                })

            # Show R1 diagnoses being targeted
            if dx_names:
                await ws.send_json({
                    "type": "info",
                    "stage": "R2",
                    "content": f"🎯 R1 target diagnoses: {' | '.join(dx_names)}",
                })

    # ── Step 2: Merge programmatic + LLM plans, deduplicate ──
    combined_plan: list[dict] = []
    seen_dedup: set[str] = set()

    def _dedup_key(q: dict) -> str:
        tool = q.get("tool", "")
        params = q.get("params", {})
        query_text = str(params.get("query", params.get("disease_name", ""))).lower().strip()
        return f"{tool}|{query_text}"

    # Programmatic queries go first (higher quality, guaranteed)
    for q in programmatic_plan:
        key = _dedup_key(q)
        if key not in seen_dedup:
            seen_dedup.add(key)
            combined_plan.append(q)

    # Suspicion-driven queries second (from R1 clinical self-critique)
    for q in suspicion_queries:
        key = _dedup_key(q)
        if key not in seen_dedup:
            seen_dedup.add(key)
            combined_plan.append(q)

    # Then LLM queries (may overlap)
    for q in research_plan:
        key = _dedup_key(q)
        if key not in seen_dedup:
            seen_dedup.add(key)
            combined_plan.append(q)

    dedup_removed = (len(programmatic_plan) + len(suspicion_queries) + len(research_plan)) - len(combined_plan)
    if dedup_removed > 0:
        logger.info("[R2-MERGE] Removed %d duplicate queries", dedup_removed)

    total_plan = len(combined_plan)
    if total_plan > 0:
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": f"📋 Combined plan: {total_plan} queries ({len(programmatic_plan)} programmatic + {len(suspicion_queries)} suspicion + {len(research_plan)} LLM, {dedup_removed} deduped)",
        })

    # Log each query for visibility
    for i, q in enumerate(combined_plan):
        tool = q.get("tool", "?")
        params = q.get("params", {})
        query_text = params.get("query", params.get("disease_name", params.get("condition", params.get("drug_name", "?"))))
        rationale = q.get("rationale", "")
        logger.info(
            "[R2-PLAN] Query %d/%d: %s → '%s' (%s)",
            i + 1, total_plan, tool, str(query_text)[:60], rationale[:50],
        )

    # ── Step 3: Execute all queries in parallel ──
    if combined_plan:
        total = len(combined_plan)
        tasks = []
        for i, query_spec in enumerate(combined_plan):
            tool_name = query_spec.get("tool", "")
            params = query_spec.get("params", {})
            rationale = query_spec.get("rationale", "")
            tasks.append(
                _execute_tool_query(ws, tool_name, params, rationale, i + 1, total)
            )
        # Run all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, dict):
                r2_evidence.append(res)
            elif isinstance(res, Exception):
                logger.error("[R2] Parallel task exception: %s", res)

    # ── Step 3.5: Safety Valve — if insufficient evidence + high R1 confidence ──
    r2_evidence = await _safety_valve_search(
        ws, diagnoses, r1_primary_confidence, r2_evidence,
    )

    # ── Step 4: Fallback — if LLM plan failed or produced nothing, use static search ──
    if not r2_evidence:
        logger.warning("[R2] No evidence from dynamic plan — falling back to static search")
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": "⚠ Falling back to static evidence search...",
        })
        knowledge_gaps = r1_json.get("knowledge_gaps", [])

        # Build fallback queries from knowledge_gaps or patient text
        if not knowledge_gaps and patient_text:
            fallback_query = _extract_search_query(patient_text)
            if fallback_query:
                knowledge_gaps = [{"search_query": fallback_query, "gap": "R1 incomplete — fallback"}]

        if HAS_PUBMED and knowledge_gaps:
            max_searches = 1 if is_fast else 3
            for i, gap in enumerate(knowledge_gaps[:max_searches]):
                query = gap.get("search_query", gap.get("gap", ""))
                if not query:
                    continue
                await ws.send_json({
                    "type": "api_call",
                    "stage": "R2",
                    "api": "PubMed",
                    "query": query,
                    "index": i + 1,
                    "total": min(len(knowledge_gaps), max_searches),
                })
                try:
                    result = await search_pubmed(query, max_results=3)
                    articles = result.get("articles", [])
                    r2_evidence.append({
                        "source": "pubmed",
                        "query": query,
                        "articles": articles,
                        "count": len(articles),
                    })
                    await ws.send_json({
                        "type": "api_result",
                        "stage": "R2",
                        "api": "PubMed",
                        "query": query,
                        "count": len(articles),
                        "articles": [
                            {"title": a.get("title", ""), "pmid": a.get("pmid", "")}
                            for a in articles[:3]
                        ],
                    })
                except Exception as exc:
                    await ws.send_json({
                        "type": "api_error",
                        "stage": "R2",
                        "api": "PubMed",
                        "error": str(exc),
                    })

    # ── Step 4: ICD-11 code resolution (always runs) ──
    icd_resolved: dict[str, list] = {}
    if HAS_ICD11 and diagnoses:
        max_icd = 1 if is_fast else 3
        for dx in diagnoses[:max_icd]:
            name = dx.get("diagnosis", "")
            if not name:
                continue
            await ws.send_json({
                "type": "api_call",
                "stage": "R2",
                "api": "WHO ICD-11",
                "query": f"Resolving ICD-11 for: {name}",
            })
            try:
                search_results = await search_icd11_who(name, max_results=3)
                icd_resolved[name] = search_results
                top_code = search_results[0]["theCode"] if search_results else "—"
                top_title = search_results[0]["title"] if search_results else "No match"
                r2_evidence.append({
                    "source": "icd11",
                    "diagnosis": name,
                    "resolved_code": top_code,
                    "resolved_title": top_title,
                    "result": search_results,
                })
                await ws.send_json({
                    "type": "api_result",
                    "stage": "R2",
                    "api": "WHO ICD-11",
                    "code": top_code,
                    "found": bool(search_results),
                    "title": top_title,
                    "diagnosis": name,
                })
            except Exception as exc:
                await ws.send_json({
                    "type": "api_error",
                    "stage": "R2",
                    "api": "WHO ICD-11",
                    "error": str(exc),
                })

    # Inject real ICD-11 codes into R1 diagnoses
    if icd_resolved:
        resolve_icd_codes(diagnoses, icd_resolved)

    # ── Evidence breakdown summary ──
    if r2_evidence:
        source_counts: dict[str, int] = {}
        total_articles = 0
        for ev in r2_evidence:
            src = ev.get("source", "unknown")
            # Normalize source name for display
            src_display = src.replace("search_", "").replace("_", " ").title()
            source_counts[src_display] = source_counts.get(src_display, 0) + 1
            total_articles += len(ev.get("articles", []))
            total_articles += len(ev.get("papers", []))
            total_articles += len(ev.get("results", []))
        breakdown = " | ".join(f"{k}: {v}" for k, v in sorted(source_counts.items()))
        summary_msg = f"📊 R2 Evidence: {len(r2_evidence)} sources, ~{total_articles} articles"
        if breakdown:
            summary_msg += f" [{breakdown}]"
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": summary_msg,
        })
        logger.info("[R2-SUMMARY] %s", summary_msg)

    await ws.send_json({
        "type": "stage_result",
        "stage": "R2",
        "data": {"evidence_count": len(r2_evidence), "evidence": r2_evidence},
    })
    await ws.send_json({"type": "stage_complete", "stage": "R2"})

    return r2_evidence, icd_resolved


def _extract_search_query(patient_text: str) -> str:
    """Extract key clinical terms from patient text for a fallback PubMed search."""
    import re as _re
    signal_patterns = [
        r'kavit[eé]r', r'nod[üu]l', r'hemoptizi', r'hemoptysis',
        r'tromb(?:oz|ophlebitis|us)', r'embol[i]', r'septik|sepsis|septicemia',
        r'ateş|fever', r'hipotansi[fv]|hypotens', r'taşikardi|tachycardia',
        r'infiltras(?:yon)?|infiltrat', r'apse|abscess', r'farenjit|pharyngitis',
        r'menenjit|meningit', r'pnömoni|pneumonia', r'jugular',
        r'sternokleidomastoid|sternocleidomastoid',
    ]
    matched_terms = []
    text_lower = patient_text.lower()
    for pat in signal_patterns:
        m = _re.search(pat, text_lower)
        if m:
            matched_terms.append(m.group())
    if not matched_terms:
        words = [w for w in text_lower.split() if len(w) > 4]
        return ' '.join(words[:5])
    return ' '.join(matched_terms[:5])
