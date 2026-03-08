"""RRRIE pipeline orchestrator — full iterative R3↔IE loop + memory.

Coordinates stages, streaming, memory retrieval/storage, and produces
the final summary sent to the WebSocket client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from fastapi import WebSocket

from src.llm.prompt_templates import (
    R3_SYSTEM_PROMPT,
    R3_DIAGNOSIS_SYSTEM_PROMPT,
    R3_TREATMENT_SYSTEM_PROMPT,
    IE_SYSTEM_PROMPT,
    FAST_MODE_INSTRUCTION,
    ITERATION_FEEDBACK_TEMPLATE,
    PERSPECTIVE_SHIFT_PROMPTS,
    MEMORY_PREAMBLE,
    adapt_prompt_for_complexity,
)
from src.llm.stage_adapter import simplify_for_ie, resolve_icd_codes
from src.memory.case_store import CaseStore, MemoryContext

from src.pipeline.streaming import (
    stream_groq_completion,
    stream_gemini_completion,
    stream_llm_completion,
    parse_json_from_response,
)
from src.pipeline.stages import run_safety, run_r1, run_r2, annotate_clinical_text_pre_r1
from src.pipeline.token_budget import TokenBudgetManager
from src.pipeline.treatment_safety import (
    IEConsensusTracker,
    build_differential_safety_block,
    validate_treatment_safety,
)
from src.pipeline.dllm_r0 import DLLMR0, R0Result
from src.pipeline.router import route as route_pipeline, PipelineConfig
from src.pipeline.iteration_ctrl import IterationController

from src.pipeline.ie_layers import run_layered_ie

from config.settings import get_settings

logger = logging.getLogger("rrrie-cdss")


# ═══════════════════════════════════════════════════════════════════
# Semantic Diagnosis Similarity — for stagnation detection
# ═══════════════════════════════════════════════════════════════════

# Common medical abbreviation → expansion mapping
_DX_ABBREVIATIONS: dict[str, str] = {
    "ami": "acute myocardial infarction",
    "mi": "myocardial infarction",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "dka": "diabetic ketoacidosis",
    "uti": "urinary tract infection",
    "copd": "chronic obstructive pulmonary disease",
    "acs": "acute coronary syndrome",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "chf": "congestive heart failure",
    "hf": "heart failure",
    "ards": "acute respiratory distress syndrome",
    "aki": "acute kidney injury",
    "ckd": "chronic kidney disease",
    "sle": "systemic lupus erythematosus",
    "gca": "giant cell arteritis",
    "ie": "infective endocarditis",
    "tb": "tuberculosis",
    "scd": "sickle cell disease",
    "t1dm": "type 1 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
}

# Filler words to strip for word overlap
_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "in", "with", "and", "or", "due", "to",
    "by", "on", "for", "from", "type", "stage", "grade", "syndrome",
    "disease", "disorder", "condition", "acute", "chronic", "severe",
})


def _diagnoses_are_similar(dx_a: str, dx_b: str, threshold: float = 0.55) -> bool:
    """Check if two diagnosis names are semantically similar using word overlap.

    Handles abbreviations (AMI ↔ Acute Myocardial Infarction),
    synonyms (Heart Attack ↔ MI), and minor wording differences.
    Returns True if Jaccard similarity of content words ≥ threshold.
    """
    if not dx_a or not dx_b:
        return False

    def _normalize(dx: str) -> set[str]:
        dx_lower = dx.lower().strip()
        # Expand known abbreviations
        expanded = _DX_ABBREVIATIONS.get(dx_lower, dx_lower)
        words = set(re.split(r"[\s\-/,()]+", expanded))
        words -= _STOP_WORDS
        words.discard("")
        return words

    words_a = _normalize(dx_a)
    words_b = _normalize(dx_b)

    if not words_a or not words_b:
        return dx_a.lower().strip() == dx_b.lower().strip()

    # Exact match shortcut
    if words_a == words_b:
        return True

    intersection = words_a & words_b
    union = words_a | words_b
    jaccard = len(intersection) / len(union) if union else 0.0

    return jaccard >= threshold


# ═══════════════════════════════════════════════════════════════════
# IE Override — Extract IE's Diagnosis Suggestion
# ═══════════════════════════════════════════════════════════════════

# Garbage phrases that indicate a non-diagnosis string (action, description, or quality assessment)
_NON_DX_PHRASES = frozenset({
    "well-supported", "high confidence", "low confidence", "needs further",
    "requires additional", "no issues", "well supported", "not applicable",
    "no critical", "no hallucination", "finalize", "iterate",
    "likely correct", "consider evaluating", "needs evaluation",
    "re-evaluation", "differential", "framework", "protocol",
    "comprehensive", "multisystem", "concurrent", "evaluate for",
    "alternative framework", "organ-system", "systematic review",
})


def _is_valid_diagnosis_name(candidate: str) -> bool:
    """Check if a string looks like a real medical diagnosis name.

    Rejects action phrases, quality descriptions, and framework suggestions
    that IE sometimes outputs instead of actual diagnosis names.
    """
    if not candidate or len(candidate) < 3:
        return False
    words = candidate.split()
    if len(words) > 12:
        return False
    candidate_lower = candidate.lower()
    if any(phrase in candidate_lower for phrase in _NON_DX_PHRASES):
        return False
    # Must contain at least one uppercase letter (proper medical term)
    if not re.search(r'[A-Z]', candidate):
        return False
    return True


def _extract_ie_diagnosis_suggestion(ie_json: dict) -> str | None:
    """Extract a specific diagnosis suggestion from IE's output.

    Checks (in order of priority):
    1. Explicit "suggested_diagnosis" field (new IE output format)
    2. Pattern matching in IE's "reasoning" text for diagnosis mentions
    3. Pattern matching in IE's "issues" for wrong_diagnosis type

    Returns a diagnosis name string, or None if IE didn't suggest anything.
    """
    import re

    # Priority 1: Explicit field
    suggested = ie_json.get("suggested_diagnosis")
    if suggested and isinstance(suggested, str) and suggested.strip().lower() not in (
        "null", "none", "", "unknown", "n/a",
    ):
        candidate = suggested.strip()
        if _is_valid_diagnosis_name(candidate):
            return candidate

    # Priority 2: Parse reasoning text for diagnosis mentions
    reasoning = ie_json.get("reasoning", "")
    if reasoning:
        # Patterns like "this is actually X", "correct diagnosis is X",
        # "should be X", "suggests X", "consider X"
        # NOTE: all patterns use [A-Za-z] (not [A-Z]) to handle lowercase model outputs
        patterns = [
            r"(?:correct|actual|true|real|proper|likely|primary)\s+diagnosis\s+(?:is|should\s+be|appears?\s+to\s+be|seems?\s+to\s+be|would\s+be)\s+(?:actually\s+)?['\"]?([A-Za-z][a-zA-Z\s\-]{3,40}?)['\"]?(?:\.|,|$)",
            # Known toxic substances — high priority, precise match
            r"((?:methanol|ethylene\s+glycol|diethylene\s+glycol|toxic\s+alcohol)\s+(?:toxicity|poisoning|ingestion))",
            # "this is/looks like/consistent with/more consistent with X poisoning/syndrome/etc."
            r"(?:this\s+(?:is|case\s+is)|this\s+looks?\s+like|(?:more\s+)?consistent\s+with|strongly\s+suggests?)\s+(?:actually\s+)?['\"]?([A-Za-z][a-zA-Z\s\-]{3,40}?)['\"]?\s+(?:toxicity|poisoning|syndrome|disease|disorder|deficiency|infection|encephalopathy)",
            r"re-?evaluation\s+(?:of\s+the\s+case\s+)?as\s+(?:a\s+)?['\"]?([A-Za-z][a-zA-Z\s\-]{3,40}?)(?:\s+toxicity|\s+poisoning)?['\"]?(?:\.|,|\s|$)",
            # "should consider X", "points to X", "more likely X"
            r"(?:should\s+consider|points?\s+(?:to|toward)|more\s+likely)\s+['\"]?([A-Za-z][a-zA-Z\s\-]{3,40}?)['\"]?(?:\s+(?:as\s+the|rather|instead)|\.|,|$)",
            # "diagnosis: X" or "diagnosis — X" (structured output)
            r"(?:suggested|alternative|revised|proposed)\s+diagnosis[\s:—\-]+['\"]?([A-Za-z][a-zA-Z\s\-]{3,40}?)['\"]?(?:\.|,|$)",
        ]
        for pat in patterns:
            m = re.search(pat, reasoning, re.IGNORECASE)
            if m:
                # Return the captured group (or full match if no group)
                suggestion = m.group(1) if m.lastindex else m.group(0)
                suggestion = suggestion.strip().rstrip(".,;:")
                # Strip trailing common English filler words
                suggestion = re.sub(
                    r"\s+(?:instead|rather|perhaps|also|too|but|and|or|as\s+well)$",
                    "", suggestion, flags=re.IGNORECASE,
                )
                if len(suggestion) > 3 and _is_valid_diagnosis_name(suggestion):
                    return suggestion

    # Priority 3: Check issues for wrong_diagnosis type with alternative
    for issue in ie_json.get("issues", []):
        detail = issue.get("detail", "")
        issue_type = issue.get("type", "")
        if issue_type in ("wrong_diagnosis", "incomplete_diagnosis"):
            # Try to extract the suggested alternative from the detail text
            for pat in [
                r"(?:should\s+be\s+(?:evaluated|re-?evaluated|considered)\s+as|should\s+be|consider|evaluate|re-evaluate\s+as)\s+['\"]?([A-Za-z][a-zA-Z\s\-]{3,40}?)['\"]?(?:\s+as\s|\.\s|\.|,|;|$)",
                r"((?:methanol|ethylene\s+glycol|diethylene\s+glycol|toxic\s+alcohol)[\s\w]*(?:toxicity|poisoning|ingestion))",
                r"(?:likely|actual|correct)\s+diagnosis\s+(?:is|being)\s+['\"]?([A-Za-z][a-zA-Z\s\-]{3,40}?)['\"]?(?:\.|,|;|$)",
            ]:
                m = re.search(pat, detail, re.IGNORECASE)
                if m:
                    suggestion = m.group(1) if m.lastindex else m.group(0)
                    suggestion = suggestion.strip().rstrip(".,;:")
                    suggestion = re.sub(
                        r"\s+(?:instead|rather|perhaps|also|too|but|and|or|as\s+well)$",
                        "", suggestion, flags=re.IGNORECASE,
                    )
                    if _is_valid_diagnosis_name(suggestion):
                        return suggestion

    return None


IE_OVERRIDE_DIRECTIVE = """
## 🚨 IE OVERRIDE — MANDATORY DIAGNOSIS RECONSIDERATION

The quality checker (IE) has suggested that the correct diagnosis may be:
### → {ie_suggested_diagnosis}

This suggestion came from the IE evaluator after analyzing symptoms, evidence,
and your previous reasoning. The IE found your current diagnosis
("{prev_diagnosis}" at {prev_confidence} confidence) to be INADEQUATE.

### YOU MUST DO THE FOLLOWING:
1. EVALUATE "{ie_suggested_diagnosis}" as a SERIOUS candidate — do not dismiss it.
2. Check: does "{ie_suggested_diagnosis}" explain ALL major symptoms,
   INCLUDING any drug-triggered worsening (Drug-Exacerbation Rule)?
3. If YES and it explains MORE symptoms than your current diagnosis → adopt it.
4. If NO → you MUST explain SPECIFICALLY why "{ie_suggested_diagnosis}" does NOT fit,
   citing concrete symptoms or findings that contradict it.
   "I don't think so" or "unlikely" is NOT acceptable.
5. CRITICAL CROSS-CHECK — before accepting IE's suggestion, verify:
   (a) Does IE's suggestion account for any medication that WORSENED symptoms?
       If NOT, but an R1 differential DOES → prefer the R1 differential.
   (b) Does IE's suggestion address ALL of R1's top 3 differential diagnoses?
       If IE's suggestion ignores an R1 diagnosis that better explains a paradox → reject it.
   (c) Is IE's suggestion based on only ONE interpretation of an ambiguous sign?
       If so, consider other interpretations before adopting.
6. If an R1 differential diagnosis explains a drug-triggered worsening that NEITHER
   your current diagnosis NOR IE's suggestion explains → THAT R1 diagnosis should be primary.
7. If your current diagnosis is "Unknown" or has ≤ 0.10 confidence,
   you MUST adopt "{ie_suggested_diagnosis}" — an imperfect diagnosis beats no diagnosis.

### CONTEXT FROM IE:
{ie_reasoning}

### ISSUES THAT LED TO THIS OVERRIDE:
{issues_text}

Evaluate this override thoroughly. The IE identified a diagnostic concern — take it seriously,
but apply clinical judgment informed by ALL available evidence including drug-exacerbation paradoxes.
"""


# ── R2 Evidence Compression ─────────────────────────────────────────
def _compress_evidence_for_r3(r2_evidence: list[dict], max_chars: int = 4000) -> str:
    """Compress R2 evidence to fit within R3's context budget.

    Strips bulky raw_result fields, keeps essential fields (source, query,
    rationale, article titles/abstracts, counts). Truncates if still too long.
    """
    compressed = []
    for ev in r2_evidence:
        entry: dict[str, Any] = {
            "source": ev.get("source", ""),
            "query": ev.get("query", ""),
            "rationale": ev.get("rationale", ""),
            "count": ev.get("count", 0),
        }
        # Include article titles + truncated abstracts for PubMed/Europe PMC
        articles = ev.get("articles", [])
        if articles:
            entry["articles"] = []
            for a in articles[:5]:  # max 5 articles per query
                art = {"title": a.get("title", ""), "pmid": a.get("pmid", "")}
                abstract = a.get("abstract", "")
                if abstract:
                    art["abstract"] = abstract[:400] + ("..." if len(abstract) > 400 else "")
                entry["articles"].append(art)

        # Include web search answer
        if ev.get("answer"):
            entry["answer"] = ev["answer"][:500]
        if ev.get("results"):
            entry["results"] = [
                {"title": r.get("title", ""), "url": r.get("url", "")}
                for r in ev["results"][:3]
            ]

        # Include Wikipedia summary
        if ev.get("summary"):
            entry["summary"] = ev["summary"][:600]

        # Include clinical trials
        if ev.get("studies"):
            entry["studies"] = [
                {"title": s.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", str(s)[:100])}
                for s in ev["studies"][:3]
            ]

        # Include Semantic Scholar
        if ev.get("papers"):
            entry["papers"] = [
                {"title": p.get("title", ""), "citationCount": p.get("citationCount", 0)}
                for p in ev["papers"][:3]
            ]

        compressed.append(entry)

    text = json.dumps(compressed, indent=2, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [evidence truncated to fit context]"
        logger.info("[R3] R2 evidence truncated from %d to %d chars", len(text), max_chars)
    return text


async def run_rrrie_chat(
    ws: WebSocket,
    patient_text: str,
    *,
    llm_client: Any,
    groq_client: Any,
    gemini_client: Any = None,
    llama_server_url: str,
    memory: CaseStore,
    thinking_enabled: bool = True,
    local_only: bool = False,
    super_thinking: bool = False,
    deep_thinking: bool = False,
    expected_output: dict | None = None,
) -> None:
    """Execute the full RRRIE pipeline with real-time streaming.

    Parameters:
        ws:               Active WebSocket for real-time events
        patient_text:     Raw patient story
        llm_client:       LlamaCppClient instance (local 4B)
        groq_client:      GroqClient instance (cloud 70B)
        gemini_client:    GeminiClient instance (Gemini Flash/Pro)
        llama_server_url: Base URL for llama-server
        memory:           CaseStore singleton (3-tier memory)
        thinking_enabled: True = Thinking Mode, False = Fast Mode
        local_only:       True = all stages use local Qwen 3.5-4B
        super_thinking:   True = ALL stages use Gemini 3 Flash
        deep_thinking:    True = ALL stages use Gemini 3 Pro (with thinking)
        expected_output:  Optional Ground Truth dict for Post-Mortem Learning Loop
    """
    total_start = time.time()
    is_fast = not thinking_enabled

    # ── Gemini availability check (needed early for R1/R2) ──
    use_gemini = (
        (super_thinking or deep_thinking)
        and gemini_client is not None
        and getattr(gemini_client, "is_available", False)
    )
    gemini_use_pro = deep_thinking  # Pro for deep, Flash for super

    # ── Dynamic Token Budget Manager ──
    settings = get_settings()
    budget = TokenBudgetManager(
        ctx_window=llm_client.max_ctx if hasattr(llm_client, 'max_ctx') else settings.num_ctx,
        is_fast=is_fast,
        max_iterations=settings.max_rrrie_iterations,
    )

    # ══════════════════════════════════════════════════════════════
    # PHASE 0: PARALLEL PRE-PROCESSING
    #   R0 (DLLM on 0.8B), Safety+Zebra, Drug Lookup run concurrently.
    #   R0 uses a separate GPU server, Safety/Drug are CPU/API-bound.
    #   After all complete, R0 red flags are merged into Safety results.
    # ══════════════════════════════════════════════════════════════
    r0_result: R0Result | None = None
    pipeline_cfg: PipelineConfig | None = None

    async def _run_r0() -> R0Result | None:
        """Run DLLM R0 analysis on 0.8B model (separate server)."""
        try:
            from src.llm.llama_cpp_client import DLLMClient
            dllm_client = DLLMClient(base_url=settings.dllm_api_url)
            if dllm_client.is_healthy():
                await ws.send_json({
                    "type": "stage_start",
                    "stage": "R0",
                    "title": "R0 — DLLM Deep Analysis 🧠",
                    "description": "Running 5-layer deep reasoning on 0.8B model...",
                })
                dllm_engine = DLLMR0()
                result = await dllm_engine.analyze(patient_text)
                layers_str = "→".join(f"L{l}" for l in result.layers_run)
                await ws.send_json({
                    "type": "info",
                    "stage": "R0",
                    "content": (
                        f"✓ DLLM R0 complete: {layers_str} in {result.total_time:.1f}s — "
                        f"complexity={result.complexity}, urgency={result.urgency}, "
                        f"{len(result.red_flags)} red flags, "
                        f"{len(result.suggested_differentials)} prelim DDx"
                    ),
                })
                await ws.send_json({"type": "stage_complete", "stage": "R0"})
                logger.info(
                    "[R0] Done: %s (%.1fs) — complexity=%s, urgency=%s",
                    layers_str, result.total_time,
                    result.complexity, result.urgency,
                )
                return result
            else:
                logger.info("[R0] DLLM server not available — skipping R0")
                return None
        except Exception as exc:
            logger.warning("[R0] DLLM analysis failed (non-fatal): %s", exc)
            return None

    async def _run_drug_lookup() -> str:
        """Resolve patient drugs from pharmacology APIs."""
        try:
            from src.pipeline.drug_lookup import resolve_patient_drugs, format_drug_facts
            resolved_drugs = await resolve_patient_drugs(patient_text)
            if resolved_drugs:
                facts = format_drug_facts(resolved_drugs)
                drug_summary = ", ".join(
                    f"{d.original_name}→{d.generic_name} ({', '.join(d.drug_class[:1]) or 'resolving...'})"
                    for d in resolved_drugs
                )
                await ws.send_json({
                    "type": "info",
                    "stage": "SAFETY",
                    "content": f"💊 Resolved {len(resolved_drugs)} drug(s) from external APIs: {drug_summary}",
                })
                logger.info("[DRUG-LOOKUP] Resolved %d drugs: %s", len(resolved_drugs), drug_summary)
                return facts
            return ""
        except Exception as exc:
            logger.warning("[DRUG-LOOKUP] Drug resolution failed (non-fatal): %s", exc)
            return ""

    # Run R0, Safety (base, without R0 red flags), and Drug Lookup in parallel
    r0_task = asyncio.create_task(_run_r0())
    safety_task = asyncio.create_task(run_safety(ws, patient_text, r0_result=None))
    drug_task = asyncio.create_task(_run_drug_lookup())

    r0_result, safety, drug_facts_text = await asyncio.gather(
        r0_task, safety_task, drug_task,
    )

    # Route pipeline based on R0 assessment
    if r0_result is not None:
        pipeline_cfg = route_pipeline(r0_result.complexity, r0_result.urgency)
        logger.info(
            "[ROUTER] Pipeline config: complexity=%s, max_iter=%d, min_iter=%d",
            pipeline_cfg.complexity, pipeline_cfg.max_iterations, pipeline_cfg.min_iterations,
        )

    # Merge R0 red flags into Safety results (post-parallel)
    red_flags = safety["red_flags"]
    if r0_result is not None:
        for rf in getattr(r0_result, "red_flags", []):
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
                "type": "info",
                "stage": "SAFETY",
                "content": f"🔗 R0 merged {len(getattr(r0_result, 'red_flags', []))} DLLM red flags into safety results.",
            })

    zebra_matches = safety["zebra_matches"]
    zebra_alert_text = safety["zebra_alert_text"]

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: R1 — Reasoned Analysis
    # ══════════════════════════════════════════════════════════════
    # Enrich patient text with R0 findings for R1
    r1_patient_text = patient_text
    if r0_result is not None:
        r0_hints = []
        if r0_result.suggested_differentials:
            r0_hints.append(
                "\n\n📋 DLLM PRELIMINARY ANALYSIS (from 0.8B deep reasoning):\n"
                "Suggested differentials: " + ", ".join(r0_result.suggested_differentials)
            )
        if r0_result.connections:
            r0_hints.append("Clinical connections: " + "; ".join(r0_result.connections[:5]))
        if r0_result.patterns:
            pattern_names = [p.get("name", "?") for p in r0_result.patterns[:3]]
            r0_hints.append("Detected patterns: " + ", ".join(pattern_names))
        if r0_hints:
            r1_patient_text += "\n".join(r0_hints)

        # Pre-R1 Clinical Text Annotation: translate non-English terms before R1
        if r0_result.language != "en":
            try:
                annotation = await annotate_clinical_text_pre_r1(
                    patient_text=patient_text,
                    r0_language=r0_result.language,
                    r0_entities=r0_result.entities,
                    groq_client=groq_client,
                    llm_client=llm_client,
                    llama_server_url=llama_server_url,
                    local_only=local_only,
                )
                if annotation:
                    r1_patient_text += annotation
                    await ws.send_json({
                        "type": "info",
                        "stage": "R1",
                        "content": "🌐 Non-English text detected — clinical terms annotated with English equivalents for R1.",
                    })
            except Exception as exc:
                logger.warning("[PRE-R1] Clinical annotation failed (non-fatal): %s", exc)

    r1_json, r1_result, r1_model_label = await run_r1(
        ws, r1_patient_text,
        red_flags=red_flags,
        zebra_alert_text=zebra_alert_text,
        drug_facts=drug_facts_text,
        is_fast=is_fast,
        groq_client=groq_client,
        gemini_client=gemini_client if use_gemini else None,
        gemini_use_pro=gemini_use_pro,
        llm_client=llm_client,
        llama_server_url=llama_server_url,
        thinking_enabled=thinking_enabled,
        local_only=local_only,
        budget=budget,
        complexity=pipeline_cfg.complexity if pipeline_cfg else "moderate",
    )

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: R2 — Reinforced Fact-Finding
    # ══════════════════════════════════════════════════════════════
    r2_evidence, icd_resolved = await run_r2(
        ws, r1_json,
        is_fast=is_fast,
        patient_text=patient_text,
        drug_facts=drug_facts_text,
        llm_client=llm_client,
        llama_server_url=llama_server_url,
        groq_client=groq_client,
        gemini_client=gemini_client if use_gemini else None,
        gemini_use_pro=gemini_use_pro,
        local_only=local_only,
        super_thinking=super_thinking or deep_thinking,
    )

    # ══════════════════════════════════════════════════════════════
    # MEMORY RETRIEVAL
    # ══════════════════════════════════════════════════════════════
    memory_ctx = memory.retrieve_context(patient_text)
    if not memory_ctx.is_empty:
        mem_stats = memory.get_stats()
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": (
                f"🧠 Memory active — {mem_stats['tier3_principles']} principles, "
                f"{mem_stats['tier2_patterns']} patterns, "
                f"{mem_stats['tier1_cases']} past cases. "
                f"Injecting relevant experience into analysis..."
            ),
        })
        logger.info(
            "[MEMORY] Retrieved: %d principles, %d patterns, %d similar cases",
            len(memory_ctx.principles), len(memory_ctx.patterns),
            len(memory_ctx.similar_cases),
        )

    memory_prompt_text = ""
    if not memory_ctx.is_empty:
        memory_prompt_text = MEMORY_PREAMBLE + memory_ctx.format_for_prompt()

    # ══════════════════════════════════════════════════════════════
    # PHASE 2.5: PARADOX DETECTION (drug → worsening analysis)
    # ══════════════════════════════════════════════════════════════
    from src.core.paradox_resolver import detect_paradoxes, format_paradox_directive, format_paradox_for_ie

    paradoxes = detect_paradoxes(patient_text)
    paradox_directive = format_paradox_directive(
        paradoxes,
        r1_differentials=r1_json.get("differential_diagnoses", []) if r1_json else None,
    )
    paradox_ie_text = format_paradox_for_ie(paradoxes)

    if paradoxes:
        paradox_summary = ", ".join(f'"{p.drug_or_intervention}"' for p in paradoxes)
        await ws.send_json({
            "type": "info",
            "stage": "R3",
            "content": (
                f"⚡ Drug-exacerbation paradox detected: {paradox_summary} → symptoms worsened. "
                f"Injecting investigation directive into R3."
            ),
        })
        logger.info(
            "[PARADOX] Detected %d drug-exacerbation paradox(es): %s",
            len(paradoxes), paradox_summary,
        )

    # ══════════════════════════════════════════════════════════════
    # PHASE 3+4: ITERATIVE R3 ↔ IE LOOP  (Layer C — Dynamic)
    # ══════════════════════════════════════════════════════════════
    settings = get_settings()

    # Use Router-derived config if R0 ran, else fall back to settings
    if pipeline_cfg is not None:
        max_iterations = pipeline_cfg.max_iterations
        min_iterations = pipeline_cfg.min_iterations
        stagnation_threshold = getattr(settings, 'stagnation_threshold', 2)
        logger.info(
            "[ROUTER] Pipeline config: complexity=%s, max_iter=%d, min_iter=%d",
            pipeline_cfg.complexity, max_iterations, min_iterations,
        )
    else:
        max_iterations = settings.max_rrrie_iterations
        min_iterations = getattr(settings, 'min_rrrie_iterations', 2)
        stagnation_threshold = getattr(settings, 'stagnation_threshold', 2)

    if is_fast:
        max_iterations = 2
        min_iterations = 1
    elif local_only:
        # Local mode: cap max iterations but respect Router's min for complex/critical cases
        max_iterations = min(max_iterations, 4)
        # Only reduce min_iterations if Router didn't set a higher minimum
        if pipeline_cfg is None:
            min_iterations = min(min_iterations, 1)

    # ── Smart Iteration Controller ──
    iter_ctrl = IterationController(
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        confidence_target=pipeline_cfg.confidence_target if pipeline_cfg else settings.confidence_threshold,
        stagnation_limit=stagnation_threshold,
    )

    # ── Model routing flags ──
    if use_gemini:
        # Gemini mode: ALL cloud stages use Gemini (Flash or Pro)
        use_groq_r3 = False
        use_groq_ie = False
        gemini_label = f"☁️ Gemini {'Pro' if gemini_use_pro else 'Flash'}"
        r3_model_label = gemini_label
        ie_model_label = gemini_label
    else:
        use_groq_r3 = groq_client.is_available and not is_fast and not local_only
        use_groq_ie = super_thinking and groq_client.is_available and not is_fast
        r3_model_label = f"☁️ {groq_client.model}" if use_groq_r3 else "💻 Local 4B"
        ie_model_label = f"☁️ {groq_client.model}" if use_groq_ie else "💻 Local 4B"

    r3_json: dict = {}
    r3_result: dict = {}
    ie_json: dict = {}
    ie_result: dict = {}
    iteration_history: list[dict] = []
    final_iteration = 1

    # ── Stagnation tracking ──
    prev_primary_diagnoses: list[str] = []  # track primary dx per iteration
    stagnation_count = 0  # consecutive same-dx iterations
    perspective_used = False  # whether we've injected a perspective shift
    r2_research_done = False  # one-shot R2 re-search tracker
    ie_suggested_dx: str | None = None  # IE's suggested diagnosis (if any)
    ie_consensus_tracker = IEConsensusTracker()  # Layer 1: treatment safety

    for iteration in range(1, max_iterations + 1):
        is_last_iteration = (iteration == max_iterations)

        # ── R3 — Reasoning Synthesis ──────────────────────────────
        iter_label = f" (iter {iteration}/{max_iterations})" if max_iterations > 1 else ""
        await ws.send_json({
            "type": "stage_start",
            "stage": "R3",
            "title": f"R3 — Reasoning Synthesis{iter_label}" + (" ⚡" if is_fast else " 🧠"),
            "description": (
                f"{'Quick synthesis...' if is_fast else 'Deep synthesis — integrating R1 hypotheses with R2 evidence...'}"
                f" [{r3_model_label}]"
            ),
        })

        # Build R3 prompt — always start with original patient text
        r2_compressed = _compress_evidence_for_r3(r2_evidence)
        r3_base_context = f"""## Original Patient Presentation
{patient_text}

## R1 Analysis Results
{json.dumps(r1_json, indent=2, ensure_ascii=False)}

## R2 Evidence Collected
{r2_compressed}
"""
        # If R1 JSON is empty/unusable, include raw R1 text as fallback
        if not r1_json.get("differential_diagnoses"):
            raw_r1 = r1_result.get("clean_content", "")
            if raw_r1 and len(raw_r1) > 100:
                r3_base_context += f"""
## R1 Raw Output (JSON parse failed — use this as best-effort input)
{raw_r1[:3000]}
"""
                logger.info("[R3] Injecting raw R1 text (%d chars) as fallback", len(raw_r1))

        if memory_prompt_text:
            r3_base_context += "\n" + memory_prompt_text + "\n"
        if drug_facts_text:
            r3_base_context += "\n" + drug_facts_text + "\n"
        if paradox_directive:
            r3_base_context += "\n" + paradox_directive + "\n"
        r3_base_context += (
            "\nSynthesize the above R1 hypotheses with R2 evidence. "
            "Update confidence levels based on evidence support."
        )
        if is_fast:
            r3_base_context = FAST_MODE_INSTRUCTION + "\n\n" + r3_base_context

        # On iterations > 1: inject IE feedback (or perspective shift if stuck)
        if iteration > 1 and ie_json:
            prev_issues = ie_json.get("issues", [])
            issues_text = "\n".join(
                f"- [{iss.get('severity', '?').upper()}] {iss.get('type', '?')}: {iss.get('detail', '?')}"
                for iss in prev_issues
            ) if prev_issues else "No specific issues listed (but IE said ITERATE)."

            prev_dx = r3_json.get("primary_diagnosis", {})
            prev_dx_name = prev_dx.get("diagnosis", "Unknown")
            prev_confidence = prev_dx.get("confidence", 0.0)

            # Check stagnation: same primary diagnosis repeated? (semantic similarity)
            if prev_primary_diagnoses and _diagnoses_are_similar(prev_dx_name, prev_primary_diagnoses[-1]):
                stagnation_count += 1
            else:
                stagnation_count = 0

            # Decide: normal feedback, perspective shift, or evidence-fix?
            # ── HIGH-CONFIDENCE GATE ──
            # If the model is confident (≥0.75) AND IE's issues are mainly about
            # evidence/hallucination (not wrong diagnosis), a perspective shift
            # would DESTROY the correct answer. Instead, give targeted feedback.
            ie_issue_types = {iss.get("type", "") for iss in prev_issues} if prev_issues else set()
            evidence_only_issues = ie_issue_types <= {
                "hallucination", "contradictory_stats", "evidence_gap",
                "missing_history", "contraindication",
            }
            high_conf_stagnation = (
                stagnation_count >= stagnation_threshold
                and prev_confidence >= 0.75
                and evidence_only_issues
            )

            if high_conf_stagnation:
                # Don't shift — the diagnosis is likely correct, only evidence is broken
                feedback = (
                    f"## ⚠️ EVIDENCE REFINEMENT (Iteration {iteration})\n\n"
                    f"Your diagnosis '{prev_dx_name}' (confidence {prev_confidence}) may be CORRECT,\n"
                    f"but the quality checker found these evidence/citation issues:\n\n"
                    f"{issues_text}\n\n"
                    f"### MANDATORY FIXES:\n"
                    f"1. If R2 found ZERO articles: your citations list MUST be empty [].\n"
                    f"   Do NOT invent PMIDs. Remove any PMID not in the R2 Evidence section.\n"
                    f"2. Check your symptom coverage — explain ALL patient symptoms.\n"
                    f"3. Keep your current diagnosis unless clinical reasoning is genuinely flawed.\n"
                    f"4. A correct diagnosis with empty citations is BETTER than a wrong diagnosis with fake citations.\n"
                    f"5. Review treatment ORDERING: in malnourished/bariatric patients,\n"
                    f"   ALWAYS give thiamine BEFORE glucose — glucose without thiamine can be FATAL.\n"
                )
                await ws.send_json({
                    "type": "info",
                    "stage": "R3",
                    "content": (
                        f"🔧 Evidence refinement mode — '{prev_dx_name}' at {prev_confidence} confidence, "
                        f"fixing citation/evidence issues without changing diagnosis."
                    ),
                })
                logger.info(
                    "[LAYER-C] High-confidence stagnation (%s at %.2f) — evidence-fix mode, NOT perspective shift.",
                    prev_dx_name, prev_confidence,
                )
            elif stagnation_count >= stagnation_threshold and not perspective_used:
                # PERSPECTIVE SHIFT — pick a prompt based on iteration
                shift_idx = min(stagnation_count - stagnation_threshold, len(PERSPECTIVE_SHIFT_PROMPTS) - 1)
                feedback = PERSPECTIVE_SHIFT_PROMPTS[shift_idx].format(
                    iteration=iteration,
                    stagnation_count=stagnation_count,
                    prev_diagnosis=prev_dx_name,
                    prev_confidence=prev_confidence,
                    issues_text=issues_text,
                )
                perspective_used = True

                await ws.send_json({
                    "type": "info",
                    "stage": "R3",
                    "content": (
                        f"🔄 PERSPECTIVE SHIFT activated — model stuck on '{prev_dx_name}' "
                        f"for {stagnation_count} iterations. Forcing new reasoning angle."
                    ),
                })
                logger.info(
                    "[LAYER-C] Stagnation detected (%d repeats of '%s'). Injecting perspective shift.",
                    stagnation_count, prev_dx_name,
                )
            elif stagnation_count >= stagnation_threshold and perspective_used:
                # Already tried one perspective, try next
                shift_idx = min(stagnation_count - stagnation_threshold, len(PERSPECTIVE_SHIFT_PROMPTS) - 1)
                feedback = PERSPECTIVE_SHIFT_PROMPTS[shift_idx].format(
                    iteration=iteration,
                    stagnation_count=stagnation_count,
                    prev_diagnosis=prev_dx_name,
                    prev_confidence=prev_confidence,
                    issues_text=issues_text,
                )
                await ws.send_json({
                    "type": "info",
                    "stage": "R3",
                    "content": f"🔄 Trying another perspective shift (angle #{shift_idx + 1})...",
                })
            else:
                # Normal iteration feedback
                feedback = ITERATION_FEEDBACK_TEMPLATE.format(
                    iteration=iteration,
                    issues_text=issues_text,
                    ie_reasoning=ie_json.get("reasoning", "No reasoning provided."),
                    prev_diagnosis=prev_dx_name,
                    prev_confidence=prev_confidence,
                )

            # ── IE OVERRIDE: If IE suggested a specific diagnosis, inject MANDATORY directive ──
            if ie_suggested_dx:
                override_text = IE_OVERRIDE_DIRECTIVE.format(
                    ie_suggested_diagnosis=ie_suggested_dx,
                    prev_diagnosis=prev_dx_name,
                    prev_confidence=prev_confidence,
                    ie_reasoning=ie_json.get("reasoning", "No reasoning provided."),
                    issues_text=issues_text,
                )
                feedback = override_text + "\n\n" + feedback
                await ws.send_json({
                    "type": "info",
                    "stage": "R3",
                    "content": (
                        f"🚨 IE OVERRIDE: IE suggests '{ie_suggested_dx}' as correct diagnosis. "
                        f"R3 MUST evaluate this as primary candidate."
                    ),
                })
                logger.info(
                    "[IE-OVERRIDE] Injecting mandatory directive: IE suggests '%s' (prev: '%s' at %.2f)",
                    ie_suggested_dx, prev_dx_name, prev_confidence,
                )

            r3_base_context = feedback + "\n\n" + r3_base_context

        # Two-Phase Architecture: diagnosis-only prompt in thinking mode,
        # full prompt (with treatment) only in fast mode.
        r3_system_raw = R3_SYSTEM_PROMPT if is_fast else R3_DIAGNOSIS_SYSTEM_PROMPT
        # Adaptive prompt trimming: only trim for local model (cloud has plenty of context)
        r3_is_local = not (use_gemini or use_groq_r3)
        r3_system = adapt_prompt_for_complexity(r3_system_raw, pipeline_cfg.complexity if pipeline_cfg and r3_is_local else "moderate")
        r3_messages = [
            {"role": "system", "content": r3_system},
            {"role": "user", "content": r3_base_context},
        ]

        # Dynamic token allocation from pool
        r3_prompt_est = len(r3_base_context) // 3  # rough char→token estimate

        if use_gemini:
            r3_max_tokens = budget.allocate("R3", iteration=iteration, prompt_tokens=r3_prompt_est, is_groq=True)
            try:
                r3_result = await stream_gemini_completion(
                    ws, gemini_client, r3_messages, stage="R3",
                    max_tokens=r3_max_tokens, use_pro=gemini_use_pro,
                )
            except Exception as exc:
                logger.warning("[R3] Gemini failed (%s), falling back to Groq/local", exc)
                await ws.send_json({
                    "type": "info", "stage": "R3",
                    "content": f"⚠ Gemini unavailable ({exc}), falling back...",
                })
                if groq_client.is_available:
                    r3_result = await stream_groq_completion(
                        ws, groq_client, r3_messages, stage="R3", max_tokens=r3_max_tokens,
                    )
                else:
                    local_r3_max = budget.allocate("R3", iteration=iteration, prompt_tokens=r3_prompt_est, is_groq=False)
                    r3_result = await stream_llm_completion(
                        ws, llm_client, llama_server_url, r3_messages,
                        stage="R3", max_tokens=local_r3_max, thinking_enabled=thinking_enabled,
                        budget_managed=True,
                    )
                r3_model_label = "☁️ Groq (fallback)" if groq_client.is_available else "💻 Local 4B (fallback)"
        elif use_groq_r3:
            r3_max_tokens = budget.allocate("R3", iteration=iteration, prompt_tokens=r3_prompt_est, is_groq=True)
            try:
                r3_result = await stream_groq_completion(
                    ws, groq_client, r3_messages, stage="R3", max_tokens=r3_max_tokens,
                )
            except Exception as exc:
                logger.warning("[R3] Groq failed (%s), falling back to local model", exc)
                await ws.send_json({
                    "type": "info", "stage": "R3",
                    "content": f"⚠ Groq unavailable ({exc}), falling back to local model...",
                })
                local_r3_max = budget.allocate("R3", iteration=iteration, prompt_tokens=r3_prompt_est, is_groq=False)
                r3_result = await stream_llm_completion(
                    ws, llm_client, llama_server_url, r3_messages,
                    stage="R3", max_tokens=local_r3_max, thinking_enabled=thinking_enabled,
                    budget_managed=True,
                )
                r3_model_label = "💻 Local 4B (fallback)"
        else:
            local_r3_max = budget.allocate("R3", iteration=iteration, prompt_tokens=r3_prompt_est, is_groq=False)
            r3_result = await stream_llm_completion(
                ws, llm_client, llama_server_url, r3_messages,
                stage="R3", max_tokens=local_r3_max, thinking_enabled=thinking_enabled,
                budget_managed=True,
            )

        # ── Safety net: if R3 returned empty (likely 400 context overflow), retry with truncated prompt ──
        if not r3_result.get("clean_content", "").strip():
            logger.warning("[R3] Empty response — retrying with truncated context")
            await ws.send_json({
                "type": "info", "stage": "R3",
                "content": "⚠ R3 context too large — retrying with compressed input...",
            })
            # Aggressively truncate: halve evidence, trim patient text
            r2_short = _compress_evidence_for_r3(r2_evidence, max_chars=3000)
            r3_short_context = f"""## Patient (abbreviated)
{patient_text[:1500]}

## R1 Top Diagnoses
{json.dumps([d for d in r1_json.get('differential_diagnoses', [])[:3]], indent=1, ensure_ascii=False)}

## R2 Evidence (compressed)
{r2_short}
"""
            if drug_facts_text:
                r3_short_context += f"\n{drug_facts_text}\n"
            r3_short_context += "\nSynthesize the above. Update confidence levels based on evidence support."
            if is_fast:
                r3_short_context = FAST_MODE_INSTRUCTION + "\n\n" + r3_short_context
            r3_messages = [
                {"role": "system", "content": r3_system},
                {"role": "user", "content": r3_short_context},
            ]
            retry_max = budget.allocate("R3", iteration=iteration, prompt_tokens=len(r3_short_context) // 3, is_groq=False)
            r3_result = await stream_llm_completion(
                ws, llm_client, llama_server_url, r3_messages,
                stage="R3", max_tokens=retry_max, thinking_enabled=thinking_enabled,
                budget_managed=True,
            )

        r3_json = parse_json_from_response(r3_result["clean_content"])

        # Inject real ICD-11 codes into R3 output
        if icd_resolved:
            primary = r3_json.get("primary_diagnosis")
            if primary:
                pname = primary.get("diagnosis", "")
                if pname in icd_resolved and icd_resolved[pname]:
                    best = icd_resolved[pname][0]
                    primary["icd11_code"] = best.get("theCode", "")
                    primary["icd11_title"] = best.get("title", "")
                    primary["icd11_source"] = "WHO API"
            updated = r3_json.get("updated_diagnoses", [])
            if updated:
                resolve_icd_codes(updated, icd_resolved)

        # Report R3 actual usage to budget manager
        r3_alloc_used = r3_result.get("completion_tokens", 0)
        r3_allocated = r3_max_tokens if (use_gemini or use_groq_r3) else local_r3_max
        budget.report("R3", allocated=r3_allocated, used=r3_alloc_used)

        # Broadcast token pool status to UI
        await ws.send_json({"type": "token_pool", "data": budget.get_status()})

        await ws.send_json({
            "type": "stage_result",
            "stage": "R3",
            "data": r3_json,
            "stats": {
                "tokens": r3_result["completion_tokens"],
                "time": r3_result["elapsed"],
                "tok_per_sec": r3_result["tok_per_sec"],
                "iteration": iteration,
            },
        })
        await ws.send_json({"type": "stage_complete", "stage": "R3"})

        # ── IE — Iterative Evaluation ─────────────────────────────
        await ws.send_json({
            "type": "stage_start",
            "stage": "IE",
            "title": f"IE — Iterative Evaluation{iter_label}" + (" ⚡" if is_fast else " 🧠"),
            "description": (
                ("Quick quality check..." if is_fast else
                 "Checklist review — symptom coverage, hallucinations, evidence gaps...")
                + f" [{ie_model_label}]"
            ),
        })

        ie_context = simplify_for_ie(
            r1_json=r1_json,
            r3_json=r3_json,
            r2_evidence=r2_evidence,
            patient_text=patient_text,
            memory_context=memory_ctx if not memory_ctx.is_empty else None,
            iteration=iteration,
            drug_facts=drug_facts_text,
            paradox_text=paradox_ie_text,
        )
        if is_fast:
            ie_context = FAST_MODE_INSTRUCTION + "\n\n" + ie_context

        ie_messages = [
            {"role": "system", "content": IE_SYSTEM_PROMPT},
            {"role": "user", "content": ie_context},
        ]

        ie_prompt_est = len(ie_context) // 3
        _used_layered_ie = False

        if use_gemini:
            # Gemini mode: IE uses Gemini Flash/Pro
            ie_max_tokens = budget.allocate("IE", iteration=iteration, prompt_tokens=ie_prompt_est, is_groq=True)
            try:
                ie_result = await stream_gemini_completion(
                    ws, gemini_client, ie_messages, stage="IE",
                    max_tokens=ie_max_tokens, use_pro=gemini_use_pro,
                )
            except Exception as exc:
                logger.warning("[IE] Gemini failed (%s), falling back", exc)
                await ws.send_json({
                    "type": "info", "stage": "IE",
                    "content": f"⚠ Gemini unavailable ({exc}), falling back...",
                })
                ie_max_tokens = budget.allocate("IE", iteration=iteration, prompt_tokens=ie_prompt_est, is_groq=False)
                ie_result = await stream_llm_completion(
                    ws, llm_client, llama_server_url, ie_messages,
                    stage="IE", max_tokens=ie_max_tokens, thinking_enabled=thinking_enabled,
                    budget_managed=True,
                )
        elif use_groq_ie:
            # Super Thinking: IE uses Groq 70B for deeper evaluation
            ie_max_tokens = budget.allocate("IE", iteration=iteration, prompt_tokens=ie_prompt_est, is_groq=True)
            try:
                ie_result = await stream_groq_completion(
                    ws, groq_client, ie_messages, stage="IE", max_tokens=ie_max_tokens,
                )
            except Exception as exc:
                logger.warning("[IE] Groq failed (%s), falling back to local model", exc)
                await ws.send_json({
                    "type": "info", "stage": "IE",
                    "content": f"⚠ Groq unavailable ({exc}), falling back to local model...",
                })
                ie_max_tokens = budget.allocate("IE", iteration=iteration, prompt_tokens=ie_prompt_est, is_groq=False)
                ie_result = await stream_llm_completion(
                    ws, llm_client, llama_server_url, ie_messages,
                    stage="IE", max_tokens=ie_max_tokens, thinking_enabled=thinking_enabled,
                    budget_managed=True,
                )
        else:
            # Standard: IE uses local model — LAYERED approach for 4B model
            # Decomposes the monolithic 9-check IE into 3 focused micro-checks
            # that the 4B model can handle without truncation.
            ie_thinking_local = thinking_enabled if not local_only else False

            async def _local_ie_call(messages, max_tokens):
                """Wrapper for layered IE calls using local LLM."""
                return await stream_llm_completion(
                    ws, llm_client, llama_server_url, messages,
                    stage="IE", max_tokens=max_tokens,
                    thinking_enabled=ie_thinking_local,
                    budget_managed=True,
                )

            def _local_budget_alloc(stage, iteration, prompt_tokens):
                return budget.allocate(stage, iteration=iteration, prompt_tokens=prompt_tokens, is_groq=False)

            layered_result = await run_layered_ie(
                patient_text=patient_text,
                r1_json=r1_json,
                r3_json=r3_json,
                r2_evidence=r2_evidence,
                drug_facts=drug_facts_text,
                paradox_text=paradox_ie_text,
                call_fn=_local_ie_call,
                budget_allocate=_local_budget_alloc,
                iteration=iteration,
            )

            ie_json = layered_result
            ie_max_tokens = 0  # layered IE manages its own budget per-layer
            # Build a synthetic ie_result for stats compatibility
            ie_result = {
                "completion_tokens": layered_result.get("_completion_tokens", 0),
                "elapsed": layered_result.get("_elapsed", 0),
                "tok_per_sec": round(
                    layered_result.get("_completion_tokens", 0) / max(layered_result.get("_elapsed", 0.1), 0.1), 1
                ),
            }
            _used_layered_ie = True

        if not _used_layered_ie:
            ie_json = parse_json_from_response(ie_result["clean_content"])

        # ── Extract IE's diagnosis suggestion (if any) ──
        ie_suggested_dx = _extract_ie_diagnosis_suggestion(ie_json)
        if ie_suggested_dx:
            logger.info(
                "[IE-SUGGESTION] IE suggests diagnosis: '%s' (iteration %d)",
                ie_suggested_dx, iteration,
            )
            await ws.send_json({
                "type": "info",
                "stage": "IE",
                "content": f"💡 IE suggests alternative diagnosis: {ie_suggested_dx}",
            })

        # ── Record for consensus tracking (treatment safety Layer 1) ──
        ie_consensus_tracker.record(
            ie_suggested_dx,
            reasoning=ie_json.get("reasoning", ""),
            confidence=ie_json.get("confidence", 0.0),
        )

        await ws.send_json({
            "type": "stage_result",
            "stage": "IE",
            "data": ie_json,
            "stats": {
                "tokens": ie_result["completion_tokens"],
                "time": ie_result["elapsed"],
                "tok_per_sec": ie_result["tok_per_sec"],
                "iteration": iteration,
            },
        })
        await ws.send_json({"type": "stage_complete", "stage": "IE"})

        # Report IE actual usage to budget manager
        ie_alloc_used = ie_result.get("completion_tokens", 0)
        budget.report("IE", allocated=ie_max_tokens, used=ie_alloc_used)

        # Broadcast token pool status to UI
        await ws.send_json({"type": "token_pool", "data": budget.get_status()})

        # Record iteration history
        current_dx = r3_json.get("primary_diagnosis", {}).get("diagnosis", "Unknown")
        prev_primary_diagnoses.append(current_dx)

        iteration_history.append({
            "iteration": iteration,
            "r3_tokens": r3_result["completion_tokens"],
            "r3_time": r3_result["elapsed"],
            "ie_tokens": ie_result["completion_tokens"],
            "ie_time": ie_result["elapsed"],
            "ie_decision": ie_json.get("decision", "FINALIZE"),
            "ie_confidence": ie_json.get("confidence", 0),
            "ie_issues_count": len(ie_json.get("issues", [])),
            "primary_diagnosis": current_dx,
            "stagnation_count": stagnation_count,
            "perspective_shift": perspective_used,
            "budget_pool_remaining": budget.pool_remaining,
            "budget_pool_utilization": round(budget.pool_utilization, 3),
        })
        final_iteration = iteration

        # ── Feed the IterationController ──
        ie_decision = ie_json.get("decision", "FINALIZE").upper()
        ie_confidence = ie_json.get("confidence", 0.0)
        ie_issues = ie_json.get("issues", [])
        ie_max_sev = "low"
        for iss in ie_issues:
            sev = iss.get("severity", "low").lower()
            if sev == "critical":
                ie_max_sev = "critical"
                break
            if sev == "high" and ie_max_sev != "critical":
                ie_max_sev = "high"
            elif sev == "moderate" and ie_max_sev not in ("critical", "high"):
                ie_max_sev = "moderate"
        iter_ctrl.record(
            iteration=iteration,
            primary_diagnosis=current_dx,
            confidence=ie_confidence,
            ie_verdict=ie_decision,
            ie_issue_count=len(ie_issues),
            ie_max_severity=ie_max_sev,
        )

        # ── Decision: FINALIZE or ITERATE ─────────────────────────

        # Dynamic early exit: if IE says FINALIZE and we've done enough iterations
        if ie_decision == "FINALIZE" and iteration >= min_iterations:
            if iteration > 1:
                await ws.send_json({
                    "type": "info",
                    "stage": "IE",
                    "content": f"✅ IE satisfied after {iteration} iteration(s). Finalizing.",
                })
            break
        elif ie_decision == "FINALIZE" and iteration < min_iterations:
            # Force at least min_iterations
            await ws.send_json({
                "type": "info",
                "stage": "IE",
                "content": (
                    f"⚠ IE says FINALIZE at iter {iteration}, but min iterations = {min_iterations}. "
                    f"Continuing for robustness."
                ),
            })
            continue
        elif is_last_iteration:
            # ── LAST-RESORT IE OVERRIDE ──
            # If R3 still has Unknown/0% but IE suggested a specific diagnosis,
            # adopt IE's suggestion to avoid returning a blank diagnosis.
            current_primary = r3_json.get("primary_diagnosis", {})
            current_dx_name = current_primary.get("diagnosis", "Unknown")
            current_conf = current_primary.get("confidence", 0.0)

            is_blank_diagnosis = (
                current_dx_name.lower() in ("unknown", "unknown diagnosis", "0", "")
                or current_conf <= 0.05
            )

            if is_blank_diagnosis and ie_suggested_dx:
                logger.warning(
                    "[IE-OVERRIDE] LAST RESORT: R3 has '%s' at %.2f — adopting IE suggestion '%s'",
                    current_dx_name, current_conf, ie_suggested_dx,
                )
                # Override R3's primary diagnosis with IE's suggestion
                r3_json["primary_diagnosis"] = {
                    "diagnosis": ie_suggested_dx,
                    "confidence": max(current_conf, 0.35),  # At least 35% — it's a reasoned guess
                    "reasoning_chain": [
                        f"R3 failed to identify diagnosis after {max_iterations} iterations.",
                        f"IE evaluator independently suggested: {ie_suggested_dx}",
                        f"IE reasoning: {ie_json.get('reasoning', 'N/A')}",
                        "This diagnosis was adopted as a last-resort override.",
                    ],
                    "explains_symptoms": [],
                    "unexplained_symptoms": [],
                    "evidence_support": "ie_override",
                    "ie_override": True,
                }
                # Also add to updated_diagnoses
                existing_updated = r3_json.get("updated_diagnoses", [])
                existing_updated.insert(0, {
                    "diagnosis": ie_suggested_dx,
                    "initial_confidence": 0.0,
                    "updated_confidence": 0.35,
                    "evidence_summary": f"IE override: {ie_json.get('reasoning', 'IE suggested this diagnosis')}",
                })
                r3_json["updated_diagnoses"] = existing_updated

                await ws.send_json({
                    "type": "info",
                    "stage": "IE",
                    "content": (
                        f"🚨 IE OVERRIDE ACTIVATED: R3 returned '{current_dx_name}' at {current_conf:.0%}. "
                        f"Adopting IE's suggestion: '{ie_suggested_dx}' at 35%. "
                        f"An imperfect diagnosis is better than no diagnosis."
                    ),
                })
            else:
                await ws.send_json({
                    "type": "info",
                    "stage": "IE",
                    "content": (
                        f"⚠ IE still recommends iteration but max iterations ({max_iterations}) reached. "
                        f"Forcing FINALIZE with best available result."
                    ),
                })

            logger.warning(
                "[LAYER-C] Max iterations reached (%d). IE wanted ITERATE. Forcing FINALIZE.",
                max_iterations,
            )
            break
        else:
            critical_count = sum(
                1 for iss in ie_json.get("issues", [])
                if iss.get("severity", "").lower() == "critical"
            )

            # ── IE-TRIGGERED R2 RE-SEARCH ──
            # If IE keeps finding evidence_gap issues and we haven't re-searched yet,
            # trigger a targeted R2 re-run using R3's current primary diagnosis.
            # This fixes cases where R2's initial queries were off-topic.
            ie_issue_types_now = {
                iss.get("type", "") for iss in ie_json.get("issues", [])
            }
            has_evidence_gap = bool(ie_issue_types_now & {
                "evidence_gap", "hallucination", "contradictory_stats",
            })
            current_primary = r3_json.get("primary_diagnosis", {}).get("diagnosis", "")

            # Use IE's suggested diagnosis for re-search if R3's primary is weak
            research_target = current_primary
            if ie_suggested_dx and (
                not current_primary
                or current_primary.lower() in ("unknown", "unknown diagnosis")
                or r3_json.get("primary_diagnosis", {}).get("confidence", 0) < 0.1
            ):
                research_target = ie_suggested_dx
                logger.info(
                    "[LAYER-C] Using IE suggestion '%s' for re-search (R3 primary '%s' is weak)",
                    ie_suggested_dx, current_primary,
                )

            if (has_evidence_gap and iteration >= 2 and not r2_research_done
                    and research_target and not is_fast):
                r2_research_done = True
                logger.info(
                    "[LAYER-C] IE found evidence gaps at iter %d. Triggering targeted R2 re-search for '%s'",
                    iteration, research_target,
                )
                await ws.send_json({
                    "type": "info",
                    "stage": "R2",
                    "content": (
                        f"🔬 IE detected evidence gaps — running targeted re-search "
                        f"for '{research_target}'..."
                    ),
                })

                # Import the tool execution function
                from src.pipeline.stages import _execute_tool_query, _init_tool_dispatch
                _init_tool_dispatch()

                # Generate targeted queries for the research target (R3 primary or IE suggestion)
                targeted_queries = [
                    {
                        "tool": "search_pubmed",
                        "params": {"query": research_target},
                        "rationale": f"Targeted re-search: PubMed evidence for {research_target}",
                    },
                    {
                        "tool": "search_europe_pmc",
                        "params": {"query": research_target, "max_results": 5},
                        "rationale": f"Targeted re-search: Europe PMC for {research_target}",
                    },
                    {
                        "tool": "search_wikipedia_medical",
                        "params": {"disease_name": research_target},
                        "rationale": f"Targeted re-search: Wikipedia overview for {research_target}",
                    },
                ]

                total_re = len(targeted_queries)
                import asyncio as _asyncio
                re_tasks = []
                for i, q in enumerate(targeted_queries):
                    re_tasks.append(
                        _execute_tool_query(
                            ws, q["tool"], q["params"], q["rationale"],
                            i + 1, total_re,
                        )
                    )
                re_results = await _asyncio.gather(*re_tasks, return_exceptions=True)
                new_evidence_count = 0
                for res in re_results:
                    if isinstance(res, dict):
                        r2_evidence.append(res)
                        new_evidence_count += 1

                # Recompress evidence for next R3 iteration
                r2_compressed = _compress_evidence_for_r3(r2_evidence)

                await ws.send_json({
                    "type": "info",
                    "stage": "R2",
                    "content": (
                        f"✓ Targeted re-search complete: {new_evidence_count} new evidence sources "
                        f"added. Total evidence: {len(r2_evidence)} items."
                    ),
                })
                logger.info(
                    "[LAYER-C] Targeted R2 re-search added %d evidence items. Total: %d",
                    new_evidence_count, len(r2_evidence),
                )

            await ws.send_json({
                "type": "info",
                "stage": "IE",
                "content": (
                    f"🔄 IE recommends revision — "
                    f"{len(ie_json.get('issues', []))} issues found "
                    f"({critical_count} critical). "
                    f"Re-running R3 with feedback (iteration {iteration + 1}/{max_iterations})..."
                ),
            })
            logger.info(
                "[LAYER-C] Iteration %d → ITERATE. Issues: %d (critical: %d). Starting iter %d.",
                iteration, len(ie_json.get("issues", [])), critical_count, iteration + 1,
            )

            # ── IterationController supplementary guard ──
            if not iter_ctrl.should_continue(iteration):
                logger.info("[IterCtrl] Controller says STOP at iter %d — overriding IE ITERATE.", iteration)
                await ws.send_json({
                    "type": "info",
                    "stage": "IE",
                    "content": (
                        f"🧠 Smart controller detected plateau/target met at iter {iteration}. "
                        f"Finalizing despite IE request for more iterations."
                    ),
                })
                break

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: EVIDENCE-BASED TREATMENT PLAN (Two-Phase Architecture)
    #   Diagnosis is now LOCKED. Search for treatment evidence
    #   (including contraindications), then generate a safe plan.
    #   This prevents small models from hallucinating drug protocols.
    # ══════════════════════════════════════════════════════════════
    tx_r3_tokens = 0
    tx_r3_time = 0.0

    primary_dx_name = r3_json.get("primary_diagnosis", {}).get("diagnosis", "")
    primary_dx_conf = r3_json.get("primary_diagnosis", {}).get("confidence", 0.0)
    has_valid_dx = (
        primary_dx_name
        and primary_dx_name.lower() not in ("unknown", "unknown diagnosis", "")
        and primary_dx_conf > 0.10
    )

    if not is_fast and has_valid_dx:
        # ── Phase 4a: Treatment Evidence Search ──────────────────
        await ws.send_json({
            "type": "stage_start",
            "stage": "R2",
            "title": "R2 — Treatment Evidence Search 💊",
            "description": (
                f"Searching treatment guidelines + contraindications "
                f"for '{primary_dx_name}'..."
            ),
        })

        from src.pipeline.stages import _execute_tool_query, _init_tool_dispatch
        _init_tool_dispatch()

        tx_queries = [
            {
                "tool": "search_pubmed",
                "params": {"query": f'"{primary_dx_name}" AND treatment'},
                "rationale": f"Treatment protocols for {primary_dx_name}",
            },
            {
                "tool": "web_search",
                "params": {
                    "query": (
                        f"{primary_dx_name} treatment guidelines "
                        f"contraindicated drugs what NOT to do"
                    ),
                },
                "rationale": f"Treatment guidelines + contraindications for {primary_dx_name}",
            },
            {
                "tool": "search_europe_pmc",
                "params": {
                    "query": f"{primary_dx_name} contraindicated adverse treatment",
                    "max_results": 5,
                },
                "rationale": f"Contraindication evidence for {primary_dx_name}",
            },
            {
                "tool": "search_wikipedia_medical",
                "params": {"disease_name": primary_dx_name},
                "rationale": f"Wikipedia treatment overview for {primary_dx_name}",
            },
        ]

        total_tx = len(tx_queries)
        tx_tasks = []
        for i, q in enumerate(tx_queries):
            tx_tasks.append(
                _execute_tool_query(
                    ws, q["tool"], q["params"], q["rationale"],
                    i + 1, total_tx,
                )
            )
        tx_results = await asyncio.gather(*tx_tasks, return_exceptions=True)
        tx_evidence: list[dict] = []
        for res in tx_results:
            if isinstance(res, dict):
                tx_evidence.append(res)

        tx_evidence_text = _compress_evidence_for_r3(tx_evidence, max_chars=6000)

        tx_sources_with_results = sum(
            1 for e in tx_evidence if e.get("count", 0) > 0
        )
        await ws.send_json({
            "type": "info",
            "stage": "R2",
            "content": (
                f"✓ Treatment evidence search complete: "
                f"{tx_sources_with_results}/{len(tx_evidence)} sources returned data."
            ),
        })
        await ws.send_json({"type": "stage_complete", "stage": "R2"})

        logger.info(
            "[TREATMENT-R2] Searched treatment evidence for '%s': %d sources, %d with results.",
            primary_dx_name, len(tx_evidence), tx_sources_with_results,
        )

        # ── Phase 4b: Treatment Plan Generation ──────────────────
        await ws.send_json({
            "type": "stage_start",
            "stage": "R3",
            "title": "R3 — Treatment Plan Generation 💊",
            "description": (
                f"Generating evidence-based treatment for "
                f"'{primary_dx_name}'... [{r3_model_label}]"
            ),
        })

        current_meds = r1_json.get(
            "patient_summary", {}
        ).get("current_medications", [])

        # ── Build differential safety block (Layer 2) ──────────
        ie_consensus = ie_consensus_tracker.consensus_diagnosis
        ie_consensus_count = ie_consensus_tracker.consensus_count
        ie_total_iters = ie_consensus_tracker.total_iterations

        diff_safety_block = build_differential_safety_block(
            differentials=r3_json.get("updated_diagnoses", []),
            primary_diagnosis=primary_dx_name,
            primary_confidence=primary_dx_conf,
            ie_consensus=ie_consensus,
            ie_consensus_count=ie_consensus_count,
            ie_reasoning=ie_consensus_tracker.latest_reasoning,
            ie_total_iterations=ie_total_iters,
        )

        if diff_safety_block:
            logger.info(
                "[TREATMENT-SAFETY] Differential safety block injected. "
                "IE consensus: %s (%d/%d iters). Dangerous diffs: %d.",
                ie_consensus or 'none', ie_consensus_count, ie_total_iters,
                len([d for d in r3_json.get('updated_diagnoses', [])
                     if d.get('updated_confidence', 0) > 0.20
                     and d.get('diagnosis', '').lower() != primary_dx_name.lower()]),
            )
            await ws.send_json({
                "type": "info",
                "stage": "R3",
                "content": (
                    f"🛡️ Treatment Safety: Injecting differential safety checks. "
                    + (f"IE consensus: '{ie_consensus}' ({ie_consensus_count}/{ie_total_iters} iters). "
                       if ie_consensus else "")
                    + f"Active differentials will be cross-checked against all prescriptions."
                ),
            })

        tx_context = f"""## Confirmed Diagnosis
{primary_dx_name} (confidence: {primary_dx_conf:.0%})

## Diagnosis Reasoning
{json.dumps(r3_json.get("primary_diagnosis", {}).get("reasoning_chain", []), indent=2, ensure_ascii=False)}

## Original Patient Presentation
{patient_text}

## Treatment Evidence (from R2 — USE ONLY THESE)
{tx_evidence_text}

## Patient Current Medications
{json.dumps(current_meds, ensure_ascii=False)}
{diff_safety_block}
Generate a safe, evidence-based treatment plan.
ONLY use drugs mentioned in the Treatment Evidence above.
Check for contraindications before recommending any drug.
If the evidence mentions drugs that should NOT be used for this diagnosis, list them
in contraindication_notes.
"""
        if drug_facts_text:
            tx_context += f"\n## Verified Drug Facts (from RxNorm/OpenFDA)\n{drug_facts_text}\n"

        tx_messages = [
            {"role": "system", "content": R3_TREATMENT_SYSTEM_PROMPT},
            {"role": "user", "content": tx_context},
        ]

        tx_prompt_est = len(tx_context) // 3

        if use_gemini:
            tx_max_tokens = budget.allocate("TX", iteration=1, prompt_tokens=tx_prompt_est, is_groq=True)
            try:
                tx_result = await stream_gemini_completion(
                    ws, gemini_client, tx_messages,
                    stage="R3", max_tokens=tx_max_tokens, use_pro=gemini_use_pro,
                )
            except Exception as exc:
                logger.warning("[TX-R3] Gemini failed (%s), falling back", exc)
                await ws.send_json({
                    "type": "info", "stage": "R3",
                    "content": f"⚠ Gemini unavailable ({exc}), falling back...",
                })
                if groq_client.is_available:
                    tx_result = await stream_groq_completion(
                        ws, groq_client, tx_messages, stage="R3", max_tokens=tx_max_tokens,
                    )
                else:
                    tx_max_tokens = budget.allocate("TX", iteration=1, prompt_tokens=tx_prompt_est, is_groq=False)
                    tx_result = await stream_llm_completion(
                        ws, llm_client, llama_server_url, tx_messages,
                        stage="R3", max_tokens=tx_max_tokens,
                        thinking_enabled=thinking_enabled, budget_managed=True,
                    )
        elif use_groq_r3:
            tx_max_tokens = budget.allocate("TX", iteration=1, prompt_tokens=tx_prompt_est, is_groq=True)
            try:
                tx_result = await stream_groq_completion(
                    ws, groq_client, tx_messages,
                    stage="R3", max_tokens=tx_max_tokens,
                )
            except Exception as exc:
                logger.warning("[TX-R3] Groq failed (%s), falling back to local", exc)
                await ws.send_json({
                    "type": "info", "stage": "R3",
                    "content": f"⚠ Groq unavailable ({exc}), using local model for treatment...",
                })
                tx_max_tokens = budget.allocate("TX", iteration=1, prompt_tokens=tx_prompt_est, is_groq=False)
                tx_result = await stream_llm_completion(
                    ws, llm_client, llama_server_url, tx_messages,
                    stage="R3", max_tokens=tx_max_tokens,
                    thinking_enabled=thinking_enabled,
                    budget_managed=True,
                )
        else:
            tx_max_tokens = budget.allocate("TX", iteration=1, prompt_tokens=tx_prompt_est, is_groq=False)
            tx_result = await stream_llm_completion(
                ws, llm_client, llama_server_url, tx_messages,
                stage="R3", max_tokens=tx_max_tokens,
                thinking_enabled=thinking_enabled,
                budget_managed=True,
            )

        tx_json = parse_json_from_response(tx_result["clean_content"])

        # Report TX usage to budget manager
        budget.report("TX", allocated=tx_max_tokens, used=tx_result.get("completion_tokens", 0))
        await ws.send_json({"type": "token_pool", "data": budget.get_status()})

        # ── Layer 3: Post-treatment safety validation ─────────
        all_differentials = r3_json.get("updated_diagnoses", [])
        if tx_json.get("treatment_plan") and all_differentials:
            safety_result = await validate_treatment_safety(
                treatment_json=tx_json,
                primary_diagnosis=primary_dx_name,
                differentials=all_differentials,
                ie_consensus=ie_consensus,
                ws=ws,
                llm_client=llm_client,
                llama_server_url=llama_server_url,
                groq_client=groq_client,
                use_groq=use_groq_r3,
            )
            if safety_result.get("modified"):
                logger.warning(
                    "[TREATMENT-SAFETY] Treatment plan modified by safety gate. "
                    "Blocked drugs: %s",
                    safety_result.get('blocked_drugs', []),
                )

        # Merge treatment into r3_json
        if tx_json.get("treatment_plan"):
            r3_json["treatment_plan"] = tx_json["treatment_plan"]
            logger.info(
                "[TREATMENT] Merged evidence-based treatment plan for '%s'.",
                primary_dx_name,
            )
        else:
            logger.warning("[TREATMENT] Treatment R3 returned no treatment_plan JSON.")

        # Merge treatment citations (avoid duplicates)
        existing_citations = r3_json.get("citations", [])
        tx_citations = tx_json.get("citations", [])
        if tx_citations:
            existing_pmids = {c.get("pmid") for c in existing_citations}
            for c in tx_citations:
                if c.get("pmid") and c["pmid"] not in existing_pmids:
                    existing_citations.append(c)
            r3_json["citations"] = existing_citations

        tx_r3_tokens = tx_result.get("completion_tokens", 0)
        tx_r3_time = tx_result.get("elapsed", 0.0)

        await ws.send_json({
            "type": "stage_result",
            "stage": "R3",
            "data": {"treatment_plan": r3_json.get("treatment_plan", {})},
            "stats": {
                "tokens": tx_r3_tokens,
                "time": tx_r3_time,
                "tok_per_sec": tx_result.get("tok_per_sec", 0),
                "phase": "treatment",
            },
        })
        await ws.send_json({"type": "stage_complete", "stage": "R3"})

        await ws.send_json({
            "type": "info",
            "stage": "R3",
            "content": (
                f"💊 Evidence-based treatment plan generated for '{primary_dx_name}' "
                f"using {tx_sources_with_results} evidence sources."
            ),
        })

    elif is_fast:
        # Fast mode: R3 already generated treatment in the loop (R3_SYSTEM_PROMPT)
        logger.info("[TREATMENT] Fast mode — treatment was in-loop (single-phase R3).")
    elif not has_valid_dx:
        logger.warning(
            "[TREATMENT] Skipping treatment phase — diagnosis is '%s' at %.2f.",
            primary_dx_name, primary_dx_conf,
        )

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: POST-MORTEM LEARNING LOOP (Test Cases Only)
    # ══════════════════════════════════════════════════════════════
    post_mortem_json = None
    if expected_output:
        await ws.send_json({
            "type": "stage_start",
            "stage": "POST_MORTEM",
            "title": "🎓 Post-Mortem Learning Loop",
            "description": f"Comparing AI diagnosis against Ground Truth [{ie_model_label}]"
        })
        
        from src.llm.prompt_templates import POST_MORTEM_SYSTEM_PROMPT
        pm_context = f"""
## 1. AI Final Diagnosis
{json.dumps(r3_json.get("primary_diagnosis", {}), indent=2, ensure_ascii=False)}

## 2. AI Reasoning Chain
{json.dumps(r3_json.get("primary_diagnosis", {}).get("reasoning_chain", []), indent=2, ensure_ascii=False)}

## 3. GROUND TRUTH (Expected Output)
{json.dumps(expected_output, indent=2, ensure_ascii=False)}
"""
        pm_messages = [
            {"role": "system", "content": POST_MORTEM_SYSTEM_PROMPT},
            {"role": "user", "content": pm_context}
        ]

        try:
            # Re-use IE's model allocation logic (Local or Groq)
            pm_prompt_est = len(pm_context) // 3
            if use_gemini:
                pm_max_tokens = budget.allocate("IE", iteration=1, prompt_tokens=pm_prompt_est, is_groq=True)
                pm_result = await stream_gemini_completion(
                    ws, gemini_client, pm_messages, stage="POST_MORTEM", max_tokens=pm_max_tokens, use_pro=gemini_use_pro
                )
            elif use_groq_ie:
                pm_max_tokens = budget.allocate("IE", iteration=1, prompt_tokens=pm_prompt_est, is_groq=True)
                pm_result = await stream_groq_completion(
                    ws, groq_client, pm_messages, stage="POST_MORTEM", max_tokens=pm_max_tokens,
                )
            else:
                pm_max_tokens = budget.allocate("IE", iteration=1, prompt_tokens=pm_prompt_est, is_groq=False)
                pm_result = await stream_llm_completion(
                    ws, llm_client, llama_server_url, pm_messages, stage="POST_MORTEM",
                    max_tokens=pm_max_tokens, thinking_enabled=thinking_enabled, budget_managed=True
                )
                
            post_mortem_json = parse_json_from_response(pm_result["clean_content"])
            
            await ws.send_json({
                "type": "stage_result",
                "stage": "POST_MORTEM",
                "data": {"post_mortem": post_mortem_json},
            })
            await ws.send_json({"type": "stage_complete", "stage": "POST_MORTEM"})
            
            # Print pearl to console
            pearl = post_mortem_json.get("clinical_pearl", "")
            if pearl:
                logger.info("[POST-MORTEM] 🎓 Clinical Pearl generated: %s", pearl)
                await ws.send_json({
                    "type": "info",
                    "stage": "SUMMARY",
                    "content": f"🎓 Clinical Pearl: {pearl}",
                })
                
        except Exception as exc:
            logger.warning("[POST-MORTEM] Failed: %s", exc)
            await ws.send_json({"type": "error", "content": f"Post-Mortem failed: {exc}"})


    # ══════════════════════════════════════════════════════════════
    # POST-PIPELINE: Store in memory + consolidation
    # ══════════════════════════════════════════════════════════════
    try:
        primary_dx = r3_json.get("primary_diagnosis", {}).get("diagnosis", "Unknown")
        r3_conf = r3_json.get("primary_diagnosis", {}).get("confidence", 0.0)
        ie_conf = ie_json.get("confidence", 0.0)
        ie_dec = ie_json.get("decision", "FINALIZE")
        ie_issues_list = ie_json.get("issues", [])

        memory.store_case(
            patient_text=patient_text,
            primary_diagnosis=primary_dx,
            r3_confidence=r3_conf,
            ie_decision=ie_dec,
            ie_confidence=ie_conf,
            ie_issues=ie_issues_list,
            iteration_count=final_iteration,
            r1_model=r1_model_label,
            r3_model=r3_model_label,
        )

        if memory.should_consolidate():
            consolidation_stats = memory.consolidate()
            if consolidation_stats["cases_processed"] > 0:
                await ws.send_json({
                    "type": "info",
                    "stage": "IE",
                    "content": (
                        f"🧠 Memory consolidation: "
                        f"{consolidation_stats['cases_processed']} cases distilled → "
                        f"{consolidation_stats['patterns_created']} new patterns, "
                        f"{consolidation_stats['patterns_reinforced']} reinforced, "
                        f"{consolidation_stats['principles_promoted']} promoted to core."
                    ),
                })
                logger.info("[MEMORY] Consolidation: %s", consolidation_stats)
    except Exception as exc:
        logger.warning("[MEMORY] Failed to store case: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    total_elapsed = round(time.time() - total_start, 1)

    total_r3_tokens = sum(h["r3_tokens"] for h in iteration_history) + tx_r3_tokens
    total_r3_time = sum(h["r3_time"] for h in iteration_history) + tx_r3_time
    total_ie_tokens = sum(h["ie_tokens"] for h in iteration_history)
    total_ie_time = sum(h["ie_time"] for h in iteration_history)

    final_summary = {
        "primary_diagnosis": r3_json.get("primary_diagnosis", {}),
        "differential_diagnoses": r3_json.get("updated_diagnoses", r1_json.get("differential_diagnoses", [])),
        "treatment_plan": r3_json.get("treatment_plan", {}),
        "evaluation": {
            "decision": ie_json.get("decision", "FINALIZE"),
            "confidence": ie_json.get("confidence", 0.0),
            "reasoning": ie_json.get("reasoning", ie_json.get("self_critique", "")),
            "issues": ie_json.get("issues", [])
        },
        "post_mortem": post_mortem_json,  # Injected Ground Truth Evaluation
        "zebra_flags": [
            {"disease": z.disease, "icd11": z.icd11, "confidence": z.confidence,
             "key_question": z.key_question}
            for z in zebra_matches
        ] if zebra_matches else [],
        "mode": "fast" if is_fast else "thinking",
        "r1_model": r1_model_label,
        "r3_model": r3_model_label,
        "total_time": total_elapsed,
        "iterations": final_iteration,
        "iteration_history": iteration_history,
        "memory_stats": memory.get_stats(),
        "token_budget": budget.get_summary(),
        "stages": {
            "R1": {
                "time": r1_result["elapsed"],
                "tokens": r1_result["completion_tokens"],
                "tok_s": r1_result["tok_per_sec"],
            },
            "R3": {
                "time": round(total_r3_time, 2),
                "tokens": total_r3_tokens,
                "tok_s": round(total_r3_tokens / total_r3_time, 1) if total_r3_time > 0 else 0,
            },
            "IE": {
                "time": round(total_ie_time, 2),
                "tokens": total_ie_tokens,
                "tok_s": round(total_ie_tokens / total_ie_time, 1) if total_ie_time > 0 else 0,
            },
        },
    }

    # ── Pipeline summary notification ──
    primary_dx_name = r3_json.get("primary_diagnosis", {}).get("diagnosis", "Unknown")
    primary_dx_conf = r3_json.get("primary_diagnosis", {}).get("confidence", 0.0)
    ie_final_dec = ie_json.get("decision", "FINALIZE")
    r2_re_label = " + R2 re-search" if r2_research_done else ""

    pipeline_summary_msg = (
        f"📋 Pipeline complete in {total_elapsed}s | "
        f"{final_iteration} iteration(s){r2_re_label} | "
        f"Dx: {primary_dx_name} ({primary_dx_conf:.0%}) | "
        f"IE: {ie_final_dec} | "
        f"R1: {r1_result['elapsed']:.1f}s, R3: {total_r3_time:.1f}s, IE: {total_ie_time:.1f}s"
    )
    await ws.send_json({
        "type": "info",
        "stage": "SUMMARY",
        "content": pipeline_summary_msg,
    })
    logger.info("[PIPELINE] %s", pipeline_summary_msg)

    await ws.send_json({
        "type": "final_result",
        "data": final_summary,
    })
