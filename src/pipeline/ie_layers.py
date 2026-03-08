"""Layered IE Evaluation — decomposes IE into focused micro-checks.

Instead of one monolithic prompt with 9 checks (which the 4B model struggles
to handle in a single context window), this module splits IE into 3 layers:

  Layer A: Hallucination + Symptom Coverage (critical accuracy checks)
  Layer B: Evidence + Treatment Safety (citation & drug verification)
  Layer C: Decision Synthesis (integrates A+B findings → FINALIZE/ITERATE)

Each layer gets a focused prompt (~300 tokens system) and produces simple JSON.
Layer C receives the outputs of A+B and makes the final decision.

This REPLACES the monolithic IE call when layered_ie=True.
The original IE prompt is still used as fallback.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("rrrie-cdss")

# ═══════════════════════════════════════════════════════════════
# Layer A: Hallucination + Symptom Coverage
# ═══════════════════════════════════════════════════════════════

LAYER_A_SYSTEM = """You are a medical fact-checker. Check for HALLUCINATIONS, SYMPTOM COVERAGE, and DIAGNOSTIC CATEGORY GAPS.

═══ GROUNDING RULES (MANDATORY — apply BEFORE any hallucination check) ═══
1. TRIAGE VITALS ARE VERIFIED DATA. Age, sex, blood pressure, heart rate, temperature,
   and any structured data from "TRIAGE VITALS" section are MEDICAL FACTS recorded by
   clinical staff. If R1 restates them (even in different language or format, e.g.
   "24 y, Kadın" → "24-year-old female"), that is NOT a hallucination.
2. PARAPHRASING is NOT HALLUCINATION. If R1 restates a patient's words in clinical
   language (e.g. patient says "I fainted" → R1 writes "Syncope", or patient says
   "I haven't eaten" → R1 writes "History of fasting"), that is clinical interpretation,
   NOT fabrication.
3. TRANSLATION is NOT HALLUCINATION. If the patient speaks in one language (e.g. Turkish)
   and R1 reports findings in English, matching concepts are valid clinical mappings.
   Example: "açlıktan tansiyonum düştü" = "low blood pressure from hunger" — NOT hallucination.
4. A true HALLUCINATION is information that has ZERO basis in the patient text, triage data,
   or reasonable clinical inference from stated symptoms. Only flag something as hallucination
   if you cannot find ANY related phrase in the original text or triage data.
5. Before listing ANY hallucination, mentally verify: "Is there ANY phrase in the Original
   Patient Text or Triage Vitals that could be the source of this finding?" If YES → NOT
   a hallucination. Only if the answer is definitively NO → flag it.
═══════════════════════════════════════════════════════════════════════════

TASK 1 — HALLUCINATION CHECK:
Compare "R1 Reported Findings" (key_positives) against "Original Patient Text" AND "Triage Vitals".
Apply GROUNDING RULES above. Only flag findings with ZERO basis in the text or triage data.

TASK 2 — SYMPTOM COVERAGE:
For each patient symptom, is it explained by the primary diagnosis?
If a MAJOR symptom is unexplained → that is a coverage gap.

TASK 3 — DIAGNOSTIC CATEGORY AUDIT (demographic-aware):
Given the patient's demographics (age, sex, reproductive status), check if ALL relevant
medical specialty categories are represented in the differential diagnosis:
  Categories: Cardiovascular, Pulmonary, GI, GU/Gynecological, Neurological,
              Hematological, Endocrine, Infectious, Surgical, Toxicological
Rules:
- If the patient's demographics make a specific category HIGHLY RELEVANT
  (e.g., reproductive-age female → GU/GYN MUST be considered,
   elderly on anticoagulants → Hematological/Hemorrhagic MUST be considered,
   child with abdominal pain → Surgical/Pediatric MUST be considered)
  and ZERO diagnoses from that category appear → this is a CRITICAL gap.
- A missing demographic-relevant category means the diagnostic FRAME may be wrong.

OUTPUT — valid JSON only:
{
  "hallucinations": ["finding NOT in patient text"],
  "unexplained_symptoms": ["symptom not covered by diagnosis"],
  "missing_categories": ["medical specialty category that SHOULD be represented but is NOT"],
  "history_complete": true,
  "has_critical_issue": true
}"""

# ═══════════════════════════════════════════════════════════════
# Layer B: Evidence + Treatment Safety
# ═══════════════════════════════════════════════════════════════

LAYER_B_SYSTEM = """You are a medical evidence reviewer. Check EVIDENCE quality and TREATMENT SAFETY only.

TASK 1 — CITATION CHECK:
Compare R3's cited PMIDs against R2's actual PMIDs.
Any PMID cited by R3 that is NOT in R2's actual list → FABRICATED citation.
If R2 returned zero articles → R3's citations MUST be empty.

TASK 2 — TREATMENT SAFETY:
Check drug doses, contraindications, and cofactor ordering.
If patient has nutritional risk → thiamine MUST come before glucose.
If a drug could worsen the patient's condition → flag it.

TASK 3 — DIFFERENTIAL ACCOUNTABILITY:
Did R3 address R1's top 3 differential diagnoses?
Any silently dropped differential → flag it.

OUTPUT — valid JSON only:
{
  "fabricated_citations": ["PMID not in R2"],
  "treatment_issues": ["specific safety concern"],
  "dropped_differentials": ["dx from R1 that R3 ignored"],
  "has_critical_issue": true
}"""

# ═══════════════════════════════════════════════════════════════
# Layer C: Decision Synthesis
# ═══════════════════════════════════════════════════════════════

LAYER_C_SYSTEM = """You are the final quality gate for a clinical assessment.

You receive the results of two prior checks:
- Layer A: hallucination + symptom coverage + diagnostic category audit results
- Layer B: evidence + treatment safety results

Based on these findings, make the FINAL decision.

═══ LETHAL FOCUS RULE (MOST IMPORTANT — READ FIRST) ═══
Your job is NOT to make the assessment academically perfect or textbook-complete.
Your SOLE purpose is: "Will the current diagnosis and plan KILL or SERIOUSLY HARM the patient?"

- Missing diagnostic CATEGORIES alone do NOT warrant ITERATE.
  A well-supported diagnosis that explains ALL major symptoms is CORRECT even if the
  differential list doesn't cover every medical specialty (Cardiovascular, Neurological, etc.).
- "Tunnel vision" is only a problem if it MISSES A LETHAL ALTERNATIVE that explains
  symptoms BETTER. If the primary diagnosis already explains all major findings with
  confidence ≥ 0.70 → category gaps are MINOR documentation issues, not diagnostic failures.
- Do NOT force ITERATE just because some organ-system categories are unrepresented.
  Forcing the model to consider irrelevant categories DESTROYS correct diagnoses.
- ITERATE only when you identify a CONCRETE threat to patient safety:
  (a) Hallucinated findings that change the clinical picture
  (b) A MAJOR symptom that the primary diagnosis CANNOT explain
  (c) A wrong drug or lethal contraindication
  (d) A clearly BETTER diagnosis being ignored despite strong evidence
═══════════════════════════════════════════════════════════

RULES:
- If ANY hallucination found → decision = "ITERATE"
- If ANY major symptom is genuinely unexplained by the primary diagnosis → decision = "ITERATE"
- If ANY fabricated citation → decision = "ITERATE"
- If ANY critical treatment safety issue → decision = "ITERATE"
- If primary diagnosis is "Unknown" or 0% → decision = "ITERATE" and suggest a diagnosis
- If missing categories but diagnosis is well-supported (confidence ≥ 0.70) and
  all major symptoms are explained → these are MINOR issues, decision = "FINALIZE"
- If only minor issues → decision = "FINALIZE"
- Set confidence based on severity of issues found (0.0-1.0)

OUTPUT — valid JSON only:
{
  "issues": [
    {"severity": "critical|major|minor", "type": "string", "detail": "string"}
  ],
  "confidence": 0.0,
  "decision": "FINALIZE|ITERATE",
  "reasoning": "1-2 sentence justification",
  "suggested_diagnosis": null,
  "alternative_framework": null
}"""


def _build_layer_a_context(
    patient_text: str,
    r1_json: dict,
    r3_json: dict,
) -> str:
    """Build focused context for Layer A (hallucination + symptom coverage)."""
    import re as _re

    # Truncate patient text intelligently
    pt = patient_text.strip()
    if len(pt) > 1800:
        pt = pt[:1200] + "\n[...]\n" + pt[-500:]

    # ── Extract triage vitals from patient text for explicit grounding ──
    triage_facts: list[str] = []
    # Age patterns: "24 y", "24 yaş", "24-year-old", "Age: 24"
    age_m = _re.search(r'(\d{1,3})\s*(?:y(?:aş)?(?:ında)?|year[s\-]?\s*old|yo\b)', pt, _re.IGNORECASE)
    if age_m:
        triage_facts.append(f"Age: {age_m.group(1)}")
    # Sex patterns: "Kadın", "Erkek", "Female", "Male", "F/M"
    sex_m = _re.search(r'\b(Kadın|Erkek|Female|Male|Kadin)\b', pt, _re.IGNORECASE)
    if sex_m:
        sex_val = sex_m.group(1).capitalize()
        if sex_val in ("Kadın", "Kadin"):
            sex_val = "Female (Kadın)"
        elif sex_val == "Erkek":
            sex_val = "Male (Erkek)"
        triage_facts.append(f"Sex: {sex_val}")
    # BP: "85/50", "Tansiyon: 85/50"
    bp_m = _re.search(r'(?:Tansiyon|BP|Blood\s*Pressure)[:\s]*(\d{2,3}/\d{2,3})', pt, _re.IGNORECASE)
    if not bp_m:
        bp_m = _re.search(r'\b(\d{2,3}/\d{2,3})\s*mmHg', pt, _re.IGNORECASE)
    if bp_m:
        triage_facts.append(f"BP: {bp_m.group(1)} mmHg")
    # HR: "128/dk", "Nabız: 128", "HR: 128"
    hr_m = _re.search(r'(?:Nab[ıi]z|HR|Heart\s*Rate|Pulse)[:\s]*(\d{2,3})', pt, _re.IGNORECASE)
    if hr_m:
        triage_facts.append(f"HR: {hr_m.group(1)} bpm")
    # Temp: "36.4°C", "Ateş: 36.4"
    temp_m = _re.search(r'(?:Ate[şs]|Temp|Temperature)[:\s]*([\d.]+)\s*°?[CF]?', pt, _re.IGNORECASE)
    if temp_m:
        triage_facts.append(f"Temp: {temp_m.group(1)}°C")

    triage_section = ""
    if triage_facts:
        triage_section = (
            "\n\n## TRIAGE VITALS (VERIFIED MEDICAL DATA — NOT hallucination)\n"
            + "\n".join(f"  ✓ {f}" for f in triage_facts)
            + "\nThese are FACTS. R1 restating them in any language/format is NOT a hallucination."
        )

    r1_summary = r1_json.get("patient_summary", {})
    positives = r1_summary.get("key_positives", [])

    primary = r3_json.get("primary_diagnosis", {})
    dx_name = primary.get("diagnosis", "Unknown")
    dx_conf = primary.get("confidence", 0.0)
    explained = primary.get("explains_symptoms", [])
    unexplained = primary.get("unexplained_symptoms", [])

    return f"""## Original Patient Text
{pt}{triage_section}

## R1 Reported Findings (key_positives)
{json.dumps(positives[:12], ensure_ascii=False)}

## Primary Diagnosis: {dx_name} ({dx_conf:.0%})
Explains: {', '.join(explained[:8]) if explained else 'not listed'}
Does NOT explain: {', '.join(unexplained[:5]) if unexplained else 'none listed'}

## Relevant History
{r1_summary.get('relevant_history', 'Not recorded')}

Check for hallucinations (applying GROUNDING RULES) and symptom coverage gaps."""


def _build_layer_b_context(
    r1_json: dict,
    r3_json: dict,
    r2_evidence: list[dict],
    drug_facts: str = "",
    paradox_text: str = "",
) -> str:
    """Build focused context for Layer B (evidence + treatment + differentials)."""
    # R3 citations
    citations = r3_json.get("citations", [])
    cited_pmids = [c.get("pmid", "?") for c in citations[:5]]

    # R2 actual PMIDs
    real_pmids = set()
    for e in r2_evidence:
        for a in e.get("articles", []):
            pmid = a.get("pmid", "")
            if pmid:
                real_pmids.add(str(pmid))
        for p in e.get("papers", []):
            pid = p.get("paperId", p.get("pmid", ""))
            if pid:
                real_pmids.add(str(pid))

    # Treatment summary
    treatment = r3_json.get("treatment_plan", {})
    tx_lines = []
    for drug in treatment.get("pharmacological", [])[:5]:
        d = drug.get("drug", "?")
        dose = drug.get("dose", "?")
        route = drug.get("route", "?")
        tx_lines.append(f"- {d} {dose} ({route})")
    for action in treatment.get("immediate_actions", [])[:3]:
        tx_lines.append(f"- URGENT: {action}")

    # R1 differentials
    r1_diffs = r1_json.get("differential_diagnoses", [])
    diff_lines = []
    for d in r1_diffs[:3]:
        diff_lines.append(f"- #{d.get('rank', '?')}: {d.get('diagnosis', '?')} ({d.get('confidence', 0):.0%})")

    # R3 updated differentials
    r3_updated = r3_json.get("updated_diagnoses", [])
    r3_diff_names = [d.get("diagnosis", "").lower() for d in r3_updated]

    sections = []
    sections.append(f"## R3 Cited PMIDs: {', '.join(cited_pmids) if cited_pmids else 'none'}")
    sections.append(f"## R2 Actual PMIDs: {', '.join(sorted(real_pmids)) if real_pmids else 'NONE — zero articles found'}")

    if tx_lines:
        sections.append(f"## Treatment Plan\n" + "\n".join(tx_lines))

    if drug_facts:
        compact = [l.strip() for l in drug_facts.split("\n") if l.strip().startswith(("💊", "Class:", "⚠️", "🚫"))]
        if compact:
            sections.append("## Drug Facts\n" + "\n".join(compact[:6]))

    if paradox_text:
        sections.append(paradox_text)

    sections.append(f"## R1 Top 3 Differentials\n" + "\n".join(diff_lines))
    sections.append(f"## R3 Addressed Diagnoses: {', '.join(r3_diff_names[:5]) if r3_diff_names else 'none listed'}")

    return "\n\n".join(sections) + "\n\nCheck citations, treatment safety, and differential accountability."


def _build_layer_c_context(
    layer_a_json: dict,
    layer_b_json: dict,
    primary_diagnosis: str,
    confidence: float,
) -> str:
    """Build context for Layer C (decision synthesis from A+B results)."""
    missing_cats = layer_a_json.get("missing_categories", [])
    missing_section = ""
    if missing_cats:
        missing_section = f"\n- Missing diagnostic categories: {json.dumps(missing_cats, ensure_ascii=False)}"

    return f"""## Primary Diagnosis: {primary_diagnosis} ({confidence:.0%})

## Layer A Results (Hallucination + Symptom Coverage + Category Audit)
- Hallucinations found: {json.dumps(layer_a_json.get('hallucinations', []), ensure_ascii=False)}
- Unexplained symptoms: {json.dumps(layer_a_json.get('unexplained_symptoms', []), ensure_ascii=False)}{missing_section}
- History complete: {layer_a_json.get('history_complete', True)}
- Has critical issue: {layer_a_json.get('has_critical_issue', False)}

## Layer B Results (Evidence + Treatment Safety)
- Fabricated citations: {json.dumps(layer_b_json.get('fabricated_citations', []), ensure_ascii=False)}
- Treatment issues: {json.dumps(layer_b_json.get('treatment_issues', []), ensure_ascii=False)}
- Dropped differentials: {json.dumps(layer_b_json.get('dropped_differentials', []), ensure_ascii=False)}
- Has critical issue: {layer_b_json.get('has_critical_issue', False)}

Synthesize findings and decide: FINALIZE or ITERATE?
If diagnosis is wrong or unknown, suggest the correct diagnosis."""


async def run_layered_ie(
    *,
    patient_text: str,
    r1_json: dict,
    r3_json: dict,
    r2_evidence: list[dict],
    drug_facts: str = "",
    paradox_text: str = "",
    call_fn: Any,
    budget_allocate: Any,
    iteration: int = 1,
) -> dict:
    """Run layered IE evaluation (3 micro-checks instead of 1 monolithic).

    Args:
        patient_text: Original patient text.
        r1_json: Parsed R1 output.
        r3_json: Parsed R3 output.
        r2_evidence: R2 evidence list.
        drug_facts: Verified drug facts string.
        paradox_text: Paradox summary string.
        call_fn: Async callable(messages, max_tokens) → dict with 'clean_content'.
        budget_allocate: Callable(stage, iteration, prompt_tokens) → int max_tokens.
        iteration: Current iteration number.

    Returns:
        Combined IE JSON result (same schema as monolithic IE).
    """
    from src.pipeline.streaming import parse_json_from_response

    primary = r3_json.get("primary_diagnosis", {})
    dx_name = primary.get("diagnosis", "Unknown")
    dx_conf = primary.get("confidence", 0.0)

    # ── Layer A: Hallucination + Symptom Coverage ──
    a_context = _build_layer_a_context(patient_text, r1_json, r3_json)
    a_messages = [
        {"role": "system", "content": LAYER_A_SYSTEM},
        {"role": "user", "content": a_context},
    ]
    a_max = budget_allocate("IE", iteration=iteration, prompt_tokens=len(a_context) // 3)
    a_result = await call_fn(a_messages, min(a_max, 800))
    layer_a_json = parse_json_from_response(a_result["clean_content"])
    if not layer_a_json:
        layer_a_json = {"hallucinations": [], "unexplained_symptoms": [], "history_complete": True, "has_critical_issue": False}

    logger.info(
        "[IE-LAYER-A] Hallucinations=%d, Unexplained=%d, Critical=%s",
        len(layer_a_json.get("hallucinations", [])),
        len(layer_a_json.get("unexplained_symptoms", [])),
        layer_a_json.get("has_critical_issue", False),
    )

    # ── Layer B: Evidence + Treatment + Differentials ──
    b_context = _build_layer_b_context(r1_json, r3_json, r2_evidence, drug_facts, paradox_text)
    b_messages = [
        {"role": "system", "content": LAYER_B_SYSTEM},
        {"role": "user", "content": b_context},
    ]
    b_max = budget_allocate("IE", iteration=iteration, prompt_tokens=len(b_context) // 3)
    b_result = await call_fn(b_messages, min(b_max, 800))
    layer_b_json = parse_json_from_response(b_result["clean_content"])
    if not layer_b_json:
        layer_b_json = {"fabricated_citations": [], "treatment_issues": [], "dropped_differentials": [], "has_critical_issue": False}

    logger.info(
        "[IE-LAYER-B] FabCitations=%d, TxIssues=%d, DroppedDiffs=%d, Critical=%s",
        len(layer_b_json.get("fabricated_citations", [])),
        len(layer_b_json.get("treatment_issues", [])),
        len(layer_b_json.get("dropped_differentials", [])),
        layer_b_json.get("has_critical_issue", False),
    )

    # ── Layer C: Decision Synthesis ──
    c_context = _build_layer_c_context(layer_a_json, layer_b_json, dx_name, dx_conf)
    c_messages = [
        {"role": "system", "content": LAYER_C_SYSTEM},
        {"role": "user", "content": c_context},
    ]
    c_max = budget_allocate("IE", iteration=iteration, prompt_tokens=len(c_context) // 3)
    c_result = await call_fn(c_messages, min(c_max, 600))
    layer_c_json = parse_json_from_response(c_result["clean_content"])
    if not layer_c_json:
        layer_c_json = {
            "issues": [], "confidence": 0.5, "decision": "ITERATE",
            "reasoning": "IE layer C failed to parse — defaulting to ITERATE.",
            "suggested_diagnosis": None,
        }

    # ── Merge all layer results into standard IE output ──
    # Build symptom_audit from Layer A
    symptom_audit = []
    for s in layer_a_json.get("unexplained_symptoms", []):
        symptom_audit.append({"symptom": s, "explained": False, "by_diagnosis": None})

    # Merge issues from all layers
    all_issues = list(layer_c_json.get("issues", []))

    # Add Layer A findings as issues if not already in C
    for h in layer_a_json.get("hallucinations", []):
        all_issues.append({"severity": "critical", "type": "hallucination", "detail": f"Hallucinated finding: {h}"})
    for s in layer_a_json.get("unexplained_symptoms", []):
        # Only add if not already covered in C's issues
        if not any(s.lower() in i.get("detail", "").lower() for i in all_issues):
            all_issues.append({"severity": "critical", "type": "unexplained_symptom", "detail": f"Unexplained: {s}"})
    for mc in layer_a_json.get("missing_categories", []):
        all_issues.append({"severity": "minor", "type": "missing_category", "detail": f"Missing diagnostic category: {mc}"})

    # Add Layer B findings
    for fc in layer_b_json.get("fabricated_citations", []):
        all_issues.append({"severity": "critical", "type": "hallucination", "detail": f"Fabricated PMID: {fc}"})
    for ti in layer_b_json.get("treatment_issues", []):
        all_issues.append({"severity": "critical", "type": "contraindication", "detail": ti})
    for dd in layer_b_json.get("dropped_differentials", []):
        all_issues.append({"severity": "critical", "type": "dropped_differential", "detail": f"R1 dx dropped: {dd}"})

    # Deduplicate issues by detail text
    seen = set()
    unique_issues = []
    for iss in all_issues:
        key = iss.get("detail", "").lower().strip()
        if key not in seen:
            seen.add(key)
            unique_issues.append(iss)

    # Aggregate token usage from all layers
    total_tokens = (
        a_result.get("completion_tokens", 0) +
        b_result.get("completion_tokens", 0) +
        c_result.get("completion_tokens", 0)
    )
    total_time = (
        a_result.get("elapsed", 0) +
        b_result.get("elapsed", 0) +
        c_result.get("elapsed", 0)
    )

    combined = {
        "symptom_audit": symptom_audit,
        "issues": unique_issues,
        "confidence": layer_c_json.get("confidence", 0.5),
        "decision": layer_c_json.get("decision", "ITERATE"),
        "reasoning": layer_c_json.get("reasoning", ""),
        "suggested_diagnosis": layer_c_json.get("suggested_diagnosis"),
        "questions_for_clinician": [],
        # Aggregated stats
        "_ie_layers": {
            "layer_a": layer_a_json,
            "layer_b": layer_b_json,
            "layer_c": layer_c_json,
        },
        "_completion_tokens": total_tokens,
        "_elapsed": total_time,
    }

    logger.info(
        "[IE-LAYERED] Decision=%s, Confidence=%.2f, Issues=%d, Tokens=%d, Time=%.1fs",
        combined["decision"], combined["confidence"],
        len(unique_issues), total_tokens, total_time,
    )

    return combined
