"""End-to-End Medical Test — Full RRRIE pipeline with REAL WHO ICD-11 + PubMed data.

Runs the entire R1→R2→R3→IE pipeline using:
  - Qwen/Qwen3.5-4B (GGUF Q4_K_M, adaptive thinking) for local inference (R1, R3, IE)
  - Real WHO ICD-11 API for code verification & correction
  - Real NCBI PubMed API for evidence gathering (R2)

Modes:
  Structured (classic JSON cases from tests/test_cases/):
    py tests/test_e2e_medical.py                                    # 3 default structured cases
    py tests/test_e2e_medical.py --all                             # All 23 structured cases
    py tests/test_e2e_medical.py --case pneumonia                  # Single structured case

  Narrative (dynamic patient stories — different each run):
    py tests/test_e2e_medical.py --narrative                       # 3 random narrative cases
    py tests/test_e2e_medical.py --narrative --count 5             # 5 random narrative cases
    py tests/test_e2e_medical.py --narrative --all                 # All 20 narrative cases
    py tests/test_e2e_medical.py --narrative --case rabies         # Specific narrative case
    py tests/test_e2e_medical.py --narrative --hard                # Only hard-difficulty cases

  Mixed:
    py tests/test_e2e_medical.py --mix                             # 2 structured + 3 narrative (random)

    py -m pytest tests/test_e2e_medical.py -v -s -k test_e2e      # Via pytest
"""

from __future__ import annotations

import asyncio
import json
import os
import re as _re
import sys
import time
from pathlib import Path
from typing import Any

# ── Path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports ─────────────────────────────────────────────────────────
from src.llm.llama_cpp_client import LlamaCppClient, LlamaCppChatResponse
from src.llm.prompt_templates import R1_SYSTEM_PROMPT, R3_SYSTEM_PROMPT, IE_SYSTEM_PROMPT
from src.utils.safety_checks import detect_red_flags, check_vitals_red_flags
from src.tools.pubmed_tool import search_pubmed
from src.utils.medical_codes import (
    lookup_icd11_who,
    search_icd11_who,
    is_valid_icd11,
    normalize_icd11,
    COMMON_ICD11,
)

# Narrative case bank — dynamic patient stories
from tests.narrative_cases import (
    NARRATIVE_CASES,
    get_random_cases,
    get_case_by_keyword,
    get_cases_by_difficulty,
    get_mixed_difficulty_set,
    list_all_cases,
)

# ── Constants ───────────────────────────────────────────────────────
CASES_DIR = PROJECT_ROOT / "tests" / "test_cases"

# Default: 3 representative cases (fast run). Use --all for all 23.
DEFAULT_CASES = [
    "pneumonia_case.json",       # Resp — CA40.0
    "sepsis_case.json",          # Infectious — 1G40
    "stroke_case.json",          # Neuro — 8B11
]

ALL_CASES = sorted([
    f.name for f in CASES_DIR.glob("*_case.json") if f.name != "__init__.py"
])


# ═════════════════════════════════════════════════════════════════════
# WHO ICD-11 VERIFICATION
# ═════════════════════════════════════════════════════════════════════

async def verify_icd11_code(code: str, diagnosis_text: str) -> dict:
    """Verify an ICD-11 code via the real WHO API. If invalid, search by diagnosis text.

    Returns:
        Dict with verified_code, who_title, method, valid (bool).
    """
    result = {
        "original_code": code,
        "diagnosis_text": diagnosis_text,
        "verified_code": code,
        "who_title": "",
        "method": "unverified",
        "valid": False,
    }

    # Step 1: Try direct code lookup via WHO codeinfo
    if code and is_valid_icd11(code):
        lookup = await lookup_icd11_who(code)
        if lookup.get("found"):
            result["verified_code"] = lookup["code"]
            result["who_title"] = lookup.get("title", "")
            result["method"] = "codeinfo_lookup"
            result["valid"] = True
            result["definition"] = lookup.get("definition", "")
            result["browserUrl"] = lookup.get("browserUrl", "")
            return result

    # Step 2: Code not found — search WHO by diagnosis text
    if diagnosis_text:
        search_results = await search_icd11_who(diagnosis_text, max_results=3)
        if search_results:
            top = search_results[0]
            result["verified_code"] = top.get("theCode", code)
            result["who_title"] = top.get("title", "")
            result["method"] = "mms_search"
            result["valid"] = True
            result["search_score"] = top.get("score", 0)
            result["alternatives"] = [
                {"code": r["theCode"], "title": r["title"], "score": r["score"]}
                for r in search_results[1:]
            ]
            return result

    # Step 3: Fallback to local COMMON_ICD11 dictionary
    normalized = normalize_icd11(code) if code else None
    if normalized and normalized in COMMON_ICD11:
        result["verified_code"] = normalized
        result["who_title"] = COMMON_ICD11[normalized]
        result["method"] = "local_dict"
        result["valid"] = True
        return result

    return result


async def verify_all_diagnoses(diagnoses: list[dict]) -> list[dict]:
    """Verify/correct ICD-11 codes for all diagnoses in parallel."""
    tasks = []
    for diag in diagnoses:
        code = diag.get("icd11_code", "")
        text = diag.get("diagnosis", "")
        tasks.append(verify_icd11_code(code, text))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    verified = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            verified.append({
                "original_code": diagnoses[i].get("icd11_code", ""),
                "valid": False,
                "error": str(r),
            })
        else:
            verified.append(r)
    return verified


# ═════════════════════════════════════════════════════════════════════
# STAGE RUNNERS
# ═════════════════════════════════════════════════════════════════════

def load_case(filename: str) -> dict:
    """Load a clinical test case from JSON."""
    path = CASES_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_patient_text(patient_data: dict) -> tuple[str, list[str]]:
    """Format patient data into prompt text with red flag detection."""
    text = "\n".join(f"{k}: {v}" for k, v in patient_data.items())

    # Detect red flags
    red_flags = detect_red_flags(
        chief_complaint=patient_data.get("chief_complaint", ""),
        symptoms=patient_data.get("symptoms", []),
    )
    vitals = patient_data.get("vitals")
    if vitals:
        vital_flags = check_vitals_red_flags(
            spo2=vitals.get("spo2"),
            temperature=vitals.get("temperature"),
            heart_rate=vitals.get("heart_rate"),
            respiratory_rate=vitals.get("respiratory_rate"),
            blood_pressure=vitals.get("blood_pressure"),
        )
        red_flags.extend(vital_flags)

    if red_flags:
        text += "\n\n⚠️ DETECTED RED FLAGS:\n" + "\n".join(f"- {f}" for f in red_flags)

    return text, red_flags


def run_r1(client: LlamaCppClient, patient_text: str) -> dict:
    """R1 — Reasoned Analysis: generate differential diagnosis."""
    messages = [
        {"role": "system", "content": R1_SYSTEM_PROMPT},
        {"role": "user", "content": patient_text},
    ]
    response = client.chat(messages=messages, temperature=0.0, max_tokens=1536, json_mode=True, stage="R1")
    return _parse_json_response(response.content, "R1")


async def run_r2_evidence(patient_data: dict, r1_output: dict) -> list[dict]:
    """R2 — Reinforced Fact-Finding: gather real PubMed evidence.

    Directly calls PubMed API for each knowledge gap (bypasses Groq cloud).
    """
    evidence = []
    knowledge_gaps = r1_output.get("knowledge_gaps", [])

    # Always search for the primary diagnosis
    hypotheses = r1_output.get("differential_diagnoses", [])
    if hypotheses:
        primary = hypotheses[0].get("diagnosis", "")
        if primary:
            knowledge_gaps.insert(0, {
                "gap": f"Latest guidelines for {primary}",
                "search_query": f"{primary} treatment guidelines",
                "target_api": "pubmed",
                "importance": "critical",
            })

    # Search PubMed for top 3 knowledge gaps
    for gap in knowledge_gaps[:3]:
        query = gap.get("search_query", gap.get("gap", ""))
        if not query:
            continue
        try:
            result = await search_pubmed(query=query, max_results=3)
            if result and "error" not in result:
                evidence.append({
                    "type": "search_pubmed",
                    "query": {"query": query},
                    "data": result,
                })
        except Exception as exc:
            evidence.append({
                "type": "search_pubmed",
                "query": {"query": query},
                "data": {"error": str(exc)},
            })

    if not evidence:
        evidence.append({
            "type": "llm_summary",
            "content": "No external evidence found. Recommendations based on clinical reasoning only.",
        })

    return evidence


def run_r3(client: LlamaCppClient, r1_output: dict, evidence: list[dict],
           patient_data: dict, who_verified: list[dict]) -> dict:
    """R3 — Reasoning Synthesis: merge R1 hypotheses + R2 evidence + WHO codes."""
    hypotheses = r1_output.get("differential_diagnoses", [])

    parts = []
    parts.append("## Patient Overview")
    parts.append(f"- Age: {patient_data.get('age', 'Unknown')}")
    parts.append(f"- Sex: {patient_data.get('sex', 'Unknown')}")
    parts.append(f"- Chief Complaint: {patient_data.get('chief_complaint', 'Unknown')}")
    if patient_data.get("allergies"):
        parts.append(f"- ALLERGIES: {', '.join(patient_data['allergies'])}")
    if patient_data.get("medications"):
        parts.append(f"- Current Medications: {', '.join(patient_data['medications'])}")
    if patient_data.get("lab_results"):
        labs = patient_data["lab_results"]
        parts.append("- Lab Results:")
        for k, v in labs.items():
            parts.append(f"    {k}: {v}")

    parts.append("\n## R1 Differential Diagnoses (WHO ICD-11 Verified)")
    for i, h in enumerate(hypotheses):
        diag = h.get("diagnosis", "Unknown")
        conf = h.get("confidence", 0)
        original_icd = h.get("icd11_code", "")

        # Use WHO-verified code if available
        who_code = original_icd
        who_title = ""
        if i < len(who_verified) and who_verified[i].get("valid"):
            who_code = who_verified[i]["verified_code"]
            who_title = who_verified[i]["who_title"]

        parts.append(f"- {diag} (WHO ICD-11: {who_code} — {who_title}, Confidence: {conf})")
        if h.get("supporting_factors"):
            parts.append(f"  Supporting: {', '.join(h['supporting_factors'])}")

    parts.append(f"\n## R2 Evidence ({len(evidence)} items)")
    for i, ev in enumerate(evidence, 1):
        ev_type = ev.get("type", "unknown")
        if ev_type == "llm_summary":
            parts.append(f"{i}. [LLM Summary] {ev.get('content', '')[:300]}")
        else:
            data = ev.get("data", {})
            articles = data.get("articles", [])
            parts.append(f"{i}. [PubMed] Query: {ev.get('query', {}).get('query', '')}")
            for art in articles[:2]:
                parts.append(f"   - {art.get('title', 'N/A')}")

    parts.append("\n## TASK")
    parts.append("Synthesize ALL the above into a final clinical assessment.")
    parts.append("Use the WHO-verified ICD-11 codes in your response.")

    messages = [
        {"role": "system", "content": R3_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]
    response = client.chat(messages=messages, temperature=0.0, max_tokens=1792, json_mode=True, stage="R3")
    return _parse_json_response(response.content, "R3")


def run_ie(client: LlamaCppClient, r3_output: dict, evidence: list[dict], patient_data: dict) -> dict:
    """IE — Iterative Evaluation: evaluate the clinical assessment."""
    diagnosis = r3_output.get("primary_diagnosis", {})
    treatment = r3_output.get("treatment_plan", {})

    parts = []
    parts.append("## Patient Data")
    parts.append(f"- Age: {patient_data.get('age')}, Sex: {patient_data.get('sex')}")
    parts.append(f"- Chief Complaint: {patient_data.get('chief_complaint')}")
    if patient_data.get("allergies"):
        parts.append(f"- ALLERGIES: {', '.join(patient_data['allergies'])}")
    if patient_data.get("medications"):
        parts.append(f"- Medications: {', '.join(patient_data['medications'])}")
    if patient_data.get("lab_results"):
        parts.append("- Lab Results:")
        for k, v in patient_data["lab_results"].items():
            parts.append(f"    {k}: {v}")

    parts.append("\n## R3 Assessment")
    if diagnosis:
        parts.append(f"Primary Diagnosis: {diagnosis.get('diagnosis', 'N/A')}")
        parts.append(f"Confidence: {diagnosis.get('confidence', 0)}")
    if treatment:
        parts.append(f"Treatment Plan: {json.dumps(treatment, default=str, ensure_ascii=False)[:1500]}")

    parts.append(f"\n## Evidence Used ({len(evidence)} items)")
    for i, ev in enumerate(evidence[:5], 1):
        parts.append(f"  {i}. [{ev.get('type')}]")

    parts.append("\n## TASK")
    parts.append("Evaluate the clinical assessment using the 8-point checklist. Be RUTHLESS.")

    messages = [
        {"role": "system", "content": IE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]
    response = client.chat(messages=messages, temperature=0.0, max_tokens=1536, json_mode=True, stage="IE")
    return _parse_json_response(response.content, "IE")


# ═════════════════════════════════════════════════════════════════════
# SCORING & EVALUATION
# ═════════════════════════════════════════════════════════════════════

def evaluate_results(case: dict, r1: dict, r3: dict, ie: dict,
                     who_verified: list[dict], detected_red_flags: list[str]) -> dict:
    """Score RRRIE outputs against expected case answers."""
    expected = case.get("expected_output", {})
    expected_diag = expected.get("primary_diagnosis", "").lower()
    expected_codes = [c.upper() for c in expected.get("expected_icd11_codes", [])]
    expected_detections = expected.get("should_detect", [])
    expected_red_flags = expected.get("red_flags", [])

    scores = {
        "diagnosis_match": False,
        "icd11_match": False,
        "who_verified_match": False,
        "red_flags_detected": 0,
        "red_flags_total": len(expected_red_flags),
        "clinical_detections": 0,
        "clinical_detections_total": len(expected_detections),
        "ie_confidence": 0.0,
        "who_codes_valid": 0,
        "who_codes_total": 0,
    }

    # ── Check R1 differential diagnoses ──
    r1_diags = r1.get("differential_diagnoses", [])
    all_diagnoses_text = " ".join(d.get("diagnosis", "").lower() for d in r1_diags)
    all_icd_codes = [d.get("icd11_code", "").upper() for d in r1_diags]

    # ── R3 primary diagnosis ──
    r3_primary = r3.get("primary_diagnosis", {})
    r3_diag_text = r3_primary.get("diagnosis", "").lower() if r3_primary else ""
    r3_icd = r3_primary.get("icd11_code", "").upper() if r3_primary else ""

    # ── Diagnosis match (fuzzy keyword match) ──
    expected_keywords = expected_diag.split()
    if expected_keywords:
        match_count = sum(1 for kw in expected_keywords if kw in all_diagnoses_text or kw in r3_diag_text)
        scores["diagnosis_match"] = match_count >= len(expected_keywords) * 0.5

    # ── ICD-11 code match (model output) ──
    for code in expected_codes:
        if code in all_icd_codes or code == r3_icd:
            scores["icd11_match"] = True
            break

    # ── WHO-verified ICD-11 match ──
    who_codes = [v.get("verified_code", "").upper() for v in who_verified if v.get("valid")]
    scores["who_codes_valid"] = sum(1 for v in who_verified if v.get("valid"))
    scores["who_codes_total"] = len(who_verified)
    for code in expected_codes:
        for wc in who_codes:
            if wc.startswith(code) or code.startswith(wc.split(".")[0]):
                scores["who_verified_match"] = True
                break

    # ── Red flags (from safety checks + R1 output) ──
    r1_red_flags_text = " ".join(
        r1.get("patient_summary", {}).get("red_flags", [])
    ).lower()
    safety_flags_text = " ".join(detected_red_flags).lower()
    for flag in expected_red_flags:
        keywords = flag.lower().split()
        if any(kw in r1_red_flags_text or kw in safety_flags_text or kw in all_diagnoses_text
               for kw in keywords):
            scores["red_flags_detected"] += 1

    # ── Clinical detections ──
    full_text = json.dumps(r1, default=str).lower() + json.dumps(r3, default=str).lower()
    for detection in expected_detections:
        keywords = [w.lower() for w in detection.split() if len(w) > 3]
        if any(kw in full_text for kw in keywords[:max(2, len(keywords) // 2)]):
            scores["clinical_detections"] += 1

    # ── IE confidence ──
    scores["ie_confidence"] = ie.get("confidence", 0.0)

    return scores


# ═════════════════════════════════════════════════════════════════════
# HELPER
# ═════════════════════════════════════════════════════════════════════

def _parse_json_response(content: str, stage: str) -> dict:
    """Parse JSON from model output, handling markdown fences and think blocks."""
    text = content.strip()

    # Remove <think>...</think> blocks (Qwen3 thinking)
    text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()

    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ⚠ {stage} JSON parse error: {e}")
        print(f"  Raw output (first 500 chars): {content[:500]}")
        return {"_parse_error": str(e), "_raw": content[:1000]}


# ═════════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNER
# ═════════════════════════════════════════════════════════════════════

async def run_full_pipeline(case_file: str, client: LlamaCppClient) -> dict:
    """Run the full R1→WHO→R2→R3→IE pipeline on a single case."""
    case = load_case(case_file)
    case_id = case.get("case_id", "?")
    title = case.get("title", case_file)
    patient_data = case["patient_data"]
    expected_icd = case.get("icd11_expected", "")

    print(f"\n{'='*70}")
    print(f"  CASE {case_id}: {title}")
    print(f"  Expected ICD-11: {expected_icd}")
    print(f"{'='*70}")

    t_total = time.time()

    # ── R1: Reasoned Analysis ────────────────────────────────────
    print("\n  ▶ R1 — Reasoned Analysis (Qwen3.5-4B)...")
    t0 = time.time()
    patient_text, detected_red_flags = get_patient_text(patient_data)
    r1_output = run_r1(client, patient_text)
    t_r1 = time.time() - t0

    r1_diags = r1_output.get("differential_diagnoses", [])
    print(f"    ✓ {len(r1_diags)} differential diagnoses ({t_r1:.1f}s)")
    for d in r1_diags[:3]:
        print(f"      {d.get('rank', '?')}. {d.get('diagnosis', '?')} "
              f"(ICD-11: {d.get('icd11_code', '?')}, conf: {d.get('confidence', 0):.2f})")

    if detected_red_flags:
        print(f"    ⚠ Safety red flags: {detected_red_flags}")

    # ── WHO ICD-11 Verification ──────────────────────────────────
    print("\n  ▶ WHO — ICD-11 Code Verification (real WHO API)...")
    t0 = time.time()
    who_verified = await verify_all_diagnoses(r1_diags)
    t_who = time.time() - t0

    valid_count = sum(1 for v in who_verified if v.get("valid"))
    print(f"    ✓ {valid_count}/{len(who_verified)} codes verified ({t_who:.1f}s)")
    for v in who_verified[:5]:
        status = "✅" if v.get("valid") else "❌"
        method = v.get("method", "?")
        orig = v.get("original_code", "?")
        verified = v.get("verified_code", "?")
        who_title = v.get("who_title", "")
        correction = f" → {verified}" if orig != verified and v.get("valid") else ""
        print(f"      {status} {orig}{correction} — {who_title} [{method}]")

    # ── R2: Reinforced Fact-Finding ──────────────────────────────
    print("\n  ▶ R2 — Reinforced Fact-Finding (PubMed API)...")
    t0 = time.time()
    evidence = await run_r2_evidence(patient_data, r1_output)
    t_r2 = time.time() - t0

    total_articles = sum(
        len(e.get("data", {}).get("articles", []))
        for e in evidence if e.get("type") == "search_pubmed"
    )
    print(f"    ✓ {len(evidence)} sources, {total_articles} articles ({t_r2:.1f}s)")
    for ev in evidence:
        if ev.get("type") == "search_pubmed":
            q = ev.get("query", {}).get("query", "")[:50]
            n = len(ev.get("data", {}).get("articles", []))
            print(f"      📄 [{n} articles] {q}")

    # ── R3: Reasoning Synthesis ──────────────────────────────────
    print("\n  ▶ R3 — Reasoning Synthesis (Qwen3.5-4B + WHO codes)...")
    t0 = time.time()
    r3_output = run_r3(client, r1_output, evidence, patient_data, who_verified)
    t_r3 = time.time() - t0

    r3_primary = r3_output.get("primary_diagnosis", {})
    print(f"    ✓ Primary: {r3_primary.get('diagnosis', 'N/A')} "
          f"(conf: {r3_primary.get('confidence', 0)}) ({t_r3:.1f}s)")

    treatment = r3_output.get("treatment_plan", {})
    if treatment:
        imm = treatment.get("immediate_actions", [])
        pharma = treatment.get("pharmacological", [])
        print(f"    ✓ Treatment: {len(imm)} immediate actions, {len(pharma)} medications")

    # ── WHO verification of R3 codes ─────────────────────────────
    r3_diags = r3_output.get("updated_diagnoses", [])
    if r3_diags:
        print("\n  ▶ WHO — R3 Code Re-verification...")
        t0_w2 = time.time()
        r3_who = await verify_all_diagnoses(r3_diags)
        t_who2 = time.time() - t0_w2
        r3_valid = sum(1 for v in r3_who if v.get("valid"))
        print(f"    ✓ {r3_valid}/{len(r3_who)} R3 codes verified ({t_who2:.1f}s)")
    else:
        r3_who = []

    # ── IE: Iterative Evaluation ─────────────────────────────────
    print("\n  ▶ IE — Iterative Evaluation (Qwen3.5-4B)...")
    t0 = time.time()
    ie_output = run_ie(client, r3_output, evidence, patient_data)
    t_ie = time.time() - t0

    ie_conf = ie_output.get("confidence", 0)
    ie_decision = ie_output.get("decision", "?")
    critical = ie_output.get("critical_issues", [])
    major = ie_output.get("major_issues", [])
    print(f"    ✓ Confidence: {ie_conf:.2f} | Decision: {ie_decision} ({t_ie:.1f}s)")
    if critical:
        print(f"    ⚠ Critical issues: {len(critical)}")
        for ci in critical[:3]:
            print(f"      - {ci}")

    checks = ie_output.get("checks", {})
    if checks:
        pass_count = sum(1 for v in checks.values()
                        if isinstance(v, dict) and v.get("status") == "PASS")
        print(f"    ✓ Checklist: {pass_count}/{len(checks)} PASS")

    # ── Scoring ─────────────────────────────────────────────────
    t_total_elapsed = time.time() - t_total
    scores = evaluate_results(case, r1_output, r3_output, ie_output,
                              who_verified, detected_red_flags)

    print(f"\n  ── SCORING ──────────────────────────────────────")
    print(f"    Diagnosis match:      {'✅' if scores['diagnosis_match'] else '❌'}")
    print(f"    ICD-11 (model):       {'✅' if scores['icd11_match'] else '❌'}")
    print(f"    ICD-11 (WHO verified):{'✅' if scores['who_verified_match'] else '❌'}")
    print(f"    WHO codes valid:      {scores['who_codes_valid']}/{scores['who_codes_total']}")
    print(f"    Red flags:            {scores['red_flags_detected']}/{scores['red_flags_total']}")
    print(f"    Clinical detections:  {scores['clinical_detections']}/{scores['clinical_detections_total']}")
    print(f"    IE confidence:        {scores['ie_confidence']:.2f}")
    print(f"    Total time:           {t_total_elapsed:.1f}s")

    return {
        "case_id": case_id,
        "title": title,
        "expected_icd": expected_icd,
        "mode": "structured",
        "times": {"r1": t_r1, "who": t_who, "r2": t_r2, "r3": t_r3, "ie": t_ie, "total": t_total_elapsed},
        "r1_output": r1_output,
        "who_verified": who_verified,
        "r3_output": r3_output,
        "ie_output": ie_output,
        "evidence_count": len(evidence),
        "evidence_articles": total_articles,
        "detected_red_flags": detected_red_flags,
        "scores": scores,
    }


async def run_narrative_pipeline(narrative_case: dict, client: LlamaCppClient) -> dict:
    """Run the full R1→WHO→R2→R3→IE pipeline on a narrative patient story."""
    case_id = narrative_case["case_id"]
    title = narrative_case["title"]
    patient_data = narrative_case["patient_data"]
    narrative = narrative_case["narrative"]
    difficulty = narrative_case.get("difficulty", "?")
    category = narrative_case.get("category", "?")
    expected_icd = narrative_case["expected_output"].get("expected_icd11_codes", ["?"])[0]

    print(f"\n{'='*70}")
    print(f"  NARRATIVE {case_id}: {title}")
    print(f"  Difficulty: {difficulty} | Category: {category}")
    print(f"  Expected ICD-11: {expected_icd}")
    print(f"{'='*70}")
    print(f"  📖 Story: {narrative[:150]}...")

    t_total = time.time()

    # ── R1: Reasoned Analysis (uses NARRATIVE text, not structured) ─
    print("\n  ▶ R1 — Reasoned Analysis (narrative → Qwen3.5-4B)...")
    t0 = time.time()

    # Detect red flags from structured patient_data (supplement info)
    detected_red_flags = detect_red_flags(
        chief_complaint=patient_data.get("chief_complaint", ""),
        symptoms=patient_data.get("symptoms", []),
    )
    vitals = patient_data.get("vitals")
    if vitals:
        vital_flags = check_vitals_red_flags(
            spo2=vitals.get("spo2"),
            temperature=vitals.get("temperature"),
            heart_rate=vitals.get("heart_rate"),
            respiratory_rate=vitals.get("respiratory_rate"),
            blood_pressure=vitals.get("blood_pressure"),
        )
        detected_red_flags.extend(vital_flags)

    # Build R1 input — narrative with any detected red flags
    narrative_input = f"PATIENT PRESENTATION (narrative):\n{narrative}"
    if detected_red_flags:
        narrative_input += "\n\n⚠️ DETECTED RED FLAGS:\n" + "\n".join(f"- {f}" for f in detected_red_flags)

    r1_output = run_r1(client, narrative_input)
    t_r1 = time.time() - t0

    r1_diags = r1_output.get("differential_diagnoses", [])
    print(f"    ✓ {len(r1_diags)} differential diagnoses ({t_r1:.1f}s)")
    for d in r1_diags[:3]:
        print(f"      {d.get('rank', '?')}. {d.get('diagnosis', '?')} "
              f"(ICD-11: {d.get('icd11_code', '?')}, conf: {d.get('confidence', 0):.2f})")

    if detected_red_flags:
        print(f"    ⚠ Safety red flags: {detected_red_flags}")

    # ── WHO ICD-11 Verification ──────────────────────────────────
    print("\n  ▶ WHO — ICD-11 Code Verification...")
    t0 = time.time()
    who_verified = await verify_all_diagnoses(r1_diags)
    t_who = time.time() - t0

    valid_count = sum(1 for v in who_verified if v.get("valid"))
    print(f"    ✓ {valid_count}/{len(who_verified)} codes verified ({t_who:.1f}s)")
    for v in who_verified[:5]:
        status = "✅" if v.get("valid") else "❌"
        orig = v.get("original_code", "?")
        verified = v.get("verified_code", "?")
        who_title = v.get("who_title", "")
        correction = f" → {verified}" if orig != verified and v.get("valid") else ""
        print(f"      {status} {orig}{correction} — {who_title} [{v.get('method', '?')}]")

    # ── R2: PubMed Evidence ──────────────────────────────────────
    print("\n  ▶ R2 — PubMed Evidence Gathering...")
    t0 = time.time()
    evidence = await run_r2_evidence(patient_data, r1_output)
    t_r2 = time.time() - t0

    total_articles = sum(
        len(e.get("data", {}).get("articles", []))
        for e in evidence if e.get("type") == "search_pubmed"
    )
    print(f"    ✓ {len(evidence)} sources, {total_articles} articles ({t_r2:.1f}s)")

    # ── R3: Reasoning Synthesis ──────────────────────────────────
    print("\n  ▶ R3 — Reasoning Synthesis (WHO codes + evidence)...")
    t0 = time.time()
    r3_output = run_r3(client, r1_output, evidence, patient_data, who_verified)
    t_r3 = time.time() - t0

    r3_primary = r3_output.get("primary_diagnosis", {})
    print(f"    ✓ Primary: {r3_primary.get('diagnosis', 'N/A')} "
          f"(conf: {r3_primary.get('confidence', 0)}) ({t_r3:.1f}s)")

    # ── WHO R3 re-verify ─────────────────────────────────────────
    r3_diags = r3_output.get("updated_diagnoses", [])
    if r3_diags:
        print("\n  ▶ WHO — R3 Code Re-verification...")
        t0_w2 = time.time()
        r3_who = await verify_all_diagnoses(r3_diags)
        t_who2 = time.time() - t0_w2
        r3_valid = sum(1 for v in r3_who if v.get("valid"))
        print(f"    ✓ {r3_valid}/{len(r3_who)} R3 codes verified ({t_who2:.1f}s)")
    else:
        r3_who = []

    # ── IE: Iterative Evaluation ─────────────────────────────────
    print("\n  ▶ IE — Iterative Evaluation...")
    t0 = time.time()
    ie_output = run_ie(client, r3_output, evidence, patient_data)
    t_ie = time.time() - t0

    ie_conf = ie_output.get("confidence", 0)
    ie_decision = ie_output.get("decision", "?")
    critical = ie_output.get("critical_issues", [])
    print(f"    ✓ Confidence: {ie_conf:.2f} | Decision: {ie_decision} ({t_ie:.1f}s)")
    if critical:
        for ci in critical[:3]:
            print(f"      ⚠ {ci}")

    # ── Scoring ──────────────────────────────────────────────────
    t_total_elapsed = time.time() - t_total

    # Build a scoring-compatible case dict from narrative_case
    scoring_case = {
        "expected_output": narrative_case["expected_output"],
    }
    scores = evaluate_results(scoring_case, r1_output, r3_output, ie_output,
                              who_verified, detected_red_flags)

    # Extra: check if the correct rare diagnosis was even considered
    expected_diag = narrative_case["expected_output"]["primary_diagnosis"].lower()
    r1_diag_text = " ".join(d.get("diagnosis", "").lower() for d in r1_diags)
    r3_diag_text = (r3_primary.get("diagnosis", "") if r3_primary else "").lower()
    rare_detected = any(
        kw in r1_diag_text or kw in r3_diag_text
        for kw in expected_diag.split() if len(kw) > 3
    )
    scores["rare_diagnosis_detected"] = rare_detected

    print(f"\n  ── SCORING ──────────────────────────────────────")
    print(f"    Rare diagnosis detected: {'✅' if rare_detected else '❌'} ({expected_diag})")
    print(f"    Diagnosis match:         {'✅' if scores['diagnosis_match'] else '❌'}")
    print(f"    ICD-11 (WHO verified):   {'✅' if scores['who_verified_match'] else '❌'}")
    print(f"    WHO codes valid:         {scores['who_codes_valid']}/{scores['who_codes_total']}")
    print(f"    Red flags:               {scores['red_flags_detected']}/{scores['red_flags_total']}")
    print(f"    Clinical detections:     {scores['clinical_detections']}/{scores['clinical_detections_total']}")
    print(f"    IE confidence:           {scores['ie_confidence']:.2f}")
    print(f"    Total time:              {t_total_elapsed:.1f}s")

    return {
        "case_id": case_id,
        "title": title,
        "expected_icd": expected_icd,
        "mode": "narrative",
        "difficulty": difficulty,
        "category": category,
        "times": {"r1": t_r1, "who": t_who, "r2": t_r2, "r3": t_r3, "ie": t_ie, "total": t_total_elapsed},
        "r1_output": r1_output,
        "who_verified": who_verified,
        "r3_output": r3_output,
        "ie_output": ie_output,
        "evidence_count": len(evidence),
        "evidence_articles": total_articles,
        "detected_red_flags": detected_red_flags,
        "scores": scores,
    }


# ═════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict]) -> None:
    """Print aggregate summary of all test results."""
    structured = [r for r in results if r.get("mode") == "structured"]
    narrative = [r for r in results if r.get("mode") == "narrative"]

    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS — {len(results)} Cases")
    if structured:
        print(f"    Structured: {len(structured)}")
    if narrative:
        print(f"    Narrative:  {len(narrative)} (dynamic)")
    print(f"  Data Sources: WHO ICD-11 API (real) + NCBI PubMed (real)")
    print(f"{'='*70}")

    n = len(results)
    total_diag = sum(1 for r in results if r["scores"]["diagnosis_match"])
    total_icd_model = sum(1 for r in results if r["scores"]["icd11_match"])
    total_icd_who = sum(1 for r in results if r["scores"]["who_verified_match"])
    total_who_valid = sum(r["scores"]["who_codes_valid"] for r in results)
    total_who_total = sum(r["scores"]["who_codes_total"] for r in results)
    total_rf_det = sum(r["scores"]["red_flags_detected"] for r in results)
    total_rf = sum(r["scores"]["red_flags_total"] for r in results)
    total_cd_det = sum(r["scores"]["clinical_detections"] for r in results)
    total_cd = sum(r["scores"]["clinical_detections_total"] for r in results)
    avg_conf = sum(r["scores"]["ie_confidence"] for r in results) / max(n, 1)
    avg_time = sum(r["times"]["total"] for r in results) / max(n, 1)
    total_articles = sum(r["evidence_articles"] for r in results)

    print(f"\n  {'Metric':<35} {'Score':>12}")
    print(f"  {'─'*49}")
    print(f"  {'Diagnosis Accuracy':<35} {total_diag}/{n:>9}")
    print(f"  {'ICD-11 Model Output Match':<35} {total_icd_model}/{n:>9}")
    print(f"  {'ICD-11 WHO Verified Match':<35} {total_icd_who}/{n:>9}")
    print(f"  {'WHO API Code Validation':<35} {total_who_valid}/{total_who_total:>9}")
    print(f"  {'Red Flag Detection':<35} {total_rf_det}/{total_rf:>9}")
    print(f"  {'Clinical Detection':<35} {total_cd_det}/{total_cd:>9}")
    print(f"  {'Avg IE Confidence':<35} {avg_conf:>12.2f}")
    print(f"  {'PubMed Articles Retrieved':<35} {total_articles:>12}")
    print(f"  {'Avg Time / Case':<35} {avg_time:>10.1f}s")

    print(f"\n  Per-Case Breakdown:")
    print(f"  {'Case':<10} {'Mode':<5} {'Diag':<6} {'ICD':<6} {'WHO':<6} {'RF':<6} {'CD':<6} {'Conf':<6} {'Time':>6}")
    print(f"  {'─'*63}")
    for r in results:
        s = r["scores"]
        mode_tag = "N" if r.get("mode") == "narrative" else "S"
        print(
            f"  {r['case_id']:<10} "
            f"{mode_tag:<5} "
            f"{'✅' if s['diagnosis_match'] else '❌':<6} "
            f"{'✅' if s['icd11_match'] else '❌':<6} "
            f"{'✅' if s['who_verified_match'] else '❌':<6} "
            f"{s['red_flags_detected']}/{s['red_flags_total']:<4} "
            f"{s['clinical_detections']}/{s['clinical_detections_total']:<4} "
            f"{s['ie_confidence']:<6.2f} "
            f"{r['times']['total']:>5.1f}s"
        )

    # ── Narrative-specific: rare diagnosis detection ──
    if narrative:
        print(f"\n  Narrative Case Analysis (Rare/Tricky Diagnosis Detection):")
        print(f"  {'Case':<10} {'Difficulty':<9} {'Expected Diagnosis':<35} {'Detected':>8}")
        print(f"  {'─'*70}")
        rare_ok = 0
        for r in narrative:
            detected = r["scores"].get("rare_diagnosis_detected", False)
            if detected:
                rare_ok += 1
            expected_diag = r.get("title", "?").split("—")[0].strip()
            diff = r.get("difficulty", "?")
            print(f"  {r['case_id']:<10} {diff:<9} {expected_diag:<35} {'✅' if detected else '❌':>8}")
        print(f"\n  Rare Diagnosis Detection Rate: {rare_ok}/{len(narrative)} "
              f"({rare_ok/len(narrative)*100:.0f}%)" if narrative else "")

    # ── WHO verification detail ──
    print(f"\n  WHO ICD-11 Verification Detail:")
    for r in results:
        print(f"  {r['case_id']} (expected: {r['expected_icd']}):")
        for v in r["who_verified"][:3]:
            status = "✅" if v.get("valid") else "❌"
            print(f"    {status} {v.get('original_code','')} → "
                  f"{v.get('verified_code','')} | {v.get('who_title','')} [{v.get('method','')}]")

    # Save detailed results to JSON
    output_path = PROJECT_ROOT / "tests" / "e2e_results.json"
    slim_results = []
    for r in results:
        slim = {
            "case_id": r["case_id"],
            "title": r["title"],
            "mode": r.get("mode", "structured"),
            "difficulty": r.get("difficulty", ""),
            "category": r.get("category", ""),
            "expected_icd": r["expected_icd"],
            "times": r["times"],
            "scores": r["scores"],
            "r1_diagnoses": r["r1_output"].get("differential_diagnoses", [])[:5],
            "who_verified": [
                {
                    "original": v.get("original_code"),
                    "verified": v.get("verified_code"),
                    "title": v.get("who_title"),
                    "method": v.get("method"),
                    "valid": v.get("valid"),
                }
                for v in r["who_verified"]
            ],
            "r3_primary": r["r3_output"].get("primary_diagnosis"),
            "r3_treatment": r["r3_output"].get("treatment_plan"),
            "ie_confidence": r["ie_output"].get("confidence"),
            "ie_decision": r["ie_output"].get("decision"),
            "ie_critical_issues": r["ie_output"].get("critical_issues", []),
            "detected_red_flags": r["detected_red_flags"],
            "evidence_articles": r["evidence_articles"],
        }
        slim_results.append(slim)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(slim_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  📄 Results saved to: {output_path}")


# ═════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ═════════════════════════════════════════════════════════════════════

async def main():
    """Run the full end-to-end test suite."""
    import argparse
    parser = argparse.ArgumentParser(description="RRRIE-CDSS End-to-End Test")
    parser.add_argument("--all", action="store_true", help="Run ALL cases (structured or narrative)")
    parser.add_argument("--case", type=str, help="Run a specific case by keyword (e.g. pneumonia, rabies)")
    parser.add_argument("--narrative", action="store_true", help="Use narrative patient stories (different each run)")
    parser.add_argument("--count", type=int, default=3, help="Number of random narrative cases (default: 3)")
    parser.add_argument("--hard", action="store_true", help="Only hard-difficulty narrative cases")
    parser.add_argument("--mix", action="store_true", help="Mixed mode: 2 structured + 3 narrative (random)")
    parser.add_argument("--list", action="store_true", help="List all available narrative cases and exit")
    args, _ = parser.parse_known_args()

    # ── List mode ────────────────────────────────────────────────
    if args.list:
        print("\n  Available Narrative Cases:")
        print(f"  {'ID':6s} | {'Diff':6s} | {'Category':25s} | Title")
        print(f"  {'-'*80}")
        print(list_all_cases())
        print(f"\n  Total: {len(NARRATIVE_CASES)} narrative cases")
        print(f"  Also available: {len(ALL_CASES)} structured JSON cases in tests/test_cases/")
        return []

    # ── Determine mode and case list ─────────────────────────────
    structured_files: list[str] = []
    narrative_list: list[dict] = []

    if args.mix:
        # Mixed: 2 structured + 3 narrative (random each run)
        import random
        structured_files = random.sample(ALL_CASES, min(2, len(ALL_CASES)))
        narrative_list = get_random_cases(3)
    elif args.narrative:
        if args.case:
            matches = get_case_by_keyword(args.case)
            if not matches:
                print(f"No narrative case matching '{args.case}'. Use --list to see all.")
                return []
            narrative_list = matches
        elif args.all:
            narrative_list = NARRATIVE_CASES[:]
        elif args.hard:
            narrative_list = get_cases_by_difficulty("hard")
            if args.count < len(narrative_list):
                import random
                narrative_list = random.sample(narrative_list, args.count)
        else:
            narrative_list = get_random_cases(args.count)
    else:
        # Classic structured mode
        if args.case:
            matches = [f for f in ALL_CASES if args.case.lower() in f.lower()]
            if not matches:
                print(f"No case matching '{args.case}'. Available: {ALL_CASES}")
                return []
            structured_files = matches
        elif args.all:
            structured_files = ALL_CASES
        else:
            structured_files = DEFAULT_CASES

    total_cases = len(structured_files) + len(narrative_list)

    print("=" * 70)
    print("  RRRIE-CDSS End-to-End Medical Test")
    print("  Model: Qwen/Qwen3.5-4B (GGUF Q4_K_M via llama.cpp, adaptive thinking)")
    print("  Pipeline: R1 → WHO Verify → R2 (PubMed) → R3 → IE")
    if structured_files:
        print(f"  Structured cases: {len(structured_files)}")
    if narrative_list:
        print(f"  Narrative cases: {len(narrative_list)} (dynamic selection)")
        for nc in narrative_list:
            print(f"    • {nc['case_id']}: {nc['title']} [{nc['difficulty']}]")
    print(f"  Total: {total_cases} cases")
    print(f"  Data: REAL WHO ICD-11 API + REAL PubMed API")
    print("=" * 70)

    # Load model once (singleton)
    print("\n  Loading Qwen3.5-4B (GGUF Q4_K_M via llama.cpp)...")
    t0 = time.time()
    client = LlamaCppClient.get_instance("Qwen/Qwen3.5-4B")
    print(f"  ✓ Model loaded in {time.time()-t0:.1f}s | VRAM: {client.get_vram_usage():.2f} GB")

    results = []

    # Run structured cases
    for case_file in structured_files:
        try:
            result = await run_full_pipeline(case_file, client)
            results.append(result)
        except Exception as exc:
            print(f"\n  ❌ CASE FAILED: {case_file} — {exc}")
            import traceback
            traceback.print_exc()

    # Run narrative cases
    for nc in narrative_list:
        try:
            result = await run_narrative_pipeline(nc, client)
            results.append(result)
        except Exception as exc:
            print(f"\n  ❌ NARRATIVE FAILED: {nc['case_id']} — {exc}")
            import traceback
            traceback.print_exc()

    if results:
        print_summary(results)

    return results


# pytest entry point
import pytest

@pytest.mark.asyncio
@pytest.mark.slow
async def test_e2e_pipeline():
    """Pytest wrapper."""
    await main()


if __name__ == "__main__":
    asyncio.run(main())
