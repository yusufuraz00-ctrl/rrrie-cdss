"""Stage Adapter — bridges intelligence gap between cloud 70B and local 4B.

Strategy D Architecture:
    R1 (☁️ 70B) → R2 (📚 APIs) → R3 (☁️ 70B) → [🔧 This Adapter] → IE (💻 4B)

The 70B model generates sophisticated, verbose JSON. The 4B model struggles with
large context windows and complex nested structures. This adapter:

1. Extracts only the fields IE needs for its checklist evaluation
2. Flattens nested structures into simple key-value pairs
3. Keeps total IE input under ~1500 tokens
4. Preserves the original patient text verbatim for hallucination checking
5. Injects relevant memory context (patterns + similar cases) when available

Design principle: "No LLM interprets another LLM's full output" — Python does the bridging.
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.case_store import MemoryContext


def simplify_for_ie(
    r1_json: dict[str, Any],
    r3_json: dict[str, Any],
    r2_evidence: list[dict[str, Any]],
    patient_text: str,
    memory_context: "MemoryContext | None" = None,
    iteration: int = 1,
    drug_facts: str = "",
    paradox_text: str = "",
) -> str:
    """Build a simplified, flat context string for the IE (4B local) model.

    Target: <1500 tokens total input to IE (system prompt ~400 + this context ~1100).

    Args:
        r1_json: Parsed R1 output (differential diagnoses + patient summary).
        r3_json: Parsed R3 output (primary diagnosis + treatment plan).
        r2_evidence: List of R2 evidence dicts (PubMed + ICD-11 results).
        patient_text: Original patient presentation text (verbatim).
        memory_context: Optional MemoryContext from 3-tier memory system.
        iteration: Current iteration number (1 = first pass).
        drug_facts: Verified drug facts from pharmacology API (RxNorm/OpenFDA).
        paradox_text: Compact paradox summary for IE checks (from paradox_resolver).

    Returns:
        Formatted string ready to be used as IE user message.
    """
    sections = []

    # ── Iteration header (if re-running IE after R3 revision) ──────
    if iteration > 1:
        sections.append(
            f"## ⚠️ ITERATION {iteration} — This is a REVISED analysis.\n"
            f"The previous version had issues. Check if they are now fixed."
        )

    # ── Section 1: Original Patient Text (verbatim — for hallucination check) ──
    # Use a generous limit to ensure vital signs, exam findings, and imaging
    # results at the END of the text are never lost (IE will flag them as
    # "hallucinations" if they are truncated away).
    pt = patient_text.strip()
    if len(pt) > 2000:
        # Keep first 1200 + last 600 so both history and exam/labs survive
        pt = pt[:1200] + "\n[...middle truncated...]\n" + pt[-600:]
    sections.append(f"## Original Patient Text\n{pt}")

    # ── Section 2: Patient Symptoms (flat list from R1) ──
    r1_summary = r1_json.get("patient_summary", {})
    positives = r1_summary.get("key_positives", [])
    if positives:
        symptoms_str = "\n".join(f"- {s}" for s in positives[:10])
        sections.append(f"## Patient Symptoms (from R1)\n{symptoms_str}")

    # ── Section 3: R1 Recorded History (comprehensive) ──
    history = r1_summary.get("relevant_history", "Not recorded")
    negatives = r1_summary.get("key_negatives", [])
    medications = r1_summary.get("current_medications", [])
    red_flags = r1_summary.get("red_flags", [])

    history_lines = [f"Past History: {history}"]
    if medications:
        history_lines.append(f"Medications: {', '.join(medications[:5])}")
    if red_flags:
        history_lines.append(f"Red Flags: {', '.join(red_flags[:5])}")
    if negatives:
        history_lines.append(f"Key Negatives: {', '.join(negatives[:5])}")
    sections.append(f"## Recorded History (from R1)\n" + "\n".join(history_lines))

    # ── Section 4: Primary Diagnosis (flat — from R3) ──
    primary = r3_json.get("primary_diagnosis", {})
    dx_name = primary.get("diagnosis", "Unknown")
    dx_conf = primary.get("confidence", 0.0)
    dx_evidence = primary.get("evidence_support", "unknown")
    explained = primary.get("explains_symptoms", [])
    unexplained = primary.get("unexplained_symptoms", [])

    dx_section = f"## Primary Diagnosis (from R3)\n"
    dx_section += f"- Diagnosis: {dx_name}\n"
    dx_section += f"- Confidence: {dx_conf}\n"
    dx_section += f"- Evidence support: {dx_evidence}\n"
    if explained:
        dx_section += f"- Explains: {', '.join(explained[:5])}\n"
    if unexplained:
        dx_section += f"- Does NOT explain: {', '.join(unexplained[:5])}\n"
    sections.append(dx_section)

    # ── Section 5: Treatment Summary (flat — from R3) ──
    treatment = r3_json.get("treatment_plan", {})
    if treatment:
        tx_lines = []
        for action in treatment.get("immediate_actions", [])[:3]:
            tx_lines.append(f"- URGENT: {action}")
        for drug in treatment.get("pharmacological", [])[:3]:
            d = drug.get("drug", "?")
            dose = drug.get("dose", "?")
            route = drug.get("route", "?")
            tx_lines.append(f"- Drug: {d} {dose} ({route})")
        if tx_lines:
            sections.append(f"## Treatment Plan (from R3)\n" + "\n".join(tx_lines))

    # ── Section 5.5: Verified Drug Facts (from pharmacology API) ──
    if drug_facts:
        # Compact version for IE (4B model has limited context)
        # Keep only the essential class + mechanism lines
        compact_lines = []
        for line in drug_facts.split("\n"):
            stripped = line.strip()
            if stripped.startswith("💊") or stripped.startswith("Class:") or \
               stripped.startswith("Mechanism:") or stripped.startswith("⚠️") or \
               stripped.startswith("🚫"):
                compact_lines.append(stripped)
        if compact_lines:
            sections.append(
                "## Verified Drug Facts (from RxNorm/OpenFDA — TRUST THESE)\n"
                + "\n".join(compact_lines[:10])
            )

    # ── Section 5.6: Paradox info (for CHECK 8 — PARADOX AWARENESS) ──
    if paradox_text:
        sections.append(paradox_text)

    # ── Section 5.7: R1 Top 3 Differentials (for CHECK 7 — DIFFERENTIAL ACCOUNTABILITY) ──
    r1_diffs = r1_json.get("differential_diagnoses", [])
    if r1_diffs:
        diff_lines = ["## R1 Differential Diagnoses (top 3 — R3 must address all)"]
        for d in r1_diffs[:3]:
            rank = d.get("rank", "?")
            name = d.get("diagnosis", "?")
            conf = d.get("confidence", 0.0)
            diff_lines.append(f"- #{rank}: {name} ({conf:.0%})")
        sections.append("\n".join(diff_lines))

    # ── Section 6: R2 Evidence Stats (critical for evidence-gap detection) ──
    # Match both 'pubmed' (static fallback) and 'search_pubmed' (dynamic dispatch)
    _is_pubmed = lambda e: "pubmed" in e.get("source", "").lower()
    _is_epmc = lambda e: "europe_pmc" in e.get("source", "").lower()
    _is_s2 = lambda e: "semantic_scholar" in e.get("source", "").lower()
    _is_wiki = lambda e: "wikipedia" in e.get("source", "").lower()
    _is_web = lambda e: "web_search" in e.get("source", "").lower()

    pubmed_queries = [e for e in r2_evidence if _is_pubmed(e)]
    epmc_queries = [e for e in r2_evidence if _is_epmc(e)]
    s2_queries = [e for e in r2_evidence if _is_s2(e)]
    wiki_queries = [e for e in r2_evidence if _is_wiki(e)]
    web_queries = [e for e in r2_evidence if _is_web(e)]

    total_articles = sum(e.get("count", 0) for e in pubmed_queries)
    total_articles += sum(e.get("count", 0) for e in epmc_queries)
    total_articles += sum(e.get("count", 0) for e in s2_queries)
    total_results = total_articles + sum(1 for e in wiki_queries if e.get("count", 0) > 0)
    total_results += sum(e.get("count", 0) for e in web_queries)
    successful_sources = sum(1 for e in r2_evidence if e.get("count", 0) > 0 and e.get("source") != "icd11")

    ev_section = f"## R2 Evidence Stats\n"
    ev_section += f"- Total search queries: {len(r2_evidence)}\n"
    ev_section += f"- PubMed: {len(pubmed_queries)} queries, {sum(e.get('count', 0) for e in pubmed_queries)} articles\n"
    ev_section += f"- Europe PMC: {len(epmc_queries)} queries, {sum(e.get('count', 0) for e in epmc_queries)} articles\n"
    ev_section += f"- Semantic Scholar: {len(s2_queries)} queries, {sum(e.get('count', 0) for e in s2_queries)} papers\n"
    ev_section += f"- Wikipedia: {len(wiki_queries)} queries\n"
    ev_section += f"- Web search: {len(web_queries)} queries\n"
    ev_section += f"- Sources with results: {successful_sources}/{len(r2_evidence)}\n"
    ev_section += f"- Total articles/papers: {total_articles}\n"
    if total_results == 0:
        ev_section += "- WARNING: No supporting literature found for any hypothesis.\n"
    sections.append(ev_section)

    # ── Section 7: R3 Citations (for fabrication check) ──
    citations = r3_json.get("citations", [])
    if citations:
        # List PMIDs R3 claims to cite
        pmid_list = [c.get("pmid", "?") for c in citations[:5]]
        sections.append(f"## R3 Cited PMIDs\n{', '.join(pmid_list)}")

        # List PMIDs that actually came from R2 (scan ALL article sources)
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
        if real_pmids:
            sections.append(f"## R2 Actual PMIDs\n{', '.join(sorted(real_pmids))}")
        else:
            sections.append("## R2 Actual PMIDs\nNone — R2 returned zero articles.")

    # ── Section 8: Memory Context (from 3-tier experience system) ──
    if memory_context and not memory_context.is_empty:
        # Only inject patterns and similar cases for IE (principles go to R3)
        mem_lines = []
        if memory_context.patterns:
            mem_lines.append("## Historical Patterns")
            for p in memory_context.patterns[:3]:
                lesson = p.get("lesson", "")
                conf = p.get("confidence", 0.5)
                mem_lines.append(f"- [{conf:.0%}] {lesson}")
        if memory_context.similar_cases:
            for sc in memory_context.similar_cases[:1]:
                sc_dx = sc.get("primary_diagnosis", "?")
                sc_ie = sc.get("ie_decision", "?")
                sc_issues = sc.get("ie_issues_summary", "None")
                mem_lines.append(
                    f"## Similar Past Case\n"
                    f"- Diagnosis: {sc_dx}, IE decision: {sc_ie}\n"
                    f"- Issues found: {sc_issues}"
                )
        if mem_lines:
            sections.append("\n".join(mem_lines))

    return "\n\n".join(sections)


def resolve_icd_codes(
    diagnoses: list[dict[str, Any]],
    icd_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Inject WHO API-resolved ICD-11 codes into diagnosis dicts.

    This replaces LLM-generated (hallucinated) ICD codes with real codes
    from the WHO ICD-11 API.

    Args:
        diagnoses: List of diagnosis dicts (from R1 or R3) with "diagnosis" field.
        icd_results: Mapping of diagnosis_name → list of WHO API results.
            Each result: {"theCode": "4A44", "title": "Giant cell arteritis", "score": 0.95}

    Returns:
        Same diagnoses list with "icd11_code" and "icd11_title" fields added/updated.
    """
    for dx in diagnoses:
        name = dx.get("diagnosis", "")
        if not name:
            continue

        results = icd_results.get(name, [])
        if results and len(results) > 0:
            best = results[0]  # Highest score from WHO API
            dx["icd11_code"] = best.get("theCode", "")
            dx["icd11_title"] = best.get("title", "")
            dx["icd11_source"] = "WHO API"
        else:
            dx["icd11_code"] = ""
            dx["icd11_title"] = "Not found via WHO API"
            dx["icd11_source"] = "WHO API (no match)"

    return diagnoses
