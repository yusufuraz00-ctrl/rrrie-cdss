"""Paradox Resolver — detects drug/intervention → symptom worsening patterns.

This module analyses patient text for the critical diagnostic pattern:
    "Patient received treatment X → symptoms WORSENED"

This is one of the strongest diagnostic clues in medicine.
A drug that triggers or exacerbates a disease often IDENTIFIES that disease.

Design philosophy (user requirement):
    ✦ Pattern-based text analysis, NOT hardcoded disease rules
    ✦ The module detects the PARADOX (drug → worsening)
    ✦ The LLM determines which disease the paradox implies
    ✦ Works for English and Turkish patient texts
    ✦ "mantık iyileştirmesi" — pure logic improvement, zero disease-specific code
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Paradox:
    """A detected drug/intervention → worsening event."""

    drug_or_intervention: str
    worsening_description: str
    source_text: str  # Original sentence/phrase that triggered detection


# ── Pre-compiled patterns (executed once at import) ────────────────

_WORSENING_PATTERNS: list[re.Pattern[str]] = []
_RESPIRATORY_PATTERNS: list[re.Pattern[str]] = []

def _compile_patterns() -> None:
    """Compile regex patterns once at import time."""
    global _WORSENING_PATTERNS, _RESPIRATORY_PATTERNS  # noqa: PLW0603

    # ── Generic "took X, got worse" patterns ──────────────────────
    worsening_raw = [
        # English — "gave/took/received [drug] but ... made worse / worsened"
        r'(?:gave|administered|prescribed|started|received|took|given)\s+(?:her |him |me |us |them )?([a-zA-Z][a-zA-Z\s\-]{2,38}?)\s+(?:but|however|yet|and(?:\s+then)?)\s+(?:(?:it|she|he|they|the\s+patient|symptoms?)\s+)?(?:(?:got|became|grew)\s+worse|worsened|made\s+(?:it|her|him|things|the\s+patient|symptoms?)\s+worse)',
        # English — "X made it/her/him worse"
        r'(?:the\s+|those\s+|these\s+)?([a-zA-Z][a-zA-Z\s\-]{2,38}?)\s+(?:made\s+(?:it|her|him|things|symptoms?|the\s+patient|the\s+condition|everything)\s+worse|worsened\s+the\s+(?:symptoms?|condition|patient))',
        # English — "after taking/receiving/given X, ... worsened"
        r'(?:after|following|upon|since)\s+(?:taking|receiving|being\s+given|starting|using)\s+([a-zA-Z][a-zA-Z\s\-]{2,38}?)\s*[,.]?\s*(?:symptoms?\s+)?(?:worsened|got\s+worse|deteriorat|exacerbat|aggravat|intensifi|increased|worsening)',
        # English — "worsening after X"
        r'(?:worsening|deterioration|exacerbation|decline)\s+(?:after|following|since|upon)\s+([a-zA-Z][a-zA-Z\s\-]{2,38})',
        # English — "X triggered/provoked/precipitated"
        r'(?:the\s+)?([a-zA-Z][a-zA-Z\s\-]{2,38}?)\s+(?:triggered|provoked|precipitated|induced|caused)\s+(?:the|a|an)?\s*(?:crisis|attack|episode|flare|exacerbation|worsening|deterioration)',

        # Turkish — "[drug] verdiler/aldı ama daha kötü/kötüleşti" (with ASCII variants)
        r'([a-zA-ZçğıöşüÇĞİÖŞÜ][a-zA-ZçğıöşüÇĞİÖŞÜ\s\-]{2,38}?)\s+(?:verdiler|verdi|ald[ıi]|i[çc]ti|kulland[ıi])\s*[,.]?\s*(?:ama|fakat|ancak|lakin)\s*(?:daha\s+k[öo]t[üu]|k[öo]t[üu]le[şs]ti|a[ğg][ıi]rla[şs]t[ıi]|artt[ıi]|fenala[şs]t[ıi]|rahats[ıi]zland[ıi]|daha\s+k[öo]t[üu]\s+oldu)',
        # Turkish — "[drug] sonrası kötüleşti" (with ASCII variants)
        r'([a-zA-ZçğıöşüÇĞİÖŞÜ][a-zA-ZçğıöşüÇĞİÖŞÜ\s\-]{2,38}?)\s+(?:sonras[ıi]|sonras[ıi]nda|ald[ıi]ktan\s+sonra|verdikten\s+sonra|i[çc]tikten\s+sonra|kulland[ıi]ktan\s+sonra)\s+(?:k[öo]t[üu]le[şs]ti|a[ğg][ıi]rla[şs]t[ıi]|artt[ıi]|nefes\s+al[ae]m(?:ad[ıi]|[ıi]yor)|daha\s+k[öo]t[üu]|fenala[şs]t[ıi])',
        # Turkish — "[drug] işe yaramadı / daha kötü yaptı" (with ASCII variants)
        r'([a-zA-ZçğıöşüÇĞİÖŞÜ][a-zA-ZçğıöşüÇĞİÖŞÜ\s\-]{2,38}?)\s+(?:i[şs]e\s+yaramad[ıi]|fayda\s+etmedi|daha\s+da\s+k[öo]t[üu]\s+yapt[ıi]|daha\s+k[öo]t[üu]\s+hale\s+getirdi)',
        # Turkish — "kötüleşme [drug] sonrasında" (with ASCII variants)
        r'(?:k[öo]t[üu]le[şs]me|a[ğg][ıi]rla[şs]ma|fenala[şs]ma)\s+([a-zA-ZçğıöşüÇĞİÖŞÜ][a-zA-ZçğıöşüÇĞİÖŞÜ\s\-]{2,38}?)\s+(?:sonras[ıi]|sonras[ıi]nda|y[üu]z[üu]nden|nedeniyle)',
    ]

    # ── Drug + respiratory distress patterns ──────────────────────
    respiratory_raw = [
        # Turkish — "[drug] verdi, nefes alamadı / boğuldu" (with ASCII variants)
        r'([a-zA-ZçğıöşüÇĞİÖŞÜ][a-zA-ZçğıöşüÇĞİÖŞÜ\s\-]{2,38}?)\s+(?:verdi|verdiler|ald[ıi]|i[çc]ti)\s*[,.]?\s*(?:nefes\s+al[ae]m(?:ad[ıi]|[ıi]yor)|bo[ğg]ul|solunum\s+s[ıi]k[ıi]nt[ıi]|nefes\s+darl[ıi][ğg][ıi])',
        # English — "after X, couldn't breathe"
        r'(?:after|following)\s+([a-zA-Z][a-zA-Z\s\-]{2,38}?)\s*[,.]?\s*(?:couldn.?t\s+breathe|respiratory\s+distress|breathing\s+(?:worsened|difficulty|problems?)|dyspnea\s+(?:worsened|developed))',
        # English — "X caused breathing difficulty"
        r'([a-zA-Z][a-zA-Z\s\-]{2,38}?)\s+(?:caused|led\s+to|resulted\s+in)\s+(?:respiratory\s+distress|breathing\s+difficulty|dyspnea|respiratory\s+failure)',
    ]

    _WORSENING_PATTERNS.extend(re.compile(p, re.IGNORECASE) for p in worsening_raw)
    _RESPIRATORY_PATTERNS.extend(re.compile(p, re.IGNORECASE) for p in respiratory_raw)


# Compile at import time
_compile_patterns()

# ── Filler words to strip from captured drug names ────────────────
_FILLER_RE = re.compile(
    r'^(?:the|a|an|some|her|his|my|their|those|these|that|this|'
    r'says?|said|told|mention|'
    r'bu|şu|o|bir|onun|benim|bizim|annesi|babas[ıi]|'
    r'hastanede|doktor|doktora|hemşire)\s+',
    re.IGNORECASE,
)


def _clean_drug_name(raw: str) -> str:
    """Clean up a captured drug/intervention name."""
    name = raw.strip().rstrip('.,;:!?')
    # Iteratively strip leading filler words (may need multiple passes)
    for _ in range(3):
        prev = name
        name = _FILLER_RE.sub('', name)
        if name == prev:
            break
    # Remove trailing filler
    name = re.sub(
        r'\s+(?:to|for|from|with|and|or|but|also|too|as\s+well|'
        r'için|ile|ve|veya|da|de|ama)$',
        '', name, flags=re.IGNORECASE,
    )
    return name.strip()


def _is_substring_of_existing(drug: str, seen: set[str]) -> bool:
    """Check if drug is a longer phrase containing an already-seen drug name."""
    drug_lower = drug.lower()
    for existing in seen:
        # If the new capture contains an already-seen drug → skip (duplicate)
        if existing in drug_lower and len(drug_lower) > len(existing) + 5:
            return True
        # If an existing capture contains this new one → also skip
        if drug_lower in existing and len(existing) > len(drug_lower) + 5:
            return True
    return False


def detect_paradoxes(patient_text: str) -> list[Paradox]:
    """Scan patient text for drug/intervention → worsening patterns.

    Returns list of Paradox objects. Empty if no paradoxes detected.
    Language-agnostic: works for English and Turkish patient texts.

    This function detects the PATTERN of worsening, not specific diseases.
    The LLM will interpret what disease the paradox implies.
    """
    paradoxes: list[Paradox] = []
    seen: set[str] = set()

    def _try_add(drug_raw: str, description: str, source: str) -> None:
        drug = _clean_drug_name(drug_raw)
        drug_key = drug.lower()
        if drug and len(drug) > 2 and drug_key not in seen and not _is_substring_of_existing(drug, seen):
            seen.add(drug_key)
            paradoxes.append(Paradox(
                drug_or_intervention=drug,
                worsening_description=description,
                source_text=source,
            ))

    # ── Main worsening patterns ───────────────────────────────────
    for pat in _WORSENING_PATTERNS:
        for m in pat.finditer(patient_text):
            drug_raw = m.group(1)
            _try_add(drug_raw, m.group(0).strip(), m.group(0).strip())

    # ── Respiratory distress patterns ─────────────────────────────
    for pat in _RESPIRATORY_PATTERNS:
        for m in pat.finditer(patient_text):
            drug_raw = m.group(1)
            description = f"respiratory distress after {_clean_drug_name(drug_raw)}"
            _try_add(drug_raw, description, m.group(0).strip())

    return paradoxes


def format_paradox_directive(paradoxes: list[Paradox]) -> str:
    """Format detected paradoxes into a persistent directive for R3.

    This directive is injected into EVERY R3 iteration and CANNOT be
    overridden by IE suggestions or perspective shifts. It represents
    a hard clinical signal extracted from the patient text.

    Returns empty string if no paradoxes detected.
    """
    if not paradoxes:
        return ""

    lines = [
        "## ⚡ DRUG-EXACERBATION PARADOX DETECTED (PERSISTENT — DO NOT IGNORE)",
        "",
        "The following medication/treatment → worsening patterns were detected in the patient text.",
        "These are CRITICAL diagnostic clues. A drug that worsens a condition often IDENTIFIES that condition.",
        "",
    ]

    for i, p in enumerate(paradoxes, 1):
        lines.append(f"### Paradox {i}: \"{p.drug_or_intervention}\" → worsening")
        lines.append(f"- Detected pattern: \"{p.worsening_description}\"")
        lines.append(f"- Original text fragment: \"{p.source_text}\"")
        lines.append("")

    lines.extend([
        "### MANDATORY INVESTIGATION (applies to ALL R3 iterations):",
        "1. For EACH paradox above: identify which diseases on R1's differential list are",
        "   TRIGGERED or EXACERBATED by the named drug/intervention.",
        "2. Any diagnosis that IS triggered by this drug MUST receive ELEVATED ranking,",
        "   even if R1 ranked it low or gave it low confidence.",
        "3. Any diagnosis that would NOT be affected by this drug → its likelihood of",
        "   being correct is REDUCED (it cannot explain the paradox).",
        "4. If NONE of your current differential explains the paradox → you are missing",
        "   the correct diagnosis. Expand your differential to include conditions known",
        "   to be triggered by the named drug/intervention.",
        "5. This directive is PERSISTENT — it applies to every iteration and CANNOT be",
        "   overridden by IE suggestions or perspective shifts.",
        "6. In your reasoning_chain, you MUST explicitly state which diagnosis is",
        "   supported or ruled out by each paradox.",
        "",
    ])

    return "\n".join(lines)


def format_paradox_for_ie(paradoxes: list[Paradox]) -> str:
    """Format a compact paradox summary for the IE (4B model) context.

    IE needs to verify that R3 addressed the paradox. This gives IE
    the information to do CHECK 8 — PARADOX AWARENESS.
    """
    if not paradoxes:
        return ""

    lines = ["## ⚡ Drug-Exacerbation Paradoxes (from patient text)"]
    for p in paradoxes:
        lines.append(f"- \"{p.drug_or_intervention}\" made symptoms worse: {p.worsening_description}")
    lines.append("→ R3 MUST explain which diagnosis is triggered by this drug. If not → critical issue.")
    return "\n".join(lines)
