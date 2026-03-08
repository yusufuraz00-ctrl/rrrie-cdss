"""Drug Lookup Pipeline — Extract drug names from patient text + resolve pharmacology.

This module:
  1. Extracts drug name candidates from unstructured patient text
  2. Resolves each via external APIs (RxNorm + OpenFDA + web search)
  3. Formats resolved drug profiles as "Hard Facts" for LLM prompt injection

Design:
  NO hardcoded drug lists. Drug detection uses contextual pattern matching
  (dosage units, medication keywords, etc.). All pharmacological data comes
  from live API queries + persistent cache.

  The cache grows with every new case — the system "learns" drugs over time,
  just like a clinician building experience. Second encounter = instant.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from src.tools.pharmacology_tool import resolve_drug, DrugInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
# Drug Name Extraction (Pattern-Based — No Drug Dictionary)
# ═════════════════════════════════════════════════════════════════════

# Dosage pattern: "Dideral 40mg", "Propranolol 80 mg", "Aspirin 100mg"
_DOSAGE_PATTERN = re.compile(
    r'(\b[A-Za-zığüşöçİĞÜŞÖÇ][a-zA-ZığüşöçİĞÜŞÖÇ]{2,25})\s*'
    r'(?:\d+\s*(?:mg|mcg|µg|ml|g|iu|units?|tablets?|caps?|tb|amp|cc)\b)',
    re.IGNORECASE,
)

# Medication context: "taking Dideral", "started on IV Labetalol", "prescribed Aspirin"
# NOTE: \b prevents matching inside words (e.g. "on" inside "Concor")
# Route words MUST be followed by at least one space to prevent "impaired" → "IM" + "paired"
_ROUTE_WORDS = r'(?:(?:IV|IM|SC|PO|oral|topical|sublingual|rectal|inhaled|nebulized)\s+)?'
_MEDICATION_CONTEXT_EN = re.compile(
    r'\b(?:taking|prescribed|receiving|given|administered|started\s+on|'
    r'currently\s+on|on\s+(?:daily|regular)|using|medication|drug|medicine|dose\s+of|'
    r'treated\s+with|therapy\s+with|history\s+of)\s*'
    r'[:\-]?\s*' + _ROUTE_WORDS +
    r'([A-Z][a-zA-ZığüşöçİĞÜŞÖÇ]{2,25})',
    re.IGNORECASE,
)

# Turkish medication context (\b prevents false matches inside compound words)
_MEDICATION_CONTEXT_TR = re.compile(
    r'\b(?:aldığı|kullandığı|kullanan|kullanıyor|kullanmakta|'
    r'reçete|ilaç|tedavi|verildi|başlandı|başlanmış|'
    r'uygulandı|uygulanmakta)\s*'
    r'[:\-]?\s*' + _ROUTE_WORDS +
    r'([A-Za-zığüşöçİĞÜŞÖÇ][a-zA-ZığüşöçİĞÜŞÖÇ]{2,25})',
    re.IGNORECASE,
)

# Medication list pattern: "Medications: Aspirin, Metoprolol, Lisinopril"
_DRUG_LIST_PATTERN = re.compile(
    r'(?:medications?|meds|drugs?|current\s+meds|ilaçlar?|tedaviler?|ilaç\s+listesi)\s*'
    r'[:\-]\s*'
    r'([^.\n]{5,200})',
    re.IGNORECASE,
)

# Route-of-administration prefix: "IV Labetalol", "oral Metoprolol", "PO Aspirin"
_ROUTE_PREFIX_PATTERN = re.compile(
    r'\b(?:IV|IM|SC|PO|oral|topical|sublingual|rectal|inhaled|nebulized)\s+'
    r'([A-Z][a-zA-ZığüşöçİĞÜŞÖÇ]{2,25})',
    re.IGNORECASE,
)

# Turkish reverse-order: "Labetalol IV başlandı", "Concor kullanıyor"
# (Drug name BEFORE verb — Turkish SOV order)
_TR_REVERSE_PATTERN = re.compile(
    r'([A-Za-zığüşöçİĞÜŞÖÇ][a-zA-ZığüşöçİĞÜŞÖÇ]{2,25})\s+'
    r'(?:(?:IV|IM|SC|PO)\s+)?'
    r'(?:başlandı|başlanmış|başlanıyor|verildi|verilmekte|verilmiş|'
    r'uygulandı|uygulanıyor|uygulanmakta|'
    r'kesildi|kesilmiş|bırakıldı)',
    re.IGNORECASE,
)

# Parenthetical mentions: "Dideral (propranolol)" or "(Dideral)"
_PAREN_DRUG_PATTERN = re.compile(
    r'([A-Z][a-zA-Z]{2,25})\s*\([^)]*\)|'  # Before parens: "Dideral (a beta-blocker)"
    r'\(([A-Z][a-zA-Z]{2,25})\)',            # Inside parens: "(Propranolol)"
    re.IGNORECASE,
)

# Common non-drug words that pattern matching might catch
_NON_DRUG_WORDS = frozenset({
    # ── English pronouns / determiners / prepositions ──
    "his", "her", "she", "him", "its", "they", "them", "their", "our",
    "your", "who", "which", "what", "that", "this", "these", "those",
    "the", "was", "had", "has", "with", "and", "for", "from",
    "been", "being", "were", "also", "only", "after", "before",
    "during", "about", "since", "into", "onto", "upon", "over",
    "through", "between", "among", "within", "without", "each",
    "every", "some", "many", "much", "more", "most", "other",
    "another", "both", "either", "neither", "all", "any", "few",
    # ── English common verbs / auxiliaries ──
    "given", "started", "receiving", "prescribed", "taking",
    "having", "doing", "going", "coming", "getting", "making",
    "looking", "feeling", "working", "trying", "becoming", "moving",
    "running", "walking", "standing", "sitting", "lying", "sleeping",
    "causing", "showing", "involving", "including", "following",
    "leading", "producing", "resulting", "occurring", "appearing",
    "remaining", "continuing", "developing", "improving", "worsening",
    # ── Time / quantity words ──
    "days", "weeks", "months", "years", "hours", "minutes", "times",
    "once", "twice", "daily", "weekly", "monthly", "hourly",
    # ── Medical non-drug substances / procedures ──
    "fluids", "fluid", "saline", "dextrose", "infusion", "bolus",
    "transfusion", "irrigation", "solution", "suspension", "cream",
    "ointment", "drops", "patch", "injection", "tablet", "capsule",
    "paired", "repair", "preparation", "supplement", "vitamin",
    # ── English medical / anatomy / clinical terms ──
    "patient", "history", "physical", "examination", "hospital", "clinic",
    "doctor", "nurse", "blood", "pressure", "temperature", "heart", "rate",
    "respiratory", "oxygen", "saturation", "white", "cell", "count",
    "presented", "admitted", "diagnosed", "treatment", "therapy",
    "condition", "disease", "disorder", "syndrome", "infection",
    "normal", "abnormal", "elevated", "decreased", "increased",
    "severe", "moderate", "mild", "acute", "chronic", "bilateral",
    "positive", "negative", "present", "absent", "right", "left",
    "upper", "lower", "report", "result", "level", "value",
    "pulse", "breath", "chest", "abdomen", "head", "neck", "back",
    "skin", "wound", "mass", "test", "scan", "ultrasound",
    "surgery", "procedure", "diagnosis", "prognosis", "assessment",
    "evaluation", "intervention", "monitoring", "screening",
    "emergency", "critical", "stable", "progressive", "recurrent",
    "intermittent", "persistent", "refractory", "idiopathic",
    # ── Symptoms / clinical descriptors (CRITICAL — these caused API hallucinations) ──
    "impaired", "drooping", "swollen", "swelling", "enlarged", "reduced",
    "weakness", "numbness", "tingling", "paralysis", "tremor", "seizure",
    "headache", "dizziness", "vertigo", "fatigue", "malaise", "lethargy",
    "nausea", "vomiting", "diarrhea", "constipation", "bloating",
    "coughing", "wheezing", "sneezing", "bleeding", "bruising",
    "itching", "redness", "pallor", "cyanosis", "jaundice", "edema",
    "tenderness", "stiffness", "spasm", "cramping", "burning",
    "throbbing", "aching", "stabbing", "shooting", "radiating",
    "worsened", "improved", "unchanged", "fluctuating", "progressive",
    "difficulty", "inability", "unable", "limited", "restricted",
    "painful", "painless", "tender", "diffuse", "localized", "focal",
    "blurred", "double", "dimmed", "distorted", "muffled",
    "hoarse", "slurred", "garbled", "confused", "disoriented",
    "agitated", "lethargic", "drowsy", "unresponsive", "altered",
    "febrile", "afebrile", "diaphoretic", "tachycardic", "bradycardic",
    "hypertensive", "hypotensive", "tachypneic", "dyspneic", "orthopneic",
    "distended", "rigid", "flaccid", "spastic", "atrophic",
    "inflamed", "infected", "necrotic", "gangrenous", "ulcerated",
    "obstructed", "perforated", "ruptured", "displaced", "fractured",
    "congested", "consolidated", "infiltrated", "calcified", "fibrotic",
    # ── Body parts / anatomical terms ──
    "arm", "arms", "leg", "legs", "hand", "hands", "foot", "feet",
    "finger", "fingers", "toe", "toes", "eye", "eyes", "ear", "ears",
    "nose", "mouth", "lip", "lips", "tongue", "throat", "jaw",
    "face", "forehead", "cheek", "chin", "scalp", "skull",
    "shoulder", "elbow", "wrist", "hip", "knee", "ankle",
    "spine", "rib", "ribs", "pelvis", "femur", "tibia",
    "lung", "lungs", "liver", "kidney", "kidneys", "spleen",
    "pancreas", "bladder", "intestine", "colon", "rectum",
    "brain", "nerve", "nerves", "muscle", "muscles", "bone", "bones",
    "joint", "joints", "tendon", "ligament", "cartilage",
    "artery", "vein", "capillary", "vessel", "vessels",
    "mucosa", "serosa", "peritoneum", "pleura", "meninges",
    # ── Turkish medical ──
    "hasta", "hastane", "muayene", "fizik", "tanı", "teşhis",
    "bulgu", "sonuç", "değer", "ölçüm", "kontrol", "takip",
    "nefes", "kalp", "ateş", "ağrı", "bulantı", "kusma",
    "şişlik", "kızarıklık", "morarma", "sarılık", "ödem",
    "baş", "göz", "kulak", "burun", "boğaz", "göğüs", "karın",
    "kol", "bacak", "sırt", "bel", "boyun", "omuz", "diz",
    # ── Turkish verbs (used as context keywords — not drug names) ──
    "başlandı", "başlanmış", "başlanıyor", "verildi", "verilmekte", "verilmiş",
    "uygulandı", "uygulanıyor", "uygulanmakta", "kullanıyor", "kullanmakta",
    "kullanan", "kullandığı", "aldığı", "kesildi", "kesilmiş", "bırakıldı",
    "reçete", "tedavi",
    # ── Turkish symptoms / descriptors ──
    "güçsüzlük", "uyuşma", "karıncalanma", "felç", "titreme", "nöbet",
    "baş ağrısı", "sersemlik", "halsizlik", "yorgunluk", "ishal", "kabızlık",
    "kaşıntı", "hassasiyet", "sertlik", "kramp", "yanma",
    "bulanık", "çift", "şiddetli", "hafif", "orta", "kronik", "akut",
    # ── Likely lab abbreviations (all-caps short) ──
    "WBC", "RBC", "HGB", "PLT", "CRP", "ESR", "BUN", "ALT", "AST", "GFR",
    "INR", "PTT", "ABG", "BMP", "CBC", "ECG", "EKG", "MRI", "CT",
    "LDH", "TSH", "HBA", "PSA", "AFP", "CEA", "BNP", "DDimer",
})

# ── Morphological filter: English word endings that real drugs almost never have ──
# Drug names rarely end in common English adjective/verb suffixes.
# This catches thousands of false positives like "impaired", "drooping", "swollen" etc.
_NON_DRUG_SUFFIXES = re.compile(
    r'(?:ness|ment|tion|sion|ling|ting|ning|ring|king|ving|ying|ping|bing|'
    r'ated|ized|ised|ured|ored|ered|ised|ised|ived|oved|'
    r'able|ible|ical|ious|eous|ular|ular|ence|ance|'
    r'ful|less|ward|wise|like|ment|ship|hood|ness|'
    r'edly|ingly|ously|ively|ally|ably|ibly|'
    r'ing|ism|ist)$',
    re.IGNORECASE,
)

# Suffixes that ARE common in drug names — override the morphological filter
_DRUG_SAFE_SUFFIXES = re.compile(
    r'(?:olol|pril|artan|statin|mab|nib|pine|zole|done|pam|lam|'
    r'oxin|mycin|cillin|cycline|floxacin|azole|conazole|'
    r'triptan|lukast|gliptin|tide|mide|nide|'
    r'amine|azine|idine|ophen|profen|oxicam|'
    r'barbital|diazepine|morphine|codone|'
    r'ase|plase|kinase|mase)$',
    re.IGNORECASE,
)


def extract_drug_candidates(text: str) -> list[str]:
    """Extract potential drug names from patient text using context clues.

    Pure pattern-based detection — no drug dictionary. Uses:
      1. Dosage patterns: "Dideral 40mg"
      2. Medication context (EN+TR): "taking Dideral", "kullandığı Concor"
      3. Medication list parsing: "Meds: X, Y, Z"
      4. Parenthetical mentions: "Dideral (propranolol)"

    Returns deduplicated candidate list to validate via external APIs.
    """
    candidates: set[str] = set()

    # ── Pattern 1: Drug name + dosage unit ──
    for m in _DOSAGE_PATTERN.finditer(text):
        name = m.group(1).strip()
        if _is_viable_candidate(name):
            candidates.add(name)

    # ── Pattern 2: Medication context (English) ──
    for m in _MEDICATION_CONTEXT_EN.finditer(text):
        name = m.group(1).strip()
        if _is_viable_candidate(name):
            candidates.add(name)

    # ── Pattern 3: Medication context (Turkish) ──
    for m in _MEDICATION_CONTEXT_TR.finditer(text):
        name = m.group(1).strip()
        if _is_viable_candidate(name):
            candidates.add(name)

    # ── Pattern 4: Medication lists ──
    for m in _DRUG_LIST_PATTERN.finditer(text):
        list_text = m.group(1)
        for item in re.split(r'[,;/]', list_text):
            words = item.strip().split()
            if words:
                name = words[0].strip(".,;:()")
                if _is_viable_candidate(name):
                    candidates.add(name)

    # ── Pattern 5: Route-of-administration prefix ──
    for m in _ROUTE_PREFIX_PATTERN.finditer(text):
        name = m.group(1).strip()
        if _is_viable_candidate(name):
            candidates.add(name)

    # ── Pattern 6: Turkish reverse-order (Drug + verb) ──
    for m in _TR_REVERSE_PATTERN.finditer(text):
        name = m.group(1).strip()
        if _is_viable_candidate(name):
            candidates.add(name)

    # ── Pattern 7: Parenthetical mentions ──
    for m in _PAREN_DRUG_PATTERN.finditer(text):
        for grp in m.groups():
            if grp and _is_viable_candidate(grp.strip()):
                candidates.add(grp.strip())

    if candidates:
        logger.info(
            "[DRUG-EXTRACT] Found %d candidate(s): %s",
            len(candidates), ", ".join(sorted(candidates)),
        )
    return list(candidates)


def _is_viable_candidate(name: str) -> bool:
    """Check if a word could plausibly be a drug name.

    Three-layer filter:
      1. Blocklist: explicit non-drug words (symptoms, anatomy, pronouns, etc.)
      2. Morphology: reject common English adjective/verb suffixes (-ed, -ing, etc.)
         UNLESS the word also has a drug-safe suffix (-olol, -pril, etc.)
      3. Structural: reject short words, numbers, lab abbreviations
    """
    if not name or len(name) < 3:
        return False
    low = name.lower()
    if low in _NON_DRUG_WORDS:
        return False
    if not name[0].isalpha():
        return False
    # Reject all-uppercase abbreviations ≤ 5 chars (likely lab values)
    if name.isupper() and len(name) <= 5:
        return False
    # Reject pure numbers
    if name.isdigit():
        return False

    # ── Layer 2: Morphological filter ──
    # If the word ends in a common English suffix (-ed, -ing, -ness, etc.)
    # AND does NOT end in a known drug suffix (-olol, -mycin, etc.), reject it.
    if _NON_DRUG_SUFFIXES.search(low) and not _DRUG_SAFE_SUFFIXES.search(low):
        logger.debug("[DRUG-FILTER] Rejected '%s' — common English morphology", name)
        return False

    # ── Layer 3: Reject very short all-lowercase words (≤4 chars) ──
    # Real drug names ≤4 chars are almost always capitalized (e.g., "Tylenol" not "his")
    if len(name) <= 4 and name.islower():
        return False

    return True


def _validate_resolution(candidate: str, resolved: "DrugInfo") -> bool:
    """Post-resolution validation: ensure the resolved drug relates to the candidate.

    Prevents phantom matches like "impaired"→"tramadol" or "drooping"→"ropivacaine"
    where the API returned a seemingly random drug for a non-drug word.
    """
    original = candidate.lower().strip()
    generic = (resolved.generic_name or "").lower().strip()
    brand = (resolved.original_name or "").lower().strip()

    # If the candidate IS the generic or brand name (or close), it's valid
    if original == generic or original == brand:
        return True

    # Check if candidate is a substring of generic/brand or vice versa (≥4 chars overlap)
    if len(original) >= 4:
        if original in generic or generic in original:
            return True
        if original in brand or brand in original:
            return True

    # Check first-3-char match (e.g., "Aspirin" → "aspirin")
    if len(original) >= 3 and len(generic) >= 3 and original[:3] == generic[:3]:
        return True

    # If nothing matched, the API returned a phantom drug — reject
    logger.info(
        "[DRUG-VALIDATE] Rejected phantom resolution: '%s' → '%s' (no name similarity)",
        candidate, resolved.generic_name,
    )
    return False


# ═════════════════════════════════════════════════════════════════════
# Full Pipeline: Extract → Resolve → Format
# ═════════════════════════════════════════════════════════════════════

async def resolve_patient_drugs(text: str) -> list[DrugInfo]:
    """Full pipeline: extract drug candidates from text and resolve each via APIs.

    Runs resolutions in parallel for speed. Includes post-resolution validation
    to reject phantom matches (e.g., "impaired"→"tramadol").
    Returns successfully resolved + validated DrugInfo list.
    """
    candidates = extract_drug_candidates(text)
    if not candidates:
        return []

    # Resolve all candidates in parallel (API calls are I/O-bound)
    async def _safe_resolve(name: str) -> tuple[str, Optional[DrugInfo]]:
        try:
            result = await resolve_drug(name)
            return (name, result)
        except Exception as e:
            logger.warning("[DRUG-LOOKUP] Error resolving '%s': %s", name, e)
            return (name, None)

    results = await asyncio.gather(*[_safe_resolve(name) for name in candidates])

    # Post-resolution validation: reject phantom matches
    resolved: list[DrugInfo] = []
    rejected: list[str] = []
    for candidate_name, drug_info in results:
        if drug_info is None:
            continue
        if _validate_resolution(candidate_name, drug_info):
            resolved.append(drug_info)
        else:
            rejected.append(f"{candidate_name}→{drug_info.generic_name}")

    if rejected:
        logger.info(
            "[DRUG-LOOKUP] Rejected %d phantom resolution(s): %s",
            len(rejected), ", ".join(rejected),
        )

    if resolved:
        logger.info(
            "[DRUG-LOOKUP] Resolved %d/%d candidates: %s",
            len(resolved), len(candidates),
            ", ".join(f"{d.original_name}→{d.generic_name}" for d in resolved),
        )
    else:
        logger.info(
            "[DRUG-LOOKUP] None of %d candidates resolved as known drugs: %s",
            len(candidates), ", ".join(candidates),
        )

    return resolved


def format_drug_facts(drugs: list[DrugInfo]) -> str:
    """Format resolved drugs as a "Verified Drug Facts" section for LLM prompt injection.

    This section is clearly labeled as verified external data that the LLM
    MUST NOT contradict or hallucinate over.
    """
    if not drugs:
        return ""

    lines = [
        "## 💊 VERIFIED DRUG FACTS (from RxNorm/OpenFDA — DO NOT CONTRADICT)",
        "The following drug information was retrieved from RxNorm (National Library of Medicine)",
        "and OpenFDA (U.S. Food & Drug Administration). These are VERIFIED pharmaceutical facts.",
        "You MUST use these exact drug classes and mechanisms in your analysis.",
        "Do NOT substitute your own drug knowledge when it conflicts with these facts.",
        "",
    ]

    for drug in drugs:
        lines.append(drug.format_for_prompt())
        lines.append("")

    lines.append(
        "⚠️ PHARMACOLOGICAL SAFETY RULE: If the patient is already taking a drug from one class "
        "AND the treatment plan includes another drug from the SAME or CONFLICTING class, "
        "you MUST flag this as a potential drug duplication or dangerous interaction. "
        "For example: Beta-blocker + Beta-blocker = Dangerous. "
        "Beta-blocker + Pheochromocytoma = LETHAL (unopposed alpha stimulation)."
    )

    return "\n".join(lines)
