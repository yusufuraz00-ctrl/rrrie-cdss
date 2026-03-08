"""Safety checks — input sanitization, red flag detection, disclaimer enforcement."""

from __future__ import annotations

import re


# ── Red-flag keywords that suggest emergency ────────────────────────────
RED_FLAG_KEYWORDS = [
    # Cardiovascular emergencies
    "chest pain",
    "tearing pain",
    "bp differential",
    # Neurological emergencies
    "severe headache",
    "thunderclap headache",
    "loss of consciousness",
    "syncope",
    "seizure",
    "stroke",
    "altered mental status",
    "confusion",
    "weakness in legs",
    "saddle anesthesia",
    "urinary retention",
    "descending paralysis",
    # Respiratory emergencies
    "hemoptysis",
    "coughing blood",
    "difficulty breathing",
    "respiratory distress",
    "stridor",
    # Infectious / exposure emergencies
    "hydrophobia",
    "difficulty swallowing water",
    "can't swallow",
    "animal bite",
    "dog bite",
    "petechial rash",
    "neck stiffness",
    "nuchal rigidity",
    "trismus",
    "crepitus",
    # Surgical emergencies
    "rebound tenderness",
    "pain out of proportion",
    "tracheal deviation",
    # Psychiatric / toxicity emergencies
    "suicidal",
    "anaphylaxis",
    "unresponsive",
    "severe bleeding",
    "hemorrhage",
    "clonus",
]

# ── Turkish red-flag keywords ───────────────────────────────────────────
RED_FLAG_KEYWORDS_TR = [
    # Kardiyovasküler aciller
    "göğüs ağrısı",
    "yırtılma tarzında ağrı",
    "kan basıncı farkı",
    "çarpıntı",
    # Nörolojik aciller
    "şiddetli baş ağrısı",
    "bilinç kaybı",
    "bayılma",
    "nöbet",
    "felç",
    "inme",
    "bilinç bulanıklığı",
    "konfüzyon",
    "bacaklarda güçsüzlük",
    "idrar tutamama",
    "inen paralizi",
    # Solunum acilleri
    "kan tükürme",
    "hemoptizi",
    "nefes darlığı",
    "solunum sıkıntısı",
    # Enfeksiyöz aciller
    "su içememe",
    "yutma güçlüğü",
    "hayvan ısırığı",
    "köpek ısırığı",
    "peteşiyal döküntü",
    "ense sertliği",
    # Cerrahi aciller
    "defans",
    "rebound hassasiyet",
    "trakeal deviasyon",
    # Psikiyatrik / toksikolojik aciller
    "intihar",
    "anafilaksi",
    "yanıtsız",
    "şiddetli kanama",
    "kanama",
]

# Pre-compile all red-flag patterns with word boundaries for fast matching
_RED_FLAG_PATTERNS: list[tuple[str, re.Pattern]] = []
for _kw in RED_FLAG_KEYWORDS + RED_FLAG_KEYWORDS_TR:
    _RED_FLAG_PATTERNS.append(
        (_kw, re.compile(r'(?<!\w)' + re.escape(_kw) + r'(?!\w)', re.IGNORECASE))
    )

RED_FLAG_VITALS = {
    "spo2_low": 92.0,       # SpO2 < 92% → critical
    "temp_high": 40.0,      # Temperature > 40°C → hyperpyrexia
    "hr_high": 130,         # HR > 130 → tachycardia concern
    "hr_low": 45,           # HR < 45  → bradycardia concern
    "rr_high": 30,          # RR > 30  → tachypnea
    "sbp_low": 90,          # Systolic BP < 90 → shock concern
}


def detect_red_flags(chief_complaint: str, symptoms: list[str]) -> list[str]:
    """Detect red-flag keywords in patient data (EN + TR, word-boundary safe)."""
    all_text = (chief_complaint + " " + " ".join(symptoms)).lower()
    found = []
    for kw, pattern in _RED_FLAG_PATTERNS:
        if pattern.search(all_text):
            found.append(f"RED FLAG detected: '{kw}' in patient data")
    return found


# ── Demographic-aware severity escalation ───────────────────────────
# Adaptive: [demographic pattern] + [critical vitals] → mandatory specialty alert.
# NOT disease-specific — catches any demographic-vital mismatch.

# Compiled patterns for demographic extraction (EN + TR)
_AGE_PATTERN = re.compile(
    r'(?:(\d{1,3})\s*(?:yaş|yaşında|yo|y/o|year[s]?\s*old|year[s]?|y\b))',
    re.IGNORECASE,
)
_SEX_FEMALE_PATTERN = re.compile(
    r'\b(?:female|kadın|kız|woman|F\b|bayan)',
    re.IGNORECASE,
)
_SEX_MALE_PATTERN = re.compile(
    r'\b(?:male|erkek|adam|M\b|bay)\b',
    re.IGNORECASE,
)
_ANTICOAGULANT_PATTERN = re.compile(
    r'\b(?:warfarin|coumadin|heparin|enoxaparin|rivaroxaban|apixaban|edoxaban|'
    r'dabigatran|kumadin|kan\s*sulandırıcı|antikoagül)',
    re.IGNORECASE,
)


def detect_demographic_severity(patient_text: str, vitals_flags: list[str]) -> list[str]:
    """Cross-reference patient demographics with critical vitals for severity escalation.

    Adaptive pattern: when shock/critical vitals appear alongside specific demographics,
    generate mandatory specialty consideration alerts. NOT disease-specific.

    Args:
        patient_text: Raw patient text.
        vitals_flags: Already-detected vital sign red flags from check_vitals_red_flags().

    Returns:
        List of demographic-aware severity alerts.
    """
    alerts: list[str] = []
    text_lower = patient_text.lower()

    # Extract age
    age_match = _AGE_PATTERN.search(patient_text)
    age = int(age_match.group(1)) if age_match else None

    # Detect sex
    is_female = bool(_SEX_FEMALE_PATTERN.search(patient_text))
    is_male = bool(_SEX_MALE_PATTERN.search(patient_text))

    # Detect shock pattern in vitals flags
    has_shock = any("shock" in f.lower() or "systolic bp" in f.lower() for f in vitals_flags)
    has_tachycardia = any("heart rate" in f.lower() and ">" in f for f in vitals_flags)
    has_critical_vitals = has_shock or has_tachycardia

    # Also check raw text for vital shock patterns not caught by structured vitals
    bp_match = re.search(r'(?:bp|ta|tansiyon|blood\s*pressure)\s*[:\s]*(\d+)/(\d+)', text_lower)
    hr_match = re.search(r'(?:hr|nabız|pulse|heart\s*rate)\s*[:\s]*(\d+)', text_lower)
    if bp_match:
        sbp = int(bp_match.group(1))
        if sbp < 90:
            has_shock = True
            has_critical_vitals = True
    if hr_match:
        hr = int(hr_match.group(1))
        if hr > 120:
            has_tachycardia = True
            has_critical_vitals = True

    # Has abdominal/pelvic pain
    has_abdominal_pain = bool(re.search(
        r'(?:karın|abdom|pelvi|batın|alt\s*karın|lower\s*abdomen|pelvic)\s*(?:ağrı|pain|hassas|tender)',
        text_lower,
    ))

    # Has syncope/LOC
    has_syncope = bool(re.search(
        r'\b(?:syncope|bayılma|bilinç\s*kaybı|loss\s*of\s*consciousness|fainted|passed\s*out)\b',
        text_lower,
    ))

    if not has_critical_vitals:
        return alerts

    # ─ Pattern 1: Reproductive-age female + shock → OB/GYN mandate
    if is_female and age is not None and 12 <= age <= 55 and has_shock:
        alerts.append(
            "🚨 DEMOGRAPHIC SEVERITY: Shock in reproductive-age female — "
            "MUST rule out obstetric/gynecological emergency "
            "(ectopic pregnancy, ovarian torsion, hemorrhage, ruptured cyst). "
            "OB/GYN differential is MANDATORY."
        )

    # ─ Pattern 2: Female + shock + abdominal pain → heightened OB/GYN
    if is_female and has_abdominal_pain and (has_shock or has_syncope):
        alerts.append(
            "🚨 DEMOGRAPHIC SEVERITY: Female + abdominal pain + hemodynamic instability — "
            "Gynecological/obstetric cause MUST be top differential until excluded. "
            "Referred shoulder pain in this context = hemoperitoneum until proven otherwise."
        )

    # ─ Pattern 3: Elderly + shock + anticoagulant → occult hemorrhage
    on_anticoagulant = bool(_ANTICOAGULANT_PATTERN.search(patient_text))
    if age is not None and age >= 65 and has_shock and on_anticoagulant:
        alerts.append(
            "🚨 DEMOGRAPHIC SEVERITY: Shock in anticoagulated elderly patient — "
            "MUST rule out occult hemorrhage (GI, retroperitoneal, intracranial). "
            "Anticoagulation + shock = hemorrhagic emergency until proven otherwise."
        )

    # ─ Pattern 4: Child + abdominal pain + shock → surgical emergency
    if age is not None and age < 14 and has_abdominal_pain and has_critical_vitals:
        alerts.append(
            "🚨 DEMOGRAPHIC SEVERITY: Pediatric patient + abdominal pain + critical vitals — "
            "MUST rule out surgical emergency (intussusception, volvulus, appendiceal perforation). "
            "Pediatric surgical differential is MANDATORY."
        )

    # ─ Pattern 5: Young adult + shock + syncope → vascular/hemorrhagic
    if age is not None and 15 <= age <= 45 and has_shock and has_syncope:
        alerts.append(
            "🚨 DEMOGRAPHIC SEVERITY: Young adult + shock + syncope — "
            "MUST rule out hemorrhagic and vascular emergencies "
            "(ruptured aneurysm, ectopic, splenic rupture, aortic dissection). "
            "Do NOT attribute to vasovagal without excluding hemorrhage."
        )

    return alerts


def check_vitals_red_flags(
    spo2: float | None = None,
    temperature: float | None = None,
    heart_rate: int | None = None,
    respiratory_rate: int | None = None,
    blood_pressure: str | None = None,
) -> list[str]:
    """Check vital signs for red-flag thresholds."""
    flags = []

    if spo2 is not None and spo2 < RED_FLAG_VITALS["spo2_low"]:
        flags.append(f"CRITICAL: SpO2 {spo2}% < {RED_FLAG_VITALS['spo2_low']}%")

    if temperature is not None and temperature > RED_FLAG_VITALS["temp_high"]:
        flags.append(f"CRITICAL: Temperature {temperature}°C > {RED_FLAG_VITALS['temp_high']}°C")

    if heart_rate is not None:
        if heart_rate > RED_FLAG_VITALS["hr_high"]:
            flags.append(f"WARNING: Heart rate {heart_rate}bpm > {RED_FLAG_VITALS['hr_high']}bpm")
        if heart_rate < RED_FLAG_VITALS["hr_low"]:
            flags.append(f"WARNING: Heart rate {heart_rate}bpm < {RED_FLAG_VITALS['hr_low']}bpm")

    if respiratory_rate is not None and respiratory_rate > RED_FLAG_VITALS["rr_high"]:
        flags.append(f"WARNING: RR {respiratory_rate} > {RED_FLAG_VITALS['rr_high']}")

    if blood_pressure is not None:
        match = re.match(r"(\d+)/(\d+)", blood_pressure)
        if match:
            sbp = int(match.group(1))
            if sbp < RED_FLAG_VITALS["sbp_low"]:
                flags.append(
                    f"CRITICAL: Systolic BP {sbp}mmHg < {RED_FLAG_VITALS['sbp_low']}mmHg (shock risk)"
                )

    return flags


def sanitize_input(text: str) -> str:
    """Basic input sanitization — remove potential injection patterns."""
    # Remove common prompt injection patterns
    patterns = [
        r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"you\s+are\s+now\s+(?:acting\s+as|a)\s+",
        r"system\s*:\s*",
        r"<\s*/?script\s*>",
    ]
    sanitized = text
    for pattern in patterns:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


MEDICAL_DISCLAIMER = (
    "⚠️ DISCLAIMER: This system does not provide medical advice. "
    "All outputs are for informational purposes only and do not replace "
    "professional medical evaluation. Clinical decisions must always be "
    "made by a qualified healthcare professional."
)
