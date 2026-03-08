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
