"""Thin Safety Net — Universal Life-Threatening Threshold Checks.

This is NOT a "Logic Gate" or "Rule Engine". It does NOT reject diagnoses.
It ONLY checks for universally agreed-upon life-threatening lab/vital thresholds
and generates WARNINGS (not rejections).

Only ~10 rules, all based on immutable medical facts:
  - Potassium > 6.5 → cardiac arrest risk
  - SpO2 < 88% → critical hypoxia
  - etc.

All intelligent learning is handled by the ICL Engine.
This is just a thin "seatbelt" for patient safety.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("rrrie-cdss.safety_net")


@dataclass
class SafetyAlert:
    """A safety alert triggered by a critical threshold."""
    parameter: str      # e.g., "potassium"
    value: float        # e.g., 6.8
    threshold: str      # e.g., "> 6.5"
    alert: str          # Human-readable alert message
    severity: str       # "critical" | "warning"


# Universal life-threatening thresholds — these NEVER change
UNIVERSAL_CHECKS = [
    {
        "parameter": "potassium",
        "condition": "gt",
        "threshold": 6.5,
        "alert": "🚨 CRITICAL HYPERKALEMIA (K⁺ > 6.5) — Immediate cardiac arrest risk. "
                 "Requires urgent intervention: calcium gluconate, insulin+glucose, kayexalate.",
        "severity": "critical",
        "aliases": ["potassium", "k+", "k⁺", "serum_potassium", "serum potassium"],
    },
    {
        "parameter": "potassium",
        "condition": "lt",
        "threshold": 2.5,
        "alert": "🚨 SEVERE HYPOKALEMIA (K⁺ < 2.5) — Fatal arrhythmia risk. "
                 "Requires urgent IV potassium replacement with cardiac monitoring.",
        "severity": "critical",
        "aliases": ["potassium", "k+", "k⁺", "serum_potassium", "serum potassium"],
    },
    {
        "parameter": "spo2",
        "condition": "lt",
        "threshold": 88.0,
        "alert": "🚨 CRITICAL HYPOXIA (SpO₂ < 88%) — Tissue hypoxia. "
                 "Immediate supplemental oxygen, assess for respiratory failure.",
        "severity": "critical",
        "aliases": ["spo2", "spo₂", "o2_sat", "oxygen saturation", "o2sat"],
    },
    {
        "parameter": "heart_rate",
        "condition": "gt",
        "threshold": 150,
        "alert": "⚠️ SEVERE TACHYCARDIA (HR > 150) — Hemodynamic instability risk. "
                 "Assess for underlying cause: sepsis, PE, SVT, hemorrhage.",
        "severity": "warning",
        "aliases": ["heart_rate", "hr", "pulse", "heart rate"],
    },
    {
        "parameter": "heart_rate",
        "condition": "lt",
        "threshold": 40,
        "alert": "🚨 SEVERE BRADYCARDIA (HR < 40) — Cardiogenic shock risk. "
                 "Assess for heart block, prepare for atropine/pacing.",
        "severity": "critical",
        "aliases": ["heart_rate", "hr", "pulse", "heart rate"],
    },
    {
        "parameter": "systolic_bp",
        "condition": "lt",
        "threshold": 80,
        "alert": "🚨 HYPOTENSIVE SHOCK (SBP < 80) — Organ perfusion danger. "
                 "Immediate IV fluids, vasopressors if needed, identify source.",
        "severity": "critical",
        "aliases": ["systolic_bp", "sbp", "systolic", "systolic blood pressure"],
    },
    {
        "parameter": "temperature",
        "condition": "gt",
        "threshold": 41.0,
        "alert": "🚨 HYPERPYREXIA (T > 41°C) — Brain damage risk. "
                 "Aggressive cooling, consider neuroleptic malignant syndrome, heat stroke.",
        "severity": "critical",
        "aliases": ["temperature", "temp", "body temperature", "body_temp"],
    },
    {
        "parameter": "sodium",
        "condition": "lt",
        "threshold": 120,
        "alert": "🚨 SEVERE HYPONATREMIA (Na < 120) — Cerebral edema and seizure risk. "
                 "Careful correction to avoid osmotic demyelination.",
        "severity": "critical",
        "aliases": ["sodium", "na", "na+", "serum_sodium", "serum sodium"],
    },
    {
        "parameter": "glucose",
        "condition": "lt",
        "threshold": 40,
        "alert": "🚨 SEVERE HYPOGLYCEMIA (Glucose < 40) — Neuroglycopenia. "
                 "Immediate IV dextrose, assess for insulin overdose.",
        "severity": "critical",
        "aliases": ["glucose", "blood_sugar", "blood sugar", "blood_glucose", "bg"],
    },
    {
        "parameter": "ph",
        "condition": "lt",
        "threshold": 7.1,
        "alert": "🚨 SEVERE ACIDOSIS (pH < 7.1) — Multi-organ failure risk. "
                 "Identify cause: DKA, lactic acidosis, renal failure. Consider bicarbonate.",
        "severity": "critical",
        "aliases": ["ph", "blood_ph", "arterial_ph"],
    },
]


def check_patient_safety(patient_text: str, parsed_vitals: dict | None = None) -> list[SafetyAlert]:
    """Check patient data against universal safety thresholds.
    
    Extracts numeric values from patient text using regex patterns,
    then checks against UNIVERSAL_CHECKS.
    
    Args:
        patient_text: Raw patient clinical text
        parsed_vitals: Optional pre-parsed vitals dict (e.g., from R0)
    
    Returns:
        List of SafetyAlert for any triggered thresholds
    """
    alerts = []
    
    # Extract all numeric values from patient text
    extracted = _extract_lab_values(patient_text)
    
    # Merge with parsed vitals if available
    if parsed_vitals:
        for key, val in parsed_vitals.items():
            if isinstance(val, (int, float)):
                extracted[key.lower()] = val

    # Check each universal rule
    for check in UNIVERSAL_CHECKS:
        value = None
        matched_alias = None
        
        # Try each alias for this parameter
        for alias in check["aliases"]:
            if alias in extracted:
                value = extracted[alias]
                matched_alias = alias
                break
        
        if value is None:
            continue
        
        triggered = False
        if check["condition"] == "gt" and value > check["threshold"]:
            triggered = True
        elif check["condition"] == "lt" and value < check["threshold"]:
            triggered = True
        
        if triggered:
            threshold_str = f"> {check['threshold']}" if check["condition"] == "gt" else f"< {check['threshold']}"
            alert = SafetyAlert(
                parameter=check["parameter"],
                value=value,
                threshold=threshold_str,
                alert=check["alert"],
                severity=check["severity"],
            )
            alerts.append(alert)
            logger.warning(
                "[SAFETY-NET] 🚨 %s ALERT: %s = %.1f (%s)",
                alert.severity.upper(), check["parameter"], value, threshold_str,
            )

    return alerts


def format_safety_alerts(alerts: list[SafetyAlert]) -> str:
    """Format safety alerts for prompt injection or UI display."""
    if not alerts:
        return ""
    
    lines = ["🛡️ SAFETY NET ALERTS:"]
    for a in alerts:
        lines.append(f"  {a.alert}")
    return "\n".join(lines)


def _extract_lab_values(text: str) -> dict[str, float]:
    """Extract numeric lab/vital values from clinical text.
    
    Uses regex patterns to find patterns like:
      - "potassium: 6.1 mEq/L"
      - "K+: 6.1"
      - "spo2: 88%"
      - "temperature: 36.9"
      - "blood_pressure: 165/95"  (extracts systolic)
    """
    values = {}
    text_lower = text.lower()
    
    # Pattern: "parameter: value" or "parameter = value" or "parameter value"
    # Covers most clinical text formats
    patterns = [
        # Standard "key: value unit" format
        (r'(?:potassium|k\+|k⁺)\s*[:=]?\s*([\d.]+)', 'potassium'),
        (r'(?:sodium|na\+?)\s*[:=]?\s*([\d.]+)', 'sodium'),
        (r'(?:spo2|spo₂|o2.?sat)\s*[:=]?\s*([\d.]+)', 'spo2'),
        (r'(?:heart.?rate|hr|pulse)\s*[:=]?\s*(\d+)', 'heart_rate'),
        (r'(?:temperature|temp)\s*[:=]?\s*([\d.]+)', 'temperature'),
        (r'(?:glucose|blood.?sugar|bg)\s*[:=]?\s*([\d.]+)', 'glucose'),
        (r'(?:blood.?pressure|bp)\s*[:=]?\s*(\d+)\s*/\s*\d+', 'systolic_bp'),
        (r'(?:ph|blood.?ph)\s*[:=]?\s*([\d.]+)', 'ph'),
        (r'(?:creatinine|cr)\s*[:=]?\s*([\d.]+)', 'creatinine'),
        (r'(?:bicarbonate|hco3|bicarb)\s*[:=]?\s*([\d.]+)', 'bicarbonate'),
    ]
    
    for pattern, param_name in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                values[param_name] = float(match.group(1))
            except ValueError:
                pass
    
    return values
