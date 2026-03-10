"""Dynamic Narrative Case Bank — challenging clinical scenarios as patient stories.

Each case is a realistic ER-style narrative that tests the system's ability to:
- Detect rare/dangerous conditions from subtle behavioral clues
- Parse free-text patient stories (Turkish ER format)
- Identify misleading presentations (looks like X but is actually Y)
- Handle drug interactions and contraindications
- Catch pathognomonic signs buried in noise

Usage:
    from tests.narrative_cases import get_random_cases, get_case_by_id, ALL_NARRATIVE_CASES
"""

from __future__ import annotations

import random
from typing import Any


# ═════════════════════════════════════════════════════════════════════
# NARRATIVE CASE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════

NARRATIVE_CASES: list[dict[str, Any]] = [
    # ── N-001: RABIES (user's case) ─────────────────────────────────
    {
        "case_id": "N-001",
        "title": "Rabies — Hydrophobia masked as panic attack",
        "difficulty": "hard",
        "category": "infectious",
        "narrative": (
            "28 y, Male. Presents to ER with dyspnea and severe palpitations. "
            "Patient cannot sit still, keeps pacing the room, and appears extremely agitated. "
            "Reports tingling and numbness in right arm — 'my arm is going numb, am I having a heart attack?'. "
            "Temp: 38.5°C, HR: 110, BP: 140/90, RR: 22, SpO2: 97%. "
            "When the triage nurse offered water, the patient brought the cup to his mouth but immediately "
            "gagged and spit it out saying 'my throat locks up, I can't swallow'. "
            "ECG: normal sinus rhythm, no ST changes. "
            "Patient repeatedly asks for a sedative injection. "
            "On further questioning, patient mentions being bitten by a stray dog 6 weeks ago on the right arm "
            "but 'it wasn't that bad, I just washed it with soap'."
        ),
        "patient_data": {
            "age": 28,
            "sex": "male",
            "chief_complaint": "Dyspnea, palpitations, agitation, right arm numbness",
            "symptoms": [
                "Dyspnea",
                "Palpitations",
                "Agitation — cannot sit still",
                "Right arm tingling and numbness",
                "Difficulty swallowing water (hydrophobia)",
                "Fever",
            ],
            "vitals": {
                "temperature": 38.5,
                "heart_rate": 110,
                "respiratory_rate": 22,
                "blood_pressure": "140/90",
                "spo2": 97.0,
            },
            "history": "Dog bite 6 weeks ago on right arm, wound washed with soap only, no post-exposure prophylaxis",
            "medications": [],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Rabies",
            "expected_icd11_codes": ["1D47"],
            "should_detect": [
                "Hydrophobia — pathognomonic for rabies",
                "Recent animal bite history",
                "Need for post-exposure prophylaxis evaluation",
                "Agitation as neurological symptom, not anxiety",
            ],
            "red_flags": ["Hydrophobia", "Agitation", "Fever with animal bite history"],
        },
    },

    # ── N-002: AORTIC DISSECTION ────────────────────────────────────
    {
        "case_id": "N-002",
        "title": "Aortic Dissection — Tearing back pain with BP differential",
        "difficulty": "hard",
        "category": "cardiovascular",
        "narrative": (
            "62 y, Male. Brought to ER by ambulance with sudden onset severe chest and back pain. "
            "Patient describes the pain as 'tearing' that started between his shoulder blades and migrated "
            "to the front of his chest. Onset was sudden about 45 minutes ago while lifting a heavy box. "
            "PMH: uncontrolled hypertension (doesn't take meds regularly), smoker 40 pack-years. "
            "Vitals: BP right arm 180/110, BP left arm 145/85 (35mmHg difference!), HR: 95, "
            "RR: 20, SpO2: 96%, Temp: 36.8°C. "
            "Patient is diaphoretic, pale, clutching his chest. "
            "ECG shows mild LVH but no acute ST changes. "
            "D-dimer markedly elevated. Chest X-ray shows widened mediastinum."
        ),
        "patient_data": {
            "age": 62,
            "sex": "male",
            "chief_complaint": "Sudden tearing chest and back pain",
            "symptoms": [
                "Sudden severe tearing pain between shoulder blades",
                "Pain migrating from back to chest",
                "Diaphoresis",
                "Pallor",
            ],
            "vitals": {
                "temperature": 36.8,
                "heart_rate": 95,
                "respiratory_rate": 20,
                "blood_pressure": "180/110",
                "spo2": 96.0,
            },
            "history": "Uncontrolled hypertension, non-compliant with medications, 40 pack-year smoking",
            "medications": [],
            "allergies": [],
            "imaging": "Chest X-ray: widened mediastinum",
            "lab_results": {"D-dimer": "markedly elevated"},
        },
        "expected_output": {
            "primary_diagnosis": "Aortic dissection",
            "expected_icd11_codes": ["BD42"],
            "should_detect": [
                "BP differential between arms",
                "Tearing pain — classic dissection descriptor",
                "Widened mediastinum on CXR",
                "Need for CT angiography",
                "Avoid thrombolytics",
            ],
            "red_flags": ["Tearing chest pain", "BP differential", "Widened mediastinum"],
        },
    },

    # ── N-003: CARBON MONOXIDE POISONING ────────────────────────────
    {
        "case_id": "N-003",
        "title": "CO Poisoning — Family cluster with headaches",
        "difficulty": "medium",
        "category": "toxicology",
        "narrative": (
            "34 y, Female. Presents with her husband (36) and two children (8 and 5) to ER. "
            "All four have headaches, nausea, and dizziness that started this morning. "
            "Patient says 'we all got sick at the same time, must be food poisoning from last night's dinner'. "
            "The symptoms improve when they step outside for fresh air but worsen when they go back inside. "
            "Their gas heater was recently serviced by 'a neighbor who knows about these things'. "
            "Vitals: BP 120/80, HR 100, RR 18, SpO2 reads 99% on pulse oximeter. "
            "Patient appears cherry-red, which she attributes to 'blushing from the nausea'. "
            "Temp: 37.0°C. Neuro exam shows mild confusion."
        ),
        "patient_data": {
            "age": 34,
            "sex": "female",
            "chief_complaint": "Headache, nausea, and dizziness — entire family affected simultaneously",
            "symptoms": [
                "Headache",
                "Nausea",
                "Dizziness",
                "Symptoms improve outdoors",
                "Cherry-red skin",
                "Mild confusion",
            ],
            "vitals": {
                "temperature": 37.0,
                "heart_rate": 100,
                "respiratory_rate": 18,
                "blood_pressure": "120/80",
                "spo2": 99.0,
            },
            "history": "Gas heater recently serviced by non-professional. Entire family symptomatic simultaneously.",
            "medications": [],
            "allergies": [],
            "social_history": "Lives in small apartment with gas heating, poor ventilation",
        },
        "expected_output": {
            "primary_diagnosis": "Carbon monoxide poisoning",
            "expected_icd11_codes": ["NE61"],
            "should_detect": [
                "Family cluster — environmental exposure",
                "Symptoms improve with fresh air",
                "SpO2 falsely normal (pulse ox reads COHb as O2Hb)",
                "Cherry-red skin",
                "Gas heater as source",
                "Need for CO-oximetry, not pulse ox",
            ],
            "red_flags": ["Family cluster", "Environmental exposure", "Confusion"],
        },
    },

    # ── N-004: WERNICKE ENCEPHALOPATHY ──────────────────────────────
    {
        "case_id": "N-004",
        "title": "Wernicke Encephalopathy — Confused alcoholic, DO NOT give glucose first",
        "difficulty": "hard",
        "category": "neurological",
        "narrative": (
            "52 y, Male. Found confused wandering near a park by police. "
            "Patient is a known chronic alcoholic, last seen sober 4 days ago. "
            "On exam: confused, ataxic gait (staggering, not just drunk), bilateral lateral gaze palsy "
            "(nystagmus on horizontal gaze). "
            "Temp: 36.5°C, HR: 88, BP: 100/60, RR: 16, SpO2: 96%. "
            "Blood glucose: 55 mg/dL (low). "
            "The intern says 'let's push D50 for the hypoglycemia'. "
            "Smell of alcohol but BAC is actually 0.02% (nearly sober). "
            "BMI appears low, malnourished appearance. Last meal unknown."
        ),
        "patient_data": {
            "age": 52,
            "sex": "male",
            "chief_complaint": "Confusion, ataxia, abnormal eye movements",
            "symptoms": [
                "Confusion / altered mental status",
                "Ataxic gait",
                "Bilateral lateral gaze palsy",
                "Nystagmus",
                "Hypoglycemia (glucose 55 mg/dL)",
                "Malnourished appearance",
            ],
            "vitals": {
                "temperature": 36.5,
                "heart_rate": 88,
                "respiratory_rate": 16,
                "blood_pressure": "100/60",
                "spo2": 96.0,
            },
            "history": "Chronic alcoholism, malnourished, last sober 4 days ago",
            "medications": [],
            "allergies": [],
            "lab_results": {"glucose": "55 mg/dL", "BAC": "0.02%"},
            "social_history": "Chronic alcoholism, homeless, malnourished",
        },
        "expected_output": {
            "primary_diagnosis": "Wernicke encephalopathy",
            "expected_icd11_codes": ["6D80.0"],
            "should_detect": [
                "Classic triad: confusion + ataxia + ophthalmoplegia",
                "MUST give thiamine BEFORE glucose (glucose without thiamine can precipitate/worsen Wernicke)",
                "Chronic alcoholism as risk factor for thiamine deficiency",
                "Low BAC means this is NOT acute intoxication",
            ],
            "red_flags": ["Altered mental status", "Hypoglycemia", "Thiamine before glucose"],
        },
    },

    # ── N-005: TENSION PNEUMOTHORAX ─────────────────────────────────
    {
        "case_id": "N-005",
        "title": "Tension Pneumothorax — Post-trauma with hemodynamic collapse",
        "difficulty": "medium",
        "category": "emergency",
        "narrative": (
            "22 y, Male. Motorcycle accident 30 minutes ago. GCS 14 on arrival. "
            "Complains of severe right chest pain and increasing difficulty breathing. "
            "Vitals: HR 130, BP 85/50 and dropping, RR 32, SpO2 82%. "
            "Exam: absent breath sounds on right side, trachea deviated to LEFT, "
            "distended jugular veins, hyperresonance on right percussion. "
            "Patient becoming progressively agitated and confused. "
            "IV access established, crystalloid bolus started. "
            "The intern wants to wait for a chest X-ray before doing anything."
        ),
        "patient_data": {
            "age": 22,
            "sex": "male",
            "chief_complaint": "Post-trauma: severe chest pain, progressive dyspnea, hemodynamic instability",
            "symptoms": [
                "Severe right chest pain",
                "Progressive dyspnea",
                "Hypotension (BP 85/50)",
                "Absent breath sounds right side",
                "Tracheal deviation to left",
                "JVD — jugular vein distension",
                "Hyperresonance on right",
                "Agitation and confusion",
            ],
            "vitals": {
                "temperature": 36.8,
                "heart_rate": 130,
                "respiratory_rate": 32,
                "blood_pressure": "85/50",
                "spo2": 82.0,
            },
            "history": "Motorcycle accident 30 minutes ago",
            "medications": [],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Tension pneumothorax",
            "expected_icd11_codes": ["CB23"],
            "should_detect": [
                "Do NOT wait for imaging — clinical diagnosis, immediate needle decompression",
                "Beck's triad analogue: hypotension + JVD + absent breath sounds",
                "Tracheal deviation = mediastinal shift = tension",
                "SpO2 82% is critical",
            ],
            "red_flags": ["SpO2 < 92%", "Hypotension", "Tracheal deviation", "Tachycardia"],
        },
    },

    # ── N-006: ECTOPIC PREGNANCY ────────────────────────────────────
    {
        "case_id": "N-006",
        "title": "Ectopic Pregnancy — Masked as gastroenteritis",
        "difficulty": "hard",
        "category": "obstetric_emergency",
        "narrative": (
            "26 y, Female. Presents with lower abdominal pain and nausea for 2 days. "
            "She says 'I think I ate something bad, I've been nauseous and my stomach hurts'. "
            "Reports some vaginal spotting which she attributes to 'my period coming early — it's been irregular lately'. "
            "LMP was 7 weeks ago. She says she's 'definitely not pregnant' because she uses condoms 'most of the time'. "
            "Vitals: HR 105, BP 100/65, RR 18, Temp 37.1°C, SpO2 98%. "
            "Abdomen: tender in left lower quadrant with mild rebound. "
            "She becomes light-headed when sitting up. "
            "The triage nurse noted her as 'gastroenteritis query' and placed her in the low-priority queue."
        ),
        "patient_data": {
            "age": 26,
            "sex": "female",
            "chief_complaint": "Lower abdominal pain for 2 days, nausea, vaginal spotting",
            "symptoms": [
                "Lower abdominal pain — left lower quadrant",
                "Nausea",
                "Vaginal spotting",
                "Orthostatic light-headedness",
                "Mild rebound tenderness",
            ],
            "vitals": {
                "temperature": 37.1,
                "heart_rate": 105,
                "respiratory_rate": 18,
                "blood_pressure": "100/65",
                "spo2": 98.0,
            },
            "history": "LMP 7 weeks ago, irregular periods, inconsistent condom use",
            "medications": [],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Ectopic pregnancy",
            "expected_icd11_codes": ["JA04"],
            "should_detect": [
                "Reproductive age female with abdominal pain → must rule out ectopic",
                "LMP 7 weeks ago with spotting — NOT early period",
                "Orthostatic symptoms suggest hemodynamic compromise",
                "Need urgent beta-hCG and transvaginal ultrasound",
                "Do NOT dismiss as gastroenteritis",
            ],
            "red_flags": ["Reproductive age female", "Abdominal pain with spotting", "Orthostatic symptoms"],
        },
    },

    # ── N-007: ADDISONIAN CRISIS ────────────────────────────────────
    {
        "case_id": "N-007",
        "title": "Addisonian Crisis — Hypotension refractory to fluids",
        "difficulty": "hard",
        "category": "endocrine",
        "narrative": (
            "45 y, Female. Brought in by family for 'extreme weakness and vomiting for 3 days'. "
            "Patient has been on prednisone 10mg daily for rheumatoid arthritis for 2 years "
            "but ran out 10 days ago and 'didn't think it mattered'. "
            "Vitals: BP 75/45 (not responding to 2L crystalloid), HR 120, Temp: 37.8°C, "
            "RR 22, SpO2 94%. "
            "Exam: diffuse hyperpigmentation especially on palmar creases and gingival mucosa, "
            "severe abdominal tenderness. "
            "Labs: Na 125, K 6.2, glucose 50 mg/dL, cortisol 1.2 mcg/dL (critically low). "
            "The resident started vasopressors but BP still won't come up."
        ),
        "patient_data": {
            "age": 45,
            "sex": "female",
            "chief_complaint": "Severe weakness, vomiting, hypotension refractory to fluids",
            "symptoms": [
                "Extreme weakness for 3 days",
                "Vomiting",
                "Refractory hypotension",
                "Hyperpigmentation — palmar creases and gums",
                "Abdominal tenderness",
            ],
            "vitals": {
                "temperature": 37.8,
                "heart_rate": 120,
                "respiratory_rate": 22,
                "blood_pressure": "75/45",
                "spo2": 94.0,
            },
            "history": "Rheumatoid arthritis, chronic prednisone use (2 years), abruptly stopped 10 days ago",
            "medications": ["Prednisone 10mg daily — STOPPED 10 days ago"],
            "allergies": [],
            "lab_results": {
                "Na": "125 mEq/L (low)",
                "K": "6.2 mEq/L (high)",
                "glucose": "50 mg/dL (low)",
                "cortisol": "1.2 mcg/dL (critically low)",
            },
        },
        "expected_output": {
            "primary_diagnosis": "Addisonian crisis (adrenal insufficiency)",
            "expected_icd11_codes": ["5A74.1"],
            "should_detect": [
                "Abrupt steroid withdrawal → adrenal crisis",
                "Classic labs: hyponatremia + hyperkalemia + hypoglycemia + low cortisol",
                "Hyperpigmentation (indicates chronic primary adrenal insufficiency)",
                "Must give IV hydrocortisone immediately (100mg bolus)",
                "Vasopressors won't work without cortisol replacement",
            ],
            "red_flags": ["Refractory hypotension", "Hyperkalemia", "Hypoglycemia"],
        },
    },

    # ── N-008: CAUDA EQUINA SYNDROME ────────────────────────────────
    {
        "case_id": "N-008",
        "title": "Cauda Equina — Back pain with urinary retention",
        "difficulty": "medium",
        "category": "neurological_emergency",
        "narrative": (
            "55 y, Male. Presents with severe lower back pain radiating to both legs for 3 days. "
            "Today noticed he couldn't feel when wiping after using the toilet — 'the area around my rear is numb'. "
            "Wife noticed he's been dribbling urine without realizing it. "
            "Also has progressive bilateral leg weakness — 'my legs feel like jelly'. "
            "PMH: lumbar disc herniation L4-L5 diagnosed 2 years ago, managed conservatively. "
            "Vitals: all normal. HR 75, BP 135/85, Temp 36.7°C. "
            "Exam: decreased perianal sensation (saddle anesthesia), bilateral ankle dorsiflexion weakness (4/5), "
            "absent ankle jerks bilaterally, post-void residual >300mL on bladder scan. "
            "The patient says 'it's just my old disc acting up again, give me some painkillers and I'll be fine'."
        ),
        "patient_data": {
            "age": 55,
            "sex": "male",
            "chief_complaint": "Severe low back pain with bilateral leg weakness and urinary incontinence",
            "symptoms": [
                "Severe lower back pain radiating bilaterally",
                "Saddle anesthesia — perineal numbness",
                "Urinary retention / overflow incontinence",
                "Bilateral leg weakness",
                "Absent ankle reflexes bilaterally",
            ],
            "vitals": {
                "temperature": 36.7,
                "heart_rate": 75,
                "respiratory_rate": 16,
                "blood_pressure": "135/85",
                "spo2": 98.0,
            },
            "history": "Lumbar disc herniation L4-L5 (2 years ago)",
            "medications": [],
            "allergies": [],
            "lab_results": {"post-void residual": ">300 mL"},
        },
        "expected_output": {
            "primary_diagnosis": "Cauda equina syndrome",
            "expected_icd11_codes": ["ME84.2"],
            "should_detect": [
                "Surgical emergency — needs urgent MRI and decompression",
                "Classic triad: saddle anesthesia + urinary retention + bilateral leg weakness",
                "Do NOT just give painkillers and discharge",
                "Delay in diagnosis/surgery → permanent neurological damage",
            ],
            "red_flags": ["Saddle anesthesia", "Urinary retention", "Bilateral leg weakness"],
        },
    },

    # ── N-009: NECROTIZING FASCIITIS ────────────────────────────────
    {
        "case_id": "N-009",
        "title": "Necrotizing Fasciitis — Pain out of proportion to exam",
        "difficulty": "hard",
        "category": "surgical_emergency",
        "narrative": (
            "58 y, Female with Type 2 DM (poorly controlled, HbA1c 9.5%). "
            "Presents with a 'small scratch' on her right shin from 2 days ago that's getting worse. "
            "The wound itself looks like a minor skin break with mild surrounding redness — "
            "but the patient is screaming in agony, rating pain 10/10. "
            "The skin around the wound feels warm, slightly tense, and has a 'wooden' feel on palpation. "
            "Today she developed purple-gray discoloration extending 5cm beyond the visible redness. "
            "Vitals: HR 115, BP 90/55, Temp 39.2°C, RR 24, SpO2 95%. "
            "Labs: WBC 22,000, CRP 280, lactate 4.5, creatinine 2.3 (baseline 0.9). "
            "Crepitus palpable around the wound. "
            "The junior doctor prescribed oral antibiotics and wound dressing change."
        ),
        "patient_data": {
            "age": 58,
            "sex": "female",
            "chief_complaint": "Small leg wound with extreme pain disproportionate to appearance",
            "symptoms": [
                "Extreme pain 10/10 — disproportionate to wound appearance",
                "Purple-gray skin discoloration",
                "Wooden/tense skin texture",
                "Crepitus around wound",
                "Sepsis signs: tachycardia, hypotension, fever",
            ],
            "vitals": {
                "temperature": 39.2,
                "heart_rate": 115,
                "respiratory_rate": 24,
                "blood_pressure": "90/55",
                "spo2": 95.0,
            },
            "history": "Type 2 DM (poorly controlled, HbA1c 9.5%)",
            "medications": ["Metformin 1000mg BID", "Glipizide 5mg daily"],
            "allergies": [],
            "lab_results": {
                "WBC": "22,000",
                "CRP": "280 mg/L",
                "lactate": "4.5 mmol/L",
                "creatinine": "2.3 mg/dL (baseline 0.9)",
            },
        },
        "expected_output": {
            "primary_diagnosis": "Necrotizing fasciitis",
            "expected_icd11_codes": ["1B70"],
            "should_detect": [
                "Pain disproportionate to exam — hallmark of necrotizing fasciitis",
                "Crepitus = gas in tissue = surgical emergency",
                "Oral antibiotics are INADEQUATE — needs IV broad-spectrum + urgent surgical debridement",
                "LRINEC score calculation for necrotizing fasciitis risk",
                "DM as major risk factor",
                "Acute kidney injury (creatinine doubling)",
            ],
            "red_flags": ["Sepsis", "Hypotension", "Crepitus", "Pain disproportionate to exam"],
        },
    },

    # ── N-010: PHEOCHROMOCYTOMA ─────────────────────────────────────
    {
        "case_id": "N-010",
        "title": "Pheochromocytoma — Episodic hypertensive crises",
        "difficulty": "hard",
        "category": "endocrine",
        "narrative": (
            "38 y, Female. Presents to ER with 'the worst headache of my life' plus pounding heart "
            "and profuse sweating. She reports these episodes happen 2-3 times per week for the past month, "
            "each lasting 20-30 minutes, then 'everything goes back to normal'. "
            "Today's episode has lasted over an hour. "
            "Vitals DURING episode: BP 220/130, HR 140, diaphoretic, pale, tremulous. "
            "BP between episodes (recorded by her GP last week): 125/80. "
            "PMH: 'anxiety disorder' diagnosed 6 months ago — started on propranolol by her GP. "
            "She says 'the propranolol made me WORSE — my BP went to 250 during an attack after starting it'. "
            "Family history: father had 'some adrenal surgery' at age 42."
        ),
        "patient_data": {
            "age": 38,
            "sex": "female",
            "chief_complaint": "Episodic severe headache, palpitations, diaphoresis with hypertensive crisis",
            "symptoms": [
                "Severe episodic headache",
                "Palpitations — heart pounding",
                "Profuse diaphoresis",
                "Pallor",
                "Tremor",
                "Episodic hypertension (normal between attacks)",
                "Worsened by propranolol",
            ],
            "vitals": {
                "temperature": 37.2,
                "heart_rate": 140,
                "respiratory_rate": 22,
                "blood_pressure": "220/130",
                "spo2": 97.0,
            },
            "history": "Diagnosed with 'anxiety' 6 months ago, on propranolol (worsened BP)",
            "medications": ["Propranolol 40mg BID"],
            "allergies": [],
            "family_history": "Father — adrenal surgery at age 42",
        },
        "expected_output": {
            "primary_diagnosis": "Pheochromocytoma",
            "expected_icd11_codes": ["5A71.0"],
            "should_detect": [
                "Classic triad: headache + palpitations + diaphoresis",
                "Episodic pattern with normal BP between attacks",
                "Propranolol worsened BP (unopposed alpha stimulation — DANGEROUS)",
                "Family history of adrenal surgery suggests MEN2",
                "Need plasma metanephrines / 24h urine catecholamines",
                "Alpha-blocker FIRST (phenoxybenzamine), then beta-blocker",
            ],
            "red_flags": ["BP 220/130", "Tachycardia 140", "Propranolol paradox"],
        },
    },

    # ── N-011: ACUTE INTERMITTENT PORPHYRIA ─────────────────────────
    {
        "case_id": "N-011",
        "title": "Acute Intermittent Porphyria — Abdominal pain with psych symptoms",
        "difficulty": "hard",
        "category": "metabolic",
        "narrative": (
            "25 y, Female. Presents with severe colicky abdominal pain for 24 hours, nausea, vomiting. "
            "Has had 3 similar episodes in the past year — each time admitted for 'non-specific abdominal pain', "
            "given morphine, workup negative, discharged with IBS diagnosis. "
            "This time she also reports anxiety, insomnia for 3 days, and 'hear things that aren't there.' "
            "Her husband noted she's been confused and agitated. "
            "Recent trigger: started a new oral contraceptive (combined pill) 5 days ago. "
            "Vitals: HR 110, BP 150/95, Temp 37.0°C. "
            "Abdomen: diffuse tenderness but soft, no peritoneal signs. "
            "Urine appeared dark reddish-brown — nurse assumed UTI. "
            "When urine sample was left on the counter by the window, it turned dark port-wine color after 30 min. "
            "Labs: mild hyponatremia (Na 128), otherwise unremarkable abdominal CT."
        ),
        "patient_data": {
            "age": 25,
            "sex": "female",
            "chief_complaint": "Severe recurrent abdominal pain with psychiatric symptoms",
            "symptoms": [
                "Severe colicky abdominal pain",
                "Nausea and vomiting",
                "Anxiety",
                "Insomnia",
                "Auditory hallucinations",
                "Confusion and agitation",
                "Dark reddish-brown urine (darkens in sunlight)",
                "Tachycardia",
                "Hypertension",
            ],
            "vitals": {
                "temperature": 37.0,
                "heart_rate": 110,
                "respiratory_rate": 18,
                "blood_pressure": "150/95",
                "spo2": 98.0,
            },
            "history": "3 previous episodes of 'non-specific abdominal pain', diagnosed with IBS. Recently started combined OCP.",
            "medications": ["Combined oral contraceptive pill (started 5 days ago)"],
            "allergies": [],
            "lab_results": {"Na": "128 mEq/L (low)", "abdominal_CT": "unremarkable"},
        },
        "expected_output": {
            "primary_diagnosis": "Acute intermittent porphyria",
            "expected_icd11_codes": ["5C58.00"],
            "should_detect": [
                "Recurrent abdo pain with negative workup — think porphyria",
                "Psych symptoms + abdo pain + dark urine (port-wine in sunlight) = porphyria",
                "OCP is a known precipitant (hormonal trigger)",
                "Need urine porphobilinogen (PBG) and delta-ALA",
                "STOP the OCP immediately",
                "Treatment: IV hemin, glucose loading",
            ],
            "red_flags": ["Psychiatric symptoms", "Recurrent unexplained pain", "Dark urine"],
        },
    },

    # ── N-012: BOTULISM ─────────────────────────────────────────────
    {
        "case_id": "N-012",
        "title": "Botulism — Descending paralysis after home-canned food",
        "difficulty": "hard",
        "category": "infectious",
        "narrative": (
            "42 y, Male. Presents with blurred vision and difficulty swallowing since yesterday. "
            "Today his wife noticed his eyelids drooping and his speech is slurred. "
            "He reports 'dry mouth' and hasn't urinated since last night. "
            "3 days ago, they ate home-preserved green beans from a jar that 'smelled a bit off but we ate them anyway'. "
            "His wife has milder symptoms (blurry vision only). "
            "Vitals: HR 70, BP 130/80, Temp 36.5°C, RR 16 (but shallow breaths). "
            "Exam: bilateral ptosis, diplopia, sluggish pupils, "
            "symmetric descending weakness (facial→neck→arms→proximal legs), "
            "NORMAL sensation, DTRs diminished. "
            "Gag reflex absent. "
            "The ER doc is considering Guillain-Barré syndrome and wants to do an LP."
        ),
        "patient_data": {
            "age": 42,
            "sex": "male",
            "chief_complaint": "Progressive blurred vision, dysphagia, bilateral ptosis, descending weakness",
            "symptoms": [
                "Blurred vision / diplopia",
                "Dysphagia — difficulty swallowing",
                "Bilateral ptosis (eyelid drooping)",
                "Slurred speech (dysarthria)",
                "Dry mouth",
                "Urinary retention",
                "Symmetric descending weakness",
                "Sluggish pupils",
                "Absent gag reflex",
            ],
            "vitals": {
                "temperature": 36.5,
                "heart_rate": 70,
                "respiratory_rate": 16,
                "blood_pressure": "130/80",
                "spo2": 96.0,
            },
            "history": "Ate home-preserved green beans 3 days ago (smelled off). Wife has milder symptoms.",
            "medications": [],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Botulism",
            "expected_icd11_codes": ["1C12"],
            "should_detect": [
                "Descending paralysis (vs GBS which is ascending)",
                "Home-canned food exposure — foodborne botulism",
                "Wife also symptomatic (shared exposure)",
                "Autonomic dysfunction: dry mouth, urinary retention, sluggish pupils",
                "Need botulinum antitoxin urgently — call CDC/health dept",
                "Monitor respiratory function closely — may need intubation",
                "NOT Guillain-Barré: GBS is ascending, sensory involvement, post-infectious",
            ],
            "red_flags": ["Descending paralysis", "Absent gag reflex", "Respiratory compromise risk"],
        },
    },

    # ── N-013: THYROID STORM ────────────────────────────────────────
    {
        "case_id": "N-013",
        "title": "Thyroid Storm — Post-surgery with dangerously high fever",
        "difficulty": "medium",
        "category": "endocrine_emergency",
        "narrative": (
            "35 y, Female. Post-op day 1 after cholecystectomy. Was stable overnight but suddenly developed "
            "high fever (40.5°C), severe agitation, and rapid irregular pulse. "
            "She keeps pulling out her IV lines and trying to get out of bed. "
            "PMH: Graves' disease diagnosed 1 year ago — was on methimazole but STOPPED 3 weeks ago "
            "because 'I read online it could cause liver damage and I was having surgery'. "
            "Vitals: HR 165 (irregularly irregular — new atrial fibrillation), BP 170/60 (wide pulse pressure), "
            "Temp 40.5°C, RR 28. "
            "Exam: exophthalmos, lid lag, diffuse goiter with bruit, "
            "hyperreflexia, warm moist skin, profuse diaphoresis. "
            "Staff are concerned about 'post-operative sepsis' and started broad-spectrum antibiotics."
        ),
        "patient_data": {
            "age": 35,
            "sex": "female",
            "chief_complaint": "Post-surgical: high fever, severe agitation, rapid irregular pulse",
            "symptoms": [
                "High fever 40.5°C",
                "Severe agitation — pulling IV lines",
                "Atrial fibrillation (new onset, HR 165)",
                "Wide pulse pressure (170/60)",
                "Exophthalmos, lid lag",
                "Goiter with bruit",
                "Hyperreflexia",
                "Profuse diaphoresis",
            ],
            "vitals": {
                "temperature": 40.5,
                "heart_rate": 165,
                "respiratory_rate": 28,
                "blood_pressure": "170/60",
                "spo2": 94.0,
            },
            "history": "Graves' disease (1 year). Stopped methimazole 3 weeks ago before surgery.",
            "medications": ["Methimazole — STOPPED 3 weeks ago"],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Thyroid storm",
            "expected_icd11_codes": ["5A00.1"],
            "should_detect": [
                "Surgery is a classic precipitant of thyroid storm",
                "Stopped methimazole before surgery — left uncontrolled hyperthyroidism",
                "Burch-Wartofsky score calculation",
                "Treatment: PTU > methimazole (blocks conversion), propranolol (rate control), hydrocortisone, cooling",
                "This is NOT sepsis — Graves' history + classic thyroid storm signs",
            ],
            "red_flags": ["Temp > 40°C", "HR > 130", "New atrial fibrillation", "Agitation"],
        },
    },

    # ── N-014: MENINGITIS (BACTERIAL) ───────────────────────────────
    {
        "case_id": "N-014",
        "title": "Bacterial Meningitis — College student with petechiae",
        "difficulty": "medium",
        "category": "infectious_emergency",
        "narrative": (
            "19 y, Female. College freshman, brought by roommate who says 'she's been getting worse all day'. "
            "Started with headache and fever this morning. By afternoon, she was confused, vomiting, "
            "and can't tolerate light in the dorm room. "
            "On ER exam: Temp 39.8°C, HR 120, BP 95/55, RR 22. "
            "Meningeal signs: nuchal rigidity, positive Kernig's and Brudzinski's. "
            "NON-BLANCHING PETECHIAL RASH on trunk and extremities — some coalescing into purpura. "
            "GCS has dropped to 12 since arrival (was 14 in triage). "
            "The intern wants to send her for CT head first before LP. "
            "Her vaccination record: did NOT receive meningococcal vaccine."
        ),
        "patient_data": {
            "age": 19,
            "sex": "female",
            "chief_complaint": "Headache, fever, confusion, neck stiffness, petechial rash",
            "symptoms": [
                "Severe headache",
                "High fever 39.8°C",
                "Photophobia",
                "Nuchal rigidity",
                "Positive Kernig's and Brudzinski's signs",
                "Non-blanching petechial rash (coalescing into purpura)",
                "Confusion (GCS 12, declining)",
                "Vomiting",
            ],
            "vitals": {
                "temperature": 39.8,
                "heart_rate": 120,
                "respiratory_rate": 22,
                "blood_pressure": "95/55",
                "spo2": 96.0,
            },
            "history": "College freshman, no meningococcal vaccine",
            "medications": [],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Meningococcal meningitis",
            "expected_icd11_codes": ["1C1C", "1D00"],
            "should_detect": [
                "Do NOT delay antibiotics for CT/LP — petechial rash + meningism = start empiric ABx IMMEDIATELY",
                "Non-blanching petechiae with meningism = meningococcal septicemia until proven otherwise",
                "Ceftriaxone + dexamethasone empirically within minutes",
                "Close contact prophylaxis needed (roommate, etc.)",
                "Declining GCS is alarming — may need ICU",
            ],
            "red_flags": ["Petechial rash", "Declining GCS", "Hypotension", "Meningeal signs"],
        },
    },

    # ── N-015: EPIDURAL ABSCESS ─────────────────────────────────────
    {
        "case_id": "N-015",
        "title": "Spinal Epidural Abscess — Back pain in IV drug user",
        "difficulty": "hard",
        "category": "neurological_emergency",
        "narrative": (
            "33 y, Male. Presents with severe midline thoracic back pain for 5 days, progressively worsening. "
            "Today developed weakness in both legs and 'it burns when I pee'. "
            "Patient admits to IV heroin use — last injection 10 days ago. "
            "Has a healed abscess on his left antecubital fossa. "
            "Vitals: Temp 38.9°C, HR 100, BP 120/75. "
            "Exam: exquisite midline tenderness over T8-T10 spinous processes, "
            "bilateral lower extremity weakness (3/5), decreased sensation below T10 on pin-prick, "
            "bladder distension on palpation. "
            "Labs: WBC 18,000, ESR 95, CRP 150, blood cultures pending. "
            "The ER doc ordered NSAIDs for 'muscular back pain' and was about to discharge."
        ),
        "patient_data": {
            "age": 33,
            "sex": "male",
            "chief_complaint": "Severe thoracic back pain with progressive bilateral leg weakness",
            "symptoms": [
                "Severe midline thoracic back pain (5 days)",
                "Bilateral lower extremity weakness",
                "Sensory level at T10",
                "Urinary retention (bladder distension)",
                "Fever",
                "Exquisite midline spinal tenderness T8-T10",
            ],
            "vitals": {
                "temperature": 38.9,
                "heart_rate": 100,
                "respiratory_rate": 18,
                "blood_pressure": "120/75",
                "spo2": 97.0,
            },
            "history": "IV heroin use, healed antecubital abscess",
            "medications": [],
            "allergies": [],
            "lab_results": {"WBC": "18,000", "ESR": "95 mm/hr", "CRP": "150 mg/L"},
            "social_history": "IV heroin use (active)",
        },
        "expected_output": {
            "primary_diagnosis": "Spinal epidural abscess",
            "expected_icd11_codes": ["FA40.2"],
            "should_detect": [
                "IVDU + fever + spinal tenderness + neuro deficits = epidural abscess until proven otherwise",
                "Urgent MRI spine with contrast — surgical emergency",
                "Do NOT discharge — neurological deficits are progressing",
                "IV antibiotics: vancomycin + ceftriaxone empirically",
                "Neurosurgical consult for possible decompression",
                "Elevated ESR + CRP in context of IVDU",
            ],
            "red_flags": ["Progressive neurological deficit", "Fever with spinal tenderness", "Urinary retention"],
        },
    },

    # ── N-016: LUDWIG'S ANGINA ──────────────────────────────────────
    {
        "case_id": "N-016",
        "title": "Ludwig's Angina — Floor of mouth infection threatening airway",
        "difficulty": "medium",
        "category": "ENT_emergency",
        "narrative": (
            "48 y, Male with poorly controlled type 2 DM. "
            "Presents with 3-day worsening sore throat, can barely open his mouth or swallow. "
            "He had a lower molar tooth pulled 5 days ago at a local clinic. "
            "Now has bilateral submandibular swelling — the floor of his mouth is firm, swollen, "
            "and his tongue is pushed upward and backward. "
            "Vitals: Temp 39.5°C, HR 115, BP 140/85, RR 24, SpO2 93% (he's starting to drool "
            "because he can't swallow his saliva). "
            "Voice is muffled ('hot potato voice'). "
            "Stridor audible without stethoscope. "
            "The intern is considering oral antibiotics for a dental abscess."
        ),
        "patient_data": {
            "age": 48,
            "sex": "male",
            "chief_complaint": "Severe submandibular swelling with airway compromise after dental extraction",
            "symptoms": [
                "Bilateral submandibular swelling (woody, firm)",
                "Tongue elevation and posterior displacement",
                "Trismus — can barely open mouth",
                "Dysphagia — drooling saliva",
                "Muffled voice (hot potato voice)",
                "Stridor",
                "Fever",
            ],
            "vitals": {
                "temperature": 39.5,
                "heart_rate": 115,
                "respiratory_rate": 24,
                "blood_pressure": "140/85",
                "spo2": 93.0,
            },
            "history": "Poorly controlled type 2 DM, lower molar extraction 5 days ago",
            "medications": ["Metformin 1000mg BID"],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Ludwig's angina",
            "expected_icd11_codes": ["DA03.0"],
            "should_detect": [
                "Airway emergency — potential for rapid obstruction",
                "NOT a simple dental abscess — bilateral submandibular space infection",
                "Stridor + drooling = impending airway loss",
                "Oral antibiotics are INADEQUATE — needs IV ampicillin-sulbactam or clindamycin",
                "May need emergent surgical drainage and/or intubation",
                "DM as risk factor for rapid spread",
            ],
            "red_flags": ["Stridor", "SpO2 < 94%", "Drooling", "Airway compromise"],
        },
    },

    # ── N-017: TETANUS ──────────────────────────────────────────────
    {
        "case_id": "N-017",
        "title": "Tetanus — Jaw stiffness after stepping on rusty nail",
        "difficulty": "medium",
        "category": "infectious",
        "narrative": (
            "67 y, Male. Rural farmer, presents with jaw stiffness for 2 days. "
            "Initially thought he 'slept wrong' but now can barely open his mouth. "
            "Stepped on a rusty nail in the field 8 days ago — wound was deep, "
            "he cleaned it 'with some water' and wrapped it. Never went to a doctor. "
            "Last tetanus shot: 'maybe when I was in the army, 45 years ago?' "
            "Today developed painful muscle spasms in his back and neck triggered by noise. "
            "Vitals: HR 105, BP 150/95, Temp 38.0°C, RR 20. "
            "Exam: risus sardonicus (sardonic grin from facial muscle spasm), "
            "masseter rigidity (trismus), opisthotonos (arched back) triggered by door slamming. "
            "A medical student whispers 'maybe it's TMJ?'"
        ),
        "patient_data": {
            "age": 67,
            "sex": "male",
            "chief_complaint": "Progressive jaw stiffness and muscle spasms after puncture wound",
            "symptoms": [
                "Trismus — jaw stiffness, can't open mouth",
                "Risus sardonicus (facial muscle spasm)",
                "Opisthotonos (back arching spasm) triggered by stimulus",
                "Painful muscle spasms in back and neck",
                "Stimulus-triggered spasms (noise)",
            ],
            "vitals": {
                "temperature": 38.0,
                "heart_rate": 105,
                "respiratory_rate": 20,
                "blood_pressure": "150/95",
                "spo2": 97.0,
            },
            "history": "Stepped on rusty nail 8 days ago, no tetanus vaccination in 45+ years",
            "medications": [],
            "allergies": [],
            "social_history": "Rural farmer",
        },
        "expected_output": {
            "primary_diagnosis": "Tetanus",
            "expected_icd11_codes": ["1C13"],
            "should_detect": [
                "Classic presentation: trismus + risus sardonicus + opisthotonos",
                "Puncture wound + no tetanus immunization = highest risk",
                "Treatment: tetanus immunoglobulin (TIG) + toxoid vaccine",
                "Wound debridement",
                "ICU admission — risk of respiratory failure from spasm",
                "Metronidazole as antibiotic of choice",
                "NOT TMJ — trismus with generalized spasms is tetanus",
            ],
            "red_flags": ["Trismus", "Generalized spasms", "Unvaccinated with wound"],
        },
    },

    # ── N-018: OVARIAN TORSION ──────────────────────────────────────
    {
        "case_id": "N-018",
        "title": "Ovarian Torsion — Acute abdomen dismissed as menstrual cramps",
        "difficulty": "medium",
        "category": "surgical_emergency",
        "narrative": (
            "17 y, Female. Brought to ER by parents at 2 AM with sudden onset severe right lower abdominal pain "
            "that started 4 hours ago. She was fine at dinner, woke up from sleep screaming in pain. "
            "She's vomited 3 times. Pain is constant, sharp, 9/10. "
            "Her mother says 'she always has bad period cramps' but the patient reports her period isn't due for 10 days. "
            "Vitals: HR 115, BP 110/70, Temp 37.3°C, SpO2 99%. "
            "Exam: right lower quadrant tenderness with guarding. No rebound. "
            "The patient can't find a comfortable position and keeps shifting on the bed. "
            "Previous pelvic US (3 months ago): 6cm dermoid cyst on right ovary — was being 'watched'. "
            "Urine pregnancy test: negative. "
            "WBC: 12,000 (mildly elevated). "
            "The registration clerk logged this as 'menstrual pain'."
        ),
        "patient_data": {
            "age": 17,
            "sex": "female",
            "chief_complaint": "Sudden severe right lower abdominal pain with vomiting, waking from sleep",
            "symptoms": [
                "Sudden onset severe RLQ pain (9/10)",
                "Vomiting x 3",
                "Unable to find comfortable position",
                "Right lower quadrant tenderness with guarding",
                "No rebound tenderness",
            ],
            "vitals": {
                "temperature": 37.3,
                "heart_rate": 115,
                "respiratory_rate": 18,
                "blood_pressure": "110/70",
                "spo2": 99.0,
            },
            "history": "6cm dermoid cyst on right ovary (identified 3 months ago on US)",
            "medications": [],
            "allergies": [],
            "lab_results": {"WBC": "12,000", "urine_pregnancy": "negative"},
        },
        "expected_output": {
            "primary_diagnosis": "Ovarian torsion",
            "expected_icd11_codes": ["GA14.0"],
            "should_detect": [
                "Known ovarian cyst + sudden severe pain = torsion until proven otherwise",
                "NOT menstrual cramps — sudden onset, severe, with vomiting",
                "Urgent pelvic doppler ultrasound to confirm",
                "Time-sensitive — delay risks ovarian necrosis",
                "Surgical consultation for detorsion/oophorectomy",
                "Period not due for 10 days — rules out dysmenorrhea",
            ],
            "red_flags": ["Sudden severe abdominal pain", "Known ovarian cyst", "Tachycardia"],
        },
    },

    # ── N-019: MALIGNANT HYPERTHERMIA (serotonin syndrome variant) ──
    {
        "case_id": "N-019",
        "title": "Serotonin Syndrome — SSRI + tramadol interaction",
        "difficulty": "hard",
        "category": "toxicology",
        "narrative": (
            "30 y, Male. Presents with agitation, restlessness, and profuse sweating for 6 hours. "
            "PMH: major depression, chronic low back pain. "
            "Current meds: sertraline 200mg daily (SSRI), was recently prescribed tramadol 50mg QID "
            "by his orthopedist (3 days ago) — his psychiatrist was NOT informed. "
            "Vitals: Temp 39.5°C, HR 130, BP 155/95, RR 22. "
            "Exam: CLONUS (bilateral, sustained at ankles), hyperreflexia globally, "
            "dilated pupils (6mm bilateral), diaphoresis, "
            "intermittent myoclonus (involuntary muscle jerks especially in lower extremities). "
            "Tremor more prominent in lower extremities. "
            "Bowel sounds hyperactive (diarrhea). "
            "The ER attending suspects neuroleptic malignant syndrome and orders dantrolene."
        ),
        "patient_data": {
            "age": 30,
            "sex": "male",
            "chief_complaint": "Agitation, hyperthermia, clonus, and muscle jerks after SSRI + tramadol combination",
            "symptoms": [
                "Agitation and restlessness",
                "Profuse diaphoresis",
                "Fever 39.5°C",
                "Bilateral sustained ankle clonus",
                "Global hyperreflexia",
                "Dilated pupils (mydriasis)",
                "Myoclonus (involuntary muscle jerks)",
                "Lower extremity tremor",
                "Diarrhea",
            ],
            "vitals": {
                "temperature": 39.5,
                "heart_rate": 130,
                "respiratory_rate": 22,
                "blood_pressure": "155/95",
                "spo2": 96.0,
            },
            "history": "Major depression, chronic low back pain",
            "medications": ["Sertraline 200mg daily (SSRI)", "Tramadol 50mg QID (started 3 days ago)"],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Serotonin syndrome",
            "expected_icd11_codes": ["4A45.2"],
            "should_detect": [
                "SSRI + tramadol = serotonin syndrome (tramadol is a serotonergic drug)",
                "NOT NMS — NMS has rigidity and bradyreflexia; SS has clonus and hyperreflexia",
                "Hunter criteria: clonus + agitation + diaphoresis + hyperthermia",
                "Treatment: cyproheptadine (serotonin antagonist), NOT dantrolene",
                "Stop both serotonergic agents immediately",
                "Multi-prescriber problem — orthopedist didn't check psychiatric meds",
            ],
            "red_flags": ["Hyperthermia", "Clonus", "Drug interaction", "Tachycardia"],
        },
    },

    # ── N-020: MESENTERIC ISCHEMIA ──────────────────────────────────
    {
        "case_id": "N-020",
        "title": "Mesenteric Ischemia — Pain out of proportion with benign exam",
        "difficulty": "hard",
        "category": "surgical_emergency",
        "narrative": (
            "72 y, Female. Presents with severe diffuse abdominal pain for 6 hours. "
            "Pain came on suddenly after lunch. She rates it 10/10, 'the worst pain of my life'. "
            "PMH: atrial fibrillation (NOT on anticoagulation — refused warfarin because 'it's rat poison'), "
            "previous MI 5 years ago, CHF. "
            "Vitals: HR 110 (irregular), BP 100/60, Temp: 37.0°C, RR 24. "
            "THE KEY FINDING: abdomen is SOFT and essentially NON-TENDER on palpation "
            "despite the patient screaming in pain (pain out of proportion to exam). "
            "Bowel sounds initially hyperactive, now diminishing. "
            "Labs: WBC 19,000, lactate 5.8 mmol/L (elevated), metabolic acidosis (pH 7.28). "
            "The surgery resident palpated the abdomen and said 'it's soft — probably functional'."
        ),
        "patient_data": {
            "age": 72,
            "sex": "female",
            "chief_complaint": "Severe diffuse abdominal pain out of proportion to physical exam",
            "symptoms": [
                "Sudden severe abdominal pain (10/10)",
                "Pain out of proportion to exam — soft abdomen despite extreme pain",
                "Metabolic acidosis",
                "Elevated lactate",
                "Diminishing bowel sounds (initially hyperactive)",
            ],
            "vitals": {
                "temperature": 37.0,
                "heart_rate": 110,
                "respiratory_rate": 24,
                "blood_pressure": "100/60",
                "spo2": 95.0,
            },
            "history": "Atrial fibrillation (NOT anticoagulated), previous MI, CHF",
            "medications": [],
            "allergies": [],
            "lab_results": {
                "WBC": "19,000",
                "lactate": "5.8 mmol/L",
                "pH": "7.28",
            },
        },
        "expected_output": {
            "primary_diagnosis": "Acute mesenteric ischemia",
            "expected_icd11_codes": ["DB95"],
            "should_detect": [
                "PAIN OUT OF PROPORTION TO EXAM — hallmark of mesenteric ischemia",
                "AF without anticoagulation = high embolic risk → SMA embolus",
                "Elevated lactate + metabolic acidosis = bowel ischemia",
                "Urgent CT angiography of abdomen",
                "Surgical consultation — may need embolectomy or bowel resection",
                "This is NOT functional/benign despite soft abdomen",
            ],
            "red_flags": ["Pain out of proportion", "Metabolic acidosis", "Elevated lactate", "AF without anticoagulation"],
        },
    },

    # ── N-021: MECHANICAL BACK PAIN (BEL TUTULMASI) ─────────────────
    {
        "case_id": "N-021",
        "title": "Mechanical Back Pain — Cultural Idiom 'Bıçak Saplanıyor'",
        "difficulty": "hard",
        "category": "musculoskeletal",
        "narrative": (
            "Sabah yataktan kalkarken belim resmen kilitlendi abi. Eğilip çorabımı bile giyemedim. "
            "Öyle ağır bir şey de kaldırmadım aslında, dün akşam balkonda terli terli cereyanda oturmuştum ondan mı oldu bilmem. "
            "Şimdi sağa sola dönerken bıçak saplanıyor sanki, nefesimi kesiyor ağrısı. "
            "Bacağıma vuran bir sızı yok, uyuşma falan da yok ama tahta gibi tutuldum kaldım yani, doğrulurken canımdan can gidiyor. "
            "Triyaj Notu (Gözlem): 35 Yaş, Erkek. Ateş: 36.5°C, Nabız: 78/dk, Tansiyon: 120/80 mmHg. "
            "Fizik muayenede lomber bölgede (bel) kas spazmı mevcut. "
            "Bacaklarda his kaybı, güç kaybı veya refleks eksikliği yok (Nörolojik muayene normal)."
        ),
        "patient_data": {
            "age": 35,
            "sex": "male",
            "chief_complaint": "Sudden onset severe lower back pain, exacerbated by movement",
            "symptoms": [
                "Severe lower back pain",
                "Muscle spasm in lumbar region",
                "Pain on movement (sharp, 'stabbing' sensation)",
                "No radiating pain to legs",
                "No numbness or weakness",
                "History of cold draft exposure ('cereyan')",
            ],
            "vitals": {
                "temperature": 36.5,
                "heart_rate": 78,
                "respiratory_rate": 16,
                "blood_pressure": "120/80",
                "spo2": 98.0,
            },
            "history": "No heavy lifting. Cold draft exposure yesterday.",
            "medications": [],
            "allergies": [],
        },
        "expected_output": {
            "primary_diagnosis": "Mechanical low back pain (muscle spasm)",
            "expected_icd11_codes": ["ME84.2"],
            "should_detect": [
                "Idiomatic expression ('bıçak saplanması') indicates sharp musculoskeletal pain, NOT vascular/thoracic injury.",
                "'Cereyanda kalmak' indicates cold exposure/muscle spasm, a cultural trigger.",
                "Normal neuro exam rules out radiculopathy or cauda equina.",
                "No red flags for serious underlying pathology despite severe subjective pain tone."
            ],
            "red_flags": [],
        },
    },
]


# ═════════════════════════════════════════════════════════════════════
# CASE SELECTION API
# ═════════════════════════════════════════════════════════════════════

ALL_NARRATIVE_IDS = [c["case_id"] for c in NARRATIVE_CASES]


def get_case_by_id(case_id: str) -> dict | None:
    """Get a specific narrative case by ID (e.g. 'N-001')."""
    for c in NARRATIVE_CASES:
        if c["case_id"] == case_id:
            return c
    return None


def get_case_by_keyword(keyword: str) -> list[dict]:
    """Search narrative cases by keyword in title/narrative/category."""
    kw = keyword.lower()
    return [
        c for c in NARRATIVE_CASES
        if kw in c["title"].lower()
        or kw in c["narrative"].lower()
        or kw in c["category"].lower()
        or kw in c["expected_output"]["primary_diagnosis"].lower()
    ]


def get_random_cases(n: int = 3, seed: int | None = None, exclude: list[str] | None = None) -> list[dict]:
    """Select n random narrative cases. Different each time unless seeded.

    Args:
        n: Number of cases to select (max = len(NARRATIVE_CASES))
        seed: Optional seed for reproducibility. If None, truly random.
        exclude: List of case_ids to exclude from selection.
    """
    pool = NARRATIVE_CASES[:]
    if exclude:
        pool = [c for c in pool if c["case_id"] not in exclude]
    n = min(n, len(pool))

    rng = random.Random(seed)
    return rng.sample(pool, n)


def get_cases_by_difficulty(difficulty: str) -> list[dict]:
    """Get all cases of a given difficulty: 'easy', 'medium', 'hard'."""
    return [c for c in NARRATIVE_CASES if c["difficulty"] == difficulty]


def get_cases_by_category(category: str) -> list[dict]:
    """Get all cases matching a category (partial match)."""
    cat = category.lower()
    return [c for c in NARRATIVE_CASES if cat in c["category"].lower()]


def get_mixed_difficulty_set(n: int = 5) -> list[dict]:
    """Get a balanced mix of difficulties: ~30% medium, ~70% hard."""
    hard = get_cases_by_difficulty("hard")
    medium = get_cases_by_difficulty("medium")

    n_medium = max(1, n * 3 // 10)
    n_hard = n - n_medium

    rng = random.Random()
    selected = rng.sample(hard, min(n_hard, len(hard)))
    selected += rng.sample(medium, min(n_medium, len(medium)))
    rng.shuffle(selected)
    return selected[:n]


# Quick summary for display
def list_all_cases() -> str:
    """Return a formatted list of all narrative cases."""
    lines = []
    for c in NARRATIVE_CASES:
        lines.append(
            f"  {c['case_id']:6s} | {c['difficulty']:6s} | {c['category']:25s} | {c['title']}"
        )
    return "\n".join(lines)
