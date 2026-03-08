"""Zebra Detector — Pattern-matching for rare but dangerous diseases.

Runs BEFORE the LLM to detect known clinical triads/patterns that
general-purpose models frequently miss (Anchoring Bias prevention).

When a zebra pattern is detected, it is injected into the R1 prompt
as a "ZEBRA ALERT" so the LLM is forced to consider it.

Why this matters:
  - LLMs have base-rate fallacy: they jump to common diagnoses
  - Rare diseases with classic triads are missed 60-80% of the time
  - Pattern-matching catches what statistical models overlook
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ZebraMatch:
    """A detected rare disease pattern."""
    disease: str
    icd11: str
    confidence: float          # 0.0 - 1.0 based on how many criteria matched
    matched_criteria: list[str]
    total_criteria: int
    clinical_pearl: str        # One-liner the LLM should consider
    key_question: str          # Critical question to ask the clinician


# ═══════════════════════════════════════════════════════════════════════
# ZEBRA PATTERN DATABASE
# Each entry: { triggers, criteria, disease info, key_question }
# ═══════════════════════════════════════════════════════════════════════

ZEBRA_PATTERNS = [
    # ─── Acute Intermittent Porphyria (AIP) ─────────────────────────
    {
        "disease": "Acute Intermittent Porphyria (AIP)",
        "icd11": "8E80.10",
        "triggers": [
            r"kar[ıi]n\s*a[gğ]r[ıi]",                # karın ağrısı (Turkish)
            r"abdominal?\s*pain",                       # abdominal pain
            r"severe\s*(abdominal|belly|stomach)\s*pain",
        ],
        "criteria": [
            # Neuro-visceral triad
            ("abdominal_pain_soft", [
                r"(soft|yumuşak|nondistend|no\s*(rebound|guard|defans|distension))",
                r"(karın|abdomen).{0,40}(yumuşak|soft)",
                r"pain.{0,30}(out\s*of\s*proportion|disproportionate)",
                r"(defans|rebound).{0,10}(yok|negative|absent)",
            ]),
            ("autonomic_hyperactivity", [
                r"tachycard|taşikardi|heart\s*rate.{0,10}(1[2-9]\d|[2-9]\d\d)",
                r"hypertens|hipertansiyon|blood\s*pressure.{0,10}(1[5-9]\d|[2-9]\d\d)",
                r"(nabız|pulse|hr).{0,10}(1[2-9]\d|[2-9]\d\d)",
                r"(ta|bp|tansiyon).{0,10}(1[5-9]\d|[2-9]\d\d)",
            ]),
            ("neuropsychiatric", [
                r"hal[üu]sinasyon|hallucination|g[öo]lgeler|seeing things|visual\s*disturb",
                r"ajit(asyon|ated)|agitat|restless|confusion|delirium",
                r"ışık|fotofobi|photophob|light\s*sensitiv",
                r"seizure|nöbet|konvülsiyon|convulsion",
                r"psychosis|psikoz|paranoi",
            ]),
            ("trigger_fasting", [
                r"diyet|diet|detoks|detox|açlık|fast(ing)?|starv|kalori|calorie",
                r"sıvı\s*detoks|juice\s*cleanse|water\s*fast",
            ]),
            ("trigger_drug", [
                r"antibiyotik|antibiotic|sulfonamid|bactrim|trimethoprim",
                r"barbiturat|barbiturate|phenobarbital",
                r"(eski|old)\s*(ilaç|drug|antibiy)",
                r"oral\s*contracept|doğum\s*kontrol",
            ]),
            ("urine_clue", [
                r"(koyu|dark|red|reddish|brown|kırmızı|kahverengi)\s*(idrar|urin)",
                r"urin.{0,20}(dark|red|brown|port\s*wine|cola)",
            ]),
            ("young_female", [
                r"(genç|young).{0,20}(kadın|female|woman)",
                r"\b(1[5-9]|2\d|3[0-5])\s*(y|yaş|year|yo|F)\b",
                r"kadın|female|woman",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Acute Intermittent Porphyria (AIP): "
            "Unexplained severe abdominal pain (soft abdomen, no peritoneal signs) "
            "+ autonomic storm (tachycardia, hypertension) "
            "+ neuropsychiatric symptoms (hallucinations, agitation, seizures) "
            "= Classic AIP triad. "
            "Triggers: fasting/dieting, certain drugs (sulfonamides, barbiturates, OCP). "
            "CRITICAL TEST: Spot urine for porphobilinogen (PBG) — turns dark red/port-wine on standing. "
            "ICD-11: 8E80.10"
        ),
        "key_question": "İdrar rengi nedir? (Koyu kırmızı/port şarabı rengi → Porfiri) ve Kullandığı antibiyotik hangisi? (Sülfonamid/Bactrim porfirinojenik)",
        "min_criteria": 3,  # Need at least 3 of 7 criteria to trigger
    },

    # ─── Pheochromocytoma ───────────────────────────────────────────
    {
        "disease": "Pheochromocytoma",
        "icd11": "5A21",
        "triggers": [
            r"hypertens|hipertansiyon|tansiyon\s*(yüksek|kriz)",
            r"(paroxysmal|episodic|epizodik|ataklar)",
            r"blood\s*pressure.{0,10}(1[8-9]\d|[2-9]\d\d)",
        ],
        "criteria": [
            ("paroxysmal_htn", [
                r"(paroxysmal|episodic|epizodik|atak).{0,30}(hypertens|hipertansiyon|tansiyon)",
                r"(tansiyon|bp).{0,20}(kriz|crisis|spike|surge)",
            ]),
            ("classic_triad", [
                r"headache|baş\s*ağrı",
                r"sweat|terleme|diaphoresis",
                r"palpitat|çarpıntı|tachycard|taşikardi",
            ]),
            ("episodic_panic", [
                r"(panic|panik|anxiety|anksiyete).{0,20}(attack|atak)",
                r"(episode|atak|nöbet).{0,20}(minute|dakika|spontan)",
            ]),
            ("pallor_flush", [
                r"pallor|soluk|solukluk|flush|kızarma|blanch",
            ]),
            ("family_hx", [
                r"(family|aile).{0,20}(men\s*2|multiple\s*endocrine|feokromo|pheo)",
                r"(vhl|von\s*hippel|neurofibromatosis)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Pheochromocytoma: "
            "Episodic triad of headache + sweating + palpitations with paroxysmal hypertension. "
            "Do NOT attribute to 'anxiety' without ruling out pheo. "
            "CRITICAL TEST: 24-hour urine catecholamines/metanephrines or plasma free metanephrines. "
            "ICD-11: 5A21"
        ),
        "key_question": "Ataklar ne sıklıkta? Atak sırasında nabız ve tansiyon kaça çıkıyor?",
        "min_criteria": 2,
    },

    # ─── Addisonian Crisis ──────────────────────────────────────────
    {
        "disease": "Adrenal Crisis (Addisonian Crisis)",
        "icd11": "5A74.1",
        "triggers": [
            r"hypotens|hipotansiyon|şok|shock|collap",
            r"steroid.{0,20}(withdraw|kesi|stop|bırak)",
        ],
        "criteria": [
            ("hypotension_refractory", [
                r"(hypotens|hipotansiyon|bp\s*low|düşük\s*tansiyon)",
                r"(refractory|dirençli|unresponsive).{0,20}(fluid|sıvı)",
            ]),
            ("hyperpigmentation", [
                r"hyperpigment|hiperpigment|bronz|bronze|darkening\s*skin",
                r"(koyu|dark).{0,20}(cilt|skin|el|palm|knuckle)",
            ]),
            ("hyponatremia_hyperkalemia", [
                r"hyponatr|hiponatr|sodium.{0,10}(low|düşük|\d{2,3})",
                r"hyperkal|hiperkal|potassium.{0,10}(high|yüksek)",
            ]),
            ("steroid_withdrawal", [
                r"steroid.{0,20}(withdraw|kesi|stop|bırak|taper)",
                r"(prednizon|prednisone|dexamethasone|kortizon).{0,20}(bırak|stop|kesi)",
            ]),
            ("fatigue_weakness", [
                r"(profound|extreme|şiddetli).{0,20}(fatigue|weakness|halsizlik|güçsüzlük)",
                r"(nausea|bulantı|kusma|vomit).{0,20}(fatigue|halsizlik)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Adrenal Crisis: "
            "Refractory hypotension + hyponatremia + hyperkalemia + hyperpigmentation "
            "= Adrenal insufficiency until proven otherwise. "
            "History of steroid use/withdrawal is KEY. "
            "CRITICAL: IV hydrocortisone 100mg STAT — do NOT wait for cortisol results. "
            "ICD-11: 5A74.1"
        ),
        "key_question": "Son 6 ayda steroid (kortizon/prednizon) kullandı mı? Ani bırakma var mı?",
        "min_criteria": 2,
    },

    # ─── Serotonin Syndrome ─────────────────────────────────────────
    {
        "disease": "Serotonin Syndrome",
        "icd11": "NE61",
        "triggers": [
            r"(ssri|snri|antidepress|maoi|tramadol|linezolid|fentanyl)",
            r"(ilaç|drug).{0,20}(etkileşim|interaction)",
        ],
        "criteria": [
            ("serotonergic_drug", [
                r"(ssri|snri|sertralin|fluoxetin|paroxetin|venlafaxin|duloxetin)",
                r"(tramadol|fentanyl|meperidin|linezolid|maoi)",
                r"(triptophan|st\.\s*john|dextromethorphan)",
            ]),
            ("neuromuscular", [
                r"clonus|klonus|hyperreflex|hiperrefleks|tremor|myoclonus|miyoklonus",
                r"(rigidity|rijidite).{0,20}(lower|alt|bacak|leg)",
            ]),
            ("autonomic", [
                r"tachycard|taşikardi|diaphoresis|terleme|hypertherm|hipertermi",
                r"(mydriasis|dilated\s*pupil|geniş\s*pupil)",
            ]),
            ("mental_status", [
                r"agitat|ajitasyon|confusion|konfüzyon|delirium|anxiety|anksiyete",
            ]),
            ("drug_combination", [
                r"(two|iki|multiple|birden\s*fazla).{0,20}(serotonin|ssri|antidepress)",
                r"(new|yeni).{0,20}(medication|ilaç|drug).{0,20}(added|eklendi|başlandı)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Serotonin Syndrome: "
            "Serotonergic drug + (clonus/tremor + autonomic instability + altered mental status). "
            "Hunter Criteria: spontaneous clonus, inducible clonus + agitation, ocular clonus + agitation. "
            "TREATMENT: Stop all serotonergic drugs, supportive care, cyproheptadine for severe cases. "
            "ICD-11: NE61"
        ),
        "key_question": "Hangi ilaçları kullanıyor? Son zamanda yeni eklenen ilaç var mı?",
        "min_criteria": 3,
    },

    # ─── Neuroleptic Malignant Syndrome ─────────────────────────────
    {
        "disease": "Neuroleptic Malignant Syndrome (NMS)",
        "icd11": "NE61",
        "triggers": [
            r"(antipsychotic|nöroleptik|haloperidol|olanzapine|risperidone)",
            r"(lead\s*pipe|kurşun\s*boru).{0,20}(rigid|rijid)",
        ],
        "criteria": [
            ("antipsychotic_exposure", [
                r"(antipsychotic|nöroleptik|haloperidol|olanzapin|risperidon|klorpromazin)",
                r"(metoklopramid|metoclopramide|domperidon)",
            ]),
            ("hyperthermia", [
                r"(fever|ateş|hypertherm|hipertermi).{0,20}(>?\s*4[0-2]|high|yüksek)",
                r"temperature.{0,10}(4[0-2])",
            ]),
            ("lead_pipe_rigidity", [
                r"(lead\s*pipe|generalized|jeneralize).{0,20}(rigid|rijid)",
                r"(muscle|kas).{0,20}(rigid|sert|stiff)",
            ]),
            ("mental_status", [
                r"(altered|değişmiş).{0,20}(mental|bilinç|consciousness)",
                r"(stupor|katatoni|mutism|confusion|delirium)",
            ]),
            ("autonomic_instability", [
                r"(tachycard|taşikardi|diaphoresis|terleme|labile\s*bp|değişken\s*tansiyon)",
            ]),
            ("elevated_ck", [
                r"(ck|cpk|creatine\s*kinase).{0,10}(elevated|yüksek|>\s*\d{4})",
                r"rhabdomyolysis|rabdomiyoliz",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Neuroleptic Malignant Syndrome: "
            "Antipsychotic/dopamine blocker + (hyperthermia >40°C + lead-pipe rigidity + "
            "altered consciousness + autonomic instability). CK markedly elevated. "
            "TREATMENT: Stop offending drug, dantrolene/bromocriptine, ICU admission. "
            "ICD-11: NE61"
        ),
        "key_question": "Son 2 haftada antipsikotik/nöroleptik ilaç başlandı mı veya doz artırıldı mı?",
        "min_criteria": 3,
    },

    # ─── CO Poisoning ──────────────────────────────────────────────
    {
        "disease": "Carbon Monoxide Poisoning",
        "icd11": "NE60.1",
        "triggers": [
            r"headache|baş\s*ağrı",
            r"(multiple|birden|family|aile|household|ev\s*halkı).{0,20}(same|aynı|symptom|belirti)",
        ],
        "criteria": [
            ("multiple_victims", [
                r"(multiple|birden|family|aile|household|ev\s*halkı|all|hepsi).{0,20}(same|aynı|symptom|belirti|sick|hasta)",
            ]),
            ("headache_nausea", [
                r"headache|baş\s*ağrı",
                r"nausea|bulantı|vomit|kusma|dizziness|baş\s*dönmesi",
            ]),
            ("cherry_red", [
                r"cherry\s*red|kiraz\s*kırmızı|pink\s*skin|pembe\s*cilt",
            ]),
            ("exposure_source", [
                r"(gas\s*heater|doğalgaz|soba|stove|charcoal|kömür|generator|jeneratör)",
                r"(closed|kapalı|unventilated|havalandırma).{0,20}(room|oda|space|alan)",
                r"(garage|garaj|tunnel|tünel)",
            ]),
            ("normal_spo2_misleading", [
                r"spo2.{0,10}(normal|9[5-9]|100)",
                r"(pulse\s*ox|oksimetre).{0,20}(normal|mislead|yanıltıcı)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Carbon Monoxide Poisoning: "
            "Multiple people with same symptoms (headache, nausea, confusion) from same location. "
            "WARNING: Pulse oximetry is FALSELY NORMAL in CO poisoning (COHb reads as O2Hb). "
            "CRITICAL TEST: Arterial/venous COHb level. "
            "TREATMENT: 100% O2 via non-rebreather mask; hyperbaric O2 if severe. "
            "ICD-11: NE60.1"
        ),
        "key_question": "Evde/işyerinde aynı belirtiler gösteren başka kişi var mı? Isınma kaynağı nedir?",
        "min_criteria": 2,
    },

    # ─── Mesenteric Ischemia ──────────────────────────────────────────
    {
        "disease": "Acute Mesenteric Ischemia",
        "icd11": "DA65",
        "triggers": [
            r"(abdominal|karın)\s*(pain|ağrı)",
            r"pain.{0,30}(out\s*of\s*proportion|disproportionate)",
        ],
        "criteria": [
            ("pain_disproportionate", [
                r"pain.{0,30}(out\s*of\s*proportion|disproportionate)",
                r"(şiddetli|severe).{0,20}(ağrı|pain).{0,30}(yumuşak|soft|normal)",
            ]),
            ("elderly_af", [
                r"\b([6-9]\d|[1-9]\d\d)\s*(y|yaş|year|yo)\b",
                r"(atrial\s*fib|af|afib|aritmi|arrhythm)",
            ]),
            ("vascular_risk", [
                r"(atrial\s*fib|af|afib|dvt|embolism|emboli)",
                r"(peripheral\s*vascular|pvd|atherosclerosis|ateroskleroz)",
            ]),
            ("bloody_stool", [
                r"(bloody|kanlı|maroon|hematochezia|melena|melaena)",
                r"(stool|dışkı|gaita|rectal).{0,20}(blood|kan|bleed)",
            ]),
            ("lactate_elevated", [
                r"lactat.{0,10}(elevated|yüksek|high|>\s*[2-9])",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Acute Mesenteric Ischemia: "
            "Severe abdominal pain disproportionate to exam (soft abdomen) "
            "is mesenteric ischemia until proven otherwise. "
            "High risk in: elderly, AF, vascular disease, recent cardiac procedure. "
            "CRITICAL: CT angiography STAT. Elevated lactate = bowel necrosis. "
            "ICD-11: DA65"
        ),
        "key_question": "Hastada atriyal fibrilasyon veya periferik damar hastalığı öyküsü var mı?",
        "min_criteria": 2,
    },

    # ─── Non-Alcoholic Wernicke Encephalopathy (Bariatric) ─────────
    {
        "disease": "Wernicke Encephalopathy (Non-Alcoholic / Bariatric)",
        "icd11": "6D82.0",
        "triggers": [
            r"(bariatric|t[üu]p\s*mide|gastric\s*(bypass|sleeve)|mide\s*ameliyat|gast?rektomi|gastrectomy|sleeve)",
            r"(kusma|vomiting|emesis|hyperemesis).{0,50}(konf[üu]zyon|confusion|bilin[çc]|conscious|mental|delirium|sa[çc]ma)",
        ],
        "criteria": [
            ("bariatric_surgical_history", [
                r"(bariatric|t[üu]p\s*mide|gastric\s*(bypass|sleeve)|mide\s*ameliyat|mide\s*k[üu][çc][üu]ltme|gast?rektomi|gastrectomy|sleeve|roux.en.y)",
            ]),
            ("confusion_delirium", [
                r"(confusion|konf[üu]zyon|bilin[çc]\s*(de[gğ]i[sş]ikli[gğ]i|bulan[ıi]kl[ıi][gğ]i|kayb[ıi])|delirium|sa[çc]ma|sa[çc]mal[ıi]yor|tuhaf\s*davran[ıi][sş]|orientation|dezoryant|disoriented)",
            ]),
            ("ataxia_gait", [
                r"(ataxia|ataksi|denge\s*(kayb[ıi]|bozuklu[gğ]u|problemi)|gait|y[üu]r[üu]y[üu][sş]\s*(bozuklu[gğ]u|problemi)|sendele|unsteady|stagger|falling|d[üu][sş][üme])",
            ]),
            ("eye_signs", [
                r"(nystagmus|nistagmus|g[öo]z\s*se[gğ]irmesi|g[öo]z\s*titremesi|ophthalmoplegia|oftalmopleji|diplopia|[çc]ift\s*g[öo]rme|double\s*vision)",
            ]),
            ("prolonged_vomiting", [
                r"(kusma|vomiting|emesis|hyperemesis|bulant[ıi]).{0,40}(hafta|week|ay|month|g[üu]n|uzun|s[üu]redir|prolonged)",
                r"(yemek|eat|food|solid).{0,30}(yiyemiyor|tolerate|keep\s*down|alam[ıi]yor|kusma|vomit)",
            ]),
            ("nutritional_compromise", [
                r"(kilo\s*(kayb[ıi]|verme|verdi)|weight\s*loss|malnutrit|yetersiz\s*beslen|malnourish|only\s*(liquid|s[ıi]v[ıi])|sadece\s*s[ıi]v[ıi])",
                r"([sş]ekerli\s*(su|meyve|serum)|sugar|fruit\s*juice|dextrose)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Non-Alcoholic Wernicke Encephalopathy: "
            "Bariatric surgery + prolonged vomiting + confusion/ataxia/eye signs = "
            "Wernicke encephalopathy until proven otherwise, EVEN WITHOUT ALCOHOL HISTORY. "
            "CRITICAL: Give Thiamine 500mg IV TID BEFORE any glucose/dextrose — "
            "glucose without thiamine will WORSEN the encephalopathy and can cause permanent brain damage. "
            "Classic triad (confusion + ataxia + ophthalmoplegia) present in only ~30% of cases. "
            "ICD-11: 6D82.0"
        ),
        "key_question": "Bu hastaya henüz şekerli serum (glikoz/dekstroz) verildi mi? Tiamin verilmeden önce mi?",
        "min_criteria": 3,
    },

    # ─── Botulism ───────────────────────────────────────────────────
    {
        "disease": "Botulism",
        "icd11": "1C12",
        "triggers": [
            r"(descend|inen).{0,20}(paralysis|felç|weakness|güçsüzlük)",
            r"(diplopia|çift\s*görme|blurred\s*vision|bulanık\s*görme)",
        ],
        "criteria": [
            ("descending_paralysis", [
                r"(descend|inen|yukarıdan\s*aşağı).{0,20}(paralysis|felç|weakness|güçsüzlük)",
            ]),
            ("cranial_nerve", [
                r"(diplopia|çift\s*görme|ptosis|pitoz|dysphagia|disfaji|dysarthria|dizartri)",
                r"(blurred|bulanık).{0,10}(vision|görme)",
                r"(difficulty|güçlük).{0,10}(speak|konuşma|swallow|yutma)",
            ]),
            ("food_source", [
                r"(canned|konserve|preserved|ferment|turşu|homemade|ev\s*yapımı)",
                r"(honey|bal).{0,10}(infant|bebek)",
            ]),
            ("symmetric_weakness", [
                r"(bilateral|iki\s*taraflı|symmetr|simetrik).{0,20}(weakness|güçsüzlük)",
            ]),
            ("respiratory_failure", [
                r"(respiratory|solunum).{0,20}(failure|yetmezlik|difficulty|güçlük)",
                r"(dyspnea|dispne|shortness\s*of\s*breath|nefes\s*darlığı)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Botulism: "
            "Descending symmetric paralysis starting with cranial nerves (diplopia, ptosis, dysphagia) "
            "without sensory loss. Food-borne: history of canned/preserved food. "
            "CRITICAL: Antitoxin must be given BEFORE lab confirmation. "
            "ICD-11: 1C12"
        ),
        "key_question": "Son 24-72 saatte konserve/fermente gıda tüketimi var mı?",
        "min_criteria": 2,
    },
    # ─── Aortic Dissection ──────────────────────────────────────────
    {
        "disease": "Aortic Dissection",
        "icd11": "BD20",
        "triggers": [
            r"(tearing|rip|yırtılma).{0,20}(pain|ağrı|chest|göğüs|back|sırt)",
            r"(chest|göğüs|back|sırt).{0,20}(tearing|rip|yırtılma|sharp|keskin|sudden|ani)",
        ],
        "criteria": [
            ("tearing_pain", [
                r"(tearing|rip|yırtılma).{0,20}(pain|ağrı)",
                r"(sudden|ani).{0,20}(severe|şiddetli).{0,20}(chest|göğüs|back|sırt)",
            ]),
            ("bp_differential", [
                r"(bp|blood\s*pressure|tansiyon).{0,20}(differ|asym|fark)",
                r"(unequal|eşit\s*olmayan).{0,20}(pulse|nabız|bp|pressure)",
            ]),
            ("radiating_back", [
                r"(radiat|yayıl).{0,20}(back|sırt|interscapular|skapula)",
                r"(back|sırt).{0,20}(pain|ağrı)",
            ]),
            ("widened_mediastinum", [
                r"(wide|geniş).{0,20}(mediastin)",
                r"(aort|aorta).{0,20}(dilat|genişle)",
            ]),
            ("marfan_risk", [
                r"(marfan|ehlers|connective\s*tissue|bağ\s*doku)",
                r"(tall|uzun\s*boy|arm\s*span)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Aortic Dissection: "
            "Sudden tearing chest/back pain with BP differential between arms. "
            "Type A requires emergent surgery. Type B: medical management. "
            "Check bilateral arm BPs! CTA aorta is diagnostic. ICD-11: BD20"
        ),
        "key_question": "Her iki kolda tansiyon ölçüldü mü? Fark var mı?",
        "min_criteria": 2,
    },
    # ─── Necrotizing Fasciitis ──────────────────────────────────────
    {
        "disease": "Necrotizing Fasciitis",
        "icd11": "1B70",
        "triggers": [
            r"(pain\s*out\s*of\s*proportion|ağrı.{0,20}orantısız)",
            r"(rapid|hızlı).{0,20}(spread|yayıl).{0,20}(red|kızarık|erythema|eritem)",
            r"(crepitus|krepitasyon)",
        ],
        "criteria": [
            ("pain_disproportionate", [
                r"pain\s*out\s*of\s*proportion",
                r"(extreme|şiddetli|severe).{0,20}(pain|ağrı).{0,20}(skin|cilt|wound|yara)",
            ]),
            ("rapid_spread", [
                r"(rapid|hızlı|fast).{0,20}(spread|progress|yayıl|ilerle)",
            ]),
            ("systemic_toxicity", [
                r"(fever|ateş|sepsis|septic|taşikardi|tachycard)",
                r"(hypotens|hipotansiyon|shock|şok)",
            ]),
            ("skin_changes", [
                r"(crepitus|krepitasyon|subcutaneous\s*emphysema)",
                r"(bullae|bül|hemorrhagic|hemorajik).{0,20}(blister|bül|skin|cilt)",
                r"(dusky|morumsu|necrotic|nekrotik|gangr)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Necrotizing Fasciitis: "
            "Pain out of proportion to exam findings + rapidly spreading erythema + systemic toxicity. "
            "Surgical emergency requiring immediate debridement. Do NOT wait for imaging! "
            "ICD-11: 1B70"
        ),
        "key_question": "Etkilenen bölgede ağrı muayene bulgularından çok mu şiddetli?",
        "min_criteria": 2,
    },
    # ─── Tension Pneumothorax ───────────────────────────────────────
    {
        "disease": "Tension Pneumothorax",
        "icd11": "CB23.0",
        "triggers": [
            r"(tracheal\s*deviation|trakeal\s*deviasyon)",
            r"(absent|azalmış|decrease).{0,20}(breath\s*sound|solunum\s*ses)",
            r"(tension|tansiyon).{0,20}(pneumo|pnömo)",
        ],
        "criteria": [
            ("tracheal_deviation", [
                r"(tracheal|trakeal).{0,20}(deviat|deviasyon|shift|kayma)",
            ]),
            ("absent_breath_sounds", [
                r"(absent|decrease|azalmış|yok).{0,20}(breath\s*sound|solunum\s*ses)",
                r"(unilateral|tek\s*taraflı).{0,20}(silent|sessiz)",
            ]),
            ("hypotension_tachycardia", [
                r"(hypotens|hipotansiyon|sbp.{0,5}(8|7|6)\d)",
                r"(tachycard|taşikardi|hr.{0,5}1[2-9]\d)",
            ]),
            ("jvd", [
                r"(jvd|jugular\s*venous\s*distension|juguler\s*dolgunluk)",
                r"(neck\s*vein|boyun\s*ven).{0,20}(distend|dolgun)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Tension Pneumothorax: "
            "Tracheal deviation + absent breath sounds + JVD + hypotension = clinical diagnosis. "
            "Immediate needle decompression BEFORE imaging. "
            "ICD-11: CB23.0"
        ),
        "key_question": "Trakea ortada mı? Tek taraflı solunum sesi azalması var mı?",
        "min_criteria": 2,
    },
    # ─── Cauda Equina Syndrome ──────────────────────────────────────
    {
        "disease": "Cauda Equina Syndrome",
        "icd11": "ME84.2",
        "triggers": [
            r"(saddle\s*an[ae]?sthesia|perine\s*uyuşma|eyer\s*anestezi)",
            r"(urinary\s*retention|idrar\s*retansiyon)",
            r"(bilateral|iki\s*taraf).{0,20}(leg|bacak).{0,20}(weakness|güçsüzlük)",
        ],
        "criteria": [
            ("saddle_anaesthesia", [
                r"(saddle|eyer|perine|perianal).{0,20}(an[ae]sthes|numb|uyuş)",
            ]),
            ("urinary_dysfunction", [
                r"(urinary|idrar).{0,20}(retention|retansiyon|incontinence|inkontinans)",
                r"(bladder|mesane).{0,20}(dysfun|bozuk)",
            ]),
            ("bilateral_leg", [
                r"(bilateral|iki\s*taraf).{0,20}(leg|bacak|lower\s*extrem|alt\s*ekstrem).{0,20}(weak|güçsüz|numb|uyuş)",
            ]),
            ("back_pain", [
                r"(severe|şiddetli).{0,20}(back|bel|lumbar|lomber).{0,20}(pain|ağrı)",
            ]),
            ("bowel_dysfunction", [
                r"(bowel|bağırsak|fecal|fekal).{0,20}(incontinen|inkontina|dysfun|bozuk)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Cauda Equina Syndrome: "
            "Saddle anaesthesia + urinary retention + bilateral leg weakness = surgical emergency. "
            "MRI within hours, decompression within 48h to prevent permanent paralysis. "
            "ICD-11: ME84.2"
        ),
        "key_question": "Perine bölgesinde uyuşukluk var mı? İdrar yapamama şikayeti?",
        "min_criteria": 2,
    },
    # ─── Testicular Torsion ─────────────────────────────────────────
    {
        "disease": "Testicular Torsion",
        "icd11": "GB04",
        "triggers": [
            r"(testicular|testis|skrot).{0,20}(pain|ağrı|acute|akut)",
            r"(acute|ani|sudden).{0,20}(scrot|skrot|testis|testicular)",
        ],
        "criteria": [
            ("acute_scrotal_pain", [
                r"(acute|ani|sudden).{0,20}(scrot|skrot|testis|testicular).{0,20}(pain|ağrı)",
                r"(testis|testicular|skrot).{0,20}(acute|ani|sudden|severe|şiddetli).{0,20}(pain|ağrı)",
            ]),
            ("absent_reflex", [
                r"(absent|negative|yok).{0,20}(cremaster|kremaster)",
                r"(cremaster|kremaster).{0,20}(reflex|refleks).{0,20}(absent|negative|yok)",
            ]),
            ("high_riding", [
                r"(high.?riding|yüksek\s*yerleşimli|retract|yukarıda)",
                r"(horizontal|transvers).{0,20}(lie|testis)",
            ]),
            ("young_male", [
                r"(\b1[2-9]|2[0-5])\s*(year|yaş|yo|y/o)",
                r"(adolescen|ergen|young\s*male|genç\s*erkek)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Testicular Torsion: "
            "Acute scrotal pain + absent cremasteric reflex + high-riding testis. "
            "6-hour window for salvage! Surgical exploration without delay. "
            "ICD-11: GB04"
        ),
        "key_question": "Kremaster refleksi kontrol edildi mi? Testis yüksek yerleşimli mi?",
        "min_criteria": 2,
    },
    # ─── Ectopic Pregnancy Rupture ──────────────────────────────────
    {
        "disease": "Ruptured Ectopic Pregnancy",
        "icd11": "JA03.1",
        "triggers": [
            r"(woman|kadın|female).{0,40}(abdominal|karın|pelvic|pelvik).{0,20}(pain|ağrı)",
            r"(pregnant|gebe|hamile|amenor|adet\s*gecikme|missed\s*period)",
        ],
        "criteria": [
            ("amenorrhea", [
                r"(amenor|adet\s*gecikme|missed\s*period|son\s*adet)",
                r"(pregnant|gebe|hamile|hcg\s*positive|beta\s*hcg)",
            ]),
            ("abdominal_pain", [
                r"(lower|alt).{0,20}(abdominal|karın).{0,20}(pain|ağrı)",
                r"(pelvic|pelvik).{0,20}(pain|ağrı)",
            ]),
            ("vaginal_bleeding", [
                r"(vaginal|vajinal).{0,20}(bleed|kanam)",
                r"(spotting|lekelenme)",
            ]),
            ("hemodynamic_instability", [
                r"(hypotens|hipotansiyon|shock|şok|taşikardi|tachycard)",
                r"(dizzy|baş\s*dönmesi|faint|bayılma|syncop|senkop)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Ruptured Ectopic Pregnancy: "
            "Any woman of reproductive age with abdominal pain + amenorrhea + vaginal bleeding. "
            "Check beta-hCG immediately. Hemodynamic instability = surgical emergency. "
            "ICD-11: JA03.1"
        ),
        "key_question": "Beta-hCG düzeyi kontrol edildi mi? Son adet tarihi nedir?",
        "min_criteria": 2,
    },
    # ─── Malignant Hyperthermia ─────────────────────────────────────
    {
        "disease": "Malignant Hyperthermia",
        "icd11": "NE60",
        "triggers": [
            r"(anesthesia|anestezi|surgery|cerrahi|inhalation\s*agent|sevoflurane|desflurane|succinylcholine)",
        ],
        "criteria": [
            ("hyperthermia", [
                r"(temp|ateş|fever).{0,20}(>?\s*4[0-2]|rising\s*rapidly|hızla\s*yüksel)",
                r"(hypertherm|hipertermi)",
            ]),
            ("muscle_rigidity", [
                r"(muscle|kas).{0,20}(rigid|sert|spasm)",
                r"(masseter|çene).{0,20}(rigid|sert|spasm|trismus)",
            ]),
            ("metabolic_crisis", [
                r"(rhabdomyolysis|rabdomiyoliz|myoglobin|ck\s*elevat|cpk\s*elevat)",
                r"(metabolic\s*acidosis|metabolik\s*asidoz|hypercarbia|hiperkarbi|co2.{0,10}(high|yüksek))",
                r"(hyperkalemia|hiperpotasemi|potassium.{0,10}(high|yüksek))",
            ]),
            ("post_anesthesia", [
                r"(during|after|sonra).{0,20}(anesth|anestez|surgery|cerrahi|operation|operasyon)",
                r"(intraop|periop|postop)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Malignant Hyperthermia: "
            "Rapidly rising temperature + muscle rigidity + metabolic crisis during/after anesthesia. "
            "IMMEDIATE dantrolene sodium. Stop triggering agents. Cool patient. "
            "ICD-11: NE60"
        ),
        "key_question": "Hasta yakın zamanda anestezi aldı mı? Hangi ajanlar kullanıldı?",
        "min_criteria": 2,
    },
    # ─── DKA (Diabetic Ketoacidosis) ────────────────────────────────
    {
        "disease": "Diabetic Ketoacidosis (DKA)",
        "icd11": "5A10.1",
        "triggers": [
            r"(diabet|diyabet|dm|insulin|insülin|blood\s*sugar|kan\s*şekeri)",
            r"(kussmaul|fruity\s*breath|aseton\s*koku|meyve\s*koku)",
        ],
        "criteria": [
            ("hyperglycemia", [
                r"(glucose|glikoz|blood\s*sugar|kan\s*şekeri|bs).{0,10}(>?\s*[3-9]\d\d|>?\s*\d{4})",
                r"(hyperglycemi|hiperglisemi)",
            ]),
            ("ketosis", [
                r"(keton|ketone|ketonuri|ketonemia|fruity|meyve|aseton)",
                r"(dka|diabetic\s*ketoacidosis|diyabetik\s*ketoasidoz)",
            ]),
            ("acidosis", [
                r"(metabolic\s*acidosis|metabolik\s*asidoz)",
                r"(ph\s*<?\.?\s*7\.3|bicarb|hco3).{0,10}(low|düşük|<?\s*1[0-8])",
                r"(anion\s*gap|ag).{0,10}(elevat|high|yüksek|>?\s*1[4-9]|>?\s*2\d)",
            ]),
            ("kussmaul", [
                r"(kussmaul|deep\s*rapid\s*breath|derin\s*hızlı\s*solunum)",
                r"(tachypne|takipne|hyperventilat|hiperventila)",
            ]),
            ("dehydration", [
                r"(dehydrat|dehidrat|dry\s*mucous|kuru\s*mukoza|polyuri|poliüri|polydips|polidips)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Diabetic Ketoacidosis: "
            "Hyperglycemia + ketosis + metabolic acidosis. Kussmaul breathing is late sign. "
            "IV fluids, insulin drip, potassium monitoring. Watch for cerebral edema in children. "
            "ICD-11: 5A10.1"
        ),
        "key_question": "Kan şekeri, kan gazı ve keton düzeyleri kontrol edildi mi?",
        "min_criteria": 2,
    },
    # ─── TTP (Thrombotic Thrombocytopenic Purpura) ──────────────────
    {
        "disease": "Thrombotic Thrombocytopenic Purpura (TTP)",
        "icd11": "3B64.0",
        "triggers": [
            r"(thrombocytopeni|trombositopeni|low\s*platelet|düşük\s*trombosit)",
            r"(schistocyt|şistosit|hemolytic\s*anemia|hemolitik\s*anemi)",
        ],
        "criteria": [
            ("thrombocytopenia", [
                r"(thrombocytopeni|trombositopeni|platelet.{0,10}(low|düşük|<?\s*[1-9]\d\s*k?))",
            ]),
            ("maha", [
                r"(microangiopathic|mikroanjiyopatik|schistocyt|şistosit)",
                r"(hemolytic\s*anemia|hemolitik\s*anemi|ldh.{0,10}(high|elevat|yüksek))",
            ]),
            ("neuro_symptoms", [
                r"(confusion|konfüzyon|headache|baş\s*ağrı|seizure|nöbet|focal\s*deficit|fokal\s*defisit)",
            ]),
            ("renal_involvement", [
                r"(renal|böbrek).{0,20}(failure|yetmezlik|insufficien|bozukluk)",
                r"(creatinine|kreatinin).{0,10}(elevat|high|yüksek)",
            ]),
            ("fever", [
                r"(fever|ateş|temp.{0,10}3[89]|temp.{0,10}4[0-2])",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — TTP/HUS: "
            "PENTAD: thrombocytopenia + MAHA (schistocytes) + neuro symptoms + renal failure + fever. "
            "CRITICAL: Do NOT transfuse platelets! Plasma exchange is lifesaving. "
            "ICD-11: 3B64.0"
        ),
        "key_question": "Periferik yayma yapıldı mı? Şistosit görüldü mü?",
        "min_criteria": 2,
    },
    # ─── Spinal Epidural Abscess ────────────────────────────────────
    {
        "disease": "Spinal Epidural Abscess",
        "icd11": "FA00",
        "triggers": [
            r"(back|sırt|spinal|bel).{0,20}(pain|ağrı).{0,30}(fever|ateş)",
            r"(iv\s*drug|damar\s*içi|ivdu|uyuşturucu)",
        ],
        "criteria": [
            ("back_pain_fever", [
                r"(back|sırt|spine|spinal|bel).{0,20}(pain|ağrı).{0,30}(fever|ateş)",
                r"(fever|ateş).{0,30}(back|sırt|spine|spinal|bel).{0,20}(pain|ağrı)",
            ]),
            ("neuro_deficit", [
                r"(weakness|güçsüzlük|numb|uyuş|paralyz|parali|parapar|quadripar)",
                r"(motor\s*deficit|motor\s*defisit|sensory\s*loss|duyu\s*kaybı)",
            ]),
            ("risk_factor", [
                r"(iv\s*drug|ivdu|damar\s*içi|uyuşturucu|immunocompromis|immünsüpres)",
                r"(diabet|diyabet|recent\s*procedur|spinal\s*procedur|epidural\s*inject)",
                r"(bacteremia|bakteriyemi|endocarditis|endokardit)",
            ]),
            ("elevated_markers", [
                r"(esr|sedimentasyon|crp).{0,10}(elevat|high|yüksek|>?\s*[5-9]\d|>?\s*1\d\d)",
                r"(leukocyt|lökosit|wbc).{0,10}(elevat|high|yüksek)",
            ]),
        ],
        "clinical_pearl": (
            "ZEBRA ALERT — Spinal Epidural Abscess: "
            "Classic triad: back pain + fever + progressive neurological deficit. "
            "Emergent MRI with contrast. Surgical decompression + IV antibiotics. "
            "Risk factors: IVDU, immunosuppression, spinal procedures. ICD-11: FA00"
        ),
        "key_question": "IV ilaç kullanımı veya yakın zamanda spinal girişim öyküsü var mı?",
        "min_criteria": 2,
    },
]


# ═══════════════════════════════════════════════════════════════════════
# DETECTION ENGINE  (pre-compiled regexes for performance)
# ═══════════════════════════════════════════════════════════════════════

# Pre-compile all trigger and criteria patterns at import time
_COMPILED_PATTERNS: list[dict] = []
for _p in ZEBRA_PATTERNS:
    compiled = {
        "disease": _p["disease"],
        "icd11": _p["icd11"],
        "clinical_pearl": _p["clinical_pearl"],
        "key_question": _p["key_question"],
        "min_criteria": _p.get("min_criteria", 2),
        "triggers": [re.compile(t, re.IGNORECASE) for t in _p["triggers"]],
        "criteria": [
            (name, [re.compile(pat, re.IGNORECASE) for pat in pats])
            for name, pats in _p["criteria"]
        ],
    }
    _COMPILED_PATTERNS.append(compiled)


def _check_criteria(text: str, compiled_patterns: list[re.Pattern]) -> bool:
    """Check if ANY pre-compiled pattern in the group matches the text."""
    for pattern in compiled_patterns:
        if pattern.search(text):
            return True
    return False


def detect_zebras(patient_text: str) -> list[ZebraMatch]:
    """Scan patient text for rare disease patterns (zebras).

    Args:
        patient_text: Free-text patient description (any language).

    Returns:
        List of ZebraMatch objects sorted by confidence (highest first).
    """
    text = patient_text.lower()
    results: list[ZebraMatch] = []

    for pattern in _COMPILED_PATTERNS:
        # Step 1: Check if ANY trigger matches
        has_trigger = any(t.search(text) for t in pattern["triggers"])
        if not has_trigger:
            continue

        # Step 2: Count matched criteria
        matched = []
        for crit_name, crit_compiled in pattern["criteria"]:
            if _check_criteria(text, crit_compiled):
                matched.append(crit_name)

        # Step 3: If enough criteria match, it's a zebra
        min_needed = pattern["min_criteria"]
        total = len(pattern["criteria"])

        if len(matched) >= min_needed:
            confidence = round(len(matched) / total, 2)
            results.append(ZebraMatch(
                disease=pattern["disease"],
                icd11=pattern["icd11"],
                confidence=confidence,
                matched_criteria=matched,
                total_criteria=total,
                clinical_pearl=pattern["clinical_pearl"],
                key_question=pattern["key_question"],
            ))

    # Sort by confidence descending
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results


def format_zebra_alerts(zebras: list[ZebraMatch]) -> str:
    """Format zebra matches as a prompt injection for R1."""
    if not zebras:
        return ""

    parts = ["\n\n🦓 ZEBRA ALERTS (rare but dangerous patterns detected):"]
    for z in zebras:
        parts.append(f"\n  ⚠️ {z.clinical_pearl}")
        parts.append(f"     Matched: {len(z.matched_criteria)}/{z.total_criteria} criteria ({z.matched_criteria})")
        parts.append(f"     KEY QUESTION for clinician: {z.key_question}")

    parts.append(
        "\n  INSTRUCTION: You MUST include these zebra diagnoses in your differential. "
        "If the zebra explains MORE symptoms than your primary diagnosis, rank it higher."
    )
    return "\n".join(parts)
