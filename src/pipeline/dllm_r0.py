"""DLLM R0 — Deep Large Language Model, 5-Layer Clinical Reasoning Engine.

ML→DML farkı katman derinliği ise, LLM→DLLM farkı düşünce derinliğidir.

Layered architecture inspired by Deep ML:
  L1 (Extractor)    — Raw text  → Clinical entities       (thinking OFF)
  L2 (Connector)    — Entities  → Connection graph + exit  (thinking ON)
  L3 (Pattern Det.) — Conns     → Clinical patterns + DDx  (thinking ON)
  L4 (Red Flag)     — Full ctx  → Context-aware red flags  (thinking ON)
  L5 (Synthesizer)  — All       → Final decision + routing (thinking OFF)

Adaptive Depth:
  Simple   → L1 → L2 → L5           (~2.5s)
  Moderate → L1 → L2 → L3 → L5      (~3.5s)
  Complex  → L1 → L2 → L3 → L4 → L5 (~4.5s)
  Critical → L1 → L4 → L5           (~2.5s)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config.settings import get_settings
from src.llm.llama_cpp_client import DLLMClient, DLLMResponse

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# R0Result — structured output from the DLLM analysis
# ═══════════════════════════════════════════════════════════════════

@dataclass
class R0Result:
    """Structured output from the 5-layer DLLM R0 analysis."""
    entities: dict[str, Any] = field(default_factory=dict)
    red_flags: list[dict[str, Any]] = field(default_factory=list)
    urgency: str = "moderate"
    complexity: str = "moderate"
    suggested_differentials: list[str] = field(default_factory=list)
    key_questions: list[str] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    thinking_summary: str = ""
    layers_run: list[int] = field(default_factory=list)
    layer_timings: dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Layer Prompts — each layer gets a short, focused prompt
# ═══════════════════════════════════════════════════════════════════

L1_AGENT_CLEANSER_SYSTEM = """You are a Semantic Text Cleanser (Agent Alpha).
Analyze the raw patient text. Your job is to translate cultural idioms, slang, colloquailisms, and metaphors into objective, standard medical descriptions.
Example: 'cereyanda kaldım' -> 'Cold draft exposure'.
Example: 'bıçak saplanıyor' -> 'Severe, sharp, well-localized pain'.
Example: 'içim dışıma çıktı' -> 'Severe vomiting'.
DO NOT extract JSON. Output a clean, culturally normalized, professional medical narrative that accurately reflects the patient's symptoms without metaphorical exaggeration."""

L1_AGENT_EXTRACTOR_SYSTEM = """You are a Strict Clinical Extractor (Agent Beta).
Analyze the raw patient text and strictly extract objective anatomical locations, stated severities, procedures, and physical findings.
Ignore emotional tone and metaphors. Only list the raw medical facts you can definitively extract.
DO NOT output JSON yet. Output a structured bulleted list of exact clinical findings, labs, and vitals."""

L1_AGENT_AUDITOR_SYSTEM = """You are a Context & Tone Auditor (Agent Gamma).
Compare the patient's subjective narrative with any objective triage data (found in the text).
Evaluate if the patient is exaggerating (panic factor), malingering, or downplaying their symptoms.
Output a psychological and contextual audit report, specifying an 'exaggeration index' or 'urgency discount' if subjective complaints severely outweigh objective findings."""

L1_CONSENSUS_SYSTEM = """You are the L1 Consensus Node.
Synthesize the 3 agent reports into a hallucination-free JSON.
If Alpha says a symptom is a metaphor (like 'knife'), SUPPRESS the literal meaning (no trauma) and extract the normalized medical term (e.g. 'sharp pain').
Output ONLY valid JSON matching this schema:
{"symptoms":["..."],"vitals":{},"medications":[],"history":[]}"""

L2_SYSTEM = """You are a clinical connection analyzer. Given extracted entities from a patient,
find which symptoms explain each other, which drugs interact with findings,
and how history connects to current presentation.

Count INDEPENDENT symptom clusters (groups of related findings).
A cluster is a group of findings that share ONE underlying pathophysiological process.

DECISION FRAMEWORK for early_exit:
- "simple": All findings belong to ONE pathophysiological process.
  Examples: fever + myalgia + cough + fatigue = single infectious process.
  Even if there are many entities, if they ALL connect to one process → simple.
- "continue": Findings span 2+ INDEPENDENT processes, OR multi-organ involvement
  with unclear etiology, OR contradictory/unexpected findings that need deeper analysis.

Do NOT use entity count alone. Use SEMANTIC COHERENCE:
  Ask: "Can ONE disease/process explain ALL these findings?"
  If yes → cluster_count=1, early_exit="simple"
  If no → cluster_count=N, early_exit="continue"

Think step by step in your thinking phase.
Output ONLY valid JSON:
{"connections":["entity A + entity B -> clinical implication"],
 "cluster_count":N, "early_exit":"simple|continue"}"""

L3_SYSTEM = """You are a clinical pattern recognizer. Given patient entities and their connections,
identify known clinical patterns: classic triads, drug-exacerbation patterns,
temporal sequences, well-known disease presentations.

HIGH-YIELD PATTERNS TO RECOGNIZE (non-exhaustive — match if entities fit):
- Sudden fever + severe myalgia/bodyache + fatigue + cough = Influenza (Grip)
- Fever + sore throat + cervical lymphadenopathy = Pharyngitis/Tonsillitis
- Fever + productive cough + dyspnea + crackles = Pneumonia
- Fever + dysuria + urgency + frequency = Urinary Tract Infection
- Fever + diarrhea + vomiting + abdominal cramps = Acute Gastroenteritis
- Periumbilical pain → RLQ migration + fever + leukocytosis = Acute Appendicitis
- Chest pain + troponin elevation + ECG changes = Acute Coronary Syndrome
- Headache + fever + neck stiffness = Meningitis (Kernig/Brudzinski)
- Sudden severe headache ("thunderclap") = Subarachnoid Hemorrhage until proven otherwise
- Confusion + ophthalmoplegia + ataxia = Wernicke Encephalopathy (thiamine deficiency)
- Fever + new murmur + petechiae/splinter hemorrhages = Infective Endocarditis
- Flank pain + hematuria + nausea = Nephrolithiasis
- Polyuria + polydipsia + weight loss = Diabetes (Type 1 if young)
- Drug started recently + symptom onset = Drug-induced (ALWAYS consider)
- Multi-organ dysfunction + fever + altered mental status = Sepsis

{dynamic_patterns}

Think step by step in your thinking phase.
Output ONLY valid JSON:
{"patterns":[{"name":"pattern name","components":["matched components"],"confidence":0.0}],
 "preliminary_differentials":["diagnosis 1","diagnosis 2"],
 "pattern_confidence":0.0}"""

L4_SYSTEM = """You are a context-aware clinical risk analyzer. Given the FULL clinical picture,
determine which findings are ACTUALLY dangerous for THIS SPECIFIC PATIENT.

A finding is only a red flag if it CANNOT be explained by the patient's known conditions
or medications. Example: SpO2 88% in a known COPD patient may be tolerable;
SpO2 88% in a young healthy patient is critical.

Think step by step in your thinking phase.
Output ONLY valid JSON:
{"red_flags":[{"flag":"description","severity":"critical|high|moderate",
"evidence":["supporting evidence"],"context":"why dangerous in this patient"}],
 "urgency":"low|moderate|high|critical"}"""

L5_SYSTEM = """You are a clinical synthesis engine. Given all analysis layers
(entities, connections, patterns, red flags), produce a final assessment.

Determine:
- complexity: simple (1 cluster, clear), moderate (2 clusters OR unclear),
  complex (3+ clusters OR multi-system), critical (life-threatening acute)
- suggested_differentials: top 3-5 preliminary diagnoses for downstream R1
  IMPORTANT: If a pattern was detected in L3 with high confidence, it MUST
  appear as the FIRST suggested differential.
- key_questions: what the next stage should investigate
- pipeline_hint: maps to processing track
- pattern_confidence: confidence of the strongest L3 pattern match (0.0-1.0)

{experience_prior}

Output ONLY valid JSON:
{"complexity":"simple|moderate|complex|critical",
 "suggested_differentials":["dx1","dx2","dx3"],
 "key_questions":["question 1","question 2"],
 "pipeline_hint":"simple|moderate|complex|critical",
 "pattern_confidence":0.0}"""


# ═══════════════════════════════════════════════════════════════════
# DLLMR0 — The 5-Layer Engine
# ═══════════════════════════════════════════════════════════════════

# Critical vitals thresholds for emergency detection in L1 output
_EMERGENCY_VITALS = {
    "spo2": lambda v: v < 85,
    "hr": lambda v: v > 150 or v < 35,
    "sbp": lambda v: v < 70,
    "rr": lambda v: v > 35,
    "temp": lambda v: v > 41,
}

_EMERGENCY_KEYWORDS = {
    "cardiac arrest", "respiratory arrest", "unresponsive", "pulseless",
    "status epilepticus", "massive hemorrhage", "anaphylactic shock",
    "kalp durması", "solunum durması", "yanıtsız", "nabızsız",
    "anafilaktik şok", "masif kanama",
}


class DLLMR0:
    """5-Layer Deep Reasoning Engine — Qwen3.5-0.8B with adaptive depth.

    Each layer takes the previous layer's output and produces a more abstract
    clinical representation, just like DML layers produce increasingly
    abstract visual features.

    Supports:
      - Dynamic L3 pattern injection from Evidence Memory
      - Experience-weighted L5 ranking from past diagnoses
      - 4B escalation for complex/critical cases
    """

    def __init__(self, evidence_store=None) -> None:
        self.settings = get_settings()
        self.client = DLLMClient(base_url=self.settings.dllm_api_url)
        self.evidence_store = evidence_store
        self._4b_client = None  # Lazy-init for escalation

    # ── Public API ──────────────────────────────────────────────────

    async def analyze(self, patient_text: str) -> R0Result:
        """Run layered deep analysis with adaptive depth."""
        total_t0 = time.time()
        timings: dict[str, float] = {}
        all_thinking: list[str] = []

        logger.info("🧠 [DLLM R0] Starting 5-layer analysis (%d chars)...", len(patient_text))

        # ── L1: Multi-Agent Swarm Extraction (Alpha, Beta, Gamma) ──
        t0 = time.time()
        
        # We run the sub-agents concurrently to get Latent Representations
        alpha_task = asyncio.to_thread(self._run_layer, L1_AGENT_CLEANSER_SYSTEM, patient_text, 400, 0.2)
        beta_task = asyncio.to_thread(self._run_layer, L1_AGENT_EXTRACTOR_SYSTEM, patient_text, 400, 0.1)
        gamma_task = asyncio.to_thread(self._run_layer, L1_AGENT_AUDITOR_SYSTEM, patient_text, 400, 0.2)
        
        alpha_res, beta_res, gamma_res = await asyncio.gather(alpha_task, beta_task, gamma_task)
        
        if alpha_res.thinking: all_thinking.append(f"[L1-Alpha] {alpha_res.thinking}")
        if beta_res.thinking: all_thinking.append(f"[L1-Beta] {beta_res.thinking}")
        if gamma_res.thinking: all_thinking.append(f"[L1-Gamma] {gamma_res.thinking}")

        # ── L1: Consensus Synthesizer ─────────────────────────────
        consensus_input = (
            f"RAW PATIENT TEXT:\n{patient_text}\n\n"
            f"AGENT ALPHA (Cleanser) REPORT:\n{alpha_res.output}\n\n"
            f"AGENT BETA (Extractor) REPORT:\n{beta_res.output}\n\n"
            f"AGENT GAMMA (Auditor) REPORT:\n{gamma_res.output}"
        )
        l1 = await asyncio.to_thread(self._run_layer, L1_CONSENSUS_SYSTEM, consensus_input, 600, 0.1, True)
        
        timings["L1_Swarm"] = round(time.time() - t0, 2)
        l1_json = self._parse_json(l1.output, "L1_Consensus")

        if l1.thinking:
            all_thinking.append(f"[L1-Consensus] {l1.thinking}")

        # L1 retry logic
        if not l1_json:
            logger.warning("⚠ [DLLM L1 Swarm] First consensus failed — retrying")
            t0_retry = time.time()
            l1_retry = await asyncio.to_thread(self._run_layer, L1_CONSENSUS_SYSTEM, consensus_input, 650, 0.15, True)
            timings["L1_Swarm"] += round(time.time() - t0_retry, 2)
            l1_json = self._parse_json(l1_retry.output, "L1-retry")
            if l1_json:
                logger.info("✓ [DLLM L1 Swarm] Retry succeeded")
                if l1_retry.thinking:
                    all_thinking.append(f"[L1-retry] {l1_retry.thinking}")

        if not l1_json:
            logger.warning("⚠ [DLLM L1 Swarm] Failed after retry — returning empty R0Result")
            timings["L1"] = timings.get("L1_Swarm", 0) # rename for fallback compatibility
            return R0Result(layers_run=[1], layer_timings=timings,
                            total_time=round(time.time() - total_t0, 2))

        timings["L1"] = timings.pop("L1_Swarm")
        language = l1_json.get("language", "en")
        logger.info("✓ [DLLM L1 Swarm] Entities extracted: %d symptoms, %d meds, %d history items (%.1fs)",
                     len(l1_json.get("symptoms", [])), len(l1_json.get("medications", [])),
                     len(l1_json.get("history", [])), timings["L1"])

        # ── Emergency shortcut: critical vitals → L1 → L4 → L5 ─
        if self._is_emergency(l1_json, patient_text):
            logger.warning("🚨 [DLLM R0] Emergency detected — shortcutting to L4→L5")

            t0 = time.time()
            l4 = self._run_layer(
                L4_SYSTEM,
                f"ENTITIES:\n{json.dumps(l1_json, ensure_ascii=False)}\n\nNo connections or patterns available — EMERGENCY PATH.",
                max_tokens=400, temp=0.2,
            )
            timings["L4"] = round(time.time() - t0, 2)
            l4_json = self._parse_json(l4.output, "L4") or {"red_flags": [], "urgency": "critical"}
            if l4.thinking:
                all_thinking.append(f"[L4] {l4.thinking}")

            t0 = time.time()
            l5_input = self._build_l5_input(l1_json, connections=[], patterns=[], red_flags=l4_json)
            l5 = self._run_layer(L5_SYSTEM, l5_input, max_tokens=300, temp=0.1)
            timings["L5"] = round(time.time() - t0, 2)
            l5_json = self._parse_json(l5.output, "L5") or {"complexity": "critical", "pipeline_hint": "critical"}

            total = round(time.time() - total_t0, 2)
            logger.info("🚨 [DLLM R0] Emergency path done in %.1fs (L1→L4→L5)", total)
            return self._build_result(
                l1_json, {}, {}, l4_json, l5_json,
                layers=[1, 4, 5], timings=timings, total=total,
                language=language, thinking=all_thinking,
            )

        # ── L2: Connection Graph (thinking ON) ─────────────────
        t0 = time.time()
        l2_input = f"ENTITIES:\n{json.dumps(l1_json, ensure_ascii=False)}"
        l2 = self._run_layer(L2_SYSTEM, l2_input, max_tokens=500, temp=0.2)
        timings["L2"] = round(time.time() - t0, 2)
        l2_json = self._parse_json(l2.output, "L2") or {"connections": [], "cluster_count": 1, "early_exit": "continue"}
        if l2.thinking:
            all_thinking.append(f"[L2] {l2.thinking}")

        logger.info("✓ [DLLM L2] %d connections, %d clusters, exit=%s (%.1fs)",
                     len(l2_json.get("connections", [])),
                     l2_json.get("cluster_count", 0),
                     l2_json.get("early_exit", "?"), timings["L2"])

        # ── Early Exit: Simple case → L1→L2→L5 ────────────────
        if l2_json.get("early_exit") == "simple":
            t0 = time.time()
            l5_input = self._build_l5_input(l1_json, l2_json.get("connections", []), [], {})
            l5 = self._run_layer(L5_SYSTEM, l5_input, max_tokens=300, temp=0.1)
            timings["L5"] = round(time.time() - t0, 2)
            l5_json = self._parse_json(l5.output, "L5") or {"complexity": "simple", "pipeline_hint": "simple"}

            total = round(time.time() - total_t0, 2)
            logger.info("⚡ [DLLM R0] Simple case — early exit in %.1fs (L1→L2→L5)", total)
            return self._build_result(
                l1_json, l2_json, {}, {}, l5_json,
                layers=[1, 2, 5], timings=timings, total=total,
                language=language, thinking=all_thinking,
            )

        # ── L3: Pattern Detection (thinking ON) ────────────────
        t0 = time.time()
        l3_input = (
            f"ENTITIES:\n{json.dumps(l1_json, ensure_ascii=False)}\n\n"
            f"CONNECTIONS:\n{json.dumps(l2_json.get('connections', []), ensure_ascii=False)}"
        )
        # Inject dynamic patterns from Evidence Memory
        dynamic_patterns_text = ""
        if self.evidence_store:
            try:
                l1_symptoms = [s if isinstance(s, str) else s.get("symptom", str(s)) for s in l1_json.get("symptoms", [])]
                past_patterns = self.evidence_store.get_relevant_patterns(l1_symptoms, max_patterns=5)
                dynamic_patterns_text = self.evidence_store.format_patterns_for_l3(past_patterns)
            except Exception as exc:
                logger.warning("[DLLM L3] Dynamic pattern retrieval failed: %s", exc)
        l3_prompt = L3_SYSTEM.replace("{dynamic_patterns}", dynamic_patterns_text)
        l3 = self._run_layer(l3_prompt, l3_input, max_tokens=500, temp=0.2)
        timings["L3"] = round(time.time() - t0, 2)
        l3_json = self._parse_json(l3.output, "L3") or {"patterns": [], "preliminary_differentials": []}
        if l3.thinking:
            all_thinking.append(f"[L3] {l3.thinking}")

        logger.info("✓ [DLLM L3] %d patterns, %d prelim DDx (%.1fs)",
                     len(l3_json.get("patterns", [])),
                     len(l3_json.get("preliminary_differentials", [])), timings["L3"])

        # ── L4: Context-Aware Red Flags (thinking ON) ──────────
        t0 = time.time()
        l4_input = (
            f"ENTITIES:\n{json.dumps(l1_json, ensure_ascii=False)}\n\n"
            f"CONNECTIONS:\n{json.dumps(l2_json.get('connections', []), ensure_ascii=False)}\n\n"
            f"PATTERNS:\n{json.dumps(l3_json.get('patterns', []), ensure_ascii=False)}"
        )
        l4 = self._run_layer(L4_SYSTEM, l4_input, max_tokens=500, temp=0.2)
        timings["L4"] = round(time.time() - t0, 2)
        l4_json = self._parse_json(l4.output, "L4") or {"red_flags": [], "urgency": "moderate"}
        if l4.thinking:
            all_thinking.append(f"[L4] {l4.thinking}")

        logger.info("✓ [DLLM L4] %d red flags, urgency=%s (%.1fs)",
                     len(l4_json.get("red_flags", [])),
                     l4_json.get("urgency", "?"), timings["L4"])

        # ── L5: Synthesis (thinking OFF) ───────────────────────
        t0 = time.time()
        l5_input = self._build_l5_input(
            l1_json, l2_json.get("connections", []),
            l3_json.get("patterns", []), l4_json,
        )
        # Inject experience-weighted prior from Evidence Memory
        experience_prior_text = ""
        if self.evidence_store:
            try:
                top_dx = self.evidence_store.get_top_diagnoses(limit=10)
                if top_dx:
                    lines = ["EPIDEMIOLOGICAL CONTEXT from past cases:"]
                    for dx_name, count, avg_conf in top_dx:
                        lines.append(f"  - {dx_name}: {count} case(s), avg confidence {avg_conf:.0%}")
                    lines.append("Use this as a Bayesian prior — common conditions should be")
                    lines.append("considered before rare ones UNLESS symptoms specifically contradict them.")
                    experience_prior_text = "\n".join(lines)
            except Exception as exc:
                logger.warning("[DLLM L5] Experience prior retrieval failed: %s", exc)
        l5_prompt = L5_SYSTEM.replace("{experience_prior}", experience_prior_text)
        l5 = self._run_layer(l5_prompt, l5_input, max_tokens=300, temp=0.1)
        timings["L5"] = round(time.time() - t0, 2)
        l5_json = self._parse_json(l5.output, "L5") or {"complexity": "moderate", "pipeline_hint": "moderate"}

        # ── 4B Escalation: Re-run L3+L5 with 4B for complex/critical ──
        initial_complexity = l5_json.get("complexity", "moderate")
        if initial_complexity in ("complex", "critical"):
            try:
                if self._4b_client is None:
                    self._4b_client = DLLMClient(base_url="http://127.0.0.1:8080")
                if self._4b_client.is_healthy():
                    logger.info("🔄 [DLLM R0] Escalating to 4B model for %s case...", initial_complexity)
                    t0_esc = time.time()

                    # Re-run L3 with 4B
                    l3_esc = self._run_layer_with(
                        self._4b_client, l3_prompt, l3_input, max_tokens=600, temp=0.2,
                    )
                    l3_esc_json = self._parse_json(l3_esc.output, "L3-4B")
                    if l3_esc_json and l3_esc_json.get("patterns"):
                        l3_json = l3_esc_json  # Prefer 4B patterns
                        if l3_esc.thinking:
                            all_thinking.append(f"[L3-4B] {l3_esc.thinking}")

                    # Re-run L5 with 4B using upgraded L3
                    l5_input_esc = self._build_l5_input(
                        l1_json, l2_json.get("connections", []),
                        l3_json.get("patterns", []), l4_json,
                    )
                    l5_esc = self._run_layer_with(
                        self._4b_client, l5_prompt, l5_input_esc, max_tokens=400, temp=0.1,
                    )
                    l5_esc_json = self._parse_json(l5_esc.output, "L5-4B")
                    if l5_esc_json:
                        l5_json = l5_esc_json  # Prefer 4B synthesis

                    esc_time = round(time.time() - t0_esc, 2)
                    timings["4B_escalation"] = esc_time
                    logger.info("✓ [DLLM R0] 4B escalation done in %.1fs — new complexity=%s",
                                 esc_time, l5_json.get("complexity", "?"))
            except Exception as exc:
                logger.warning("[DLLM R0] 4B escalation failed (using 0.8B results): %s", exc)

        total = round(time.time() - total_t0, 2)
        logger.info("✓ [DLLM R0] Full 5-layer analysis in %.1fs — complexity=%s",
                     total, l5_json.get("complexity", "?"))

        return self._build_result(
            l1_json, l2_json, l3_json, l4_json, l5_json,
            layers=[1, 2, 3, 4, 5], timings=timings, total=total,
            language=language, thinking=all_thinking,
        )

    # ── Layer Execution ─────────────────────────────────────────

    def _run_layer(
        self, system_prompt: str, user_content: str,
        max_tokens: int = 512, temp: float = 0.2,
    ) -> DLLMResponse:
        """Execute a single DLLM layer via the 0.8B model."""
        return self._run_layer_with(self.client, system_prompt, user_content, max_tokens, temp)

    def _run_layer_with(
        self, client: DLLMClient, system_prompt: str, user_content: str,
        max_tokens: int = 512, temp: float = 0.2,
    ) -> DLLMResponse:
        """Execute a single DLLM layer via any compatible client (0.8B or 4B)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return client.chat(messages, temperature=temp, max_tokens=max_tokens)

    # ── Emergency Detection (pre-L2) ───────────────────────────

    @staticmethod
    def _is_emergency(l1_json: dict, patient_text: str) -> bool:
        """Check if L1 output or raw text indicates a life-threatening emergency."""
        # Check critical vitals from L1
        vitals = l1_json.get("vitals", {})
        for key, check_fn in _EMERGENCY_VITALS.items():
            val = vitals.get(key)
            if val is not None:
                try:
                    numeric = float(str(val).split("/")[0])  # handle "85/60" → 85
                    if check_fn(numeric):
                        return True
                except (ValueError, IndexError):
                    pass

        # Check emergency keywords in raw text
        text_lower = patient_text.lower()
        return any(kw in text_lower for kw in _EMERGENCY_KEYWORDS)

    # ── L5 Input Builder ───────────────────────────────────────

    @staticmethod
    def _build_l5_input(
        entities: dict, connections: list, patterns: list, red_flags: dict,
    ) -> str:
        """Build the synthesis input from all previous layers."""
        parts = [f"ENTITIES:\n{json.dumps(entities, ensure_ascii=False)}"]
        if connections:
            parts.append(f"CONNECTIONS:\n{json.dumps(connections, ensure_ascii=False)}")
        if patterns:
            parts.append(f"PATTERNS:\n{json.dumps(patterns, ensure_ascii=False)}")
        if red_flags:
            parts.append(f"RED FLAGS:\n{json.dumps(red_flags, ensure_ascii=False)}")
        return "\n\n".join(parts)

    # ── JSON Parsing ───────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str, layer: str) -> dict | None:
        """Parse JSON from layer output, with truncated-JSON repair."""
        if not text or not text.strip():
            logger.warning("⚠ [DLLM %s] Empty output", layer)
            return None

        s = text.strip()
        # Strip markdown fences
        if s.startswith("```json"):
            s = s[7:]
        if s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

        # Attempt 1: direct parse
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # Attempt 1.5: clean trailing commas
        try:
            import re
            cleaned_s = re.sub(r',\s*([\]}])', r'\1', s)
            if cleaned_s != s:
                return json.loads(cleaned_s)
        except json.JSONDecodeError:
            pass

        # Attempt 2: stack-based truncated JSON repair
        try:
            repaired = DLLMR0._repair_truncated_json(s)
            if repaired:
                result = json.loads(repaired)
                logger.info("✓ [DLLM %s] JSON repaired via stack-based closure", layer)
                return result
        except (json.JSONDecodeError, Exception):
            pass

        # Attempt 3: extract first JSON object from text
        try:
            start = s.index("{")
            # Find matching closing brace
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(s)):
                c = s[i]
                if escape_next:
                    escape_next = False
                    continue
                if c == "\\":
                    escape_next = True
                    continue
                if c == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(s[start : i + 1])
        except (ValueError, json.JSONDecodeError):
            pass

        logger.error("✗ [DLLM %s] JSON parse failed after all repair attempts", layer)
        logger.debug("[DLLM %s] Raw: %s", layer, s[:300])
        return None

    @staticmethod
    def _repair_truncated_json(s: str) -> str | None:
        """Attempt to repair truncated JSON by closing open brackets/strings."""
        if not s or s[0] != "{":
            # Try to find the start of JSON
            idx = s.find("{")
            if idx == -1:
                return None
            s = s[idx:]

        # Walk the string tracking open brackets
        stack = []
        in_string = False
        escape_next = False
        last_valid = 0

        for i, c in enumerate(s):
            if escape_next:
                escape_next = False
                last_valid = i
                continue
            if c == "\\":
                escape_next = True
                last_valid = i
                continue
            if c == '"':
                if in_string:
                    in_string = False
                    last_valid = i
                else:
                    in_string = True
                    last_valid = i
                continue
            if in_string:
                last_valid = i
                continue
            if c in "{[":
                stack.append(c)
                last_valid = i
            elif c == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                last_valid = i
            elif c == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
                last_valid = i
            else:
                last_valid = i

        if not stack and not in_string:
            return s  # Already valid

        # Close open constructs
        result = s[:last_valid + 1]
        if in_string:
            result += '"'
        # Close remaining brackets in reverse
        for bracket in reversed(stack):
            if bracket == "{":
                result += "}"
            elif bracket == "[":
                result += "]"
        return result

    # ── Result Builder ─────────────────────────────────────────

    @staticmethod
    def _build_result(
        l1: dict, l2: dict, l3: dict, l4: dict, l5: dict,
        *, layers: list[int], timings: dict, total: float,
        language: str, thinking: list[str],
    ) -> R0Result:
        """Assemble final R0Result from all layer outputs."""
        return R0Result(
            entities=l1,
            red_flags=l4.get("red_flags", []) if l4 else [],
            urgency=l4.get("urgency", l5.get("complexity", "moderate")) if l4 else "moderate",
            complexity=l5.get("complexity", "moderate"),
            suggested_differentials=(
                l5.get("suggested_differentials", [])
                or (l3.get("preliminary_differentials", []) if l3 else [])
            ),
            key_questions=l5.get("key_questions", []),
            connections=l2.get("connections", []) if l2 else [],
            patterns=l3.get("patterns", []) if l3 else [],
            language=language,
            thinking_summary="\n---\n".join(thinking) if thinking else "",
            layers_run=layers,
            layer_timings=timings,
            total_time=total,
        )
