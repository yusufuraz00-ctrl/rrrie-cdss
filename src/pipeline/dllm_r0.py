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

L1_SYSTEM = """You are a clinical entity extractor. Extract ALL clinical entities from the patient text.
Extract symptoms, vital signs (with numeric values), medications (with doses if mentioned),
medical history, lab results, and procedures/surgeries.
Also detect the language (en or tr).

Output ONLY valid JSON matching this schema:
{"symptoms":[], "vitals":{}, "medications":[], "history":[], "labs":{}, "procedures":[], "language":"en|tr"}

Do NOT add any explanation. Output raw JSON only."""

L2_SYSTEM = """You are a clinical connection analyzer. Given extracted entities from a patient,
find which symptoms explain each other, which drugs interact with findings,
and how history connects to current presentation.

Count INDEPENDENT symptom clusters (groups of related findings).
If cluster_count is 1 AND total entities < 8, set early_exit to "simple".
Otherwise set early_exit to "continue".

Think step by step in your thinking phase.
Output ONLY valid JSON:
{"connections":["entity A + entity B -> clinical implication"],
 "cluster_count":N, "early_exit":"simple|continue"}"""

L3_SYSTEM = """You are a clinical pattern recognizer. Given patient entities and their connections,
identify known clinical patterns: classic triads, drug-exacerbation patterns,
temporal sequences, well-known disease presentations.

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
- key_questions: what the next stage should investigate
- pipeline_hint: maps to processing track

Output ONLY valid JSON:
{"complexity":"simple|moderate|complex|critical",
 "suggested_differentials":["dx1","dx2","dx3"],
 "key_questions":["question 1","question 2"],
 "pipeline_hint":"simple|moderate|complex|critical"}"""


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
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = DLLMClient(base_url=self.settings.dllm_api_url)

    # ── Public API ──────────────────────────────────────────────────

    async def analyze(self, patient_text: str) -> R0Result:
        """Run layered deep analysis with adaptive depth."""
        total_t0 = time.time()
        timings: dict[str, float] = {}
        all_thinking: list[str] = []

        logger.info("🧠 [DLLM R0] Starting 5-layer analysis (%d chars)...", len(patient_text))

        # ── L1: Entity Extraction (thinking OFF) ────────────────
        t0 = time.time()
        l1 = self._run_layer(L1_SYSTEM, patient_text, max_tokens=400, temp=0.1)
        timings["L1"] = round(time.time() - t0, 2)
        l1_json = self._parse_json(l1.output, "L1")

        if not l1_json:
            logger.warning("⚠ [DLLM L1] Failed to extract entities — returning empty R0Result")
            return R0Result(layers_run=[1], layer_timings=timings,
                            total_time=round(time.time() - total_t0, 2))

        language = l1_json.get("language", "en")
        logger.info("✓ [DLLM L1] Entities extracted: %d symptoms, %d meds, %d history items (%.1fs)",
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
        l3 = self._run_layer(L3_SYSTEM, l3_input, max_tokens=500, temp=0.2)
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
        l5 = self._run_layer(L5_SYSTEM, l5_input, max_tokens=300, temp=0.1)
        timings["L5"] = round(time.time() - t0, 2)
        l5_json = self._parse_json(l5.output, "L5") or {"complexity": "moderate", "pipeline_hint": "moderate"}

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.client.chat(messages, temperature=temp, max_tokens=max_tokens)

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
        """Parse JSON from layer output, handling markdown wrapping."""
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

        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            logger.error("✗ [DLLM %s] JSON parse error: %s", layer, e)
            logger.debug("[DLLM %s] Raw: %s", layer, s[:300])
            return None

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
