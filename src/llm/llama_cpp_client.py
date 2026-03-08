"""llama.cpp GGUF inference client for Qwen3.5-4B -- ADAPTIVE THINKING edition.

Backend: llama-server (b8192+) running as a local HTTP server on port 8080.
Communicates via OpenAI-compatible /v1/chat/completions endpoint.

WHY llama-server INSTEAD OF llama-cpp-python:
  - llama-cpp-python 0.3.16 (latest PyPI) bundles an older llama.cpp backend
    that does NOT support the Qwen3.5 architecture (Gated DeltaNet + MoE).
  - The pre-built llama-server binary (b8192) includes full Qwen3.5 support
    with CUDA acceleration, flash attention, and graph optimisations.
  - No Windows CUDA wheels exist for llama-cpp-python.
  - llama-server provides identical OpenAI-compatible API with better performance.

QWEN3.5 THINKING:
  - Does NOT support /think or /no_think soft switches (unlike Qwen3)
  - Thinking controlled via enable_thinking parameter in chat template
  - Default: thinking ENABLED -> generates <think>...</think> blocks
  - Recommended: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5

ADAPTIVE THINKING STRATEGY:
  Token budgets (thinking + answer):
    R1 (diagnosis):   answer_budget * 2.5
    R3 (synthesis):   answer_budget * 2.0
    IE (evaluation):  answer_budget * 1.5

Expected RTX 4050 (6 GB VRAM):
  - Qwen3-4B bitsandbytes NF4 (HF): ~10 tok/s
  - Qwen3.5-4B GGUF Q4_K_M (server): ~30-60 tok/s
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests as http_requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# -- Configuration -----------------------------------------------------------
LLAMA_SERVER_URL = "http://127.0.0.1:8080"
LLAMA_SERVER_TIMEOUT = 600  # seconds per request (long R1/R3 thinking calls)

# -- Singleton Lock ----------------------------------------------------------
_lock = threading.Lock()
_instance: "LlamaCppClient | None" = None


@dataclass
class LlamaCppChatResponse:
    """Mimics OpenAI ChatCompletion structure for drop-in compatibility."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    class _Message:
        def __init__(self, content: str):
            self.content = content
            self.tool_calls = None

    class _Usage:
        def __init__(self, prompt_tokens: int, completion_tokens: int):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class _Choice:
        def __init__(self, message: "LlamaCppChatResponse._Message"):
            self.message = message

    @property
    def choices(self) -> list:
        return [self._Choice(self._Message(self.content))]

    @property
    def usage(self) -> "_Usage":
        return self._Usage(self.prompt_tokens, self.completion_tokens)


class LlamaCppClient:
    """High-speed local LLM client -- talks to llama-server via HTTP.

    Optimised for Qwen3.5-4B with adaptive thinking.
    Drop-in replacement for HFLocalClient with identical API surface
    (.chat(), .get_instance(), .is_loaded, .get_vram_usage()).
    """

    def __init__(self, model_name: str, max_ctx: int = 4096) -> None:
        self.model_name = model_name
        self.max_ctx = max_ctx
        self._session = http_requests.Session()
        self._server_url = LLAMA_SERVER_URL
        self._verify_server()

    # -- Singleton -----------------------------------------------------------

    @classmethod
    def get_instance(
        cls, model_name: str = "Qwen/Qwen3.5-4B", max_ctx: int = 4096
    ) -> "LlamaCppClient":
        """Get or create the singleton LlamaCppClient instance."""
        global _instance
        if _instance is None:
            with _lock:
                if _instance is None:
                    _instance = cls(model_name, max_ctx)
        return _instance

    # -- Server Health Check -------------------------------------------------

    def _verify_server(self) -> None:
        """Verify llama-server is running and healthy."""
        try:
            resp = self._session.get(
                f"{self._server_url}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                logger.info(
                    "llama_server_connected",
                    model=self.model_name,
                    url=self._server_url,
                    status="healthy",
                )
                return
        except http_requests.ConnectionError:
            pass

        raise ConnectionError(
            f"llama-server not running at {self._server_url}. "
            "Start it with:\n"
            "  llama-server\\llama-server.exe -m <gguf_path> -ngl 99 "
            "--host 127.0.0.1 --port 8080 -c 4096 -fa 1"
        )

    # -- Adaptive Thinking Token Multipliers ---------------------------------
    # Aggressively reduced for speed: 4B model doesn't benefit from long thinking.
    # R1: 1.8→1.2 (diagnosis = quick initial scan)
    # R3: 1.5→1.0 (synthesis = output-focused, not thinking-heavy)
    # IE: 1.0→0.8 (simple checklist, minimal thinking needed)
    THINKING_MULTIPLIERS = {
        "R1": 1.2,
        "R3": 1.0,
        "IE": 0.8,
        "default": 1.2,
    }

    # Regex to strip <think>...</think> blocks from output
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    # -- Chat Completion -----------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        stage: str = "default",
    ) -> LlamaCppChatResponse:
        """Generate a chat completion via llama-server HTTP API.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum new tokens for the ANSWER (thinking extra).
            json_mode: If True, append JSON instruction to system prompt.
            stage: RRRIE stage ("R1", "R3", "IE") for adaptive thinking budget.

        Returns:
            LlamaCppChatResponse with .choices[0].message.content and .usage.
        """
        if json_mode:
            messages = self._inject_json_instruction(messages)

        # Adaptive thinking: scale max_tokens by stage multiplier
        multiplier = self.THINKING_MULTIPLIERS.get(
            stage, self.THINKING_MULTIPLIERS["default"]
        )
        total_budget = int(max_tokens * multiplier)
        total_budget = min(total_budget, self.max_ctx // 2)

        t0 = time.time()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": total_budget,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
        }

        resp = self._session.post(
            f"{self._server_url}/v1/chat/completions",
            json=payload,
            timeout=LLAMA_SERVER_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract raw content (may include <think>...</think> blocks)
        raw_content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Strip thinking blocks, keep only the answer
        think_match = self._THINK_RE.search(raw_content)
        thinking_text = think_match.group(0) if think_match else ""
        thinking_tokens = len(thinking_text.split()) if thinking_text else 0

        clean_content = self._THINK_RE.sub("", raw_content).strip()

        elapsed = round(time.time() - t0, 2)
        tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

        logger.info(
            "gguf_inference_complete",
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens_approx=thinking_tokens,
            answer_tokens_approx=completion_tokens - thinking_tokens,
            time_s=elapsed,
            tok_per_sec=tok_per_sec,
            thinking_multiplier=multiplier,
        )

        return LlamaCppChatResponse(
            content=clean_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _inject_json_instruction(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Append a JSON-mode instruction to the system message."""
        messages = [m.copy() for m in messages]
        json_hint = (
            "\n\nIMPORTANT: You MUST respond with valid JSON only. "
            "No markdown, no explanation outside JSON."
        )

        for m in messages:
            if m["role"] == "system":
                m["content"] += json_hint
                return messages

        messages.insert(0, {"role": "system", "content": json_hint.strip()})
        return messages

    @property
    def is_loaded(self) -> bool:
        """Check if the server is running and responsive."""
        try:
            resp = self._session.get(
                f"{self._server_url}/health", timeout=2
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_vram_usage(self) -> float:
        """Return estimated VRAM usage in GB."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / 1024**3
        except ImportError:
            pass
        return 2.6  # estimated for Qwen3.5-4B Q4_K_M


# ═══════════════════════════════════════════════════════════════════════
# DLLMClient — Lightweight client for DLLM R0 (0.8B on port 8081)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DLLMResponse:
    """Response from the DLLM 0.8B model, preserving thinking + output."""
    output: str = ""
    thinking: str = ""
    raw: str = ""
    tokens: int = 0
    elapsed: float = 0.0


class DLLMClient:
    """Lightweight HTTP client for DLLM R0 reasoning engine (Qwen3.5-0.8B).

    Differences from LlamaCppClient:
      - Targets port 8081 (separate 0.8B server)
      - Preserves <think> blocks for logging (doesn't strip them)
      - Lower timeout (0.8B is fast)
      - Not a singleton — created per pipeline run
    """

    TIMEOUT = 30  # 0.8B should respond in <10s, 30s is generous

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(self, base_url: str = "http://127.0.0.1:8081") -> None:
        # Strip trailing /v1 or /v1/ to prevent double /v1/v1/ in URL
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self._base_url = url
        self._session = http_requests.Session()

    def is_healthy(self) -> bool:
        """Check if the DLLM server is running."""
        try:
            resp = self._session.get(f"{self._base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> DLLMResponse:
        """Single chat completion — returns both thinking and output."""
        t0 = time.time()
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 0.6,
            "top_p": 0.95,
            "top_k": 20,
        }
        resp = self._session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=self.TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        raw = data["choices"][0]["message"]["content"] or ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        elapsed = round(time.time() - t0, 2)

        # Separate thinking from output (keep both)
        thinking = ""
        output = raw
        think_match = self._THINK_RE.search(raw)
        if think_match:
            thinking = think_match.group(1).strip()
            output = self._THINK_RE.sub("", raw).strip()

        logger.info(
            "dllm_inference",
            tokens=tokens,
            time_s=elapsed,
            thinking_len=len(thinking),
            output_len=len(output),
        )

        return DLLMResponse(
            output=output,
            thinking=thinking,
            raw=raw,
            tokens=tokens,
            elapsed=elapsed,
        )
