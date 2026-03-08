"""LLM streaming helpers and JSON parsing utilities.

Extracted from gui/server.py for separation of concerns.
Handles:
  - Groq cloud streaming (OpenAI-compatible SSE)
  - Local llama-server streaming (with <think> block parsing)
  - Robust JSON extraction from LLM output (with truncation repair)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from fastapi import WebSocket

import httpx

logger = logging.getLogger("rrrie-cdss")

# ── Think block regex ───────────────────────────────────────────────
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ── Singleton HTTP client for local llama-server ───────────────────
# Eliminates per-call TCP connection overhead (~50-100ms per call)
_llm_http_pool: httpx.AsyncClient | None = None


def _get_llm_http() -> httpx.AsyncClient:
    """Get or create a shared httpx client for llama-server calls."""
    global _llm_http_pool
    if _llm_http_pool is None or _llm_http_pool.is_closed:
        _llm_http_pool = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _llm_http_pool


async def stream_groq_completion(
    ws: WebSocket,
    groq_client: Any,
    messages: list[dict[str, str]],
    stage: str,
    max_tokens: int = 4096,
) -> dict:
    """Stream a Groq cloud completion and forward tokens to WebSocket.

    Returns dict with: raw_content, clean_content, thinking_text,
    prompt_tokens, completion_tokens, elapsed, tok_per_sec.
    """
    t0 = time.time()
    raw_content = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async for chunk in groq_client.stream_chat(messages, max_tokens=max_tokens, temperature=0.3):
            if chunk["type"] == "token":
                token = chunk["content"]
                raw_content += token
                await ws.send_json({
                    "type": "token",
                    "stage": stage,
                    "content": token,
                    "is_thinking": False,
                })
            elif chunk["type"] == "done":
                usage = chunk.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
    except Exception as exc:
        err_str = str(exc)
        if "429" in err_str or "rate" in err_str.lower():
            logger.warning("[GROQ] Rate limit exhausted after retries for stage %s: %s", stage, exc)
            await ws.send_json({
                "type": "info",
                "stage": stage,
                "content": f"⏳ Groq API rate limit — retries exhausted. Falling back to local model...",
            })
        else:
            logger.error("[GROQ] Streaming error: %s", exc)
            await ws.send_json({
                "type": "info",
                "stage": stage,
                "content": f"⚠ Groq API error: {exc}",
            })
        raise  # Let caller handle fallback

    elapsed = round(time.time() - t0, 2)
    if completion_tokens == 0:
        completion_tokens = len(raw_content.split())
    tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

    return {
        "raw_content": raw_content,
        "clean_content": raw_content.strip(),
        "thinking_text": "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
    }


async def stream_gemini_completion(
    ws: WebSocket,
    gemini_client: Any,
    messages: list[dict[str, str]],
    stage: str,
    max_tokens: int = 8192,
    use_pro: bool = False,
) -> dict:
    """Stream a Gemini cloud completion and forward tokens to WebSocket.

    Args:
        use_pro: True = Gemini 3 Pro (Deep Thinking), False = Gemini 3 Flash (Super)

    Returns dict with: raw_content, clean_content, thinking_text,
    prompt_tokens, completion_tokens, elapsed, tok_per_sec.
    """
    t0 = time.time()
    raw_content = ""
    thinking_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    thinking_tokens = 0

    try:
        async for chunk in gemini_client.stream_chat(
            messages, max_tokens=max_tokens, temperature=0.3, use_pro=use_pro,
        ):
            if chunk["type"] == "token":
                token = chunk["content"]
                raw_content += token
                await ws.send_json({
                    "type": "token",
                    "stage": stage,
                    "content": token,
                    "is_thinking": False,
                })
            elif chunk["type"] == "thinking":
                thinking_text += chunk["content"]
                await ws.send_json({
                    "type": "thinking_token",
                    "stage": stage,
                    "content": chunk["content"],
                })
            elif chunk["type"] == "done":
                usage = chunk.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                thinking_tokens = usage.get("thinking_tokens", 0)
    except Exception as exc:
        err_str = str(exc)
        model_label = "Gemini Pro" if use_pro else "Gemini Flash"
        logger.error("[GEMINI] Streaming error (%s): %s", model_label, exc)
        await ws.send_json({
            "type": "info",
            "stage": stage,
            "content": f"⚠ {model_label} error: {exc}",
        })
        raise  # Let caller handle fallback

    elapsed = round(time.time() - t0, 2)
    if completion_tokens == 0:
        completion_tokens = len(raw_content.split())
    tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

    if thinking_tokens > 0:
        logger.info(
            "[GEMINI] %s stage=%s: %d output tokens + %d thinking tokens in %.1fs",
            "Pro" if use_pro else "Flash", stage, completion_tokens, thinking_tokens, elapsed,
        )

    return {
        "raw_content": raw_content,
        "clean_content": raw_content.strip(),
        "thinking_text": thinking_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
    }


async def stream_llm_completion(
    ws: WebSocket,
    llm_client: Any,
    llama_server_url: str,
    messages: list[dict[str, str]],
    stage: str,
    max_tokens: int = 2048,
    thinking_enabled: bool = True,
    budget_managed: bool = False,
) -> dict:
    """Call llama-server with streaming and forward tokens to WebSocket.

    Returns dict with: raw_content, clean_content, thinking_text,
    prompt_tokens, completion_tokens, elapsed, tok_per_sec.
    """
    # Adaptive thinking budget
    # When budget_managed=True, the TokenBudgetManager already allocated
    # the right amount — skip the multiplier to avoid double-inflation.
    if budget_managed:
        total_budget = max_tokens
    elif thinking_enabled:
        multiplier = llm_client.THINKING_MULTIPLIERS.get(stage, llm_client.THINKING_MULTIPLIERS["default"])
        total_budget = int(max_tokens * multiplier)
    else:
        total_budget = max_tokens
    total_budget = min(total_budget, llm_client.max_ctx // 2)

    payload = {
        "model": llm_client.model_name,
        "messages": messages,
        "max_tokens": total_budget,
        "temperature": 0.6 if not thinking_enabled else 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "stream": True,
    }

    t0 = time.time()
    raw_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    in_think_block = False
    think_buffer = ""

    try:
        _http = _get_llm_http()
        async with _http.stream(
            "POST",
            f"{llama_server_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if not token:
                        continue

                    raw_content += token

                    # ── Think-block state machine ──
                    if "<think>" in token and not in_think_block:
                        in_think_block = True
                        parts = token.split("<think>", 1)
                        if parts[0]:
                            await ws.send_json({
                                "type": "token", "stage": stage,
                                "content": parts[0], "is_thinking": False,
                            })
                        think_buffer = parts[1] if len(parts) > 1 else ""
                        if thinking_enabled:
                            await ws.send_json({"type": "thinking_start", "stage": stage})
                            if think_buffer and "</think>" not in think_buffer:
                                await ws.send_json({
                                    "type": "thinking_token", "stage": stage,
                                    "content": think_buffer,
                                })
                        continue

                    if in_think_block:
                        if "</think>" in token:
                            parts = token.split("</think>", 1)
                            if thinking_enabled and parts[0]:
                                await ws.send_json({
                                    "type": "thinking_token", "stage": stage,
                                    "content": parts[0],
                                })
                            think_buffer += parts[0]
                            in_think_block = False
                            await ws.send_json({"type": "thinking_end", "stage": stage})
                            if len(parts) > 1 and parts[1]:
                                await ws.send_json({
                                    "type": "token", "stage": stage,
                                    "content": parts[1], "is_thinking": False,
                                })
                        else:
                            think_buffer += token
                            if thinking_enabled:
                                await ws.send_json({
                                    "type": "thinking_token", "stage": stage,
                                    "content": token,
                                })
                    else:
                        await ws.send_json({
                            "type": "token", "stage": stage,
                            "content": token, "is_thinking": False,
                        })

    except Exception as exc:
        logger.error("[LLM] Streaming error at stage %s: %s", stage, exc)
        await ws.send_json({
            "type": "error", "stage": stage,
            "content": f"LLM error: {exc}",
        })
        raise  # Re-raise so the orchestrator can handle fallback

    elapsed = round(time.time() - t0, 2)
    if completion_tokens == 0:
        completion_tokens = max(1, len(raw_content) // 4)  # ~4 chars per token (was word count)
    tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

    thinking_text = ""
    think_match = THINK_RE.search(raw_content)
    if think_match:
        thinking_text = think_match.group(1)
    clean_content = THINK_RE.sub("", raw_content).strip()

    return {
        "raw_content": raw_content,
        "clean_content": clean_content,
        "thinking_text": thinking_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
    }


# ── Non-streaming LLM call (for internal R2 query generation) ──────

async def call_llm_no_stream(
    llm_client: Any,
    llama_server_url: str,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
) -> str:
    """Call llama-server WITHOUT streaming — for internal pipeline use.

    Returns the raw text response (think blocks stripped).
    Used by R2 dynamic query generation so we don't flood the WebSocket.
    """
    payload = {
        "model": llm_client.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "top_p": 0.90,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "stream": False,
    }

    t0 = time.time()
    try:
        _http = _get_llm_http()
        resp = await _http.post(
            f"{llama_server_url}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        clean = THINK_RE.sub("", raw).strip()
        elapsed = round(time.time() - t0, 2)
        logger.info("[R2-LLM] Query generation: %d chars in %.1fs", len(clean), elapsed)
        return clean
    except Exception as exc:
        logger.error("[R2-LLM] Non-streaming call failed: %s", exc)
        return ""


async def call_groq_no_stream(
    groq_client: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
) -> str:
    """Call Groq API WITHOUT streaming — for internal pipeline use.

    Returns the raw text response.
    Used by R2 dynamic query generation when not in local_only mode.
    """
    t0 = time.time()
    try:
        full_content = ""
        async for chunk in groq_client.stream_chat(messages, max_tokens=max_tokens, temperature=0.3):
            if chunk["type"] == "token":
                full_content += chunk["content"]
            elif chunk["type"] == "done":
                break
        elapsed = round(time.time() - t0, 2)
        logger.info("[R2-GROQ] Query generation: %d chars in %.1fs", len(full_content), elapsed)
        return full_content.strip()
    except Exception as exc:
        logger.error("[R2-GROQ] Non-streaming call failed: %s", exc)
        return ""


async def call_gemini_no_stream(
    gemini_client: Any,
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    use_pro: bool = False,
) -> str:
    """Call Gemini API WITHOUT streaming — for internal pipeline use.

    Returns the raw text response.
    Used by R2 dynamic query generation in Super/Deep modes.
    """
    t0 = time.time()
    try:
        result = await gemini_client.chat_complete(
            messages, max_tokens=max_tokens, temperature=0.3, use_pro=use_pro,
        )
        content = result.get("content", "").strip()
        elapsed = round(time.time() - t0, 2)
        model_label = "Pro" if use_pro else "Flash"
        logger.info("[R2-GEMINI] Query generation (%s): %d chars in %.1fs", model_label, len(content), elapsed)
        return content
    except Exception as exc:
        logger.error("[R2-GEMINI] Non-streaming call failed: %s", exc)
        return ""


def parse_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response with robust truncation repair.

    Strategies:
      1. Direct parse
      2. Find outermost { ... } block
      3. Stack-based truncation repair (handles nested brackets correctly)
      4. Progressive line removal + repair (backtracking)

    Returns empty dict on failure.
    """
    text = text.strip()
    if not text:
        logger.warning("[JSON-PARSE] Empty response text")
        return {}

    # Remove markdown fences
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find outermost { ... }
    first_brace = text.find("{")
    if first_brace == -1:
        logger.warning("[JSON-PARSE] No '{' found in response: %s", text[:200])
        return {}

    candidate = text[first_brace:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Stack-based truncation repair
    result = _repair_truncated_json(candidate)
    if result is not None:
        logger.info("[JSON-PARSE] Repaired truncated JSON via stack-based closure")
        return result

    # Strategy 4: Progressive line removal + repair (backtracking)
    lines = candidate.split("\n")
    for trim in range(1, min(len(lines), 30)):
        trimmed = "\n".join(lines[:-trim])
        result = _repair_truncated_json(trimmed)
        if result is not None:
            logger.info("[JSON-PARSE] Repaired by removing last %d lines", trim)
            return result

    logger.warning("[JSON-PARSE] All strategies failed. First 500 chars: %s", candidate[:500])
    return {}


def _repair_truncated_json(text: str) -> dict | None:
    """Attempt to repair truncated JSON using stack-based bracket closure.

    Steps:
      1. Detect and remove unclosed string literals
      2. Iteratively clean trailing incomplete elements
      3. Use a proper stack to close all open brackets/braces in correct order
    """
    text = text.rstrip()
    if not text or text[0] != '{':
        return None

    # Step 1: Detect unclosed string and truncate from its start
    in_string = False
    escape_next = False
    last_string_start = -1
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            if not in_string:
                last_string_start = i
            in_string = not in_string

    if in_string and last_string_start >= 0:
        # Truncate from the unclosed string opening
        text = text[:last_string_start].rstrip()

    # Step 2: Iteratively clean trailing incomplete elements
    for _ in range(15):
        prev = text
        text = text.rstrip()
        # Remove trailing comma
        text = re.sub(r',\s*$', '', text)
        # Remove trailing key without value:  "key":
        text = re.sub(r',?\s*"[^"]*"\s*:\s*$', '', text)
        # Remove trailing lone colon
        text = re.sub(r':\s*$', '', text)
        text = text.rstrip()
        if text == prev:
            break

    if not text:
        return None

    # Step 3: Stack-based bracket/brace closure (correct nesting order)
    stack: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    closers = {'[': ']', '{': '}'}
    text += ''.join(closers[c] for c in reversed(stack))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
