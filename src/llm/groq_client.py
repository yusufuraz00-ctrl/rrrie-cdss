"""Groq cloud LLM client — OpenAI-compatible streaming for R1 stage.

Uses Llama-3.3-70B-Versatile (or configurable model) via Groq's ultra-fast
inference API. Falls back gracefully to local llama-server if Groq is unavailable.

Groq free tier: 30 req/min, 14400 tokens/min — sufficient for R1-only usage.
Super Thinking mode uses Groq for ALL stages, so rate limiting is critical.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator

import httpx

logger = logging.getLogger("rrrie-cdss")

# Defaults
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 120  # seconds

# Rate limit / retry config
MAX_RETRIES = 3               # max retry attempts on 429
BASE_BACKOFF = 2.0            # initial backoff seconds
MIN_REQUEST_INTERVAL = 2.5    # minimum seconds between Groq requests (rate pacer)
MAX_RETRY_WAIT = 15.0         # max seconds to wait on a single retry — if server asks more, bail out


class GroqClient:
    """Async streaming client for Groq cloud inference."""

    def __init__(
        self,
        api_key: str,
        model: str = GROQ_DEFAULT_MODEL,
        api_url: str = GROQ_API_URL,
    ):
        self.api_key = api_key
        self.model = model
        self.api_url = api_url.rstrip("/")
        self._available: bool | None = None  # lazy health check
        self._last_request_time: float = 0.0  # rate pacer timestamp

    @property
    def is_available(self) -> bool:
        """Check if Groq API key is configured and non-empty."""
        if self._available is not None:
            return self._available
        self._available = bool(self.api_key and len(self.api_key) > 10)
        if self._available:
            logger.info("groq_client_ready model=%s", self.model)
        else:
            logger.info("groq_client_unavailable — no API key, will use local model")
        return self._available

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _pace_request(self) -> None:
        """Ensure minimum interval between consecutive Groq requests.

        Prevents rapid-fire requests from hitting rate limits.
        """
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            wait = MIN_REQUEST_INTERVAL - elapsed
            logger.debug("[GROQ-PACER] Waiting %.1fs before next request", wait)
            await asyncio.sleep(wait)
        self._last_request_time = time.time()

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[dict]:
        """Stream chat completions from Groq with automatic rate-limit retry.

        Yields dicts with keys:
          - {"type": "token", "content": str}      — answer token
          - {"type": "done", "usage": dict}         — final usage stats

        On 429 (rate limit): waits for retry-after header or exponential backoff,
        then retries up to MAX_RETRIES times. Other HTTP errors are raised immediately.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "stream": True,
        }

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            # Rate pacer: enforce minimum gap between requests
            await self._pace_request()

            try:
                async with httpx.AsyncClient(timeout=GROQ_TIMEOUT) as client:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    ) as response:
                        # Handle rate limiting (429)
                        if response.status_code == 429:
                            retry_after = self._parse_retry_after(response)
                            backoff = max(retry_after, BASE_BACKOFF * (2 ** attempt))
                            # If server demands an absurd wait, bail out immediately
                            if backoff > MAX_RETRY_WAIT:
                                logger.warning(
                                    "[GROQ-RATE-LIMIT] 429 — server wants %.0fs wait (> %.0fs cap). "
                                    "Giving up to trigger local fallback.",
                                    backoff, MAX_RETRY_WAIT,
                                )
                                raise httpx.HTTPStatusError(
                                    f"Rate limited: retry-after {backoff:.0f}s exceeds cap",
                                    request=response.request,
                                    response=response,
                                )
                            logger.warning(
                                "[GROQ-RATE-LIMIT] 429 on attempt %d/%d. "
                                "Waiting %.1fs (retry-after: %.1fs, backoff: %.1fs)",
                                attempt + 1, MAX_RETRIES + 1,
                                backoff, retry_after, BASE_BACKOFF * (2 ** attempt),
                            )
                            await asyncio.sleep(backoff)
                            continue

                        response.raise_for_status()

                        # Successful response — stream tokens
                        async for line in response.aiter_lines():
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

                                # Usage in final chunk
                                usage = chunk.get("usage")
                                if usage:
                                    yield {"type": "done", "usage": usage}

                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield {"type": "token", "content": token}

                        # Success — update timestamp and return
                        self._last_request_time = time.time()
                        return

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    retry_after = self._parse_retry_after(exc.response)
                    backoff = max(retry_after, BASE_BACKOFF * (2 ** attempt))
                    if backoff > MAX_RETRY_WAIT:
                        logger.warning(
                            "[GROQ-RATE-LIMIT] 429 — server wants %.0fs (> %.0fs cap). Bailing out.",
                            backoff, MAX_RETRY_WAIT,
                        )
                        raise
                    logger.warning(
                        "[GROQ-RATE-LIMIT] 429 (exception) on attempt %d/%d. Waiting %.1fs",
                        attempt + 1, MAX_RETRIES + 1, backoff,
                    )
                    await asyncio.sleep(backoff)
                    last_error = exc
                    continue
                raise  # Non-429 errors are raised immediately
            except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                backoff = BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "[GROQ-RETRY] Connection/timeout error on attempt %d/%d: %s. Waiting %.1fs",
                    attempt + 1, MAX_RETRIES + 1, exc, backoff,
                )
                await asyncio.sleep(backoff)
                last_error = exc
                continue

        # All retries exhausted
        logger.error(
            "[GROQ-RATE-LIMIT] All %d attempts exhausted. Last error: %s",
            MAX_RETRIES + 1, last_error,
        )
        if last_error:
            raise last_error
        raise RuntimeError("Groq API rate limit exceeded after all retries")

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float:
        """Extract retry-after delay from Groq's response headers."""
        retry_after = response.headers.get("retry-after", "")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        # Groq also uses x-ratelimit-reset-tokens / x-ratelimit-reset-requests
        for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            val = response.headers.get(header, "")
            if val:
                try:
                    # Value might be like "2.5s" or "1m30s" or just seconds
                    if val.endswith("s"):
                        val = val[:-1]
                    if "m" in val:
                        parts = val.split("m")
                        return float(parts[0]) * 60 + float(parts[1].rstrip("s") or 0)
                    return float(val)
                except (ValueError, IndexError):
                    pass
        return 0.0

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict:
        """Non-streaming convenience method. Returns full result dict.

        Returns:
            {"content": str, "prompt_tokens": int, "completion_tokens": int, "elapsed": float, "tok_per_sec": float}
        """
        t0 = time.time()
        content = ""
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in self.stream_chat(messages, max_tokens, temperature):
            if chunk["type"] == "token":
                content += chunk["content"]
            elif chunk["type"] == "done":
                usage = chunk.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        elapsed = round(time.time() - t0, 2)
        tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "tok_per_sec": tok_per_sec,
        }
