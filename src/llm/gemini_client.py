"""Google Gemini cloud LLM client — async streaming for Super & Deep Thinking modes.

Uses Gemini 3 Flash (Super mode) and Gemini 3 Pro (Deep Thinking mode)
via Google's genai SDK. Same yield interface as GroqClient for drop-in use.

Gemini free tier: 10 RPM / 250K TPD (Flash), 5 RPM / 25K TPD (Pro).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

from google import genai
from google.genai import types

logger = logging.getLogger("rrrie-cdss")

GEMINI_TIMEOUT = 180  # seconds — Pro can be slow on complex prompts


class GeminiClient:
    """Async streaming client for Google Gemini inference."""

    def __init__(
        self,
        api_key: str,
        flash_model: str = "gemini-3-flash-preview",
        pro_model: str = "gemini-3-pro-preview",
    ):
        self.api_key = api_key
        self.flash_model = flash_model
        self.pro_model = pro_model
        self._available: bool | None = None
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        self._available = bool(self.api_key and len(self.api_key) > 10)
        if self._available:
            logger.info(
                "gemini_client_ready flash=%s pro=%s",
                self.flash_model, self.pro_model,
            )
        else:
            logger.info("gemini_client_unavailable — no API key")
        return self._available

    @property
    def model(self) -> str:
        """Default model label (flash)."""
        return self.flash_model

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.3,
        use_pro: bool = False,
    ) -> AsyncIterator[dict]:
        """Stream chat completions from Gemini.

        Yields dicts with keys:
          - {"type": "token", "content": str}
          - {"type": "thinking", "content": str}   — Pro thinking tokens
          - {"type": "done", "usage": dict}

        Args:
            use_pro: True = Gemini 3 Pro (Deep Thinking), False = Gemini 3 Flash (Super)
        """
        model_name = self.pro_model if use_pro else self.flash_model
        client = self._get_client()

        # Convert OpenAI-style messages to Gemini format
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part(text=text)],
                ))

        # Build generation config
        gen_config_params = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        # Enable thinking for Pro model (Deep Thinking mode)
        if use_pro:
            gen_config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=8192,
            )

        config = types.GenerateContentConfig(
            **gen_config_params,
            system_instruction=system_instruction,
        )

        # Run the synchronous SDK streaming in a thread, collect all chunks
        def _stream_all():
            """Runs in a worker thread — consumes the entire stream and returns chunks."""
            chunks = []
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                chunks.append(chunk)
            return chunks

        all_chunks = await asyncio.to_thread(_stream_all)

        prompt_tokens = 0
        completion_tokens = 0
        thinking_tokens = 0

        for chunk in all_chunks:
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        # Gemini exposes thought text as a separate part
                        yield {"type": "thinking", "content": part.text or ""}
                    elif part.text:
                        yield {"type": "token", "content": part.text}

            # Usage metadata (updated each chunk, final chunk has totals)
            if chunk.usage_metadata:
                prompt_tokens = getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0
                # Gemini 3 uses 'thoughts_token_count' (with 's')
                thinking_tokens = getattr(chunk.usage_metadata, "thoughts_token_count", 0) or thinking_tokens

        yield {
            "type": "done",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thinking_tokens": thinking_tokens,
            },
        }

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.3,
        use_pro: bool = False,
    ) -> dict:
        """Non-streaming convenience method."""
        t0 = time.time()
        content = ""
        thinking_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in self.stream_chat(messages, max_tokens, temperature, use_pro=use_pro):
            if chunk["type"] == "token":
                content += chunk["content"]
            elif chunk["type"] == "thinking":
                thinking_text += chunk["content"]
            elif chunk["type"] == "done":
                usage = chunk.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        elapsed = round(time.time() - t0, 2)
        tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

        return {
            "content": content,
            "thinking_text": thinking_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "tok_per_sec": tok_per_sec,
        }
