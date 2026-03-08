"""Application settings — loaded from environment variables via .env file."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for RRRIE-CDSS."""

    # ── Local LLM (HuggingFace Transformers) ───────────────────────
    hf_model_name: str = Field(
        default="Qwen/Qwen3.5-4B",
        description="HuggingFace model name for local inference",
    )
    local_api_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        description="Local OpenAI-compatible API URL",
    )

    # ── DLLM R0 (Neuron-Logic Preprocessing) ───────────────────────
    dllm_api_url: str = Field(
        default="http://127.0.0.1:8081/v1",
        description="Local DLLM (Qwen3.5-0.8B) API URL",
    )
    dllm_model_name: str = Field(
        default="Qwen3.5-0.8B",
    )
    dllm_temperature: float = Field(
        default=0.2, 
        description="Low but non-zero temp for neuron-logic diversity",
    )
    dllm_max_tokens: int = Field(
        default=768, 
        description="Budget for thinking + JSON extraction",
    )

    # ── Cloud LLM (Groq — R2 stage) ────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_api_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible API URL",
    )
    cloud_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Cloud model name for R2 function-calling",
    )

    # ── Fallback Cloud (Together AI) ────────────────────────────────
    together_api_key: str = Field(default="", description="Together AI API key")
    together_model_name: str = Field(default="Qwen/Qwen3-32B")

    # ── Google Gemini (Super + Deep Thinking modes) ─────────────────
    google_api_key: str = Field(default="", description="Google AI API key for Gemini")
    gemini_flash_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini Flash model for Super mode",
    )
    gemini_pro_model: str = Field(
        default="gemini-3-pro-preview",
        description="Gemini Pro model for Deep Thinking mode",
    )

    # ── Medical APIs ────────────────────────────────────────────────
    ncbi_email: str = Field(default="", description="NCBI email for E-Utilities (required by NCBI policy)")
    ncbi_api_key: str = Field(default="", description="NCBI API key for higher PubMed rate limits")
    tavily_api_key: str = Field(default="", description="Tavily Search API key")

    # ── WHO ICD-11 API (OAuth2 Client Credentials) ──────────────────
    icd11_client_id: str = Field(default="", description="WHO ICD-11 API client ID")
    icd11_client_secret: str = Field(default="", description="WHO ICD-11 API client secret")

    # ── Cache ───────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # ── RRRIE Protocol ──────────────────────────────────────────────
    max_rrrie_iterations: int = Field(default=3, ge=1, le=10)
    min_rrrie_iterations: int = Field(default=2, ge=1, le=5)
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    improvement_delta: float = Field(default=0.05, ge=0.0, le=1.0)
    min_evidence_sources: int = Field(default=2, ge=1)
    stagnation_threshold: int = Field(default=2, ge=1, le=5,
        description="If same primary dx repeats this many times with no improvement, force perspective shift")

    # ── LLM Parameters ─────────────────────────────────────────────
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=256, le=16384)
    num_ctx: int = Field(default=8192, description="Model context window size (8K = optimal for 4B)")

    # ── Application ─────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
