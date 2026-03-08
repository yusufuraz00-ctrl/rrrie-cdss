"""RRRIE pipeline package — modular pipeline extracted from gui/server.py.

Modules:
    streaming     — LLM streaming helpers (Groq SSE, llama-server, JSON parser)
    stages        — Individual stage functions (Safety, R1, R2)
    orchestrator  — Full iterative R3↔IE loop + memory + final summary
"""

from src.pipeline.orchestrator import run_rrrie_chat
from src.pipeline.streaming import (
    stream_groq_completion,
    stream_llm_completion,
    parse_json_from_response,
)
from src.pipeline.stages import run_safety, run_r1, run_r2

__all__ = [
    "run_rrrie_chat",
    "stream_groq_completion",
    "stream_llm_completion",
    "parse_json_from_response",
    "run_safety",
    "run_r1",
    "run_r2",
]
