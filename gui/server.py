"""RRRIE-CDSS Chat GUI — FastAPI + WebSocket backend.

Thin server layer: FastAPI app, singleton factories, routes.
All pipeline logic lives in src/pipeline/{streaming, stages, orchestrator}.

Run:
    cd rrrie-cdss
    py -m gui.server

Then open http://localhost:7860 in a browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import httpx

# ── Path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.llama_cpp_client import LlamaCppClient, LLAMA_SERVER_URL
from src.llm.groq_client import GroqClient
from src.llm.gemini_client import GeminiClient
from src.memory.case_store import CaseStore
from config.settings import get_settings

# Pipeline orchestrator — all RRRIE logic
from src.pipeline.orchestrator import run_rrrie_chat

logger = logging.getLogger("rrrie-cdss")

# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(title="RRRIE-CDSS Chat", version="2.0")

# ── Security: CORS ──────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "http://127.0.0.1:7860"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Max input size for patient stories (50 KB)
MAX_INPUT_BYTES = 50 * 1024
# Pipeline timeout in seconds (15 minutes — complex thinking cases need time)
PIPELINE_TIMEOUT = 900

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Singletons ──────────────────────────────────────────────────────
_client: LlamaCppClient | None = None
_groq: GroqClient | None = None
_gemini: GeminiClient | None = None
_memory: CaseStore | None = None


def get_client() -> LlamaCppClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = LlamaCppClient.get_instance(
            model_name=settings.hf_model_name,
            max_ctx=settings.num_ctx,
        )
    return _client


def get_groq() -> GroqClient:
    """Get or create the Groq client singleton."""
    global _groq
    if _groq is None:
        settings = get_settings()
        _groq = GroqClient(
            api_key=settings.groq_api_key,
            model=settings.cloud_model_name,
            api_url=settings.groq_api_url,
        )
    return _groq


def get_gemini() -> GeminiClient:
    """Get or create the Gemini client singleton."""
    global _gemini
    if _gemini is None:
        settings = get_settings()
        _gemini = GeminiClient(
            api_key=settings.google_api_key,
            flash_model=settings.gemini_flash_model,
            pro_model=settings.gemini_pro_model,
        )
    return _gemini


def get_memory() -> CaseStore:
    """Get or create the 3-tier memory system singleton."""
    global _memory
    if _memory is None:
        _memory = CaseStore()
        stats = _memory.get_stats()
        logger.info(
            "[MEMORY] Online — Tier1: %d cases, Tier2: %d patterns, Tier3: %d principles",
            stats["tier1_cases"], stats["tier2_patterns"], stats["tier3_principles"],
        )
    return _memory


# ── WebSocket Endpoint ──────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """Main chat WebSocket — receives patient stories, streams RRRIE analysis."""
    await ws.accept()

    try:
        while True:
            raw_data = await ws.receive_text()
            if len(raw_data) > MAX_INPUT_BYTES:
                await ws.send_json({"type": "error", "content": f"Input too large ({len(raw_data)} bytes). Max: {MAX_INPUT_BYTES} bytes."})
                continue
            data = json.loads(raw_data)
            msg_type = data.get("type", "")

            if msg_type == "chat":
                patient_text = data.get("content", "").strip()
                thinking_enabled = data.get("thinking", True)
                local_only = data.get("local_only", False)
                super_thinking = data.get("super_thinking", False)
                deep_thinking = data.get("deep_thinking", False)
                expected_output = data.get("expected_output", None)

                if not patient_text:
                    await ws.send_json({"type": "error", "content": "Please enter a patient story."})
                    continue

                if deep_thinking:
                    mode_label = "Deep Thinking (Gemini Pro)"
                elif super_thinking:
                    mode_label = "Super (Gemini Flash)"
                elif local_only:
                    mode_label = "Local" + (" Thinking" if thinking_enabled else " Fast")
                else:
                    mode_label = "Cloud" + (" Thinking" if thinking_enabled else " Fast")
                await ws.send_json({"type": "ack", "content": f"Starting RRRIE analysis... [{mode_label}]"})

                try:
                    await asyncio.wait_for(
                        run_rrrie_chat(
                            ws, patient_text,
                            llm_client=get_client(),
                            groq_client=get_groq(),
                            gemini_client=get_gemini(),
                            llama_server_url=LLAMA_SERVER_URL,
                            memory=get_memory(),
                            thinking_enabled=thinking_enabled if not (super_thinking or deep_thinking) else True,
                            local_only=local_only if not (super_thinking or deep_thinking) else False,
                            super_thinking=super_thinking,
                            deep_thinking=deep_thinking,
                            expected_output=expected_output,
                        ),
                        timeout=PIPELINE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    await ws.send_json({
                        "type": "error",
                        "content": f"Pipeline timed out after {PIPELINE_TIMEOUT}s. Please try a shorter case or use Fast mode.",
                    })
                except Exception as exc:
                    tb = traceback.format_exc()
                    await ws.send_json({
                        "type": "error",
                        "content": f"Pipeline error: {exc}\n{tb}",
                    })

            elif msg_type == "health":
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as _http:
                        r = await _http.get(f"{LLAMA_SERVER_URL}/health")
                    status = r.json() if r.status_code == 200 else {"status": "error"}
                except Exception:
                    status = {"status": "offline"}
                await ws.send_json({"type": "health", "data": status})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"WebSocket error: {exc}")


# ── HTTP Endpoints ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main chat UI."""
    html_path = STATIC_DIR / "index.html"
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/health")
async def health():
    """Backend + llama-server health check."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as _http:
            r = await _http.get(f"{LLAMA_SERVER_URL}/health")
        llm_status = r.json() if r.status_code == 200 else {"status": "error"}
    except Exception:
        llm_status = {"status": "offline"}
    return {
        "backend": "ok",
        "llm_server": llm_status,
    }


@app.get("/api/test-cases")
async def get_test_cases():
    """Retrieve all available test cases from tests/test_cases/."""
    test_cases_dir = PROJECT_ROOT / "tests" / "test_cases"
    cases = []
    
    if test_cases_dir.exists():
        for file_path in test_cases_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    pd = data.get("patient_data", {})
                    patient_text = pd.get("patient_text", "")
                    
                    # If patient_text is empty, construct it from older WHO schema fields
                    if not patient_text:
                        parts = []
                        if "age" in pd and "sex" in pd:
                            parts.append(f"{pd['age']}yo {pd['sex']}.")
                        if "chief_complaint" in pd:
                            parts.append(f"CC: {pd['chief_complaint']}")
                        if "history" in pd:
                            parts.append(f"HPI: {pd['history']}")
                        if "symptoms" in pd and isinstance(pd["symptoms"], list):
                            parts.append(f"Symptoms: {', '.join(pd['symptoms'])}")
                        elif "symptoms" in pd:
                            parts.append(f"Symptoms: {pd['symptoms']}")
                        
                        vitals = pd.get("vitals", {})
                        if vitals and isinstance(vitals, dict):
                            vitals_str = ", ".join([f"{k}: {v}" for k,v in vitals.items()])
                            parts.append(f"Vitals: {vitals_str}")
                            
                        labs = pd.get("lab_results", {})
                        if labs and isinstance(labs, dict):
                            labs_str = ", ".join([f"{k}: {v}" for k,v in labs.items()])
                            parts.append(f"Labs: {labs_str}")
                        
                        patient_text = "\n".join(parts)
                    
                    # Ensure case has required fields to display in UI
                    case_info = {
                        "id": data.get("case_id", file_path.stem),
                        "filename": file_path.name,
                        "title": data.get("title", "Untitled Case"),
                        "patient_text": patient_text,
                        "diagnosis": data.get("expected_output", {}).get("primary_diagnosis", "Unknown"),
                        "expected_output": data.get("expected_output", {}) # Needed for Post-Mortem evaluation
                    }
                    cases.append(case_info)
            except Exception as e:
                logger.error(f"Error loading test case {file_path.name}: {e}")
                
    # Sort cases alphabetically by title or filename
    cases.sort(key=lambda x: str(x.get("title", x.get("filename"))))
    
    return {"cases": cases}


# ── Main ────────────────────────────────────────────────────────────

def main():
    """Launch the GUI server."""
    print("=" * 60)
    print("  RRRIE-CDSS — Clinical Decision Support Chat")
    print("  http://localhost:7860")
    print("=" * 60)
    uvicorn.run(
        "gui.server:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
