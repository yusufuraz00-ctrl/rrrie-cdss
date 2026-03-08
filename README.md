# RRRIE-CDSS: Autonomous Clinical Decision Support System

**Reasoned → Reinforced → Reasoning → Iterative Evaluation** cognitive protocol for evidence-based clinical decision support.

> ⚠️ **DISCLAIMER:** This system does NOT provide medical advice. All outputs are for informational purposes only and do not replace professional medical evaluation. Clinical decisions must always be made by qualified healthcare professionals.

---

## Architecture

```
Patient Data → [R0: DLLM Pre-Analysis] → [R1: Reasoned Analysis] → [R2: Reinforced Research]
                   (Qwen3.5-0.8B local)     (Groq/Gemini cloud)       (8 Medical APIs)
                                                                             │
                   ┌─────────── iterate if IE says ITERATE ←─────────────────┤
                   ↓                                                         │
              [R3: Reasoning Synthesis] ←→ [IE: Iterative Evaluation]        │
                (Groq/Gemini/Local 4B)      (Local 4B — 3-layer)            ↑
                   │                                                         │
                   └→ [Treatment Phase] → [Post-Mortem] → [Memory Store] ────┘
```

### Pipeline Stages

| Stage | Purpose | Model | Backend |
|-------|---------|-------|---------|
| **R0** — DLLM Pre-Analysis | 5-layer deep reasoning: entity extraction → pattern detection → complexity routing | Qwen3.5-0.8B | llama.cpp (local :8081) |
| **R1** — Reasoned Analysis | Differential diagnoses + knowledge gaps | Llama-3.3-70B / Gemini | Groq / Gemini (cloud) |
| **R2** — Reinforced Research | Search 8 medical databases, resolve ICD-11 codes | Deterministic APIs | PubMed, OpenFDA, WHO, etc. |
| **R3** — Reasoning Synthesis | Integrate R1 hypotheses + R2 evidence → diagnosis | Llama-3.3-70B / Gemini / Qwen3.5-4B | Groq / Gemini / llama.cpp (local :8080) |
| **IE** — Iterative Evaluation | 3-layer quality check (hallucination, evidence, synthesis) | Qwen3.5-4B | llama.cpp (local :8080) |
| **Treatment** — Pharmacological Plan | Evidence-based treatment with contraindication cross-check | Groq / Gemini / Qwen3.5-4B | Cloud or local |

### DLLM R0 — 5-Layer Deep Reasoning (Qwen3.5-0.8B)

| Layer | Purpose | Thinking |
|-------|---------|----------|
| L1 — Extractor | Raw text → clinical entities | OFF |
| L2 — Connector | Entities → connection graph + complexity exit | ON |
| L3 — Pattern Detector | Connections → clinical patterns + DDx | ON |
| L4 — Red Flag Analyzer | Patterns → urgency flags + safety alerts | ON |
| L5 — Synthesizer | All layers → structured R0 output | OFF |

Adaptive routing: Simple → L1→L2→L5 (~2.5s) | Moderate → L1→L2→L3→L5 (~3.5s) | Complex → Full L1-L5 (~4.5s) | Critical → L1→L4→L5 (~2.5s)

### Medical APIs (R2)

- **PubMed** E-Utilities — Literature search (NCBI)
- **Europe PMC** — European biomedical literature
- **ClinicalTrials.gov** v2 — Active/recruiting trials
- **OpenFDA** — Drug interactions & adverse events
- **MedlinePlus** Connect — Patient-friendly treatment info
- **Semantic Scholar** — Citation-aware academic search
- **Wikipedia** — Quick medical reference summaries
- **Tavily** — Medical web search (trusted domains)

### Key System Features

- **Iterative R3↔IE Loop**: IE evaluates R3 output → ITERATE/FINALIZE decision. Max iterations controlled by R0 complexity routing.
- **3-Layer IE**: Decomposes the quality check into Layer A (hallucination + symptom coverage), Layer B (evidence + treatment safety), Layer C (decision synthesis).
- **Adaptive Prompting**: R0 complexity assessment dynamically trims verbose prompt sections for simple cases, saving ~2000 tokens on the 8K local context window.
- **Paradox Detection**: Detects drug-exacerbation paradoxes (medication worsened symptoms) and injects investigation directives.
- **Zebra Detection**: Pattern-based rare disease detector runs pre-LLM, injects alerts into R1.
- **3-Tier Memory**: Case store with Tier 1 (raw cases) → Tier 2 (patterns) → Tier 3 (core principles).
- **Token Budget Manager**: Pool-based dynamic allocation across all stages with floor/ceiling guarantees.
- **Treatment Safety**: OpenFDA contraindication lookup + guideline verification before treatment output.
- **WebSocket Streaming**: Real-time token-by-token streaming to the browser UI.

---

## Hardware Requirements

| Component | Minimum | Used |
|-----------|---------|------|
| GPU VRAM | 6 GB | ~5.5 GB (Qwen3.5-4B + Qwen3.5-0.8B + KV-cache) |
| RAM | 16 GB | ~8 GB (Python + llama-server ×2) |
| CPU | 6-core | AMD Ryzen / Intel i5+ |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) server binary (`llama-server`)
- GGUF model files: `Qwen3.5-4B` (main) + `Qwen3.5-0.8B` (DLLM)
- [Groq API key](https://console.groq.com/) (free tier) OR [Gemini API key](https://aistudio.google.com/)

### 2. Setup

```bash
cd rrrie-cdss

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env → add GROQ_API_KEY and/or GEMINI_API_KEY
```

### 3. Start llama-server instances

```bash
# Main model (Qwen3.5-4B) on port 8080
llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf -ngl 99 -c 8192 --port 8080

# DLLM model (Qwen3.5-0.8B) on port 8081
llama-server -m models/Qwen3.5-0.8B-Q4_K_M.gguf -ngl 99 -c 4096 --port 8081
```

### 4. Run

```bash
python run.py
```

Open `http://localhost:8000` in your browser.

---

## Project Structure

```
rrrie-cdss/
├── config/
│   ├── settings.py            # Pydantic settings (env vars)
│   └── api_config.py          # API endpoint configs
├── src/
│   ├── core/                  # Core detection engines
│   │   ├── paradox_resolver.py  # Drug-exacerbation paradox detection
│   │   └── zebra_detector.py    # Rare disease pattern matcher
│   ├── llm/                   # LLM integration
│   │   ├── gemini_client.py     # Google Gemini Flash/Pro client
│   │   ├── groq_client.py       # Groq (Llama-3.3-70B) client
│   │   ├── llama_cpp_client.py  # llama.cpp server client + DLLM
│   │   ├── prompt_templates.py  # System prompts (R1, R3, IE, Treatment)
│   │   └── stage_adapter.py     # R3→IE context bridge + ICD-11 resolver
│   ├── memory/                # 3-tier case memory
│   │   └── case_store.py
│   ├── pipeline/              # RRRIE pipeline engine
│   │   ├── orchestrator.py      # Main pipeline orchestrator (~1800 lines)
│   │   ├── stages.py            # R1, R2, Safety stage runners
│   │   ├── streaming.py         # WebSocket streaming + JSON parsing
│   │   ├── dllm_r0.py           # 5-layer DLLM R0 engine
│   │   ├── ie_layers.py         # 3-layer IE decomposition
│   │   ├── router.py            # Adaptive pipeline routing (R0→config)
│   │   ├── iteration_ctrl.py    # Smart iteration controller
│   │   ├── token_budget.py      # Pool-based token budget manager
│   │   ├── drug_lookup.py       # Drug fact verification
│   │   ├── treatment_safety.py  # Contraindication + guideline checker
│   │   └── medical_knowledge.py # Domain knowledge helpers
│   ├── tools/                 # Medical API tool implementations
│   │   ├── pubmed_tool.py
│   │   ├── europe_pmc_tool.py
│   │   ├── clinical_trials_tool.py
│   │   ├── openfda_tool.py
│   │   ├── medlineplus_tool.py
│   │   ├── semantic_scholar_tool.py
│   │   ├── wikipedia_tool.py
│   │   ├── web_search_tool.py
│   │   └── pharmacology_tool.py
│   └── utils/
│       ├── logger.py            # Logging setup
│       ├── medical_codes.py     # ICD-11 code resolution (WHO API)
│       ├── rate_limiter.py      # API rate limiting
│       └── safety_checks.py     # Red flag detection, input sanitization
├── gui/                       # FastAPI + WebSocket UI
│   ├── server.py                # FastAPI app, WebSocket handler
│   └── static/                  # Frontend (HTML + JS + CSS)
│       ├── index.html
│       ├── app.js
│       └── style.css
├── tests/                     # Test suite
│   ├── test_cases/              # JSON test cases (30+ medical scenarios)
│   ├── test_e2e_medical.py      # End-to-end pipeline tests
│   └── test_tools.py            # API tool unit tests
├── run.py                     # Application entrypoint
├── pyproject.toml
└── requirements.txt
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Configuration

Environment variables (`.env` file):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes* | — | Groq API key (Llama-3.3-70B for R1/R3) |
| `GEMINI_API_KEY` | No* | — | Google Gemini API key (alternative to Groq) |
| `TAVILY_API_KEY` | No | — | Tavily key for medical web search |
| `LLAMA_SERVER_URL` | No | `http://localhost:8080` | Main llama-server endpoint |
| `DLLM_SERVER_URL` | No | `http://localhost:8081` | DLLM (0.8B) llama-server endpoint |
| `MAX_RRRIE_ITERATIONS` | No | `3` | Max R3↔IE loop iterations |
| `CONFIDENCE_THRESHOLD` | No | `0.85` | IE convergence confidence target |

\* At least one cloud API key (Groq or Gemini) is required for R1/R3 cloud inference. The system falls back to local Qwen3.5-4B if no cloud API is available.

---

## License

Research / Educational use only. Not for clinical deployment.
