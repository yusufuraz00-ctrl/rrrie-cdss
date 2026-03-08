# RRRIE-CDSS: Autonomous Clinical Decision Support System

**Reasoned → Reinforced → Reasoning → Iterative Evaluation** cognitive protocol for evidence-based clinical decision support.

> ⚠️ **DISCLAIMER:** This system does NOT provide medical advice. All outputs are for informational purposes only and do not replace professional medical evaluation. Clinical decisions must always be made by qualified healthcare professionals.

---

## Architecture

```
Patient Data → [R1: Reasoned Analysis] → [R2: Reinforced Research] → [R3: Reasoning Synthesis] → [IE: Iterative Evaluation]
                   (Ollama local)           (Groq cloud + APIs)          (Ollama local)              (Ollama local)
                                                  ↑                                                        │
                                                  └────────────── iterate if confidence < 0.85 ────────────┘
```

### RRRIE Protocol Stages

| Stage | Purpose | Model | Backend |
|-------|---------|-------|---------|
| **R1** — Reasoned Analysis | Generate differential diagnoses from patient data | Qwen3-4B | Ollama (local) |
| **R2** — Reinforced Research | Search medical databases via function-calling | Llama-3.3-70B | Groq (cloud) |
| **R3** — Reasoning Synthesis | Synthesize evidence with hypotheses | Qwen3-4B | Ollama (local) |
| **IE** — Iterative Evaluation | 8-point quality checklist, decide convergence | Qwen3-4B | Ollama (local) |

### Medical APIs

- **PubMed** E-Utilities — Literature search
- **ClinicalTrials.gov** v2 API — Active/recruiting trials
- **OpenFDA** — Drug interactions & adverse events
- **MedlinePlus** Connect — Treatment guidelines
- **WHO** GHO — Global health observatory data
- **Tavily** — Medical web search (trusted domains)

---

## Hardware Requirements

| Component | Minimum | Used |
|-----------|---------|------|
| GPU VRAM | 6 GB | ~4.2 GB (model + KV-cache + overhead) |
| RAM | 16 GB | ~10 GB (Python + Ollama + Redis) |
| CPU | 6-core | Ryzen 5 7640HS |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed
- [Groq API key](https://console.groq.com/) (free tier)
- Optional: [Tavily API key](https://tavily.com/) for web search

### 2. Setup

```bash
# Clone and enter project
cd rrrie-cdss

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
copy .env.example .env
# Edit .env → add GROQ_API_KEY (required), TAVILY_API_KEY (optional)

# Pull the local model
ollama pull qwen3:4b
```

### 3. Run

```bash
# Start Ollama (if not running)
ollama serve

# Launch Streamlit UI
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

### 4. Docker (Alternative)

```bash
# Setup and start all services
bash deployment/setup.sh

# Or manually
docker compose -f deployment/docker-compose.yml up -d
```

---

## Project Structure

```
rrrie-cdss/
├── config/
│   ├── settings.py          # Pydantic settings (env vars)
│   ├── api_config.py         # API endpoint configs
│   └── model_config.py       # Stage→model mapping
├── src/
│   ├── models/               # Pydantic data models
│   │   ├── patient.py        # PatientData, Vitals
│   │   ├── diagnosis.py      # DifferentialDiagnosis, R1Output, R3Output
│   │   ├── evidence.py       # PubMedArticle, R2Output, AggregatedEvidence
│   │   └── evaluation.py     # IEOutput, EvaluationChecks, RRRIEResult
│   ├── llm/                  # LLM integration
│   │   ├── model_client.py   # HybridLLMClient (Ollama + Groq + Together)
│   │   ├── prompt_templates.py # System prompts for each stage
│   │   └── function_caller.py # Tool-call execution handler
│   ├── core/                 # RRRIE engine
│   │   ├── state.py          # LangGraph TypedDict state
│   │   ├── convergence.py    # Iteration decision logic
│   │   ├── rrrie_engine.py   # LangGraph StateGraph builder
│   │   └── stages/           # R1, R2, R3, IE node functions
│   ├── tools/                # Medical API tools
│   │   ├── pubmed_tool.py
│   │   ├── clinical_trials_tool.py
│   │   ├── openfda_tool.py
│   │   ├── medlineplus_tool.py
│   │   ├── who_tool.py
│   │   ├── web_search_tool.py
│   │   └── tool_registry.py  # OpenAI function-calling schemas
│   ├── cache/
│   │   └── redis_cache.py    # Redis cache layer
│   └── utils/
│       ├── logger.py         # structlog JSON logger
│       ├── medical_codes.py  # ICD-11 validation + WHO API
│       └── safety_checks.py  # Red flag detection, input sanitization
├── ui/
│   ├── app.py                # Streamlit main app
│   └── components/           # UI components
├── tests/                    # pytest test suite
├── deployment/               # Docker, Ollama config
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key for R2 stage |
| `TAVILY_API_KEY` | No | — | Tavily key for web search |
| `OLLAMA_API_URL` | No | `http://localhost:11434/v1` | Ollama endpoint |
| `LOCAL_MODEL_NAME` | No | `qwen3:4b` | Local model name |
| `MAX_RRRIE_ITERATIONS` | No | `3` | Max RRRIE loop iterations |
| `CONFIDENCE_THRESHOLD` | No | `0.85` | Convergence confidence target |

---

## License

Research / Educational use only. Not for clinical deployment.
