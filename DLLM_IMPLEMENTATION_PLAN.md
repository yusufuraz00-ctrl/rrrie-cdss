# 🧠 DLLM — Deep Large Language Model Implementation Plan

## Felsefe: ML → DML = LLM → DLLM

```
ML:   Input → [tek katman] → Output          (yüzeysel özellik)
DML:  Input → [L1→L2→...→LN] → Output       (hiyerarşik soyutlama)

LLM:  Input → [tek prompt] → Output          (tek geçişli tahmin)
DLLM: Input → [L1→L2→...→LN] → Output       (katmanlı akıl yürütme)
```

**"Deep" = düşünce derinliği**, her katman bir öncekinin çıktısını alıp daha soyut bir temsil üretir. Tıpkı DML'in pikselden nesneye çıkması gibi, DLLM ham metinden klinik karara çıkar.

---

## Mevcut Durum Analizi

### Kodda Neler Var Zaten ✅
- `dllm_r0.py` — temel yapı var ama **tek monolitik prompt** (4 task tek seferde)
- `settings.py` — DLLM config mevcut (port 8081, temp 0.2, max_tokens 768)
- `run.py` — dual llama-server altyapısı hazır (main:8080 + DLLM:8081)
- `LlamaCppClient` — HTTP client mevcut, chat API çalışıyor

### Kritik Eksikler ❌
1. **`LLMEngine` class yok** — `dllm_r0.py` import ediyor ama `llama_cpp_client.py`'da sadece `LlamaCppClient` var
2. **R0 orchestrator'a bağlı değil** — `run_rrrie_chat()` hala direkt `run_safety()` çağırıyor
3. **Router yok** — her vaka aynı pipeline'dan geçiyor (min 2, max 3 iterasyon)
4. **Katmanlı reasoning yok** — R0 tek prompt, tek çağrı, 4 task birden

### Mevcut Pipeline Akışı
```
run_safety() → Drug Lookup → R1 → R2 → [R3 ↔ IE]×(2-3) → Treatment → Post-Mortem
     ↑                                       ↑
  hardcoded keyword                    statik iterasyon sayısı
  + regex zebra                        her vaka için aynı
```

---

## Yeni Mimari: 5-Katmanlı DLLM R0

### DML Analojisi

```
DML (Görüntü tanıma):         DLLM (Klinik analiz):
─────────────────────          ─────────────────────────────────
L1: Piksel → Kenar             L1: Metin → Klinik varlıklar
L2: Kenar → Şekil              L2: Varlıklar → Bağlantı grafiği
L3: Şekil → Nesne parçası      L3: Bağlantılar → Klinik paternler
L4: Nesne parçası → Nesne      L4: Paternler → Kontekst-duyarlı risk
L5: Nesne → Sınıf              L5: Risk → Karar (complexity + yönlendirme)
```

### 5-Katmanlı R0 Pipeline

```
Patient Text
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ L1: EXTRACTOR                                                │
│ temp=0.1, thinking=OFF, ~0.8s                                │
│                                                              │
│ Görev: Saf varlık çıkarımı — "Ne var?"                       │
│ → symptoms, vitals, meds, history, labs, procedures          │
│ → Dil tespiti (en/tr)                                        │
│                                                              │
│ Neden ayrı: 0.8B tek göreve odaklanınca %90+ doğruluk.       │
│ Bağlam düşünmesi gerekmez, sadece NER yapacak.               │
└──────────────┬───────────────────────────────────────────────┘
               │ entities JSON
               ▼
┌──────────────────────────────────────────────────────────────┐
│ L2: CONNECTOR                                                │
│ temp=0.2, thinking=ON, ~1.0s                                 │
│                                                              │
│ Görev: Varlıklar arası bağlantı — "Ne neyi açıklar?"        │
│ → L1 çıktısını alır                                          │
│ → "metformin + elevated troponin + MI hx → DM+CAD risk"     │
│ → "bariatric surgery + vomiting → malabsorption risk"        │
│ → Bağımsız küme sayısı (cluster_count)                       │
│                                                              │
│ + EARLY EXIT KARARI:                                         │
│   cluster_count == 1 ve entity sayısı < 8 → SIMPLE           │
│   → L3, L4 atla, direkt L5'e git                            │
│                                                              │
│ Neden ayrı: Thinking burada şart — gizli ilişkileri keşfeder│
│ DML'deki "kenarlardan şekle geçiş" = varlıklardan bağlantıya│
└──────────────┬───────────────────────────────────────────────┘
               │ connection graph + early_exit kararı
               ▼ (SIMPLE ise L5'e atlar)
┌──────────────────────────────────────────────────────────────┐
│ L3: PATTERN DETECTOR                                         │
│ temp=0.2, thinking=ON, ~1.0s                                 │
│                                                              │
│ Görev: Bilinen klinik paternleri tanı — "Bu hangi triad?"    │
│ → Wernicke triadı (confusion + ataxia + diplopia)            │
│ → ACS cluster (chest pressure + diaphoresis + MI hx)         │
│ → Addisonian crisis (hypotension + hyperkalemia + fatigue)   │
│ → Drug-exacerbation paternleri                               │
│                                                              │
│ Input: L1 entities + L2 connections                          │
│ Output: matched_patterns[] + preliminary_differentials[]     │
│                                                              │
│ Neden ayrı: Patern tespiti bağlantıdan farklı bir soyutlama │
│ seviyesi. L2 "A ve B ilişkili" der, L3 "Bu Wernicke" der.   │
│ Mevcut zebra_detector.py'nin 892 satır regexini TAMAMLAR.    │
└──────────────┬───────────────────────────────────────────────┘
               │ patterns + preliminary DDx
               ▼
┌──────────────────────────────────────────────────────────────┐
│ L4: CONTEXT-AWARE RED FLAG ANALYZER                          │
│ temp=0.2, thinking=ON, ~1.0s                                 │
│                                                              │
│ Görev: BU HASTADA bu bulgu tehlikeli mi? — "Kontekst!"       │
│ → SpO2 88% + COPD hastası → tolere edilir (NOT red flag)     │
│ → SpO2 88% + kalp yetmezliği → RED FLAG                     │
│ → HR 45 + beta-blocker kullanımı → beklenir (NOT red flag)   │
│ → HR 45 + genç sporcu → dikkat (possible red flag)           │
│                                                              │
│ Input: L1 entities + L2 connections + L3 patterns            │
│ Output: context_aware_red_flags[] (seviye + kanıt zinciri)   │
│                                                              │
│ Neden ayrı: Mevcut safety_checks.py blokta kör keyword       │
│ taraması yapıyor. L4 konteksti bildiği için AKILlı flag      │
│ üretir. Eski sistem fallback olarak kalır.                   │
│                                                              │
│ CRITICAL vaka tespitinde: Emergency track'e yönlendirir       │
└──────────────┬───────────────────────────────────────────────┘
               │ red_flags + urgency
               ▼
┌──────────────────────────────────────────────────────────────┐
│ L5: SYNTHESIZER                                              │
│ temp=0.1, thinking=OFF, ~0.7s                                │
│                                                              │
│ Görev: Her şeyi birleştir, karar ver — "Sonuç ne?"          │
│ → complexity: simple | moderate | complex | critical          │
│ → suggested_differentials: R1'e ipucu                        │
│ → key_questions: R1'in sorgulaması gerekenler                │
│ → pipeline_config_hint: Router'a öneri                       │
│                                                              │
│ Input: L1 + L2 + L3 + L4 çıktılarının hepsi                 │
│ Output: Final R0Result JSON                                  │
│                                                              │
│ Neden ayrı: Sentez katmanı, DML'deki classifier'a karşılık   │
│ gelir. Alttaki tüm soyutlamaları TEK bir karara indirir.     │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
         R0Result → Router → Pipeline
```

### Adaptive Depth (Early Exit)

DML'de "early stopping" nasılsa, DLLM'de "early exit" öyledir:

```
Vaka Tipi        Çalışan Katmanlar           Süre      Neden
─────────────    ────────────────────────    ──────    ───────────────────────
🟢 Simple        L1 → L2 → L5              ~2.5s     L2: tek küme, az entity
   (URI, sinüzit)                                      L3-L4 gereksiz

🟡 Moderate      L1 → L2 → L3 → L5         ~3.5s     L3 patern buldu, L4
   (pnömoni)                                           kontekst riski düşük

🔴 Complex       L1 → L2 → L3 → L4 → L5   ~4.5s     Tam analiz gerekli
   (multi-system)

🚨 Critical      L1 → L4 → L5              ~2.5s     L1'de acil vitaller,
   (arrest, şok)                                       direkt L4 red flag
```

Early exit kararını **L2 veriyor** (cluster_count + entity complexity). Critical path ise L1'in vitals çıktısına bakarak L2'yi atlayıp direkt L4'e gider.

### Katman Prompt Tasarımı

Her katmanın promtu **kısa ve odaklı** — 0.8B modelin güçlü olduğu alan bu:

```python
# L1 — ~150 token prompt, ~200 token output
L1_PROMPT = """Extract ALL clinical entities from this patient text.
Output ONLY valid JSON: {"symptoms":[], "vitals":{}, "medications":[], 
"history":[], "labs":{}, "procedures":[], "language":"en|tr"}"""

# L2 — ~200 token prompt (L1 JSON dahil), ~250 token output  
L2_PROMPT = """Given these clinical entities, reason about connections.
Which symptoms explain each other? Which drugs interact with findings?
Think step by step in <think> tags, then output JSON:
{"connections":[], "cluster_count":N, "early_exit":"simple|continue"}"""

# L3 — ~250 token prompt, ~200 token output
L3_PROMPT = """Given entities and connections, identify known clinical patterns.
Look for: triads, drug-exacerbation, temporal sequences, classic presentations.
Think in <think> tags. Output JSON:
{"patterns":[], "preliminary_differentials":[], "pattern_confidence":0.0-1.0}"""

# L4 — ~300 token prompt, ~250 token output
L4_PROMPT = """Given this patient's full clinical picture, determine which findings
are ACTUALLY dangerous IN THIS SPECIFIC CONTEXT. A finding is only a red flag
if it can't be explained by the patient's known conditions or medications.
Think in <think> tags. Output JSON:
{"red_flags":[{"flag":"","severity":"","evidence":[],"context":""}], "urgency":""}"""

# L5 — ~400 token prompt (tüm katmanlar), ~200 token output
L5_PROMPT = """Synthesize all analysis layers into a final assessment.
Output JSON: {"complexity":"", "suggested_differentials":[], 
"key_questions":[], "pipeline_hint":"simple|moderate|complex|critical"}"""
```

### Token Bütçesi (Katman Başına)

```
Katman    Prompt    Thinking    Output    Toplam     Thinking?
──────    ──────    ────────    ──────    ──────     ─────────
L1        ~200      0           ~200      ~400       OFF
L2        ~350      ~200        ~250      ~800       ON
L3        ~400      ~200        ~200      ~800       ON
L4        ~450      ~200        ~250      ~900       ON
L5        ~500      0           ~200      ~700       OFF
──────────────────────────────────────────────────
TOPLAM (5 katman):                        ~3600 token
TOPLAM (3 katman, simple):                ~1900 token
```

0.8B @ RTX 4050 (~120 tok/s):
- 5 katman ≈ 3600/120 = **~4.5s** (ağ gecikmeleri dahil ~5s)
- 3 katman ≈ 1900/120 = **~2.5s** (simple vakalar)

---

## Adaptive Router

### R0 → Router → Pipeline Config

```
R0 L5 çıktısı:
  complexity: "moderate"
  pipeline_hint: "moderate"
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ Router (router.py)                                       │
│                                                          │
│ R0 complexity + urgency + entity sayısı → PipelineConfig │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
    PipelineConfig {
      max_iterations: 2,
      min_iterations: 1,
      r2_tool_count: 3,
      r2_tools: ["pubmed", "europe_pmc", "semantic_scholar"],
      ie_mode: "standard",
      treatment_enabled: true,
      parallel_r1_r2: true,
    }
```

### Route Matrisi

```
Complexity    R2 Tool     Max     IE Modu        Treatment    R1↔R2       Tahmini
              Sayısı      İter                                Paralel     Süre
──────────    ────────    ────    ───────────    ─────────    ────────    ────────
🟢 Simple     2           1       skip/minimal   No           No          15-25s
🟡 Moderate   3           2       standard       Yes          Yes         60-90s
🔴 Complex    6 (full)    3       deep           Yes          Yes         120-180s
🚨 Critical   2 (hedef)   1-2     focused        Yes (acil)   No          40-60s
```

### Critical Track Özel Davranış
```
R0 urgency: "critical" tespit edildiğinde:
1. RED FLAG ALERT hemen WebSocket'e gönderilir
2. R1'e red flag + L4 analizi enjekte edilir
3. R2 sadece hedefe yönelik 2 kaynak arar (PubMed + Web)
4. R3↔IE max 2 iterasyon, IE daha toleranslı
5. Treatment generation acil protokol odaklı
```

---

## Speculative R1↔R2 Execution

### Mevcut (Sıralı)
```
R1 (20s) ────→ R2 (15s) ────→ R3...
                               = 35s (R1+R2)
```

### Yeni (R0 Sayesinde Paralel)
```
R0 (4s) ────→ R1 (20s) ────────┐
              R2 (15s) ────────┤→ R3...
              (R0'ın ön-Dx'i)  │   
                               │   = 24s (R0 + max(R1,R2))     
                               │
              ─── R1 BITTI ────┘
              Eksik Dx varsa:  
              R2-supplement (5s)
```

R0'ın `suggested_differentials` çıktısı R2'ye yeter. R1 bittikten sonra R1'in final Dx listesi ile R0'ınki karşılaştırılır — yeni teşhis varsa R2 ek arama yapar.

```python
# orchestrator.py — Speculative execution
async def _run_r1_r2_speculative(r0_result, ...):
    r1_task = asyncio.create_task(run_r1(...))
    
    # R2'yi R0'ın ön-teşhisleriyle başlat
    r2_task = asyncio.create_task(run_r2_with_hints(
        suggested_differentials=r0_result.suggested_differentials,
        ...
    ))
    
    r1_json, r2_evidence = await asyncio.gather(r1_task, r2_task)
    
    # R1'in bulduğu ama R0'ın bulamadığı teşhisler?
    r1_dx = {d["diagnosis"] for d in r1_json.get("differential_diagnoses", [])}
    r0_dx = set(r0_result.suggested_differentials)
    missing = r1_dx - r0_dx
    
    if missing:
        extra = await run_r2_supplement(list(missing), ...)
        r2_evidence.extend(extra)
    
    return r1_json, r2_evidence
```

**Kazanım: ~10-15 saniye** (moderate/complex vakalarda)

> ⚠️ **Simple ve Critical track'lerde paralel çalıştırma yapılmaz** — basit vakalarda R2 zaten küçük, critical'de sıralılık güvenlik için gerekli.

---

## Smart Iteration Controller

### Problem
```python
# Mevcut — statik
min_iterations = 2  # HER ZAMAN 2
max_iterations = 3  # HER ZAMAN 3
```

### Çözüm: Confidence Dynamics

```python
class IterationController:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.history: list[IterationSnapshot] = []
    
    def should_continue(self, iteration, ie_json, r3_json) -> bool:
        confidence = ie_json.get("confidence", 0)
        decision = ie_json.get("decision", "FINALIZE")
        
        # Snapshot kaydet
        self.history.append(IterationSnapshot(
            iteration=iteration,
            confidence=confidence,
            decision=decision,
            primary_dx=r3_json.get("primary_diagnosis", {}).get("diagnosis", ""),
        ))
        
        # Rule 1: IE FINALIZE dedi ve minimum karşılandı
        if decision == "FINALIZE" and iteration >= self.config.min_iterations:
            return False
        
        # Rule 2: Confidence plateau tespiti
        if self._is_plateauing():
            return False
        
        # Rule 3: Max iterasyona ulaşıldı
        if iteration >= self.config.max_iterations:
            return False
        
        # Rule 4: Minimum iterasyon zorunluluğu (complexity'ye göre)
        if iteration < self.config.min_iterations:
            return True
        
        return decision == "ITERATE"
    
    def _is_plateauing(self) -> bool:
        if len(self.history) < 2:
            return False
        delta = abs(self.history[-1].confidence - self.history[-2].confidence)
        same_dx = (self.history[-1].primary_dx == self.history[-2].primary_dx)
        return delta < 0.03 and same_dx  # < 3% iyileşme + aynı Dx
```

---

## Dosya Değişiklik Haritası

### Yeni Dosyalar

| Dosya | Açıklama | Tahmini Satır |
|-------|----------|---------------|
| `src/pipeline/dllm_r0.py` | **Tamamen yeniden yazılacak** — 5-katmanlı DLLM engine | ~350 |
| `src/pipeline/router.py` | Adaptive pipeline router | ~120 |
| `src/pipeline/iteration_ctrl.py` | Smart iteration controller | ~80 |

### Modifiye Edilecek Dosyalar

| Dosya | Değişiklik | Etki |
|-------|-----------|------|
| `src/llm/llama_cpp_client.py` | `DLLMClient` class ekle (8081 port, R0 parametreleri) | Orta |
| `src/pipeline/orchestrator.py` | R0 çağrısı + Router entegrasyonu + Speculative R1↔R2 + IterationController | Büyük |
| `src/pipeline/stages.py` → `run_safety()` | R0 red_flags'i kabul et, eski sistemi fallback yap | Orta |
| `src/pipeline/stages.py` → `run_r2()` | Config-driven tool selection parametresi ekle | Küçük |
| `config/settings.py` | Katman bazlı DLLM parametreleri | Küçük |
| `run.py` | Zaten hazır, minor dokunuş | Minimal |

### Dokunulmayacaklar
- `streaming.py` — pipeline-agnostik
- `token_budget.py` — zaten pool-based, router config ile uyumlu
- `treatment_safety.py` — bağımsız güvenlik
- `case_store.py` — memory layer
- `prompt_templates.py` — R1/R3/IE promptları değişmez

---

## Detaylı Implementasyon

### 1. `DLLMClient` — `llama_cpp_client.py`'ye ekleme

Mevcut `LlamaCppClient` singleton'dur ve 8080 portunu kullanır. DLLM için ayrı bir lightweight client:

```python
class DLLMClient:
    """Lightweight client for DLLM R0 reasoning engine (0.8B on port 8081).
    
    NOT a singleton — DLLM R0 her pipeline run'da taze oluşturulabilir.
    LlamaCppClient'tan farkı:
      - Farklı port (8081)
      - Thinking strip etmez (L2-L4 thinking'i loglara kaydeder)
      - Daha düşük timeout (0.8B hızlı)
      - json_mode desteği
    """
    
    TIMEOUT = 30  # 0.8B max 30s, aksi halde bir sorun var
    
    def __init__(self, base_url: str = "http://127.0.0.1:8081"):
        self.base_url = base_url
        self._session = requests.Session()
    
    def chat(self, messages, temperature=0.2, max_tokens=512,
             thinking_enabled=True) -> DLLMResponse:
        """Single chat completion call."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 0.6,
            "top_p": 0.95,
            "top_k": 20,
        }
        resp = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload, timeout=self.TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        
        raw = data["choices"][0]["message"]["content"]
        
        # Thinking'i ayır ama HER İKİSİNİ de döndür
        thinking = ""
        output = raw
        if "<think>" in raw:
            parts = raw.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            output = parts[1].strip() if len(parts) > 1 else ""
        
        return DLLMResponse(
            output=output,
            thinking=thinking,
            raw=raw,
            tokens=data.get("usage", {}).get("completion_tokens", 0),
        )
```

### 2. `dllm_r0.py` — 5-Katmanlı DLLM Engine (Yeniden Yazım)

```python
"""DLLM R0 — Deep Large Language Model, 5-Layer Reasoning Engine.

ML→DML farkı katman derinliği ise, LLM→DLLM farkı düşünce derinliğidir.

Her katman bir öncekinin çıktısını alır ve daha soyut bir temsil üretir:
  L1 (Extractor)   — Ham metin → Klinik varlıklar
  L2 (Connector)   — Varlıklar → Bağlantı grafiği + Early Exit kararı
  L3 (Pattern Det.) — Bağlantılar → Klinik paternler + Ön teşhisler
  L4 (Red Flag)    — Tüm bağlam → Kontekst-duyarlı risk analizi
  L5 (Synthesizer) — Her şey → Final karar (complexity + routing)

Adaptive Depth: Simple vakalar L1→L2→L5, Complex vakalar tüm 5 katman.
"""

class DLLMR0:
    """5-Layer Deep Reasoning Engine using Qwen3.5-0.8B."""
    
    async def analyze(self, patient_text: str) -> R0Result:
        """Run layered deep analysis with adaptive depth."""
        
        # L1: Entity Extraction (thinking OFF)
        l1 = await self._run_layer1(patient_text)
        
        # L1'de critical vitals varsa → emergency shortcut
        if self._is_emergency(l1):
            l4 = await self._run_layer4(l1, connections=[], patterns=[])
            l5 = await self._run_layer5(l1, connections=[], patterns=[], red_flags=l4)
            return self._build_result(l1, [], [], l4, l5, layers_run=[1,4,5])
        
        # L2: Connection Graph (thinking ON)
        l2 = await self._run_layer2(l1)
        
        # Early Exit: Simple case?
        if l2.get("early_exit") == "simple":
            l5 = await self._run_layer5(l1, l2, patterns=[], red_flags=[])
            return self._build_result(l1, l2, [], [], l5, layers_run=[1,2,5])
        
        # L3: Pattern Detection (thinking ON)
        l3 = await self._run_layer3(l1, l2)
        
        # L4: Context-Aware Red Flags (thinking ON)
        l4 = await self._run_layer4(l1, l2, l3)
        
        # L5: Synthesis (thinking OFF)
        l5 = await self._run_layer5(l1, l2, l3, l4)
        
        return self._build_result(l1, l2, l3, l4, l5, layers_run=[1,2,3,4,5])
```

### 3. `router.py` — Adaptive Pipeline Router

```python
"""Adaptive Pipeline Router — R0 complexity → PipelineConfig."""

@dataclass
class PipelineConfig:
    track: str                 # "simple" | "moderate" | "complex" | "critical"
    max_iterations: int
    min_iterations: int
    r2_tools: list[str]
    ie_mode: str               # "skip" | "minimal" | "standard" | "deep"
    treatment_enabled: bool
    parallel_r1_r2: bool
    
    @property
    def r2_tool_count(self) -> int:
        return len(self.r2_tools)

ROUTE_TABLE = {
    "simple": PipelineConfig(
        track="simple",
        max_iterations=1, min_iterations=1,
        r2_tools=["search_pubmed", "search_europe_pmc"],
        ie_mode="minimal", treatment_enabled=False, parallel_r1_r2=False,
    ),
    "moderate": PipelineConfig(
        track="moderate",
        max_iterations=2, min_iterations=1,
        r2_tools=["search_pubmed", "search_europe_pmc", "search_semantic_scholar"],
        ie_mode="standard", treatment_enabled=True, parallel_r1_r2=True,
    ),
    "complex": PipelineConfig(
        track="complex",
        max_iterations=3, min_iterations=2,
        r2_tools=["all"],  # tüm mevcut araçlar
        ie_mode="deep", treatment_enabled=True, parallel_r1_r2=True,
    ),
    "critical": PipelineConfig(
        track="critical",
        max_iterations=2, min_iterations=1,
        r2_tools=["search_pubmed", "web_search"],
        ie_mode="standard", treatment_enabled=True, parallel_r1_r2=False,
    ),
}

def route(r0_result: R0Result) -> PipelineConfig:
    """R0 çıktısından pipeline konfigürasyonu belirle."""
    # Urgency critical ise her zaman critical track
    if r0_result.urgency == "critical":
        return ROUTE_TABLE["critical"]
    
    # Complexity bazlı routing
    complexity = r0_result.complexity
    if complexity in ROUTE_TABLE:
        return ROUTE_TABLE[complexity]
    
    # Fallback: moderate
    return ROUTE_TABLE["moderate"]
```

### 4. `orchestrator.py` — Entegrasyon

Mevcut `run_rrrie_chat()` fonksiyonunda yapılacak değişiklikler:

```python
async def run_rrrie_chat(ws, patient_text, ...) -> None:
    # ═══════════════════════════════════════════════════════
    # PHASE 0: DLLM R0 — Deep Layered Analysis
    # ═══════════════════════════════════════════════════════
    from src.pipeline.dllm_r0 import DLLMR0
    from src.pipeline.router import route
    from src.pipeline.iteration_ctrl import IterationController
    
    dllm = DLLMR0()
    r0_result = await dllm.analyze(patient_text)
    
    # R0 başarısız olduysa → eski sisteme fallback
    if not r0_result.entities:
        safety = await run_safety(ws, patient_text)
        r0_failed = True
    else:
        # R0 red_flags'i + eski sistem red_flags'i = birleşik
        old_flags = detect_red_flags(patient_text, patient_text.split())
        merged_flags = _merge_red_flags(r0_result.red_flags, old_flags)
        r0_failed = False
    
    # ═══════════════════════════════════════════════════════
    # ROUTING — Complexity → PipelineConfig
    # ═══════════════════════════════════════════════════════
    if r0_failed:
        config = ROUTE_TABLE["moderate"]  # safe default
    else:
        config = route(r0_result)
    
    iter_ctrl = IterationController(config)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1-2: R1 + R2 (parallel if config allows)
    # ═══════════════════════════════════════════════════════
    if config.parallel_r1_r2 and r0_result.suggested_differentials:
        r1_json, r2_evidence = await _run_r1_r2_speculative(
            ws, patient_text, r0_result, config, ...
        )
    else:
        r1_json = await run_r1(ws, patient_text, ...)
        r2_evidence = await run_r2(ws, r1_json, ...)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 3-4: R3 ↔ IE Loop (IterationController)
    # ═══════════════════════════════════════════════════════
    for iteration in range(1, config.max_iterations + 1):
        r3_json = await run_r3(...)
        ie_json = await run_ie(...)
        
        if not iter_ctrl.should_continue(iteration, ie_json, r3_json):
            break
    
    # Geri kalanı mevcut kodla aynı (treatment, post-mortem, summary)
```

### 5. `run_safety()` Modifikasyonu

```python
async def run_safety(ws, patient_text, r0_result=None) -> dict:
    """R0 varsa onun red_flags'ini kullan, yoksa eski sisteme fallback."""
    
    # Eski keyword-based red flags (her zaman çalışır — güvenlik ağı)
    keyword_flags = detect_red_flags(patient_text, patient_text.split())
    
    if r0_result and r0_result.red_flags:
        # DLLM kontekst-duyarlı flags
        dllm_flags = [
            f"RED FLAG (DLLM): {rf['flag']} [severity: {rf.get('severity', '?')}] "
            f"— evidence: {', '.join(rf.get('evidence', []))}"
            for rf in r0_result.red_flags
        ]
        # Birleşik: DLLM + keyword (deduplicate)
        all_flags = list(set(dllm_flags + keyword_flags))
    else:
        all_flags = keyword_flags
    
    # Zebra detection (mevcut regex sistemi + R0 patterns)
    zebra_matches = detect_zebras(patient_text)
    # R0'ın L3 patern çıktısı zebra listesini zenginleştirir
    
    return {"red_flags": all_flags, "zebra_matches": zebra_matches, ...}
```

---

## DLLM vs Mevcut R0 Karşılaştırma

| Kriter | Mevcut R0 (Monolitik) | Yeni DLLM R0 (5-Katman) |
|--------|----------------------|--------------------------|
| **Yapı** | 1 büyük prompt, 4 task | 5 odaklı katman, her biri tek task |
| **Bağlantı kalitesi** | Orta — prompt dağıtık | Yüksek — L2 sadece bağlantıya odaklanır |
| **Entity doğruluğu** | ~80% (aynı anda çıkarım + analiz) | ~90%+ (L1 sadece NER yapıyor) |
| **Red flag kalitesi** | Keyword + basit kontekst | L4: tam bağlamsal analiz |
| **Early exit** | Yok — her vaka 4 task | Her katmanda confidence check |
| **Debug** | "R0 yanlış" → nerede? | Hangi katmanda kopukluk? → görünür |
| **Fine-tune** | Tüm R0'ı fine-tune | Sadece zayıf katmanı fine-tune |
| **DML analojisi** | Kısmen — tek katman | Tam — her katman bir soyutlama seviyesi |
| **Süre (simple)** | ~4-5s (gereksiz thinking) | ~2.5s (L1→L2→L5) |
| **Süre (complex)** | ~4-5s (yetersiz analiz) | ~4.5s (tam 5 katman, daha derin) |

---

## Performans Projeksiyonu

| Senaryo | Mevcut Pipeline | DLLM Pipeline | İyileşme |
|---------|----------------|---------------|----------|
| 🟢 Simple (URI) | ~160s | R0(2.5s) + R1(15s) + R2(5s) + R3(10s) = **~33s** | **%79** |
| 🟡 Moderate (pnömoni) | ~160s | R0(3.5s) + R1↔R2(20s) + R3↔IE×2(50s) + TX(15s) = **~89s** | **%44** |
| 🔴 Complex (multi) | ~220s | R0(4.5s) + R1↔R2(22s) + R3↔IE×3(70s) + TX(20s) = **~117s** | **%47** |
| 🚨 Critical (acil) | ~160s | R0(2.5s) + R1(15s) + R2(5s) + R3↔IE×1(25s) + TX(10s) = **~58s** | **%64** |

**Ortalama iyileşme: ~%55**

---

## Risk ve Mitigasyon

| Risk | Seviye | Mitigasyon |
|------|--------|-----------|
| L1 extraction yanlış → tüm zincir bozulur | Yüksek | L1 çıktısını mevcut `detect_red_flags()` ile çapraz doğrula. İkisi de boş bulursa "moderate" fallback. |
| 5 sıralı HTTP call → gecikme birikimi | Orta | Her call 0.8B'ye <1.5s. Connection pooling + keep-alive. Early exit ile 2-3 call'a düşürülür. |
| L2 early_exit yanlış "simple" derse | Orta | R1 tarafında zebra_detector ve paradox_resolver hala çalışıyor. Simple track'te bile minimum güvenlik. |
| 0.8B Türkçe metni yanlış parse eder | Orta | L1 prompt'una Türkçe örnek ekle. `_translate_clinical_context_for_r2` zaten mevcut. |
| İki llama-server VRAM aşar | Düşük | 4B(~2.5GB) + 0.8B(~0.5GB) = 3GB < 6GB RTX 4050 |
| LLMEngine import hatası | Kesin | `DLLMClient` olarak yeniden yaz, `LLMEngine` referanslarını kaldır |

---

## Uygulama Fazları

### Faz 1: DLLM Altyapısı (1-2 gün)
1. ✅ `LLMEngine` import hatasını düzelt → `DLLMClient` class yaz
2. `dllm_r0.py`'yi 5-katmanlı DLLM olarak yeniden yaz
3. Her katmanı izole test et (L1 entity doğruluğu, L2 bağlantı kalitesi, vb.)
4. `run_safety()`'ye R0 entegrasyonu (r0_result parametresi)
5. **Test:** Mevcut test vakalarıyla L1 extraction + L2 early_exit doğruluğu

### Faz 2: Router + Orchestrator (1 gün)
6. `router.py` oluştur
7. `iteration_ctrl.py` oluştur
8. `orchestrator.py`'da R0 → Router → Config-driven pipeline
9. R2 tool seçimini config-driven yap
10. **Test:** Simple/moderate/complex vakaların doğru yönlendirildiğini doğrula

### Faz 3: Speculative Execution + Optimizasyon (1 gün)
11. R1↔R2 parallelization (R0'ın ön-teşhisleriyle)
12. IterationController'ı orchestrator'a bağla
13. **Test:** Süre karşılaştırması + doğruluk regresyon testi

### Faz 4 (Opsiyonel — İleri): Fine-Tuning
14. L1 için medical NER LoRA fine-tune dataset hazırla
15. L2 için clinical reasoning fine-tune
16. Router thresholdlarını gerçek vaka verisiyle kalibre et

---

## Doğrulama Kontrol Listesi

Her faz sonrası:
```bash
# 1. Import kontrolü
py -c "from src.pipeline.dllm_r0 import DLLMR0; print('✓ DLLM import OK')"

# 2. Tek katman testi
py -c "
import asyncio
from src.pipeline.dllm_r0 import DLLMR0
dllm = DLLMR0()
r = asyncio.run(dllm._run_layer1('45F, chest pain, HR 110, BP 85/60'))
print(r)
"

# 3. Tam R0 testi
py -c "
import asyncio
from src.pipeline.dllm_r0 import DLLMR0
dllm = DLLMR0()
r = asyncio.run(dllm.analyze('45F, DM, MI hx, metformin, chest pressure, diaphoresis, HR 110'))
print(f'Complexity: {r.complexity}, Layers: {r.layers_run}')
"

# 4. Router testi
py -c "
from src.pipeline.router import route
from src.pipeline.dllm_r0 import R0Result
r = R0Result(complexity='simple', urgency='low')
config = route(r)
print(f'Track: {config.track}, Max iter: {config.max_iterations}')
"

# 5. E2E regresyon
py run.py --test --case pneumonia
py run.py --test --case who_botulism
py run.py --test --all
```
