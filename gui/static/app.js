/* ═══════════════════════════════════════════════════════════════
   RRRIE-CDSS Frontend — app.js
   Aligned with backend WebSocket protocol (orchestrator.py / stages.py / streaming.py)
   
   Backend event types:
     ack, stage_start, stage_complete, stage_result,
     token, thinking_start, thinking_token, thinking_end,
     info, red_flags, zebra_flags,
     api_call, api_result, api_error,
     final_result, error, health, pong
   ═══════════════════════════════════════════════════════════════ */

// ── State ──
let ws;
let isConnected = false;
let currentMessageDiv = null; // The .message-bubble for the AI response
let currentThinkBlock = null; // Active think-block div (stage engine or thinking)
let currentStage = null; // Tracker stage name
let mode = "thinking";
let localMode = false;
let pipelineStartTime = 0;
let timerInterval = null;
let currentGroundTruth = null; // Stores expected_output for post-mortem

const SERVER_URL = `ws://${window.location.host}/ws/chat`;

// ── DOM refs (resolved after DOMContentLoaded) ──
let dom = {};

function cacheDom() {
  dom = {
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    messages: document.getElementById("messages"),
    welcome: document.getElementById("welcome"),
    input: document.getElementById("inputField"),
    sendBtn: document.getElementById("sendBtn"),
    trackerStats: document.getElementById("trackerStats"),
    statTime: document.getElementById("statTime"),
    statSpeed: document.getElementById("statSpeed"),
    memBox: document.getElementById("memoryStatsBox"),
    memT1: document.getElementById("memT1"),
    memT2: document.getElementById("memT2"),
    memT3: document.getElementById("memT3"),
  };
}

// ══════════════════════════════════════════════════════════════
// Lifecycle
// ══════════════════════════════════════════════════════════════
function init() {
  cacheDom();
  connectWebSocket();
  dom.input.addEventListener("keydown", handleKeydown);
  dom.input.addEventListener("input", autoResize);
}

function connectWebSocket() {
  ws = new WebSocket(SERVER_URL);

  ws.onopen = () => {
    isConnected = true;
    setConnectionState("ready", "Connected");
  };

  ws.onclose = () => {
    isConnected = false;
    setConnectionState("error", "Disconnected. Retrying...");
    setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = () => {
    setConnectionState("error", "WebSocket Error");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleServerMessage(data);
    } catch (e) {
      console.error("[WS] Failed to parse message:", e);
    }
  };
}

function setConnectionState(cls, text) {
  dom.statusDot.className = `pulse-dot ${cls}`;
  dom.statusText.textContent = text;
  if (dom.sendBtn) dom.sendBtn.disabled = !isConnected;
}

function setMode(newMode) {
  mode = newMode;
  document
    .getElementById("btnSuperThinking")
    .classList.toggle("active", mode === "super");
  document
    .getElementById("btnDeep")
    .classList.toggle("active", mode === "deep");
  document
    .getElementById("btnThinking")
    .classList.toggle("active", mode === "thinking");
  document
    .getElementById("btnFast")
    .classList.toggle("active", mode === "fast");
  // Super/Deep modes override local toggle
  if ((mode === "super" || mode === "deep") && localMode) {
    localMode = false;
    document.getElementById("btnLocal").classList.remove("active");
  }
}

function toggleLocal() {
  localMode = !localMode;
  const btn = document.getElementById("btnLocal");
  btn.classList.toggle("active", localMode);
  btn.title = localMode
    ? "Local ON: all stages use Qwen 3.5-4B"
    : "Local OFF: cloud stages use Groq/Gemini";
  // Local mode is incompatible with Super/Deep
  if (localMode && (mode === "super" || mode === "deep")) {
    setMode("thinking");
  }
}

// ══════════════════════════════════════════════════════════════
// UI Helpers
// ══════════════════════════════════════════════════════════════
function autoResize() {
  dom.input.style.height = "auto";
  dom.input.style.height = dom.input.scrollHeight + "px";
}

function handleKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function useExample(key) {
  const examples = {
    rabies:
      "Hasta 2 gün önce kırsal alanda sokak köpeği tarafından ısırıldığını belirtiyor. Yara temizlenmiş ama aşı yapılmamış. Şu an yara yerinde karıncalanma, hafif ateş (38.2C) ve yutkunma zorluğu var. Nabız 95, tansiyon 120/80.",
    pneumonia:
      "65 yaşında erkek, bilinen KOAH hastası. 3 gündür artan nefes darlığı, pürülan balgam ve ateş (38.8C). Oksijen satürasyonu %88 oda havasında. Solunum sayısı 26/dk.",
    mi: "55 yaşında kadın hasta. Tip 2 DM ve hipertansiyon öyküsü var. Göğüs ağrısı yok ancak son 2 saattir açıklanamayan nefes darlığı, terleme, mide bulantısı ve sırta vuran huzursuzluk hissi mevcut.",
    meningitis:
      "22 yaşında üniversite öğrencisi. Şiddetli baş ağrısı, ense sertliği ve ışığa hassasiyet ile acile getirildi. Dünden beri yüksek ateşi (39.5C) var ve bilinç konfüze. Ciltte peteşiyal döküntüler dikkat çekiyor.",
  };
  if (examples[key]) {
    dom.input.value = examples[key];
    autoResize();
    dom.input.focus();
  }
}

// ── Test Case Selector ──
let fetchedCases = {};

async function toggleTestCaseDropdown(event) {
  if (event) event.stopPropagation();
  const dropdown = document.getElementById("testCaseDropdown");
  const isVisible = dropdown.style.display === "block";

  // Toggle visibility
  dropdown.style.display = isVisible ? "none" : "block";

  // Fetch if opening and not yet fetched
  if (!isVisible && Object.keys(fetchedCases).length === 0) {
    await fetchTestCases();
  }
}

// Close dropdown when clicking outside
document.addEventListener("click", function (event) {
  const dropdown = document.getElementById("testCaseDropdown");
  const btn = document.querySelector(".case-btn");
  if (
    dropdown &&
    btn &&
    !dropdown.contains(event.target) &&
    !btn.contains(event.target)
  ) {
    dropdown.style.display = "none";
  }
});

async function fetchTestCases() {
  const dropdown = document.getElementById("testCaseDropdown");
  const spinnerLine = dropdown.innerHTML; // save spinner state

  try {
    const response = await fetch("/api/test-cases");
    if (!response.ok) throw new Error("API Error");
    const data = await response.json();

    dropdown.innerHTML = ""; // clear spinner

    if (data.cases.length === 0) {
      dropdown.innerHTML =
        '<div style="padding:15px; text-align:center; color:#a6adc8;">Henüz vaka bulunamadı.</div>';
      return;
    }

    data.cases.forEach((c) => {
      fetchedCases[c.id] = c;

      const item = document.createElement("div");
      item.className = "test-case-item";
      item.onclick = () => selectTestCase(c.id);

      const badgeColor = c.id.startsWith("PUBMED-") ? "#a6e3a1" : "#cba6f7";
      const badgeText = c.id.startsWith("PUBMED-") ? "PUBMED" : "WHO";

      item.innerHTML = `
                <div class="test-case-item-title">${escapeHTML(c.title)}</div>
                <div class="test-case-item-meta">
                    <span><span style="background-color: ${badgeColor}; color: #1e1e2e; padding: 1px 6px; border-radius: 4px; font-weight: bold; font-size: 0.9em; margin-right: 5px;">${badgeText}</span> ${escapeHTML(c.id)}</span>
                </div>
            `;
      dropdown.appendChild(item);
    });
  } catch (err) {
    console.error("Test cases loaded failed:", err);
    dropdown.innerHTML =
      '<div style="padding:15px; text-align:center; color:#f38ba8;"><i class="fa-solid fa-triangle-exclamation"></i> Vaka yüklenemedi</div>';
  }
}

function selectTestCase(id) {
  const c = fetchedCases[id];
  if (c && c.patient_text) {
    dom.input.value = c.patient_text;
    currentGroundTruth = c.expected_output_raw || c.expected_output; // Cache ground truth
    autoResize();
    dom.input.focus();
  }
  document.getElementById("testCaseDropdown").style.display = "none";
}

function scrollToBottom() {
  if (dom.messages) dom.messages.scrollTop = dom.messages.scrollHeight;
}

function escapeHTML(str) {
  if (!str) return "";
  return str.replace(
    /[&<>'"]/g,
    (tag) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;",
      })[tag] || tag,
  );
}

// ══════════════════════════════════════════════════════════════
// Send Message
// ══════════════════════════════════════════════════════════════
function sendMessage() {
  if (!isConnected) return;
  const text = dom.input.value.trim();
  if (!text) return;

  // Hide welcome
  if (dom.welcome) dom.welcome.style.display = "none";

  // Reset sidebar tracker
  resetTracker();

  // Render user bubble
  appendUserMessage(text);

  // Send to backend
  ws.send(
    JSON.stringify({
      type: "chat",
      content: text,
      thinking: mode === "thinking" || mode === "super" || mode === "deep",
      local_only: localMode,
      super_thinking: mode === "super",
      deep_thinking: mode === "deep",
      expected_output: currentGroundTruth, // Injection for Post-Mortem
    }),
  );

  // Clear input
  dom.input.value = "";
  currentGroundTruth = null;
  autoResize();
  dom.sendBtn.disabled = true;

  // Create AI response container
  currentMessageDiv = createAIMessageContainer();
  currentStage = null;
  currentThinkBlock = null;

  // Start live timer
  pipelineStartTime = Date.now();
  startLiveTimer();
}

function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row message-user";
  row.innerHTML = `<div class="message-bubble">${escapeHTML(text)}</div>`;
  dom.messages.appendChild(row);
  scrollToBottom();
}

function createAIMessageContainer() {
  const row = document.createElement("div");
  row.className = "message-row message-ai";
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  row.appendChild(bubble);
  dom.messages.appendChild(row);
  scrollToBottom();
  return bubble;
}

// ══════════════════════════════════════════════════════════════
// Sidebar Tracker
// ══════════════════════════════════════════════════════════════
function resetTracker() {
  document.querySelectorAll(".timeline-step").forEach((el) => {
    el.classList.remove("active", "completed");
  });
  const r3b = document.getElementById("r3-badges");
  const ieb = document.getElementById("ie-badges");
  if (r3b) r3b.innerHTML = "";
  if (ieb) ieb.innerHTML = "";
  if (dom.trackerStats) dom.trackerStats.style.opacity = "0";
  if (dom.memBox) dom.memBox.style.opacity = "0.5";
}

function setStageActive(stageName) {
  // Complete previous
  if (currentStage) {
    const prevEl = document.getElementById(`step-${currentStage}`);
    if (prevEl) {
      prevEl.classList.remove("active");
      prevEl.classList.add("completed");
    }
  }
  currentStage = stageName;
  const newEl = document.getElementById(`step-${stageName}`);
  if (newEl) newEl.classList.add("active");
}

function completeStage(stageName) {
  const el = document.getElementById(`step-${stageName}`);
  if (el) {
    el.classList.remove("active");
    el.classList.add("completed");
  }
}

function addIterBadge(stageName, label) {
  if (stageName !== "R3" && stageName !== "IE") return;
  const box = document.getElementById(`${stageName.toLowerCase()}-badges`);
  if (!box) return;
  // Mark old badges as done
  box
    .querySelectorAll(".mini-badge.running")
    .forEach((b) => b.classList.remove("running"));
  const b = document.createElement("span");
  b.className = "mini-badge running";
  b.textContent = label;
  box.appendChild(b);
}

function startLiveTimer() {
  clearInterval(timerInterval);
  if (dom.trackerStats) dom.trackerStats.style.opacity = "1";
  timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - pipelineStartTime) / 1000).toFixed(1);
    if (dom.statTime) dom.statTime.textContent = elapsed + "s";
  }, 200);
}

function stopLiveTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ══════════════════════════════════════════════════════════════
// WebSocket Message Router — covers ALL backend event types
// ══════════════════════════════════════════════════════════════
function handleServerMessage(data) {
  switch (data.type) {
    // ── Pipeline lifecycle ──
    case "ack":
      handleAck(data);
      break;
    case "stage_start":
      handleStageStart(data);
      break;
    case "stage_complete":
      handleStageComplete(data);
      break;
    case "stage_result":
      handleStageResult(data);
      break;

    // ── LLM streaming tokens ──
    case "token":
      handleToken(data);
      break;
    case "thinking_start":
      handleThinkingStart(data);
      break;
    case "thinking_token":
      handleThinkingToken(data);
      break;
    case "thinking_end":
      handleThinkingEnd(data);
      break;

    // ── Safety events ──
    case "red_flags":
      handleRedFlags(data);
      break;
    case "zebra_flags":
      handleZebraFlags(data);
      break;

    // ── R2 API events ──
    case "api_call":
      handleApiCall(data);
      break;
    case "api_result":
      handleApiResult(data);
      break;
    case "api_error":
      handleApiError(data);
      break;

    // ── Info / status ──
    case "info":
      handleInfo(data);
      break;

    // ── Terminal events ──
    case "final_result":
      handleFinalResult(data);
      break;
    case "error":
      handleError(data);
      break;

    // ── Health / ping ──
    case "health":
    case "pong":
      break;

    default:
      console.log("[WS] Unhandled event type:", data.type, data);
  }
  scrollToBottom();
}

// ══════════════════════════════════════════════════════════════
// Event Handlers
// ══════════════════════════════════════════════════════════════

function handleAck(data) {
  if (!currentMessageDiv) return;
  appendInfoBubble(
    data.content || "Pipeline starting...",
    "fa-circle-play",
    "var(--accent-blue)",
  );
}

function handleStageStart(data) {
  const stage = data.stage; // "SAFETY", "R1", "R2", "R3", "IE"
  const title = data.title || stage;
  const desc = data.description || "";

  setStageActive(stage);

  // Extract iteration info from title: "R3 — Reasoning Synthesis (iter 2/5)"
  const iterMatch = title.match(/iter\s*(\d+)\/(\d+)/i);
  if (iterMatch) addIterBadge(stage, `Iter ${iterMatch[1]}`);

  // Create stage block in chat
  currentThinkBlock = document.createElement("div");
  currentThinkBlock.className = "think-block";
  currentThinkBlock.setAttribute("data-stage", stage);

  const iconMap = {
    SAFETY: "fa-shield-cat",
    R1: "fa-stethoscope",
    R2: "fa-book-medical",
    R3: "fa-code-branch",
    IE: "fa-clipboard-check",
    MEMORY: "fa-brain",
    POST_MORTEM: "fa-graduation-cap",
  };
  const icon = iconMap[stage] || "fa-microchip";

  currentThinkBlock.innerHTML = `
        <div class="think-header">
            <i class="fa-solid ${icon}"></i> ${escapeHTML(title)}
            <div class="spinner" style="margin-left:auto"></div>
        </div>
        <div class="think-desc">${escapeHTML(desc)}</div>
        <div class="think-content"></div>
    `;

  if (currentMessageDiv) currentMessageDiv.appendChild(currentThinkBlock);
}

function handleStageComplete(data) {
  const stage = data.stage;
  completeStage(stage);
  // Remove spinner from the corresponding think block
  if (
    currentThinkBlock &&
    currentThinkBlock.getAttribute("data-stage") === stage
  ) {
    const spinner = currentThinkBlock.querySelector(".spinner");
    if (spinner) spinner.remove();
  }
}

function handleStageResult(data) {
  // data.data contains the JSON result, data.stats has performance info
  // We could render a summary but for now the streaming tokens already show content
  const stats = data.stats || {};
  if (stats.tok_per_sec && dom.statSpeed) {
    dom.statSpeed.textContent = stats.tok_per_sec.toFixed(1) + " t/s";
  }
}

// ── LLM Token Streaming ──
function handleToken(data) {
  if (!currentThinkBlock) return;
  const ct = currentThinkBlock.querySelector(".think-content");
  if (!ct) return;
  ct.innerHTML += escapeHTML(data.content || "");
  ct.scrollTop = ct.scrollHeight;
}

function handleThinkingStart(data) {
  if (!currentThinkBlock) return;
  // Create a nested thinking sub-block inside the current stage block
  const thinkSub = document.createElement("div");
  thinkSub.className = "think-sub";
  thinkSub.innerHTML = `
        <div class="think-sub-header"><i class="fa-solid fa-brain"></i> Internal Reasoning</div>
        <div class="think-sub-content"></div>
    `;
  currentThinkBlock.querySelector(".think-content").appendChild(thinkSub);
}

function handleThinkingToken(data) {
  if (!currentThinkBlock) return;
  const subs = currentThinkBlock.querySelectorAll(".think-sub-content");
  const last = subs[subs.length - 1];
  if (last) {
    last.innerHTML += escapeHTML(data.content || "");
    last.scrollTop = last.scrollHeight;
  }
}

function handleThinkingEnd(data) {
  // No-op visually — the sub-block stays
}

// ── Safety Events ──
function handleRedFlags(data) {
  if (!currentMessageDiv) return;
  const flags = data.flags || [];
  if (!flags.length) return;
  const html = flags.map((f) => `<li>${escapeHTML(f)}</li>`).join("");
  appendInfoBubble(
    `<strong>🚨 Red Flags Detected:</strong><ul>${html}</ul>`,
    "fa-triangle-exclamation",
    "var(--accent-red)",
    true,
  );
}

function handleZebraFlags(data) {
  if (!currentMessageDiv) return;
  const zebras = data.zebras || [];
  if (!zebras.length) return;
  const html = zebras
    .map(
      (z) =>
        `<li><strong>${escapeHTML(z.disease)}</strong> (${z.icd11 || "?"}) — ` +
        `confidence: ${(z.confidence * 100).toFixed(0)}%` +
        (z.key_question ? ` — <em>${escapeHTML(z.key_question)}</em>` : "") +
        `</li>`,
    )
    .join("");
  appendInfoBubble(
    `<strong>🦓 Rare Disease Patterns:</strong><ul>${html}</ul>`,
    "fa-dna",
    "var(--accent-purple)",
    true,
  );
}

// ── R2 API Events ──
function handleApiCall(data) {
  if (!currentThinkBlock) return;
  const ct = currentThinkBlock.querySelector(".think-content");
  if (!ct) return;
  const idx = data.index ? ` [${data.index}/${data.total}]` : "";
  ct.innerHTML += `\n<span class="api-badge calling"><i class="fa-solid fa-satellite-dish"></i> ${escapeHTML(data.api)}${idx}: ${escapeHTML(data.query || "")}</span>\n`;
}

function handleApiResult(data) {
  if (!currentThinkBlock) return;
  const ct = currentThinkBlock.querySelector(".think-content");
  if (!ct) return;
  if (data.api === "PubMed") {
    const articles = data.articles || [];
    const summary = articles.length
      ? articles
          .map((a) => `  • ${a.title || "Unknown"} (PMID: ${a.pmid || "?"})`)
          .join("\n")
      : "  (no results)";
    ct.innerHTML += `<span class="api-badge done"><i class="fa-solid fa-check"></i> PubMed: ${data.count || 0} articles</span>\n${escapeHTML(summary)}\n`;
  } else if (data.api === "WHO ICD-11") {
    ct.innerHTML += `<span class="api-badge done"><i class="fa-solid fa-check"></i> ICD-11: ${escapeHTML(data.code || "?")} — ${escapeHTML(data.title || "")}</span>\n`;
  }
}

function handleApiError(data) {
  if (!currentThinkBlock) return;
  const ct = currentThinkBlock.querySelector(".think-content");
  if (!ct) return;
  ct.innerHTML += `<span class="api-badge error"><i class="fa-solid fa-xmark"></i> ${escapeHTML(data.api || "API")} error: ${escapeHTML(data.error || "Unknown")}</span>\n`;
}

// ── Info messages ──
function handleInfo(data) {
  if (!currentMessageDiv) return;
  const content = data.content || "";
  // If we have an active think block, append inside it
  if (currentThinkBlock) {
    const ct = currentThinkBlock.querySelector(".think-content");
    if (ct) {
      ct.innerHTML += `\n<span class="info-line">${escapeHTML(content)}</span>\n`;
      return;
    }
  }
  appendInfoBubble(content, "fa-circle-info", "var(--accent-blue)");
}

// ── Final Result ──
function handleFinalResult(data) {
  stopLiveTimer();

  // Mark memory stage
  setStageActive("MEMORY");
  setTimeout(() => completeStage("MEMORY"), 500);

  const result = data.data || {};
  const primary = result.primary_diagnosis || {};
  const diffs = result.differential_diagnoses || [];
  const treatment = result.treatment_plan || {};
  const evaluation = result.evaluation || {};
  const zebraFlags = result.zebra_flags || [];
  const questions = result.questions_for_clinician || [];
  const postMortem = result.post_mortem || null;
  const stages = result.stages || {};
  const memStats = result.memory_stats || {};
  const totalTime = result.total_time || 0;
  const iterations = result.iterations || 1;
  const iterHistory = result.iteration_history || [];

  // Update tracker stats
  if (dom.trackerStats) dom.trackerStats.style.opacity = "1";
  if (dom.statTime) dom.statTime.textContent = totalTime.toFixed(1) + "s";
  if (dom.statSpeed && stages.R3)
    dom.statSpeed.textContent = (stages.R3.tok_s || 0).toFixed(1) + " t/s";

  // Update memory panel
  if (dom.memBox) dom.memBox.style.opacity = "1";
  if (dom.memT1) dom.memT1.textContent = memStats.tier1_cases || 0;
  if (dom.memT2) dom.memT2.textContent = memStats.tier2_patterns || 0;
  if (dom.memT3) dom.memT3.textContent = memStats.tier3_principles || 0;

  // ── Build rich HTML output ──
  let html = "";

  // Primary diagnosis card
  html += `<div class="result-section primary-dx">`;
  html += `<h2><i class="fa-solid fa-bullseye"></i> Primary Diagnosis</h2>`;
  html += `<div class="dx-card primary">`;
  html += `<div class="dx-name">${escapeHTML(primary.diagnosis || "Unknown")}</div>`;
  html += `<div class="dx-meta">`;
  html += `<span class="confidence-badge c-${confidenceLevel(primary.confidence)}">${((primary.confidence || 0) * 100).toFixed(0)}% confidence</span>`;
  html += `<span class="evidence-badge">${escapeHTML(primary.evidence_support || "unknown")} evidence</span>`;
  if (primary.icd11_code)
    html += `<span class="icd-badge">${escapeHTML(primary.icd11_code)}</span>`;
  html += `</div>`;
  if (primary.reasoning_chain && primary.reasoning_chain.length) {
    html += `<div class="dx-reasoning">`;
    html += `<strong>Reasoning:</strong><ol>`;
    primary.reasoning_chain.forEach((step) => {
      html += `<li>${escapeHTML(step)}</li>`;
    });
    html += `</ol></div>`;
  }
  if (primary.unexplained_symptoms && primary.unexplained_symptoms.length) {
    html += `<div class="dx-gaps"><strong>⚠ Unexplained:</strong> ${primary.unexplained_symptoms.map((s) => escapeHTML(s)).join(", ")}</div>`;
  }
  html += `</div></div>`;

  // Differential diagnoses
  if (diffs.length) {
    html += `<div class="result-section">`;
    html += `<h2><i class="fa-solid fa-list-ol"></i> Differential Diagnoses</h2>`;
    diffs.forEach((dx) => {
      const conf = dx.updated_confidence || dx.confidence || 0;
      html += `<div class="dx-card">`;
      html += `<div class="dx-name">${escapeHTML(dx.diagnosis || "?")}</div>`;
      html += `<div class="dx-meta"><span class="confidence-badge c-${confidenceLevel(conf)}">${(conf * 100).toFixed(0)}%</span>`;
      if (dx.icd11_code)
        html += `<span class="icd-badge">${escapeHTML(dx.icd11_code)}</span>`;
      html += `</div>`;
      if (dx.evidence_summary)
        html += `<div class="dx-evidence">${escapeHTML(dx.evidence_summary)}</div>`;
      html += `</div>`;
    });
    html += `</div>`;
  }

  // Treatment plan
  if (treatment.immediate_actions || treatment.pharmacological) {
    html += `<div class="result-section">`;
    html += `<h2><i class="fa-solid fa-prescription"></i> Treatment Plan</h2>`;
    if (treatment.immediate_actions && treatment.immediate_actions.length) {
      html += `<h3>🔴 Immediate Actions</h3><ul>`;
      treatment.immediate_actions.forEach((a) => {
        html += `<li>${escapeHTML(a)}</li>`;
      });
      html += `</ul>`;
    }
    if (treatment.pharmacological && treatment.pharmacological.length) {
      html += `<h3>💊 Pharmacological</h3>`;
      html += `<table class="rx-table"><tr><th>Drug</th><th>Dose</th><th>Route</th><th>Duration</th><th>Source</th></tr>`;
      treatment.pharmacological.forEach((rx) => {
        html += `<tr><td>${escapeHTML(rx.drug || "")}</td><td>${escapeHTML(rx.dose || "")}</td>`;
        html += `<td>${escapeHTML(rx.route || "")}</td><td>${escapeHTML(rx.duration || "")}</td>`;
        html += `<td>${escapeHTML(rx.source || "")}</td></tr>`;
      });
      html += `</table>`;
    }
    if (treatment.monitoring && treatment.monitoring.length) {
      html += `<h3>📊 Monitoring</h3><ul>`;
      treatment.monitoring.forEach((m) => {
        html += `<li>${escapeHTML(m)}</li>`;
      });
      html += `</ul>`;
    }
    html += `</div>`;
  }

  // Zebra flags
  if (zebraFlags.length) {
    html += `<div class="result-section zebra-section">`;
    html += `<h2><i class="fa-solid fa-dna"></i> Rare Disease Alerts</h2>`;
    zebraFlags.forEach((z) => {
      html += `<div class="zebra-card"><strong>${escapeHTML(z.disease)}</strong>`;
      if (z.icd11)
        html += ` <span class="icd-badge">${escapeHTML(z.icd11)}</span>`;
      html += ` — ${(z.confidence * 100).toFixed(0)}%`;
      if (z.key_question)
        html += `<br><em>Key Q: ${escapeHTML(z.key_question)}</em>`;
      html += `</div>`;
    });
    html += `</div>`;
  }

  // Questions for clinician
  if (questions.length) {
    html += `<div class="result-section">`;
    html += `<h2><i class="fa-solid fa-user-doctor"></i> Questions for Clinician</h2><ul>`;
    questions.forEach((q) => {
      html += `<li><strong>${escapeHTML(q.question || "")}</strong>`;
      if (q.clinical_impact)
        html += ` — <em>${escapeHTML(q.clinical_impact)}</em>`;
      html += `</li>`;
    });
    html += `</ul></div>`;
  }

  // IE Evaluation summary
  html += `<div class="result-section eval-section">`;
  html += `<h2><i class="fa-solid fa-scale-balanced"></i> Quality Evaluation</h2>`;
  html += `<div class="eval-meta">`;
  html += `<span>Decision: <strong>${escapeHTML(evaluation.decision || "?")}</strong></span>`;
  html += `<span>IE Confidence: <strong>${((evaluation.confidence || 0) * 100).toFixed(0)}%</strong></span>`;
  html += `<span>Iterations: <strong>${iterations}</strong></span>`;
  html += `</div>`;
  if (evaluation.reasoning) {
    html += `<p class="eval-reasoning">${escapeHTML(evaluation.reasoning)}</p>`;
  }
  html += `</div>`;

  // Post-Mortem Learning Loop (Clinical Pearl)
  if (postMortem) {
    let pmColor = "var(--accent-blue)";
    let pmIcon = "fa-graduation-cap";
    if (postMortem.accuracy_status === "EXACT_MATCH") {
      pmColor = "var(--accent-green)";
      pmIcon = "fa-check-double";
    } else if (postMortem.accuracy_status === "PARTIAL_MATCH") {
      pmColor = "var(--accent-yellow)";
      pmIcon = "fa-code-compare";
    } else if (postMortem.accuracy_status === "MISDIAGNOSED") {
      pmColor = "var(--accent-red)";
      pmIcon = "fa-skull-crossbones";
    }

    html += `<div class="result-section pm-section" style="border-left: 4px solid ${pmColor}; padding-left: 15px; margin-top: 20px; background: rgba(0,0,0,0.2); border-radius: 8px;">`;
    html += `<h2 style="color: ${pmColor};"><i class="fa-solid ${pmIcon}"></i> Post-Mortem Evaluation</h2>`;
    
    html += `<div style="margin-bottom: 10px;">`;
    html += `<strong>Accuracy: </strong> <span style="color: ${pmColor}; font-weight: bold;">${escapeHTML(postMortem.accuracy_status || "UNKNOWN")}</span>`;
    html += `</div>`;
    
    if (postMortem.critique) {
      html += `<div style="margin-bottom: 15px;">`;
      html += `<strong>Critique:</strong><br><span style="color: #a6adc8;">${escapeHTML(postMortem.critique)}</span>`;
      html += `</div>`;
    }
    
    if (postMortem.clinical_pearl) {
      html += `<div style="background: rgba(203, 166, 247, 0.1); border: 1px solid var(--accent-purple); border-radius: 6px; padding: 12px; margin-bottom: 15px;">`;
      html += `<strong style="color: var(--accent-purple);"><i class="fa-solid fa-gem"></i> Clinical Pearl:</strong><br>`;
      html += `<span style="font-style: italic; color: #cdd6f4;">"${escapeHTML(postMortem.clinical_pearl)}"</span>`;
      html += `</div>`;
    }

    if (postMortem.key_missed_symptoms && postMortem.key_missed_symptoms.length > 0) {
      html += `<div><strong>Key Missed Symptoms:</strong><ul>`;
      postMortem.key_missed_symptoms.forEach(s => {
        html += `<li style="color: var(--accent-red);">${escapeHTML(s)}</li>`;
      });
      html += `</ul></div>`;
    }
    
    html += `</div>`;
  }

  // Pipeline performance bar
  html += `<div class="result-section perf-section">`;
  html += `<h2><i class="fa-solid fa-gauge-high"></i> Performance</h2>`;
  html += `<div class="perf-grid">`;
  ["R1", "R3", "IE"].forEach((s) => {
    const st = stages[s] || {};
    html += `<div class="perf-item"><span class="perf-label">${s}</span>`;
    html += `<span class="perf-val">${(st.time || 0).toFixed(1)}s</span>`;
    html += `<span class="perf-tok">${st.tokens || 0} tok — ${(st.tok_s || 0).toFixed(1)} t/s</span>`;
    html += `</div>`;
  });
  html += `<div class="perf-item total"><span class="perf-label">Total</span><span class="perf-val">${totalTime.toFixed(1)}s</span></div>`;
  html += `</div></div>`;

  // Append to chat
  if (currentMessageDiv) {
    const finalDiv = document.createElement("div");
    finalDiv.className = "ai-diagnosis-output";
    finalDiv.innerHTML = html;
    currentMessageDiv.appendChild(finalDiv);
  }

  // Finalize badges
  document
    .querySelectorAll(".mini-badge.running")
    .forEach((b) => b.classList.remove("running"));
  dom.sendBtn.disabled = false;
  currentStage = null;
}

function handleError(data) {
  stopLiveTimer();
  if (currentMessageDiv) {
    const err = document.createElement("div");
    err.className = "error-block";
    err.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${escapeHTML(data.content || data.message || "Unknown error")}`;
    currentMessageDiv.appendChild(err);
  }
  dom.sendBtn.disabled = false;
}

// ── Utility ──
function appendInfoBubble(content, icon, color, isHtml) {
  if (!currentMessageDiv) return;
  const div = document.createElement("div");
  div.className = "info-bubble";
  div.style.borderLeftColor = color || "var(--accent-blue)";
  div.innerHTML = `<i class="fa-solid ${icon || "fa-circle-info"}" style="color:${color}"></i> ${isHtml ? content : escapeHTML(content)}`;
  currentMessageDiv.appendChild(div);
}

function confidenceLevel(c) {
  if (!c) return "low";
  if (c >= 0.8) return "high";
  if (c >= 0.5) return "mid";
  return "low";
}

// Boot
window.onload = init;
