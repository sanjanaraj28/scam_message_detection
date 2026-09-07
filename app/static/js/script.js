// ScamShield AI - dashboard client logic
// Talks to /api/predict, /api/analytics, /api/examples. Prediction history is
// kept client-side only (per the "do not expose full message contents
// unnecessarily" requirement) using in-memory state (no localStorage, per
// artifact/browser-storage constraints of this environment - history clears
// on page reload, which is stated clearly in the UI copy).

const state = {
  history: [], // { timestamp, prediction, confidence, riskLevel, preview }
};

const el = (id) => document.getElementById(id);

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function riskClass(level) {
  if (level === "SAFE") return "safe";
  if (level === "SUSPICIOUS") return "suspicious";
  return "scam";
}

function riskEmoji(level) {
  if (level === "SAFE") return "🟢";
  if (level === "SUSPICIOUS") return "🟡";
  return "🔴";
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }
  return data;
}

function renderEmpty() {
  el("resultArea").innerHTML = `
    <div class="empty-state" id="emptyState">
      <svg width="34" height="34" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L4 5V11C4 16.5 7.4 21.2 12 22.5C16.6 21.2 20 16.5 20 11V5L12 2Z" stroke="#5f6f95" stroke-width="1.5"/></svg>
      <div>No message analyzed yet — paste a message above and click Analyze.</div>
    </div>`;
}

function renderLoading() {
  el("resultArea").innerHTML = `
    <div class="loading-row"><div class="spinner"></div> Analyzing message…</div>`;
}

function renderError(message) {
  el("resultArea").innerHTML = `<div class="error-banner">⚠ ${escapeHtml(message)}</div>`;
}

function renderResult(result) {
  const cls = riskClass(result.risk_level);
  const pct = Math.round(result.scam_probability * 100);
  const indicatorsHtml = result.indicators.length
    ? result.indicators.map((i) => `<li>${escapeHtml(i)}</li>`).join("")
    : `<li>No specific scam indicators detected in the wording.</li>`;

  el("resultArea").innerHTML = `
    <div class="result-card ${cls}">
      <div class="result-head">
        <div class="result-verdict"><span class="verdict-dot"></span> ${riskEmoji(result.risk_level)} ${result.risk_level}</div>
        <div class="risk-score">Scam probability: ${pct}%</div>
      </div>
      <div class="meter-track"><div class="meter-fill" style="width:${pct}%"></div></div>

      <div class="result-section">
        <h4>Detected indicators</h4>
        <ul class="indicator-list">${indicatorsHtml}</ul>
      </div>

      <div class="result-section">
        <h4>Recommended action</h4>
        <div class="recommendation-box">${escapeHtml(result.recommendation)}</div>
      </div>

      <div class="disclaimer-note">Prediction: statistical model (${escapeHtml(result.model_used)}) · Indicators: rule-based explanation layer, shown separately from the model's prediction.</div>
    </div>`;
}

function pushHistory(message, result) {
  state.history.unshift({
    timestamp: result.timestamp,
    riskLevel: result.risk_level,
    confidence: result.confidence,
    preview: message.length > 60 ? message.slice(0, 60) + "…" : message,
  });
  state.history = state.history.slice(0, 12);
  renderHistory();
}

function renderHistory() {
  const container = el("historyList");
  if (!state.history.length) {
    container.innerHTML = `<div class="history-empty">No predictions yet.</div>`;
    return;
  }
  container.innerHTML = state.history
    .map(
      (h) => `
      <div class="history-item">
        <span class="history-tag ${riskClass(h.riskLevel)}">${h.riskLevel}</span>
        <span class="history-msg">${escapeHtml(h.preview)}</span>
        <span class="history-time">${h.timestamp.split("T")[1] || ""}</span>
      </div>`
    )
    .join("");
}

async function analyzeMessage() {
  const message = el("messageInput").value.trim();
  if (!message) {
    renderError("Please paste a message before analyzing.");
    return;
  }
  el("analyzeBtn").disabled = true;
  renderLoading();
  try {
    const result = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    renderResult(result);
    pushHistory(message, result);
  } catch (err) {
    renderError(err.message || "Something went wrong while analyzing this message.");
  } finally {
    el("analyzeBtn").disabled = false;
  }
}

function updateCharCount() {
  el("charCount").textContent = el("messageInput").value.length;
}

async function loadAnalytics() {
  try {
    const data = await fetchJSON("/api/analytics");
    el("statGrid").innerHTML = `
      <div class="stat-box"><div class="value">${(data.accuracy * 100).toFixed(1)}%</div><div class="label">Test accuracy</div></div>
      <div class="stat-box"><div class="value">${(data.f1_macro * 100).toFixed(1)}%</div><div class="label">Macro F1 score</div></div>
      <div class="stat-box"><div class="value">${(data.recall_scam * 100).toFixed(1)}%</div><div class="label">Scam recall</div></div>
      <div class="stat-box"><div class="value">${data.dataset_size_cleaned.toLocaleString()}</div><div class="label">Unique messages used</div></div>
    `;
    el("modelBadgeWrap").innerHTML = `<span class="model-badge">Best model: ${escapeHtml(data.best_model)}</span>`;
    el("modelBadgeText").textContent = `Model ready · ${data.best_model}`;
  } catch (err) {
    el("statGrid").innerHTML = `<div class="stat-box"><div class="value">—</div><div class="label">Analytics unavailable</div></div>`;
  }
}

async function loadExamples() {
  try {
    const data = await fetchJSON("/api/examples");
    el("exampleChips").innerHTML = data.examples
      .map((ex, i) => `<span class="chip" data-idx="${i}">${escapeHtml(ex.label)}</span>`)
      .join("");
    el("exampleChips").querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const idx = parseInt(chip.dataset.idx, 10);
        el("messageInput").value = data.examples[idx].text;
        updateCharCount();
      });
    });
  } catch (err) {
    // Non-critical - examples are a convenience feature only.
  }
}

el("analyzeBtn").addEventListener("click", analyzeMessage);
el("clearBtn").addEventListener("click", () => {
  el("messageInput").value = "";
  updateCharCount();
  renderEmpty();
});
el("clearHistoryBtn").addEventListener("click", () => {
  state.history = [];
  renderHistory();
});
el("messageInput").addEventListener("input", updateCharCount);
el("messageInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyzeMessage();
});

loadAnalytics();
loadExamples();
