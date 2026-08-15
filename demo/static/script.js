// Riset Nama Gender - Demo frontend logic
const API_BASE = window.location.origin + "/api";

// DOM refs
const nameInput     = document.getElementById("name-input");
const predictBtn    = document.getElementById("predict-btn");
const deviceInfo    = document.getElementById("device-info");
const tabBtns       = document.querySelectorAll(".tab-btn");
const tabContents   = document.querySelectorAll(".tab-content");
const exampleBtns   = document.querySelectorAll(".example");
const attentionSel  = document.getElementById("attention-model");

const singleResult    = document.getElementById("single-result");
const compareResult   = document.getElementById("compare-result");
const attentionResult = document.getElementById("attention-result");

let activeTab = "single";

// Init: fetch device info
(async () => {
  try {
    const res = await fetch(`${API_BASE}/models`);
    const data = await res.json();
    deviceInfo.textContent = `${data.models.length} models loaded · device: ${data.device}`;
  } catch (e) {
    deviceInfo.textContent = "Backend offline";
    deviceInfo.style.color = "var(--color-female)";
  }
})();

// Tab switching
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const tabName = btn.dataset.tab;
    tabBtns.forEach(b => b.classList.toggle("active", b === btn));
    tabContents.forEach(c => c.classList.toggle("active", c.id === `tab-${tabName}`));
    activeTab = tabName;
    // If we already have a name, re-run for new tab
    if (nameInput.value.trim()) runPrediction();
  });
});

// Example buttons
exampleBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    nameInput.value = btn.dataset.name;
    runPrediction();
  });
});

// Main prediction dispatch
predictBtn.addEventListener("click", runPrediction);
nameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runPrediction();
});
attentionSel.addEventListener("change", () => {
  if (activeTab === "attention" && nameInput.value.trim()) runPrediction();
});

async function runPrediction() {
  const name = nameInput.value.trim();
  if (!name) return;

  predictBtn.disabled = true;

  try {
    if (activeTab === "single") {
      await renderSingle(name);
    } else if (activeTab === "compare") {
      await renderCompare(name);
    } else if (activeTab === "attention") {
      await renderAttention(name);
    }
  } catch (err) {
    console.error(err);
    showError(err.message);
  } finally {
    predictBtn.disabled = false;
  }
}

function showError(msg) {
  const target = {
    single: singleResult, compare: compareResult, attention: attentionResult,
  }[activeTab];
  target.innerHTML = `<p class="placeholder" style="color: var(--color-female);">Error: ${msg}</p>`;
}

function showLoading(target) {
  target.innerHTML = `<p class="loading">Predicting</p>`;
}

// Tab 1: Single Prediction
async function renderSingle(name) {
  showLoading(singleResult);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, model: "CharBiLSTM" }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  const probP = (data.prob_female * 100).toFixed(1);
  const probM = (data.prob_male * 100).toFixed(1);

  singleResult.innerHTML = `
    <div class="single-display">
      <div class="predicted-label ${data.label}">${data.label === "P" ? "P" : "L"}</div>
      <div class="predicted-name">${escapeHtml(data.name)}</div>
      <div class="predicted-desc">Predicted: <strong>${data.label_desc}</strong> · Model: ${data.model}</div>

      <div class="confidence-bars">
        <div class="confidence-row">
          <span class="label-mini">Laki-laki (L)</span>
          <div class="bar-bg"><div class="bar-fill L" style="width: ${probM}%;"></div></div>
          <span class="pct">${probM}%</span>
        </div>
        <div class="confidence-row">
          <span class="label-mini">Perempuan (P)</span>
          <div class="bar-bg"><div class="bar-fill P" style="width: ${probP}%;"></div></div>
          <span class="pct">${probP}%</span>
        </div>
      </div>
    </div>
  `;
}

// Tab 2: Compare 8 Models
async function renderCompare(name) {
  showLoading(compareResult);
  const res = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  // Check if all models agree
  const labels = data.predictions.map(p => p.label);
  const allAgree = labels.every(l => l === labels[0]);
  const consensus = mode(labels);

  const rows = data.predictions.map(p => {
    const level = p.model.startsWith("Char") ? "char" : "word";
    const conf = (p.confidence * 100).toFixed(1);
    const probP = (p.prob_female * 100).toFixed(1);
    const disagrees = p.label !== consensus;

    return `
      <tr class="${disagrees ? "disagree" : ""}">
        <td><strong>${p.model}</strong></td>
        <td><span class="badge ${level}">${level}-emb</span></td>
        <td><span class="badge ${p.label}">${p.label}</span></td>
        <td class="mini-bar-cell">
          <div class="mini-bar">
            <div class="mini-bar-fill ${p.label}" style="width: ${conf}%;"></div>
          </div>
        </td>
        <td><strong>${conf}%</strong></td>
      </tr>
    `;
  }).join("");

  const consensusBadge = `<span class="badge ${consensus}">${consensus}</span>`;
  const agreementText = allAgree
    ? `Semua 8 model setuju: ${consensusBadge}`
    : `Tidak semua model setuju. Konsensus: ${consensusBadge}`;

  compareResult.innerHTML = `
    <p style="margin-bottom: 16px;">
      <strong>${escapeHtml(name)}</strong> - ${agreementText}
    </p>
    <table class="compare-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Embedding</th>
          <th>Prediksi</th>
          <th class="mini-bar-cell">Confidence</th>
          <th>%</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p style="margin-top: 12px; font-size: 0.85rem; color: var(--color-muted);">
      Model yang <em>disagree</em> dari konsensus di-highlight oranye.
    </p>
  `;
}

// Tab 3: Attention Weights
async function renderAttention(name) {
  showLoading(attentionResult);
  const model = attentionSel.value;
  const res = await fetch(`${API_BASE}/attention`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, model }),
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();

  // Sort attention values to set color intensity scale
  const maxAttn = Math.max(...data.attention);

  // Render: tokens with background tinted by weight
  const tokensHtml = data.tokens.map((tok, i) => {
    const attn = data.attention[i];
    const intensity = maxAttn > 0 ? attn / maxAttn : 0;
    // yellow/orange gradient
    const bg = `rgba(245, 158, 11, ${intensity * 0.85})`;
    const display = tok === " " ? "␣" : escapeHtml(tok);
    return `
      <div class="attention-token" style="background: ${bg};" title="weight: ${attn.toFixed(4)}">
        <span class="char">${display}</span>
        <span class="weight">${(attn * 100).toFixed(1)}%</span>
      </div>
    `;
  }).join("");

  // Bar chart per token
  const barsHtml = data.tokens.map((tok, i) => {
    const attn = data.attention[i];
    const pct = maxAttn > 0 ? (attn / maxAttn * 100) : 0;
    const display = tok === " " ? "<space>" : escapeHtml(tok);
    return `
      <div class="tok-label">${display}</div>
      <div class="tok-bar"><div class="tok-bar-fill" style="width: ${pct}%;"></div></div>
      <div class="tok-pct">${(attn * 100).toFixed(2)}%</div>
    `;
  }).join("");

  attentionResult.innerHTML = `
    <div class="attention-summary">
      <strong>${escapeHtml(data.name)}</strong>:
      <span class="badge ${data.label}">${data.label}</span> ${data.label_desc}
      (${(data.confidence * 100).toFixed(1)}% confident) · Model: <strong>${data.model}</strong> · Level: ${data.level}
    </div>

    <h3 style="margin-bottom: 8px;">Token-level attention</h3>
    <p style="font-size: 0.85rem; color: var(--color-muted); margin-bottom: 12px;">
      Warna lebih gelap = model lebih memperhatikan token tersebut saat memprediksi.
    </p>
    <div class="attention-tokens">${tokensHtml}</div>

    <h3 style="margin-bottom: 8px; margin-top: 16px;">Attention bar chart</h3>
    <div class="attention-bars">${barsHtml}</div>
  `;
}

// Utils
function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function mode(arr) {
  const counts = {};
  arr.forEach(v => counts[v] = (counts[v] || 0) + 1);
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
}
