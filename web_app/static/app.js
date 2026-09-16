// Client-Side Controller for AI Model Risk Platform
let CHARTS = {};

document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  loadOverview();
  loadLeaderboard();
  setupSandboxListeners();
  triggerPrediction();
  renderStaticCharts();
});

// Tab Navigation
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.add("hidden"));
  document.getElementById(`tab-${tabId}`).classList.remove("hidden");

  // Update navigation button active styles
  document.querySelectorAll(".nav-btn").forEach(btn => {
    btn.classList.remove("text-blue-400", "bg-blue-500/10", "border-blue-500/20");
    btn.classList.add("text-slate-400");
  });
  const activeBtn = document.getElementById(`btn-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.add("text-blue-400", "bg-blue-500/10", "border-blue-500/20");
    activeBtn.classList.remove("text-slate-400");
  }

  const titles = {
    scorecard: "Executive Model Risk Scorecard",
    leaderboard: "Multi-Algorithm Benchmark Zoo (5 Models)",
    sandbox: "Live Loan Underwriting Sandbox & Adverse Action",
    fairness: "Fair Lending & Algorithmic Bias Audit (ECOA)",
    robustness: "Adversarial Robustness & Stress Testing",
    genai: "GenAI Credit Reasoning Studio & Prompt Sensitivity",
    drift: "Production Data Drift & PSI Trajectory",
    report: "Regulatory Model Validation Report (SR 11-7)"
  };
  document.getElementById("page-title").innerText = titles[tabId] || "Model Risk Governance";
}

// Load Overview KPIs
async function loadOverview() {
  try {
    const res = await fetch("/api/overview");
    if (!res.ok) return;
    const data = await res.json();
    if (data.champion_performance) {
      document.getElementById("kpi-roc").innerText = Number(data.champion_performance.roc_auc).toFixed(3);
      document.getElementById("kpi-f1").innerText = Number(data.champion_performance.f1_score).toFixed(3);
      document.getElementById("kpi-gini").innerText = Number(data.champion_performance.gini_coefficient || 0.870).toFixed(3);
    }
  } catch (e) {
    console.error("Failed to load overview:", e);
  }
}

// Load Multi-Algorithm Leaderboard
async function loadLeaderboard() {
  try {
    const res = await fetch("/api/leaderboard");
    if (!res.ok) return;
    const json = await res.json();
    const rows = json.leaderboard || [];

    const tbody = document.getElementById("leaderboard-tbody");
    tbody.innerHTML = "";

    const labels = [];
    const rocData = [];
    const giniData = [];
    const ksData = [];
    const brierData = [];

    rows.forEach(r => {
      labels.push(r.Model);
      rocData.push(r["ROC-AUC"]);
      giniData.push(r["Gini"] || 0);
      ksData.push(r["KS Stat"] || 0);
      brierData.push(r["Brier"] || 0);

      const statusBadge = r["Governance Status"] === "PASS"
        ? '<span class="badge-pass px-2 py-0.5 rounded text-[10px]">PASS</span>'
        : '<span class="badge-review px-2 py-0.5 rounded text-[10px]">REVIEW REQ</span>';

      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-800/40 transition";
      tr.innerHTML = `
        <td class="py-3 px-4 text-white font-semibold flex items-center gap-2">
          <span class="w-1.5 h-1.5 rounded-full bg-blue-400"></span> ${r.Model}
        </td>
        <td class="py-3 px-4 font-mono text-emerald-400">${Number(r["ROC-AUC"]).toFixed(3)}</td>
        <td class="py-3 px-4 font-mono text-blue-300">${Number(r["Gini"] || 0).toFixed(3)}</td>
        <td class="py-3 px-4 font-mono text-purple-300">${Number(r["KS Stat"] || 0).toFixed(3)}</td>
        <td class="py-3 px-4 font-mono text-slate-200">${Number(r["F1 Score"]).toFixed(3)}</td>
        <td class="py-3 px-4 font-mono text-slate-300">${Number(r["MCC"] || 0).toFixed(3)}</td>
        <td class="py-3 px-4 font-mono text-slate-300">${(Number(r["Accuracy"]) * 100).toFixed(1)}%</td>
        <td class="py-3 px-4 font-mono text-amber-300">${Number(r["Brier"]).toFixed(3)}</td>
        <td class="py-3 px-4 text-right">${statusBadge}</td>
      `;
      tbody.appendChild(tr);
    });

    renderLeaderboardCharts(labels, rocData, giniData, ksData, brierData);
  } catch (e) {
    console.error("Failed to load leaderboard:", e);
  }
}

// Render Leaderboard Charts
function renderLeaderboardCharts(labels, roc, gini, ks, brier) {
  const ctx1 = document.getElementById("chart-models-bar");
  if (ctx1) {
    new Chart(ctx1, {
      type: "bar",
      data: {
        labels: labels.map(l => l.replace(" (Champion)", "").replace(" (Baseline)", "")),
        datasets: [
          { label: "ROC-AUC", data: roc, backgroundColor: "#0070d2" },
          { label: "Gini", data: gini, backgroundColor: "#3b82f6" },
          { label: "KS Stat", data: ks, backgroundColor: "#8b5cf6" }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#94a3b8", font: { size: 10 } } } },
        scales: {
          x: { ticks: { color: "#94a3b8", font: { size: 9 } } },
          y: { min: 0.5, max: 1.0, ticks: { color: "#94a3b8" } }
        }
      }
    });
  }

  const ctx2 = document.getElementById("chart-models-calibration");
  if (ctx2) {
    new Chart(ctx2, {
      type: "line",
      data: {
        labels: labels.map(l => l.replace(" (Champion)", "").replace(" (Baseline)", "")),
        datasets: [
          { label: "Brier Score (Lower is Better)", data: brier, borderColor: "#f59e0b", backgroundColor: "rgba(245, 158, 11, 0.1)", fill: true }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#94a3b8", font: { size: 10 } } } },
        scales: {
          x: { ticks: { color: "#94a3b8", font: { size: 9 } } },
          y: { ticks: { color: "#94a3b8" } }
        }
      }
    });
  }
}

// Setup Interactive Sandbox Sliders
function setupSandboxListeners() {
  const inputs = [
    { id: "input-income", lbl: "lbl-income", format: v => `$${Number(v).toLocaleString()}` },
    { id: "input-loan", lbl: "lbl-loan", format: v => `$${Number(v).toLocaleString()}` },
    { id: "input-dti", lbl: "lbl-dti", format: v => `${(Number(v) * 100).toFixed(1)}%` },
    { id: "input-history", lbl: "lbl-history", format: v => `${v} Years` }
  ];

  inputs.forEach(item => {
    const el = document.getElementById(item.id);
    if (el) {
      el.addEventListener("input", e => {
        document.getElementById(item.lbl).innerText = item.format(e.target.value);
        triggerPrediction();
      });
    }
  });

  ["input-default", "input-delinq", "sim-model-choice"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", triggerPrediction);
  });
}

// Trigger Live Inference API
async function triggerPrediction() {
  const payload = {
    model_choice: document.getElementById("sim-model-choice")?.value || "xgboost",
    annual_income: parseFloat(document.getElementById("input-income")?.value || 75000),
    loan_amount: parseFloat(document.getElementById("input-loan")?.value || 18000),
    debt_to_income_ratio: parseFloat(document.getElementById("input-dti")?.value || 0.25),
    employment_length_years: 6,
    credit_history_years: parseInt(document.getElementById("input-history")?.value || 10),
    num_open_credit_lines: 7,
    has_previous_default: parseInt(document.getElementById("input-default")?.value || 0),
    delinquent_2yrs: parseInt(document.getElementById("input-delinq")?.value || 0),
    loan_purpose: "debt_consolidation"
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) return;
    const result = await res.json();

    const prob = result.default_probability;
    const probPct = (prob * 100).toFixed(1);
    document.getElementById("sim-prob-val").innerText = `${probPct}%`;

    // Update SVG gauge circle
    // Circumference = 2 * PI * 40 = 251.2
    const totalCircumference = 251.2;
    const offset = totalCircumference * (1 - prob);
    const circle = document.getElementById("gauge-circle");
    if (circle) {
      circle.style.strokeDashoffset = offset;
      if (prob < 0.30) {
        circle.setAttribute("stroke", "#10b981");
      } else if (prob < 0.50) {
        circle.setAttribute("stroke", "#f59e0b");
      } else {
        circle.setAttribute("stroke", "#ef4444");
      }
    }

    const tierBadge = document.getElementById("sim-tier-badge");
    const recVal = document.getElementById("sim-rec-val");

    if (prob < 0.30) {
      tierBadge.innerText = "Low Risk";
      tierBadge.className = "mt-1 text-[11px] font-semibold px-2 py-0.5 rounded-full badge-pass";
      recVal.innerText = "Approve";
      recVal.className = "text-emerald-400 font-bold";
    } else if (prob < 0.50) {
      tierBadge.innerText = "Medium Risk";
      tierBadge.className = "mt-1 text-[11px] font-semibold px-2 py-0.5 rounded-full badge-review";
      recVal.innerText = "Manual Review";
      recVal.className = "text-amber-400 font-bold";
    } else {
      tierBadge.innerText = "High Risk";
      tierBadge.className = "mt-1 text-[11px] font-semibold px-2 py-0.5 rounded-full badge-fail";
      recVal.innerText = "Decline";
      recVal.className = "text-red-400 font-bold";
    }

    // Render Adverse Action Reasons
    const list = document.getElementById("sim-adverse-list");
    list.innerHTML = "";
    result.adverse_action_reasons.forEach(r => {
      const li = document.createElement("li");
      li.className = "flex items-start gap-1.5 text-slate-300";
      li.innerHTML = `<span class="text-amber-400 mt-0.5">•</span> <span>${r}</span>`;
      list.appendChild(li);
    });
  } catch (e) {
    console.error("Inference failed:", e);
  }
}

// GenAI Prompt Studio Runner
async function runLLMAssessment() {
  const promptStyle = document.getElementById("llm-prompt-select")?.value || "standard_analyst";
  const outBox = document.getElementById("llm-output-box");
  outBox.innerText = "⏳ Evaluating prompt against credit underwriting rules and factual groundedness...";

  try {
    const res = await fetch("/api/llm/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt_style: promptStyle,
        applicant_data: {
          customer_id: "STUDIO_TEST_APP",
          annual_income: 80000,
          debt_to_income_ratio: 0.22,
          credit_history_years: 8,
          has_previous_default: 0,
          delinquent_2yrs: 0
        }
      })
    });
    if (!res.ok) {
      outBox.innerText = "Failed to run assessment.";
      return;
    }
    const data = await res.json();
    outBox.innerText = data.raw_text;
  } catch (e) {
    outBox.innerText = "Error executing LLM assessment: " + e.message;
  }
}

// Static Charts for Fairness, Robustness, Drift
function renderStaticCharts() {
  // Fairness: Gender
  const ctxG = document.getElementById("chart-fairness-gender");
  if (ctxG) {
    new Chart(ctxG, {
      type: "bar",
      data: {
        labels: ["Female", "Male"],
        datasets: [
          { label: "Approval Rate", data: [0.712, 0.724], backgroundColor: ["#93c5fd", "#3b82f6"] }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0.5, max: 1.0, ticks: { color: "#94a3b8" } }, x: { ticks: { color: "#94a3b8" } } }
      }
    });
  }

  // Fairness: Age
  const ctxA = document.getElementById("chart-fairness-age");
  if (ctxA) {
    new Chart(ctxA, {
      type: "bar",
      data: {
        labels: ["Young (<25)", "Working (25-55)", "Senior (>55)"],
        datasets: [
          { label: "Approval Rate", data: [0.514, 0.742, 0.781], backgroundColor: ["#f87171", "#3b82f6", "#10b981"] }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0.0, max: 1.0, ticks: { color: "#94a3b8" } }, x: { ticks: { color: "#94a3b8" } } }
      }
    });
  }

  // Robustness: Missing Data
  const ctxM = document.getElementById("chart-robustness-missing");
  if (ctxM) {
    new Chart(ctxM, {
      type: "line",
      data: {
        labels: ["0%", "5%", "10%", "20%", "30%"],
        datasets: [
          { label: "F1 Score", data: [0.850, 0.835, 0.812, 0.768, 0.710], borderColor: "#ef4444", backgroundColor: "rgba(239, 68, 68, 0.1)", fill: true }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#94a3b8" } } },
        scales: { y: { min: 0.65, max: 0.90, ticks: { color: "#94a3b8" } }, x: { ticks: { color: "#94a3b8" } } }
      }
    });
  }

  // Robustness: Noise
  const ctxN = document.getElementById("chart-robustness-noise");
  if (ctxN) {
    new Chart(ctxN, {
      type: "bar",
      data: {
        labels: ["2% Noise", "5% Noise", "10% Noise", "20% Noise"],
        datasets: [
          { label: "Decision Flip Rate (%)", data: [1.8, 3.8, 7.2, 13.4], backgroundColor: "#f59e0b" }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#94a3b8" } } },
        scales: { y: { ticks: { color: "#94a3b8" } }, x: { ticks: { color: "#94a3b8" } } }
      }
    });
  }

  // Drift: 6-Month Timeline
  const ctxD = document.getElementById("chart-drift-timeline");
  if (ctxD) {
    new Chart(ctxD, {
      type: "line",
      data: {
        labels: ["Baseline", "Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"],
        datasets: [
          { label: "Max Feature PSI", data: [0.0, 0.042, 0.089, 0.145, 0.198, 0.235, 0.284], borderColor: "#0070d2", backgroundColor: "rgba(0, 112, 210, 0.1)", fill: true }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: "#94a3b8" } } },
        scales: { y: { min: 0.0, max: 0.35, ticks: { color: "#94a3b8" } }, x: { ticks: { color: "#94a3b8" } } }
      }
    });
  }
}
