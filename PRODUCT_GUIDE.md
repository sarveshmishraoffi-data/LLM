# 🛡️ AI Model Risk & GenAI Governance Platform: Comprehensive Product Guide

## 1. Executive Summary
The **AI Model Risk & GenAI Governance Platform** is an institutional-grade Model Risk Management (MRM) and Responsible AI suite built for Tier-1 banks, credit bureaus, and fintech organizations. It implements independent model validation standards aligned with **US Federal Reserve SR 11-7**, **OCC 2011-12**, and the **Equal Credit Opportunity Act (ECOA)**.

Rather than just building predictive models, this platform acts as an **independent auditor and challenger** that evaluates whether artificial intelligence and large language models (LLMs) are safe, fair, robust, explainable, and commercially viable before deployment into high-stakes financial operations.

---

## 2. Who Uses This Product?

| Target Persona | Organization Role | How They Use the Platform |
| :--- | :--- | :--- |
| **Chief Risk Officer (CRO)** | Executive Risk Management | Reviews the **Executive Scorecard** to grant production sign-off based on regulatory traffic-light flags and the **B+ Overall Model Rating**. |
| **Model Risk Validator (Quant Auditor)** | Model Validation / MRM | Conducts independent sensitivity audits, **Stress-Testing simulations**, and cross-model stability benchmarking. |
| **Fair Lending & Compliance Officer** | Regulatory & Legal Compliance | Audits **Disparate Impact Ratios (EEOC 80% Rule)** and verifies adverse action reason codes for legal defensibility. |
| **Lead Data Scientist / ML Engineer** | Analytics & Model Development | Benchmarks 5 candidate algorithms in the **Algorithm Zoo** and inspects SHAP feature attributions for feature engineering insights. |
| **GenAI Governance / Safety Lead** | Generative AI Oversight | Audits LLM underwriting rationales against ground truth to ensure **0.0% hallucination rates** and prompt stability. |

---

## 3. End-to-End Product Workflow

```text
[1. DATA INGESTION]  -->  [2. 360° AUTOMATED AUDIT]  -->  [3. SCORECARD & ALERTS]  -->  [4. REGULATORY REPORT]
Customer Credit Data       - 5-Model Benchmark Zoo         Traffic-Light Ratings           One-Click SR 11-7
& LLM Underwriting Memos   - 10+ Financial Risk Metrics    - Green: Approved               Comprehensive Audit
                           - Fair Lending (ECOA) Audit     - Yellow: Monitor (B+)          Certificate in
                           - SHAP Explainability           - Red: Deploy Blocked           Markdown / JSON
                           - Adversarial Stress Tests
                           - GenAI Hallucination Engine
```

---

## 4. Tab-by-Tab Live Dashboard Walkthrough

### 🚦 Tab 1: Executive Scorecard
* **Purpose**: High-level executive overview for C-suite decision-makers.
* **Key Metrics**: Overall Model Rating (**B+**), Champion model name (**XGBoost**), Gini score (**0.72**), and high-level risk flags.

### 🏆 Tab 2: Algorithm Zoo (5 Models)
* **Purpose**: Multi-model benchmarking to ensure algorithm selection is mathematically justified.
* **Models**: XGBoost (Champion), LightGBM, Random Forest, Multi-Layer Perceptron (MLP Neural Net), Logistic Regression (Baseline).
* **Metrics**: ROC-AUC, PR-AUC, Gini, Kolmogorov-Smirnov (KS), Matthews Correlation Coefficient (MCC), Brier Score, and Calibration Error.

### 📋 Tab 3: Data Quality & Evidence Assessment
* **Purpose**: Verifies input data integrity before modeling.
* **Checks**: Schema conformance, missingness patterns, boundary violations, and anomaly detection.

### 🎯 Tab 4: ML Discrimination & Error Slicing
* **Purpose**: Detailed error diagnostics and threshold optimization.
* **Features**: Interactive confusion matrices, threshold trade-off curve, and sub-population error slice profiling.

### ⚖️ Tab 5: Fair Lending & Bias Audit
* **Purpose**: Independent compliance check under the **Equal Credit Opportunity Act (ECOA)**.
* **Metrics**: Disparate Impact Ratio, Demographic Parity, and Equalized Odds across protected demographic groups (Age cohorts and Gender). Flags any cohort breaching the **80% Four-Fifths rule**.

### 🔍 Tab 6: Explainability (SHAP Attributions)
* **Purpose**: Transforming "black-box" models into transparent, legally compliant decision engines.
* **Features**: Global mean $|SHAP|$ feature ranking and applicant-level waterfall reason codes required for **FCRA Adverse Action notices**.

### 🛡️ Tab 7: Robustness & Adversarial Stress Testing
* **Purpose**: Simulating macroeconomic shocks and stressed economic environments.
* **Tests**: Missing feature tolerance, Gaussian noise injection, and severe recession perturbation analysis.

### 🤖 Tab 8: GenAI / LLM Evaluation & Safety
* **Purpose**: Auditing Generative AI underwriting assistants for enterprise reliability.
* **Evaluations**:
  * **Hallucination Detection**: Extracts numerical assertions from AI-generated memos and cross-references them against ground-truth profiles (**0.0% Hallucination target**).
  * **Prompt Sensitivity**: Quantifies semantic divergence across underwriter persona rewordings.
  * **Self-Consistency**: Verifies response determinism across repeated sampling.

### 📈 Tab 9: Population Drift & Production Monitoring
* **Purpose**: Continuous post-deployment surveillance.
* **Metrics**: Population Stability Index (PSI) and Kolmogorov-Smirnov drift tracking across 6 monthly monitoring batches, with automated alert thresholds ($PSI > 0.25$).

### 📄 Tab 10: Regulatory Audit Report
* **Purpose**: Automated compliance documentation.
* **Output**: Instant download of the full, formal SR 11-7 Model Validation Report.

---

## 5. How to Run the Project

### Option A: Live Web Platform (Instant Access)
* **Streamlit Cloud Dashboard**: [https://sarveshmishraoffi-data-llm-app-ghgtnp.streamlit.app/](https://sarveshmishraoffi-data-llm-app-ghgtnp.streamlit.app/)

### Option B: Local Streamlit Dashboard
```bash
cd d:\LLM
streamlit run app.py
```
*Access via your local browser at `http://localhost:8501`.*

### Option C: Local Full-Stack Web App (FastAPI + Tailwind SPA)
```bash
uvicorn web_app.main:app --reload
```
*Access via your local browser at `http://localhost:8000`.*

### Option D: Standalone Headless Audit CLI
```bash
python run_full_audit.py
```
*Runs the full 360° audit in under 15 seconds and compiles audit reports directly into the `reports/` directory.*

### Option E: Automated Unit Test Suite
```bash
pytest tests/test_framework.py -v
```
*Runs all 11 unit and integration tests with a 100% pass rate.*

---

## 6. Real-World Business Value ($$$ Impact)
1. **Regulatory Fine Avoidance**: Non-compliance with fair lending laws can result in multimillion-dollar fines from enforcement agencies (e.g., CFPB, FTC, RBI). This tool catches disparate impact before deployment.
2. **Credit Loss Mitigation**: Stressed boundary testing prevents catastrophic portfolio defaults during unexpected macroeconomic shocks.
3. **Audit Speedup**: Replaces 3–4 weeks of manual validation audits with a 2-minute automated pipeline.
