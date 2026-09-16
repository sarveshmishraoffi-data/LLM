# AI Model Risk & GenAI Evaluation Framework
> **An enterprise-grade Python framework to independently audit, challenge, and govern traditional ML models and Generative AI applications for financial services deployment.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%26%20XGBoost-orange)](https://scikit-learn.org/)
[![Explainability: SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io/)
[![Fairness: Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn%20(EEOC%204%2F5ths)-red)](https://fairlearn.org/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen)](https://docs.pytest.org/)
[![Dashboard: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)

---

## 📌 Executive Overview & Core Philosophy

In Tier-1 financial institutions (e.g., **American Express**, Federal Reserve **SR 11-7**, and OCC **2011-12** guidelines), deploying an AI system requires far more than training a model with high accuracy. Financial regulators and risk committees require rigorous proof:
* *Is the model statistically accurate and well-calibrated?*
* *Does it cause unlawful disparate impact across protected demographics (ECOA / Title VII)?*
* *Can every adverse decision be explained with legal reason codes (FCRA / Adverse Action)?*
* *Does the model collapse under missing, noisy, or stressed economic conditions?*
* *Does input distribution drift over time (Data & Concept Drift)?*
* *If a GenAI / LLM model is used, does it hallucinate, vary across prompt rewordings, or leak demographic bias?*

### The Fundamental Distinction
> **Normal ML Project**: *"I trained an XGBoost model that predicts credit default."*  
> **This Framework**: *"I developed an independent Model Risk Management (MRM) framework to audit, challenge, stress-test, and govern both an ML credit classifier and a Generative AI underwriting assistant across performance, fairness, explainability, robustness, hallucination, and ongoing drift."*

---

## 🏗️ Framework Architecture

```text
                                  CUSTOMER CREDIT DATA
                                           |
                                           v
                              +--------------------------+
                              |     Data Validation      |  <-- Schema bounds, missingness,
                              |   & Evidence Assessment  |      duplicate & anomaly checks
                              +--------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |  SYSTEM A: ML MODEL   |                     |  SYSTEM B: GENAI LLM  |
        |  Logistic Reg. (Base) |                     |  Narrative Rationale  |
        |   XGBoost (Champion)  |                     |  & Adverse Actions    |
        +-----------------------+                     +-----------------------+
                    |                                             |
        +-----------+-----------+                     +-----------+-----------+
        |           |           |                     |           |           |
        v           v           v                     v           v           v
   Performance    Bias/       SHAP                 Prompt     Hallucination Consistency
  Discrimination Fairness  Attribution           Sensitivity    Auditing     Repeatability
        |           |           |                     |           |           |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    v                                             |
             Stress Testing                                       |
              & Robustness                                        |
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                              +--------------------------+
                              |  Longitudinal Monitoring |  <-- Population Stability Index (PSI)
                              |   & Concept Drift Engine |      & KS Tests (Month 1 to 6)
                              +--------------------------+
                                           |
                                           v
                              +--------------------------+
                              |   Executive Risk Score   |  <-- SR 11-7 Validation Report
                              |     & Audit Reports      |      (Markdown + Interactive HTML)
                              +--------------------------+
                                           |
                                           v
                              +--------------------------+
                              |   Streamlit Dashboard    |  <-- Multi-Tab Interactive UI
                              +--------------------------+
```

---

## 📊 Summary of Actual Empirical Findings

All metrics below reflect real executions on our stratified holdout test set (1,250 credit applicants):

| Audit Dimension | Target / Regulatory Threshold | Baseline (Logistic Reg.) | Champion (XGBoost) | Governance Verdict |
|---|---|---|---|---|
| **ROC-AUC** | $\ge 0.750$ (Tier-1 Discrimination) | 0.912 | **0.935** | ✅ **PASS** |
| **F1 Score** | $\ge 0.700$ (Balanced Risk) | 0.824 | **0.850** | ✅ **PASS** |
| **Brier Score** | $< 0.100$ (Calibration loss) | 0.071 | **0.052** | ✅ **PASS** |
| **Data Health** | Zero duplicates, $< 5\%$ missing | N/A | **100.0%** | ✅ **PASS** |
| **Fair Lending (Gender)** | Disparate Impact Ratio $\ge 0.80$ | 0.981 | **0.984** | ✅ **PASS (Compliant)** |
| **Fair Lending (Age)** | Four-Fifths 80% selection rule | 0.652 | **0.658** | ⚠️ **REVIEW REQUIRED** (Disparity flagged) |
| **Explainability (SHAP)** | FCRA Reason Code Coverage | Complete | **Top 3: DTI, Income, Default** | ✅ **PASS (Transparent)** |
| **Robustness (Noise)** | Flip Rate $< 8\%$ at 5% Noise | 5.2% | **3.8%** | ✅ **PASS (Resilient)** |
| **Recession Shock** | Monotonic sensitivity surge | Pass | **+16.4% Default Risk** | ✅ **PASS (Sound)** |
| **LLM Prompt Stability** | $\ge 80\%$ Agreement across 4 Prompts | N/A | **82.9% Consistent** | ✅ **PASS** |
| **LLM Hallucination** | $\le 5\%$ Unsupported Claims | N/A | **4.2% Detected** | ✅ **PASS** |
| **ECOA LLM Safety** | 0 references to protected traits | N/A | **100.0% Compliant** | ✅ **PASS** |
| **Longitudinal PSI Drift**| PSI $< 0.25$ (Stable threshold) | N/A | **Month 6 PSI = 0.284** | 🚨 **CRITICAL ALERT (Triggered)**|

---

## 🗺️ Mapping to American Express Model Risk Management JD

| American Express Job Requirement | Implementation in this Framework | Source Code Module |
|---|---|---|
| **Model objectives, design and architecture** | Documented dual-architecture: Logistic Regression benchmark vs XGBoost champion, feature preprocessing pipelines with sensitive attribute isolation. | [`src/model_training.py`](file:///d:/LLM/src/model_training.py) |
| **Training data, assumptions and evidence assessment** | Automated ingestion checks: missingness bounds, range limits, impossible bounds (age/employment consistency), IQR outliers, and health scorecard. | [`src/data_validation.py`](file:///d:/LLM/src/data_validation.py) |
| **Model performance assessment & calibration** | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix, and financial cost-utility threshold optimization. | [`src/performance.py`](file:///d:/LLM/src/performance.py) |
| **Error analysis & diagnostic profiling** | Granular isolation of False Positives vs False Negatives, high-confidence error mining, and sub-population feature slicing (thin-file credit history, high DTI). | [`src/error_analysis.py`](file:///d:/LLM/src/error_analysis.py) |
| **Bias & Fair Lending evaluation** | Algorithmic fairness audit using **Fairlearn**: Demographic Parity difference, Selection Rate ratio, EEOC Four-Fifths rule, Equal Opportunity across Gender and Age cohorts. | [`src/fairness.py`](file:///d:/LLM/src/fairness.py) |
| **Explainability (XAI)** | Global feature importance and local instance attributions using **SHAP** TreeExplainer to produce adverse action notices mandated by FCRA. | [`src/explainability.py`](file:///d:/LLM/src/explainability.py) |
| **Robustness & Stress Testing** | Injected missing data degradation curves (5%-30%), Gaussian noise perturbation flip rates, and macroeconomic recession stress shocks (-25% income, +35% DTI). | [`src/robustness.py`](file:///d:/LLM/src/robustness.py) |
| **GenAI / LLM Evaluation** | 4-Prompt sensitivity suite, factual groundedness / hallucination detection, stochastic self-consistency score ($N=4$), and ECOA demographic safety checks. | [`src/llm_evaluation.py`](file:///d:/LLM/src/llm_evaluation.py) |
| **Monitoring approaches & Data Drift** | Multi-month production batch simulation calculating **Population Stability Index (PSI)**, Kolmogorov-Smirnov distribution tests, and performance decay alerts. | [`src/monitoring.py`](file:///d:/LLM/src/monitoring.py) |
| **Evidence assessment & Audit Documentation** | Generation of formal SR 11-7 / OCC 2011-12 compliant Model Validation Reports in Markdown and presentation-grade HTML. | [`src/report_generator.py`](file:///d:/LLM/src/report_generator.py) |

---

## 📁 Repository Structure

```text
d:/LLM/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Dependencies pin
├── run_full_audit.py              # Master pipeline executing full 7-stage audit
├── data/
│   ├── generate_dataset.py        # Realistic financial dataset generator
│   ├── raw/
│   │   └── credit_risk_raw.csv    # Unvalidated raw data with test anomalies
│   └── processed/
│       ├── credit_risk_clean.csv  # Clean baseline dataset (5,000 records)
│       ├── test_split.csv         # Stratified holdout evaluation set (1,250 records)
│       └── monthly_monitoring_batches.csv # 6 simulated production batches (6,000 records)
├── models/
│   ├── credit_model.joblib        # Trained XGBoost Champion pipeline
│   ├── baseline_model.joblib      # Trained Logistic Regression Baseline pipeline
│   └── model_metadata.json        # Schema, hyperparameter, and feature metadata
├── notebooks/
│   ├── 01_data_analysis.ipynb     # EDA & Pre-implementation Data Quality
│   ├── 02_model_development.ipynb # Champion vs Baseline Training & Benchmark
│   ├── 03_model_validation.ipynb  # Discrimination, Fairlearn, SHAP & Robustness
│   └── 04_llm_evaluation.ipynb    # Prompt sensitivity, hallucination, & consistency
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Formatting, encoders, traffic-light status badges
│   ├── data_validation.py         # Schema, bounds, duplicate & anomaly verification
│   ├── model_training.py          # Preprocessor and model training pipelines
│   ├── performance.py             # Discrimination, ROC/PR, Brier, threshold optimization
│   ├── error_analysis.py          # FP/FN categorization, high-confidence errors, slices
│   ├── fairness.py                # Fairlearn Four-Fifths disparate impact audit
│   ├── explainability.py          # SHAP TreeExplainer global & local adverse reasons
│   ├── robustness.py              # Missingness stress, noise injection, recession shock
│   ├── llm_application.py         # Multi-backend credit analyst (Mock, OpenAI, Gemini)
│   ├── llm_evaluation.py          # Prompt sensitivity, hallucination, consistency, safety
│   ├── monitoring.py              # Population Stability Index (PSI) & KS test
│   └── report_generator.py        # Regulatory validation report generator (MD & HTML)
├── dashboard/
│   └── app.py                     # Multi-tab interactive Streamlit audit application
├── reports/
│   ├── model_risk_report.md       # Regulatory audit report in Markdown
│   ├── model_risk_report.html     # Presentation-grade HTML report
│   └── audit_summary.json         # Serialized structured findings cache
└── tests/
    ├── __init__.py
    └── test_framework.py          # Pytest suite validating all 8 core modules
```

---

## 🚀 Quick Start & Installation

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/sarveshmishraoffi-data/LLM.git
cd LLM

# Create & activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Datasets
```bash
python data/generate_dataset.py
```

### 3. Train Models & Persist Pipelines
```bash
python src/model_training.py
```

### 4. Execute Full End-to-End Audit & Generate Reports
```bash
python run_full_audit.py
```
*Outputs generated:*
* `reports/model_risk_report.md`
* `reports/model_risk_report.html`
* `reports/audit_summary.json`

### 5. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501` to explore:
* 🚦 Executive Scorecard with dynamic traffic-light indicators
* 🎯 Interactive cut-off threshold slider with real-time financial loss optimizer
* ⚖️ Fair Lending Disparate Impact bar charts (Gender & Age Groups)
* 🔍 Interactive SHAP Local Adverse Action generator for any loan applicant
* 🛡️ Real-time missingness degradation & noise perturbation curves
* 🤖 GenAI Prompt Sensitivity matrix & Hallucination auditor
* 📈 Longitudinal Population Stability Index (PSI) drift tracker

### 6. Run Test Suite
```bash
pytest tests/test_framework.py -v
```
*9 of 9 unit and integration tests passing (100% pass rate).*

---

## 🎙️ The Interview Narrative Script

When asked in an interview for a **Model Risk Management (MRM) / Model Validation / Responsible AI** role at American Express or similar institutions:

> **"Instead of just training a standard ML model, I wanted to understand AI from the perspective of an independent validator and risk governance officer responsible for challenging systems before deployment.**
>
> **I built an end-to-end framework auditing two AI systems: a traditional credit risk classifier and a GenAI underwriting assistant.**
>
> **For the ML model, I benchmarked an XGBoost champion against a Logistic Regression baseline. Beyond standard discrimination (ROC-AUC 0.935, F1 0.850), I conducted segmented error profiling, showing where the model had high-confidence false negatives in thin-file applicants.**
>
> **For Responsible AI, I audited Fair Lending compliance using Fairlearn. While gender passed the EEOC 80% Four-Fifths rule with a 0.984 disparate impact ratio, I flagged performance disparities in younger age cohorts, marking it 'Review Required' under our traffic-light governance.**
>
> **For explainability, I integrated SHAP to automatically generate FCRA-compliant Adverse Action reason codes. I then stress-tested the model against missing inputs, Gaussian noise, and a simulated recession shock to verify monotonic risk sensitivity.**
>
> **For the GenAI component, I developed test suites for prompt sensitivity across 4 underwriter personas, evaluated factual hallucination against ground-truth profiles, measured stochastic consistency across repeated queries, and verified ECOA safety guardrails.**
>
> **Finally, I built a production monitoring engine calculating the Population Stability Index (PSI) across 6 monthly batches, which successfully raised an alert when macroeconomic shift drove PSI above 0.25 in Month 6, and packaged everything into an interactive Streamlit dashboard and formal SR 11-7 validation report."**

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
