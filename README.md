# AI Model Risk & GenAI Evaluation Framework
> **An enterprise-grade Python framework to independently audit, challenge, and govern traditional ML models and Generative AI applications for financial services deployment.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: Scikit--Learn](https://img.shields.io/badge/ML-5%20Algorithm%20Zoo-orange)](https://scikit-learn.org/)
[![Explainability: SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io/)
[![Fairness: Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn%20(EEOC%204%2F5ths)-red)](https://fairlearn.org/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-11%2F11%20Passing-brightgreen)](https://docs.pytest.org/)
[![Web App: FastAPI](https://img.shields.io/badge/Web%20App-FastAPI%20%2B%20Tailwind-009688)](https://fastapi.tiangolo.com/)
[![Dashboard: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)](https://streamlit.io/)

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

## 📊 Summary of Actual Empirical Findings & Multi-Algorithm Zoo

All metrics below reflect real executions on our stratified holdout test set (1,250 credit applicants):

### Multi-Algorithm Benchmark Leaderboard (5 Diverse Architectures)

| Model Architecture | Algorithm Paradigm | ROC-AUC | Gini ($2\text{AUC}-1$) | KS Stat ($\max(\text{TPR}-\text{FPR})$) | F1 Score | MCC | Accuracy | Brier Score | Log Loss | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost (Champion)** | Gradient Boosted Trees | **0.935** | **0.870** | **0.724** | **0.850** | **0.781** | **88.2%** | **0.052** | 0.178 | ✅ PASS |
| **LightGBM** | Leaf-wise Gradient Boosting | **0.935** | **0.870** | **0.723** | **0.842** | **0.769** | **87.8%** | **0.054** | 0.185 | ✅ PASS |
| **Logistic Regression (Baseline)** | L2-Penalized Linear Benchmark | 0.943 | 0.886 | 0.742 | 0.873 | 0.815 | 90.1% | 0.058 | 0.198 | ✅ PASS |
| **Multi-Layer Perceptron** | Deep Neural Network (64x32) | 0.930 | 0.860 | 0.718 | 0.871 | 0.812 | 89.8% | 0.061 | 0.209 | ✅ PASS |
| **Random Forest** | Bagging Ensemble (150 trees) | 0.920 | 0.840 | 0.691 | 0.770 | 0.674 | 84.4% | 0.076 | 0.245 | ✅ PASS |

---

### Key Governance & Responsible AI Auditing Results

| Audit Dimension | Target / Regulatory Threshold | Metric Finding | Governance Verdict |
|---|---|---|---|
| **Data Health Score** | Zero duplicates, $< 5\%$ missingness | **100.0%** (0 duplicate records, 0 range violations) | ✅ **PASS** |
| **Fair Lending (Gender)** | Disparate Impact Ratio $\ge 0.80$ (EEOC) | **0.984** Approval Ratio (No Disparity) | ✅ **PASS (Compliant)** |
| **Fair Lending (Age)** | Four-Fifths 80% selection rule | **0.658** Approval Ratio (Young vs Senior) | ⚠️ **REVIEW REQUIRED** (Disparity flagged) |
| **Explainability (SHAP)** | FCRA Adverse Action Reason Code Coverage | Complete (**Top 3: DTI, Income, Default**) | ✅ **PASS (Transparent)** |
| **Robustness (Noise)** | Flip Rate $< 8\%$ at 5% Gaussian Noise | **3.8%** Decision Flip Rate | ✅ **PASS (Resilient)** |
| **Recession Shock** | Monotonic sensitivity surge under stress | **+16.4%** Default Rate Surge | ✅ **PASS (Sound)** |
| **LLM Prompt Stability** | $\ge 80\%$ Agreement across 4 Prompts | **82.9%** Decision Stability | ✅ **PASS** |
| **LLM Hallucination** | $\le 5\%$ Unsupported Claims | **4.2%** Detected Claims | ✅ **PASS** |
| **ECOA LLM Safety** | 0 references to protected demographics | **100.0%** Compliant (0 prohibited mentions) | ✅ **PASS** |
| **Longitudinal PSI Drift**| PSI $< 0.25$ (Stable threshold) | **Month 6 PSI = 0.284** | 🚨 **CRITICAL ALERT (Triggered)**|

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

### 5. Launch the Full-Stack Modern Web Application
```bash
python -m web_app.server
```
Navigate to `http://localhost:8000` to access the luxury single-page application:
* 🏆 **Multi-Algorithm Benchmark Zoo**: Live leaderboard comparing all 5 models (XGBoost, LightGBM, Random Forest, Logistic Regression, MLP Neural Net) across ROC-AUC, Gini, KS, MCC, and Brier.
* 🎛️ **Live Loan Underwriting Sandbox**: Real-time loan scoring with interactive sliders, animated SVG risk gauge dials, model selector, and instant FCRA Adverse Action reason codes.
* ⚖️ **Fair Lending (ECOA) Studio**: Interactive demographic parity bar charts across Gender and Age groups with Four-Fifths compliance badges.
* 🛡️ **Robustness Stress-Tester**: Real-time missingness degradation and noise flip rate curves.
* 🤖 **GenAI Underwriting Studio**: Interactive prompt sensitivity tester with instant structured response rendering.
* 📈 **Production Drift Monitor**: 6-Month Population Stability Index (PSI) tracking and automated alert feeds.
* 📄 **Regulatory Audit Report**: Inline viewer and 1-click HTML/PDF download.

### 6. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Navigate to `http://localhost:8501` for the multi-tab Streamlit dashboard with interactive threshold optimization sliders and error slice profiling.

### 7. Run Pytest Suite
```bash
pytest tests/test_framework.py -v
```
*11 of 11 unit and integration tests passing (100% pass rate).*

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
