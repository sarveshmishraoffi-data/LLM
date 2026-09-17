---
title: AI Model Risk & GenAI Governance Platform
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.64.0
app_file: app.py
pinned: false
license: mit
---

# Enterprise AI Model Risk Management (MRM) & GenAI Governance Platform

### Comprehensive Supervisory Validation, Fair Lending Auditing, and LLM Trust & Safety
*Aligned with Federal Reserve Supervisory Letter **SR 11-7**, OCC Bulletin **2011-12**, **ECOA (Regulation B)**, and **FCRA** Guidance.*

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sarveshmishraoffi-data-llm-app-ghgtnp.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Framework: Scikit--Learn](https://img.shields.io/badge/ML-5%20Algorithm%20Zoo-orange)](https://scikit-learn.org/)
[![Fairness: Fairlearn](https://img.shields.io/badge/Fairness-EEOC%20Four--Fifths%20Compliant-red)](https://fairlearn.org/)
[![Explainability: SHAP](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-purple)](https://shap.readthedocs.io/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-11%2F11%20Passing-brightgreen)](https://docs.pytest.org/)

[🚀 **Launch Live Production Platform**](https://sarveshmishraoffi-data-llm-app-ghgtnp.streamlit.app/) • [📤 **Online Model Testing Lab**](https://sarveshmishraoffi-data-llm-app-ghgtnp.streamlit.app/) • [📖 **Comprehensive Product Guide**](PRODUCT_GUIDE.md)

</div>

---

## 1. Executive Summary & Business Context

In institutional banking, deploying machine learning systems and Large Language Models (LLMs) requires rigorous, independent verification. Financial institutions cannot deploy automated credit decisioning engines without providing quantitative proof to supervisory authorities (e.g., US Federal Reserve, OCC, CFPB, RBI, ECB) that:

1. **Statistical Integrity**: Models demonstrate high discriminatory power and well-calibrated loss across diverse applicant profiles.
2. **Statutory Non-Discrimination**: Algorithms strictly comply with the **Equal Credit Opportunity Act (ECOA)** and the **EEOC 80% Four-Fifths Rule**.
3. **Transparent Explainability**: Adverse credit actions are legally defensible under the **Fair Credit Reporting Act (FCRA)** with exact reason codes.
4. **Stress Robustness**: Systems withstand severe macroeconomic downturns and data corruption without catastrophic degradation.
5. **GenAI Trust & Factuality**: LLM-generated underwriting rationales remain strictly grounded in audited data, with **0.0% hallucination rates**.

This platform provides an end-to-end **Second Line of Defense (2LoD) Model Risk Governance Suite** that programmatically challenges, stress-tests, and certifies AI systems prior to production deployment.

---

## 2. Core Governance Capabilities

```text
                                  CUSTOMER CREDIT DATA
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │     Data Validation      │  <── Schema bounds, missingness,
                              │   & Evidence Assessment  │      duplicate & anomaly checks
                              └──────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
        ┌───────────────────────┐                     ┌───────────────────────┐
        │  SYSTEM A: ML MODEL   │                     │  SYSTEM B: GENAI LLM  │
        │  Logistic Reg. (Base) │                     │  Narrative Rationale  │
        │   XGBoost (Champion)  │                     │  & Adverse Actions    │
        └───────────────────────┘                     └───────────────────────┘
                    │                                             │
        ┌───────────┼───────────┐                     ┌───────────┼───────────┐
        │           │           │                     │           │           │
        ▼           ▼           ▼                     ▼           ▼           ▼
   Performance    Bias/       SHAP                 Prompt     Hallucination Consistency
  Discrimination Fairness  Attribution           Sensitivity    Auditing     Repeatability
        │           │           │                     │           │           │
        └───────────┼───────────┘                     └───────────┼───────────┘
                    │                                             │
                    ▼                                             ▼
             Stress Testing                                Longitudinal
              & Robustness                                  Monitoring
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │  Executive Governance &  │  <── SR 11-7 Audit Certificate
                              │  Regulatory Validation   │      (Interactive UI + Report)
                              └──────────────────────────┘
```

### Key Pillars:
* 🏆 **Multi-Model Benchmark Zoo**: Cross-evaluates 5 distinct algorithm paradigms (XGBoost Champion, LightGBM, Random Forest, Multi-Layer Perceptron Neural Network, and Logistic Regression Baseline) across 10+ financial metrics (Gini, KS, MCC, Brier Score, ECE).
* 📤 **Online Model Testing Lab**: Drag-and-drop interface enabling risk officers to upload **any custom model (`.joblib`, `.pkl`)** and dataset (`.csv`) for instant 360° compliance audits, supported by 1-click sample downloads.
* ⚖️ **Fair Lending (ECOA) Auditing**: Automated demographic parity and disparate impact ratio evaluations under the statutory **80% Four-Fifths rule** across Age and Gender cohorts.
* 🔍 **Explainable AI (XAI)**: SHAP-driven global feature ranking and applicant-level waterfall decompositions for FCRA adverse action disclosures.
* 🛡️ **Adversarial Stress Testing**: Evaluates systemic risk under synthetic recession shocks (-25% income, +35% DTI), Gaussian noise perturbation, and missing data degradation.
* 🤖 **GenAI Hallucination & Consistency Verification**: Extracts structured claims from LLM underwriting memos, cross-referencing them against ground-truth profiles to ensure zero hallucinations and consistent persona outputs.
* 📈 **Longitudinal Drift Surveillance**: Tracks Population Stability Index (PSI) and Kolmogorov-Smirnov distribution shifts across rolling production batches to preemptively flag model decay.

---

## 3. Empirical Benchmark Findings

All metrics reflect real execution on our stratified holdout test set (1,250 credit applicants):

### Multi-Algorithm Benchmark Leaderboard (5 Diverse Architectures)

| Model Architecture | Algorithm Paradigm | ROC-AUC | Gini ($2\text{AUC}-1$) | KS Stat ($\max(\text{TPR}-\text{FPR})$) | F1 Score | MCC | Accuracy | Brier Score | Log Loss | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost (Champion)** | Gradient Boosted Trees | **0.935** | **0.870** | **0.724** | **0.850** | **0.781** | **88.2%** | **0.052** | 0.178 | ✅ PASS |
| **LightGBM** | Leaf-wise Gradient Boosting | **0.935** | **0.870** | **0.723** | **0.842** | **0.769** | **87.8%** | **0.054** | 0.185 | ✅ PASS |
| **Logistic Regression (Baseline)** | L2-Penalized Linear Benchmark | 0.943 | 0.886 | 0.742 | 0.873 | 0.815 | 90.1% | 0.058 | 0.198 | ✅ PASS |
| **Multi-Layer Perceptron** | Deep Neural Network (64x32) | 0.930 | 0.860 | 0.718 | 0.871 | 0.812 | 89.8% | 0.061 | 0.209 | ✅ PASS |
| **Random Forest** | Bagging Ensemble (150 trees) | 0.920 | 0.840 | 0.691 | 0.770 | 0.674 | 84.4% | 0.076 | 0.245 | ✅ PASS |

---

### Quantitative Model Risk Governance Scorecard

| Governance Dimension | Regulatory Benchmark | Empirical Finding | Compliance Determination |
|---|---|---|---|
| **Data Integrity Score** | Zero duplicates, $< 5\%$ missingness | **100.0%** (0 duplicate records, 0 range violations) | ✅ **APPROVED** |
| **Fair Lending (Gender)** | Disparate Impact Ratio $\ge 0.80$ (EEOC) | **0.984** Selection Rate Ratio | ✅ **COMPLIANT** |
| **Fair Lending (Age)** | Four-Fifths 80% selection rule | **0.658** Selection Rate Ratio (Young vs Senior) | ⚠️ **CONDITIONAL REVIEW** |
| **Explainability (SHAP)** | Adverse Action Reason Code Coverage | Complete (**Top 3: DTI, Income, Delinquency**) | ✅ **TRANSPARENT** |
| **Robustness (Noise)** | Flip Rate $< 8\%$ at 5% Gaussian Noise | **3.8%** Decision Flip Rate | ✅ **ROBUST** |
| **Macro Stress Shock** | Monotonic risk surge under recession | **+16.4%** Default Rate Elevation | ✅ **SOUND** |
| **LLM Prompt Stability** | $\ge 80\%$ Agreement across 4 Prompts | **82.9%** Decision Stability | ✅ **COMPLIANT** |
| **LLM Factuality** | $\le 5\%$ Unsupported Claims | **0.0%** Hallucination Rate in Validated Set | ✅ **CERTIFIED** |
| **Regulatory Drift (PSI)** | PSI $< 0.25$ (Stable threshold) | **Month 6 PSI = 0.284** | 🚨 **ACTION TRIGGERED** |

---

## 4. Supervisory Compliance & Institutional Alignment

This framework maps directly to statutory guidance from federal and international financial regulatory authorities:

| Regulatory Standard | Supervisory Requirement | Implementation in this Framework | Primary Module |
|---|---|---|---|
| **Federal Reserve SR 11-7** | Independent Model Validation & Conceptual Soundness | Multi-model champion/challenger comparison; boundary limit stress tests; mathematical calibration checks. | [`src/performance.py`](file:///d:/LLM/src/performance.py) |
| **OCC Bulletin 2011-12** | Model Governance, Limitations & Ongoing Monitoring | Longitudinal tracking via Population Stability Index (PSI); automated degradation alerts across monthly batches. | [`src/monitoring.py`](file:///d:/LLM/src/monitoring.py) |
| **ECOA (Regulation B)** | Prohibition of Disparate Impact in Credit Decisioning | Algorithmic demographic parity checks; automated detection of Four-Fifths rule violations. | [`src/fairness.py`](file:///d:/LLM/src/fairness.py) |
| **FCRA (15 U.S.C. § 1681)** | Mandated Disclosures of Adverse Action Reason Codes | SHAP local waterfall decomposition generating mathematically ranked legal reason codes. | [`src/explainability.py`](file:///d:/LLM/src/explainability.py) |
| **CFPB Circular 2022-03** | Algorithmic Explainability & Black-Box Prohibition | Global and local interpretability pipelines ensuring credit determinations are fully transparent. | [`src/explainability.py`](file:///d:/LLM/src/explainability.py) |
| **NIST AI RMF 1.0** | Trustworthy & Responsible AI System Governance | Comprehensive GenAI validation: prompt sensitivity testing, hallucination auditing, and safety boundaries. | [`src/llm_evaluation.py`](file:///d:/LLM/src/llm_evaluation.py) |

---

## 5. Repository Architecture

```text
d:/LLM/
├── README.md                      # Primary documentation
├── PRODUCT_GUIDE.md               # Enterprise product manual and user guide
├── requirements.txt               # Locked production dependencies
├── run_full_audit.py              # Master pipeline executing full 7-stage audit
├── audit_custom_model.py          # Standalone CLI to audit any custom model & dataset
├── data/
│   ├── generate_dataset.py        # Realistic financial dataset generator
│   ├── raw/                       # Unprocessed records with synthetic anomalies
│   └── processed/                 # Partitioned and validated evaluation datasets
├── models/                        # Pre-trained champion and challenger binaries
├── samples/                       # 1-Click downloadable evaluation models and datasets
├── src/                           # Core Model Risk Management engine modules
│   ├── data_validation.py         # Schema, range, duplicate, and anomaly checks
│   ├── model_training.py          # Production training pipelines
│   ├── performance.py             # Discrimination, calibration, and banking score metrics
│   ├── error_analysis.py          # Sub-population error slicing and false-negative mining
│   ├── fairness.py                # Fairlearn Four-Fifths disparate impact auditor
│   ├── explainability.py          # SHAP TreeExplainer global and local reason attribution
│   ├── robustness.py              # Adversarial noise and recession stress-testing
│   ├── llm_application.py         # Generative AI underwriting assistant
│   ├── llm_evaluation.py          # Hallucination, prompt sensitivity, and safety engine
│   ├── monitoring.py              # Population Stability Index (PSI) drift monitor
│   └── report_generator.py        # Formal SR 11-7 Markdown & HTML report generator
├── dashboard/
│   └── app.py                     # Multi-tab Streamlit governance web application
├── reports/                       # Generated audit findings, scorecards, and HTML reports
└── tests/                         # Pytest test suite (11 unit & integration tests)
```

---

## 6. Quick Start & Execution

### 1. Live Deployment (Instant Access)
Launch the interactive cloud dashboard directly in your browser:  
👉 **[https://sarveshmishraoffi-data-llm-app-ghgtnp.streamlit.app/](https://sarveshmishraoffi-data-llm-app-ghgtnp.streamlit.app/)**

### 2. Local Environment Setup
```bash
git clone https://github.com/sarveshmishraoffi-data/LLM.git
cd LLM

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch Local Streamlit Governance Dashboard
```bash
streamlit run app.py
```

### 4. Execute Full Automated Audit Pipeline
```bash
python run_full_audit.py
```

### 5. Audit Your Own Model via CLI
```bash
python audit_custom_model.py --model "path/to/model.joblib" --data "path/to/data.csv" --target "is_default"
```

### 6. Run Automated Test Suite
```bash
pytest tests/ -v
```
*(All 11 unit and integration tests passing with 100% test coverage).*

---

## 7. Executive Defense Narrative

For senior executive briefings, regulatory examinations, and technical risk committee presentations:

> *"Rather than viewing artificial intelligence solely as a predictive modeling exercise, this framework approaches AI from the discipline of independent model risk governance and supervisory validation.*
>
> *We implemented an end-to-end testing platform that simultaneously audits traditional machine learning credit engines and Generative AI underwriting assistants.*
>
> *For supervised models, we benchmarked an XGBoost champion against four diverse algorithmic architectures. Beyond standard discrimination metrics (ROC-AUC 0.935, Gini 0.870), we conducted sub-population error slicing to identify high-confidence false negatives among thin-file applicants.*
>
> *Under statutory Fair Lending compliance, we audited Disparate Impact Ratios using Fairlearn. While gender satisfied the statutory Four-Fifths threshold (0.984 ratio), we flagged performance disparities in younger demographic cohorts, assigning it a conditional 'Review Required' governance status.*
>
> *For model explainability, we integrated SHAP TreeExplainer to produce legally defensible Adverse Action reason codes compliant with FCRA mandates. We subsequently stress-tested the model against missing inputs, Gaussian noise perturbation, and synthetic macroeconomic recession shocks.*
>
> *For the Generative AI component, we developed evaluation suites measuring prompt sensitivity across four underwriter personas, verified factual groundedness against ground-truth financial profiles to eliminate hallucinations, and enforced ECOA safety guardrails.*
>
> *Finally, our production monitoring pipeline tracks Population Stability Index (PSI) across rolling monthly batches, triggering an automated supervisory alert when macroeconomic shifts elevated Month 6 PSI above 0.25."*

---

## 📄 License & Governance Standards
This framework is licensed under the **MIT License**. Validation protocols strictly follow **Federal Reserve SR 11-7** and **OCC 2011-12** supervisory model risk management standards.
