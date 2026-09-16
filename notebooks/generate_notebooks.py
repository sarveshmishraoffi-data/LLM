"""
Script to generate the 4 standard Jupyter Notebooks for the AI Model Risk Framework.
"""

import os
import json


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.10"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }


def generate_all_notebooks(output_dir="notebooks"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 01_data_analysis.ipynb
    nb1 = make_notebook([
        md_cell("# 01. Exploratory Data Analysis & Data Quality Validation\n**Project**: AI Model Risk & GenAI Evaluation Framework\n**Focus**: Pre-implementation data integrity, schema validation, anomaly checks (SR 11-7)"),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath('..'))
from src.data_validation import DataValidator"""),
        md_cell("## 1. Load Raw & Clean Datasets"),
        code_cell("""raw_df = pd.read_csv('../data/raw/credit_risk_raw.csv')
clean_df = pd.read_csv('../data/processed/credit_risk_clean.csv')

print(f"Raw unvalidated dataset shape: {raw_df.shape}")
print(f"Clean sanitized dataset shape: {clean_df.shape}")
clean_df.head()"""),
        md_cell("## 2. Execute Data Validation Engine & Health Scorecard"),
        code_cell("""validator = DataValidator()

# Run checks on raw dataset (demonstrating anomaly detection)
raw_scorecard = validator.generate_quality_scorecard(raw_df)
print("=== Raw Dataset Scorecard ===")
print(f"Health Score: {raw_scorecard['data_health_score']}%")
print(f"Status: {raw_scorecard['governance_status']}")

# Run checks on clean dataset
clean_scorecard = validator.generate_quality_scorecard(clean_df)
print("\\n=== Clean Dataset Scorecard ===")
print(f"Health Score: {clean_scorecard['data_health_score']}%")
print(f"Status: {clean_scorecard['governance_status']}")"""),
        md_cell("## 3. Financial Distribution Analysis"),
        code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(clean_df['annual_income'], kde=True, ax=axes[0, 0], color='#002663')
axes[0, 0].set_title('Annual Income Distribution')

sns.histplot(clean_df['debt_to_income_ratio'], kde=True, ax=axes[0, 1], color='#0070d2')
axes[0, 1].set_title('Debt-to-Income Ratio Distribution')

sns.countplot(x='has_previous_default', data=clean_df, ax=axes[1, 0], palette='Blues')
axes[1, 0].set_title('Historical Default Record Frequency')

sns.countplot(x='risk_label', data=clean_df, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Target Risk Distribution (0 = Safe, 1 = Default)')
plt.tight_layout()
plt.show()"""),
        md_cell("## 4. Protected Demographic Distributions"),
        code_cell("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
clean_df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[0], colors=['#93c5fd', '#3b82f6'])
axes[0].set_title('Gender Distribution')

clean_df['age_group'].value_counts().plot(kind='bar', ax=axes[1], color='#1e3a8a')
axes[1].set_title('Age Group Distribution')
plt.tight_layout()
plt.show()""")
    ])

    # 2. 02_model_development.ipynb
    nb2 = make_notebook([
        md_cell("# 02. Credit Model Development & Champion-Challenger Benchmarking\n**Project**: AI Model Risk & GenAI Evaluation Framework\n**Focus**: Baseline (Logistic Regression) vs Champion (XGBoost) modeling with Fair Lending separation"),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from src.model_training import CreditModelTrainer
from src.performance import PerformanceEvaluator"""),
        md_cell("## 1. Prepare Data & Train Pipelines"),
        code_cell("""trainer = CreditModelTrainer(data_path='../data/processed/credit_risk_clean.csv')
splits = trainer.prepare_data(test_size=0.25, random_state=42)
models = trainer.train_models(splits)
trainer.save_artifacts(models_dir='../models', data_splits=splits)"""),
        md_cell("## 2. Evaluate Baseline vs Champion Discrimination"),
        code_cell("""X_test = splits['X_test']
y_test = splits['y_test']

base_probs = models['baseline_logistic'].predict_proba(X_test)[:, 1]
champ_probs = models['champion_xgboost'].predict_proba(X_test)[:, 1]

evaluator = PerformanceEvaluator(default_threshold=0.5)
comparison_df = evaluator.compare_models(y_test, {
    'Baseline (Logistic Regression)': base_probs,
    'Champion (XGBoost)': champ_probs
})
comparison_df"""),
        md_cell("## 3. Discrimination ROC and Precision-Recall Curves"),
        code_cell("""import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

fpr_b, tpr_b, _ = roc_curve(y_test, base_probs)
fpr_c, tpr_c, _ = roc_curve(y_test, champ_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_c, tpr_c, label=f'Champion XGBoost (AUC = {auc(fpr_c, tpr_c):.3f})', color='#002663', lw=2)
plt.plot(fpr_b, tpr_b, label=f'Baseline Logistic (AUC = {auc(fpr_b, tpr_b):.3f})', color='#f59e0b', lw=2, linestyle='--')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Discrimination Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()""")
    ])

    # 3. 03_model_validation.ipynb
    nb3 = make_notebook([
        md_cell("# 03. Comprehensive Model Validation & Stress Testing\n**Project**: AI Model Risk & GenAI Evaluation Framework\n**Focus**: Performance, Error Profiling, Fairlearn Bias Audit, SHAP, and Robustness"),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.abspath('..'))
from src.performance import PerformanceEvaluator
from src.error_analysis import ErrorAnalyzer
from src.fairness import FairnessAuditor
from src.explainability import ModelExplainer
from src.robustness import RobustnessTester"""),
        md_cell("## 1. Load Test Holdout Data & Champion Model"),
        code_cell("""test_df = pd.read_csv('../data/processed/test_split.csv')
champion_model = joblib.load('../models/credit_model.joblib')

feature_cols = [
    'annual_income', 'loan_amount', 'employment_length_years', 'credit_history_years',
    'num_open_credit_lines', 'debt_to_income_ratio', 'has_previous_default',
    'delinquent_2yrs', 'loan_purpose'
]

X_test = test_df[feature_cols]
y_test = test_df['risk_label']
y_prob = champion_model.predict_proba(X_test)[:, 1]
test_df['pred_prob'] = y_prob"""),
        md_cell("## 2. Fair Lending Algorithmic Bias Audit (Fairlearn)"),
        code_cell("""auditor = FairnessAuditor(four_fifths_threshold=0.80)
bias_results = auditor.run_full_bias_audit(test_df)

print(f"Overall Fairness Status: {bias_results['overall_status']}")
for attr, res in bias_results['audited_attributes'].items():
    print(f"\\nAttribute: {attr}")
    print(f"  Disparate Impact Ratio: {res['disparate_impact_ratio']}")
    print(f"  Status: {res['overall_governance_status']}")
    print(f"  Finding: {res['finding_summary']}")"""),
        md_cell("## 3. Explainability & Reason Codes (SHAP)"),
        code_cell("""explainer = ModelExplainer(champion_model, background_data=X_test)
global_imp = explainer.compute_global_importance(X_test.iloc[:200])
print("=== Top 5 Global Risk Drivers ===")
print(global_imp.head(5))

print("\\n=== Local Reason Code Explanation for Applicant #1 ===")
local_exp = explainer.explain_instance(X_test.iloc[0])
print(f"Predicted Default Probability: {local_exp['predicted_probability']:.1%}")
print("Top Adverse Factors (+Risk):", [f['feature'] for f in local_exp['top_risk_increasing_factors']])
print("Top Mitigating Factors (-Risk):", [f['feature'] for f in local_exp['top_risk_mitigating_factors']])"""),
        md_cell("## 4. Robustness & Stress Resilience Testing"),
        code_cell("""robustness = RobustnessTester(champion_model)
rob_card = robustness.generate_robustness_scorecard(X_test, y_test)
print(f"Overall Robustness Status: {rob_card['overall_robustness_status']}")
print("\\nMissing Data Stress Results:")
print(pd.DataFrame(rob_card['missing_data_results']))
print("\\nNoise Perturbation Results:")
print(pd.DataFrame(rob_card['noise_perturbation_results']))
print(f"\\nRecession Macro Stress Shock: {rob_card['macro_stress_shock']['narrative']}")""")
    ])

    # 4. 04_llm_evaluation.ipynb
    nb4 = make_notebook([
        md_cell("# 04. GenAI / LLM Evaluation & Governance\n**Project**: AI Model Risk & GenAI Evaluation Framework\n**Focus**: Prompt Sensitivity, Factual Groundedness / Hallucination, Consistency, Safety Guardrails"),
        code_cell("""import os
import sys
import pandas as pd
import json

sys.path.insert(0, os.path.abspath('..'))
from src.llm_application import MockLLMEngine
from src.llm_evaluation import LLMEvaluator"""),
        md_cell("## 1. Initialize Engine & Test Cohort"),
        code_cell("""engine = MockLLMEngine(seed=42, inject_hallucination_rate=0.06)
evaluator = LLMEvaluator(llm_engine=engine)

test_df = pd.read_csv('../data/processed/test_split.csv')
sample_profiles = test_df.iloc[:25].to_dict(orient='records')
print(f"Loaded {len(sample_profiles)} test applicant profiles for GenAI auditing.")"""),
        md_cell("## 2. Prompt Sensitivity Testing"),
        code_cell("""prompt_res = evaluator.test_prompt_sensitivity(sample_profiles)
print(f"Prompt Stability Rate: {prompt_res['prompt_stability_pct']}%")
print(f"Governance Status: {prompt_res['governance_status']}")
pd.DataFrame(prompt_res['sample_profile_results']).head()"""),
        md_cell("## 3. Hallucination & Groundedness Audit"),
        code_cell("""halluc_res = evaluator.test_hallucination_and_groundedness(sample_profiles)
print(f"Hallucination Incident Rate: {halluc_res['hallucination_rate_pct']}%")
print(f"Groundedness Score: {halluc_res['groundedness_score']}")
print(f"Status: {halluc_res['governance_status']}")
if halluc_res['flagged_cases']:
    print("Flagged hallucination example:", halluc_res['flagged_cases'][0])"""),
        md_cell("## 4. Self-Consistency Testing"),
        code_cell("""consist_res = evaluator.test_self_consistency(sample_profiles, repeats=4, temperature=0.5)
print(f"Mean Consistency Score: {consist_res['mean_consistency_pct']}%")
print(f"Status: {consist_res['governance_status']}")"""),
        md_cell("## 5. Fair Lending Guardrails (ECOA Compliance)"),
        code_cell("""safety_res = evaluator.test_fair_lending_safety_guardrails(sample_profiles)
print(f"ECOA Guardrail Compliance Rate: {safety_res['compliance_rate_pct']}%")
print(f"Status: {safety_res['governance_status']}")""")
    ])

    files = [
        ("01_data_analysis.ipynb", nb1),
        ("02_model_development.ipynb", nb2),
        ("03_model_validation.ipynb", nb3),
        ("04_llm_evaluation.ipynb", nb4)
    ]

    for fname, nb in files:
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Generated notebook: {path}")


if __name__ == "__main__":
    generate_all_notebooks()
