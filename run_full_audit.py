"""
Master Audit Runner & Orchestration Pipeline
Executes the full end-to-end AI Model Risk & GenAI Evaluation:
1. Data Validation
2. Performance & Error Profiling
3. Fair Lending Bias Audit
4. SHAP Explainability
5. Robustness & Stress Testing
6. GenAI / LLM Evaluation Suite
7. Longitudinal Drift Monitoring
8. Regulatory Risk Report Generation (Markdown + HTML)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.utils import logger, NpEncoder
from src.data_validation import DataValidator
from src.performance import PerformanceEvaluator
from src.error_analysis import ErrorAnalyzer
from src.fairness import FairnessAuditor
from src.explainability import ModelExplainer
from src.robustness import RobustnessTester
from src.llm_application import MockLLMEngine
from src.llm_evaluation import LLMEvaluator
from src.monitoring import DriftMonitor
from src.report_generator import ReportGenerator


def run_full_audit():
    logger.info("=" * 60)
    logger.info("Starting AI Model Risk & GenAI Evaluation Audit")
    logger.info("=" * 60)

    # 1. Data Validation
    logger.info("[1/7] Running Data Validation Engine...")
    validator = DataValidator()
    raw_df = pd.read_csv("data/raw/credit_risk_raw.csv")
    clean_df = pd.read_csv("data/processed/credit_risk_clean.csv")
    val_scorecard = validator.generate_quality_scorecard(clean_df)
    logger.info(f"Data Health Score: {val_scorecard['data_health_score']}% | Status: {val_scorecard['governance_status']}")

    # Load Holdout Test Set & Models
    test_df = pd.read_csv("data/processed/test_split.csv")
    champion_model = joblib.load("models/credit_model.joblib")
    baseline_model = joblib.load("models/baseline_model.joblib")

    feature_cols = [
        "annual_income", "loan_amount", "employment_length_years", "credit_history_years",
        "num_open_credit_lines", "debt_to_income_ratio", "has_previous_default",
        "delinquent_2yrs", "loan_purpose"
    ]

    X_test = test_df[feature_cols]
    y_test = test_df["risk_label"].values

    # Model Predictions
    champ_probs = champion_model.predict_proba(X_test)[:, 1]
    base_probs = baseline_model.predict_proba(X_test)[:, 1]
    test_df["pred_prob"] = champ_probs

    # 2. Performance & Error Profiling
    logger.info("[2/7] Evaluating Discrimination & Error Patterns...")
    perf_eval = PerformanceEvaluator(default_threshold=0.5)
    champ_perf = perf_eval.evaluate_model(y_test, champ_probs)
    base_perf = perf_eval.evaluate_model(y_test, base_probs)
    curves = perf_eval.compute_curves(y_test, champ_probs)
    opt_thresh = perf_eval.find_optimal_threshold(y_test, champ_probs)

    err_analyzer = ErrorAnalyzer(threshold=0.5)
    analyzed_df = err_analyzer.segment_errors(test_df)
    high_conf_errors = err_analyzer.get_high_confidence_errors(analyzed_df, top_n=5)
    slice_reports = err_analyzer.analyze_feature_slices(analyzed_df)
    logger.info(f"Champion ROC-AUC: {champ_perf['roc_auc']} | F1: {champ_perf['f1_score']}")

    # 3. Fair Lending Bias Audit
    logger.info("[3/7] Conducting Algorithmic Fairness & Bias Audit (Fairlearn)...")
    fairness_auditor = FairnessAuditor(four_fifths_threshold=0.80)
    bias_audit = fairness_auditor.run_full_bias_audit(
        analyzed_df,
        y_true_col="risk_label",
        prob_col="pred_prob",
        protected_attributes=["gender", "age_group"]
    )
    logger.info(f"Fairness Audit Status: {bias_audit['overall_status']}")

    # 4. Explainability (SHAP)
    logger.info("[4/7] Computing SHAP Attributions & Feature Importances...")
    explainer = ModelExplainer(champion_model, background_data=X_test)
    global_imp = explainer.compute_global_importance(X_test.iloc[:300])
    sample_explanation = explainer.explain_instance(X_test.iloc[0])
    logger.info(f"Top 3 Global Risk Drivers: {global_imp['Feature'].head(3).tolist()}")

    # 5. Robustness & Stress Testing
    logger.info("[5/7] Executing Robustness Stress-Tester (Missingness, Noise, Shock)...")
    robustness_tester = RobustnessTester(champion_model, decision_threshold=0.5)
    rob_scorecard = robustness_tester.generate_robustness_scorecard(X_test, test_df["risk_label"])
    logger.info(f"Robustness Status: {rob_scorecard['overall_robustness_status']}")

    # 6. GenAI / LLM Evaluation
    logger.info("[6/7] Running GenAI Evaluation Suite (Prompt Sensitivity, Hallucination, Consistency)...")
    llm_engine = MockLLMEngine(seed=42, inject_hallucination_rate=0.06)
    llm_evaluator = LLMEvaluator(llm_engine=llm_engine)
    eval_profiles = test_df.iloc[:35].to_dict(orient="records")
    llm_audit = llm_evaluator.run_full_llm_evaluation(eval_profiles)
    logger.info(f"LLM Governance Status: {llm_audit['overall_governance_status']}")

    # 7. Production Monitoring & Drift
    logger.info("[7/7] Simulating 6-Month Production Longitudinal Drift Monitoring...")
    monthly_batches = pd.read_csv("data/processed/monthly_monitoring_batches.csv")
    drift_monitor = DriftMonitor()
    drift_audit = drift_monitor.monitor_longitudinal_batches(
        baseline_df=clean_df,
        monthly_batches_df=monthly_batches,
        model_pipeline=champion_model,
        feature_cols=feature_cols
    )
    feature_drift_df = drift_monitor.evaluate_feature_drift(clean_df, monthly_batches[monthly_batches["batch_month"] == "Month_6"])
    logger.info(f"Monitoring Drift Status: {drift_audit['overall_monitoring_status']}")

    # 8. Report Generation
    logger.info("Generating Formal Regulatory Risk Assessment Reports...")
    report_gen = ReportGenerator(output_dir="reports")
    perf_summary = {
        "champion_performance": champ_perf,
        "baseline_performance": base_perf,
        "curves": curves,
        "optimal_threshold": opt_thresh
    }
    with open("models/model_metadata.json", "r") as f:
        metadata = json.load(f)

    report_paths = report_gen.generate_full_report(
        validation_results=val_scorecard,
        performance_results=perf_summary,
        fairness_results=bias_audit,
        explainability_results={"global_importance": global_imp.to_dict(orient="records"), "sample_local": sample_explanation},
        robustness_results=rob_scorecard,
        llm_results=llm_audit,
        monitoring_results=drift_audit,
        model_metadata=metadata
    )

    # Save summary dictionary for Streamlit Dashboard caching
    summary_data = {
        "validation_scorecard": val_scorecard,
        "champion_performance": champ_perf,
        "baseline_performance": base_perf,
        "optimal_threshold": opt_thresh,
        "fairness_audit": {
            "overall_status": bias_audit["overall_status"],
            "traffic_light": bias_audit["traffic_light"],
            "audited_attributes": {
                k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "by_group_dataframe"}
                for k, v in bias_audit["audited_attributes"].items()
            }
        },
        "robustness_scorecard": rob_scorecard,
        "llm_audit": llm_audit,
        "monitoring_audit": {
            "overall_status": drift_audit["overall_monitoring_status"],
            "alerts": drift_audit["alerts_triggered"],
            "timeline": drift_audit["monitoring_timeline_df"].to_dict(orient="records")
        },
        "global_feature_importance": global_imp.to_dict(orient="records"),
        "report_paths": report_paths
    }

    summary_path = "reports/audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, cls=NpEncoder)
    logger.info(f"Audit summary cached to {summary_path}")

    logger.info("=" * 60)
    logger.info("Master Audit Completed Successfully!")
    logger.info(f"Markdown Report: {report_paths['markdown_report_path']}")
    logger.info(f"HTML Report:     {report_paths['html_report_path']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_full_audit()
