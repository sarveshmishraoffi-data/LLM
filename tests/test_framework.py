"""
Comprehensive Pytest Suite for Model Risk & GenAI Evaluation Framework
Tests all 8 core framework modules:
- Data Validation
- Model Pipeline
- Discrimination Performance
- Fair Lending Bias Audit
- SHAP Explainability
- Robustness Stress-Testing
- GenAI & LLM Evaluation
- Drift & PSI Calculation
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_validation import DataValidator
from src.model_training import CreditModelTrainer
from src.performance import PerformanceEvaluator
from src.error_analysis import ErrorAnalyzer
from src.fairness import FairnessAuditor
from src.explainability import ModelExplainer
from src.robustness import RobustnessTester
from src.llm_application import MockLLMEngine
from src.llm_evaluation import LLMEvaluator
from src.monitoring import DriftMonitor


@pytest.fixture(scope="module")
def sample_credit_data():
    """Create a minimal clean and dirty synthetic dataset fixture."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "customer_id": [f"TEST_{i}" for i in range(n)],
        "age": np.random.randint(20, 70, size=n),
        "age_group": np.random.choice(["Young (<25)", "Working Age (25-55)", "Senior (>55)"], size=n),
        "gender": np.random.choice(["Female", "Male"], size=n),
        "annual_income": np.random.uniform(25000, 120000, size=n),
        "loan_amount": np.random.uniform(3000, 40000, size=n),
        "loan_purpose": np.random.choice(["debt_consolidation", "credit_card", "home_improvement"], size=n),
        "employment_length_years": np.random.randint(1, 10, size=n),
        "credit_history_years": np.random.randint(2, 20, size=n),
        "num_open_credit_lines": np.random.randint(1, 15, size=n),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.55, size=n),
        "has_previous_default": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "delinquent_2yrs": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "risk_label": np.random.choice([0, 1], size=n, p=[0.75, 0.25])
    })
    # Ensure physical consistency
    df["employment_length_years"] = np.minimum(df["employment_length_years"], np.maximum(df["age"] - 18, 0))
    return df


def test_data_validation_clean_and_dirty(sample_credit_data):
    """Test DataValidator on clean vs dirty data with injected anomalies."""
    validator = DataValidator()
    
    # Clean check
    clean_score = validator.generate_quality_scorecard(sample_credit_data)
    assert clean_score["data_health_score"] >= 90
    assert clean_score["governance_status"] == "PASS"

    # Dirty check with negative age and duplicate
    dirty_df = sample_credit_data.copy()
    dirty_df.loc[0, "age"] = -10
    dirty_df = pd.concat([dirty_df, dirty_df.iloc[:2]], ignore_index=True)

    dirty_score = validator.generate_quality_scorecard(dirty_df)
    assert dirty_score["checks"]["range_validity"]["status"] == "FAIL"
    assert dirty_score["checks"]["duplicates"]["status"] == "FAIL"
    assert dirty_score["data_health_score"] < clean_score["data_health_score"]


def test_model_training_and_inference(sample_credit_data, tmp_path):
    """Test CreditModelTrainer fits pipeline, outputs probabilities in [0, 1], and persists."""
    csv_path = str(tmp_path / "temp_credit.csv")
    sample_credit_data.to_csv(csv_path, index=False)

    trainer = CreditModelTrainer(data_path=csv_path)
    splits = trainer.prepare_data(test_size=0.30, random_state=42)
    models = trainer.train_models(splits)

    assert "champion_xgboost" in models
    assert "baseline_logistic" in models

    champ = models["champion_xgboost"]
    X_test = splits["X_test"]
    probs = champ.predict_proba(X_test)[:, 1]

    assert len(probs) == len(X_test)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_performance_metrics():
    """Verify discrimination metric calculation and optimal threshold optimizer."""
    evaluator = PerformanceEvaluator()
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    perf = evaluator.evaluate_model(y_true, y_prob, threshold=0.5)
    assert perf["accuracy"] == 1.0
    assert perf["f1_score"] == 1.0
    assert perf["roc_auc"] == 1.0
    assert perf["brier_score"] < 0.10

    opt = evaluator.find_optimal_threshold(y_true, y_prob)
    assert "optimal_threshold" in opt
    assert 0.0 < opt["optimal_threshold"] < 1.0


def test_error_analysis(sample_credit_data):
    """Verify error segmentation into TP, TN, FP, FN and high confidence error isolation."""
    analyzer = ErrorAnalyzer(threshold=0.5)
    df = sample_credit_data.copy()
    df["pred_prob"] = np.random.uniform(0, 1, size=len(df))

    segmented = analyzer.segment_errors(df)
    assert "error_category" in segmented.columns
    assert set(segmented["error_category"].unique()).issubset({"True Negative", "False Positive", "False Negative", "True Positive"})

    high_conf = analyzer.get_high_confidence_errors(segmented, top_n=3)
    assert "high_confidence_false_negatives" in high_conf
    assert "high_confidence_false_positives" in high_conf


def test_fairness_disparate_impact(sample_credit_data):
    """Verify Fairlearn fairness auditing and Four-Fifths rule calculation."""
    auditor = FairnessAuditor(four_fifths_threshold=0.80)
    df = sample_credit_data.copy()
    df["pred_prob"] = np.random.uniform(0.1, 0.9, size=len(df))

    audit = auditor.run_full_bias_audit(df, protected_attributes=["gender", "age_group"])
    assert "overall_status" in audit
    assert "gender" in audit["audited_attributes"]
    assert "disparate_impact_ratio" in audit["audited_attributes"]["gender"]
    assert audit["audited_attributes"]["gender"]["disparate_impact_ratio"] > 0.0


def test_shap_explainability(sample_credit_data):
    """Verify SHAP explainer extracts global importances and local reason codes."""
    trainer = CreditModelTrainer()
    preprocessor = trainer.build_preprocessor()
    
    # Train quick champion model
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    X = sample_credit_data[trainer.NUMERICAL_FEATURES + trainer.CATEGORICAL_FEATURES]
    y = sample_credit_data["risk_label"]

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(n_estimators=10, max_depth=3, random_state=42))
    ])
    pipe.fit(X, y)

    explainer = ModelExplainer(pipe, background_data=X.iloc[:20])
    global_imp = explainer.compute_global_importance(X.iloc[:30])
    assert len(global_imp) > 0
    assert "Feature" in global_imp.columns
    assert "Mean_Abs_SHAP" in global_imp.columns

    local_exp = explainer.explain_instance(X.iloc[0])
    assert "predicted_probability" in local_exp
    assert "top_risk_increasing_factors" in local_exp
    assert "top_risk_mitigating_factors" in local_exp


def test_robustness_perturbation(sample_credit_data):
    """Verify robustness stress-testing under missingness and noise injection."""
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    trainer = CreditModelTrainer()
    X = sample_credit_data[trainer.NUMERICAL_FEATURES + trainer.CATEGORICAL_FEATURES]
    y = sample_credit_data["risk_label"]

    pipe = Pipeline([
        ("preprocessor", trainer.build_preprocessor()),
        ("classifier", XGBClassifier(n_estimators=10, max_depth=3, random_state=42))
    ])
    pipe.fit(X, y)

    robustness = RobustnessTester(pipe)
    scorecard = robustness.generate_robustness_scorecard(X, y)

    assert "overall_robustness_status" in scorecard
    assert len(scorecard["missing_data_results"]) > 0
    assert len(scorecard["noise_perturbation_results"]) > 0
    assert "macro_stress_shock" in scorecard


def test_llm_evaluation_suite():
    """Verify GenAI evaluation suite (prompt sensitivity, hallucination check, consistency)."""
    engine = MockLLMEngine(seed=42, inject_hallucination_rate=0.10)
    evaluator = LLMEvaluator(llm_engine=engine)

    test_profiles = [
        {
            "customer_id": "TEST_CUST_1",
            "annual_income": 85000,
            "debt_to_income_ratio": 0.18,
            "credit_history_years": 10,
            "has_previous_default": 0,
            "delinquent_2yrs": 0
        },
        {
            "customer_id": "TEST_CUST_2",
            "annual_income": 32000,
            "debt_to_income_ratio": 0.48,
            "credit_history_years": 3,
            "has_previous_default": 1,
            "delinquent_2yrs": 2
        }
    ]

    llm_audit = evaluator.run_full_llm_evaluation(test_profiles)
    assert "overall_governance_status" in llm_audit
    assert "prompt_sensitivity" in llm_audit["components"]
    assert "hallucination_and_groundedness" in llm_audit["components"]
    assert "self_consistency" in llm_audit["components"]
    assert "safety_guardrails" in llm_audit["components"]
    assert llm_audit["components"]["safety_guardrails"]["compliance_rate_pct"] == 100.0


def test_drift_psi_calculation():
    """Verify mathematical calculation of Population Stability Index (PSI)."""
    monitor = DriftMonitor()

    # Identical distributions should have PSI close to 0.0
    expected = np.random.normal(50, 10, size=1000)
    actual_same = np.random.normal(50, 10, size=1000)
    psi_stable = monitor.calculate_psi(expected, actual_same)
    assert psi_stable < 0.10  # Must be in Stable range

    # Drastic distribution shift should trigger PSI > 0.25 (Significant Drift)
    actual_drifted = np.random.normal(75, 10, size=1000)
    psi_drifted = monitor.calculate_psi(expected, actual_drifted)
    assert psi_drifted >= 0.25  # Must trigger Critical Alert


def test_advanced_banking_metrics():
    """Verify Gini, KS statistic, Balanced Accuracy, MCC, and Calibration ECE."""
    evaluator = PerformanceEvaluator()
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.95])

    perf = evaluator.evaluate_model(y_true, y_prob)

    assert "gini_coefficient" in perf
    assert perf["gini_coefficient"] == 1.0  # Since AUC is 1.0, Gini = 2*1 - 1 = 1.0
    assert "ks_statistic" in perf
    assert perf["ks_statistic"] == 1.0
    assert "balanced_accuracy" in perf
    assert perf["balanced_accuracy"] == 1.0
    assert "mcc" in perf
    assert perf["mcc"] == 1.0
    assert "expected_calibration_error" in perf
    assert perf["expected_calibration_error"] >= 0.0


def test_multi_model_zoo_training(sample_credit_data, tmp_path):
    """Verify CreditModelTrainer trains and outputs predictions across all 5 algorithms."""
    csv_path = str(tmp_path / "temp_credit_zoo.csv")
    sample_credit_data.to_csv(csv_path, index=False)

    trainer = CreditModelTrainer(data_path=csv_path)
    splits = trainer.prepare_data(test_size=0.25, random_state=42)
    models = trainer.train_models(splits)

    expected_models = ["baseline_logistic", "random_forest", "champion_xgboost", "lightgbm", "neural_net"]
    for m_name in expected_models:
        assert m_name in models
        pipeline = models[m_name]
        preds = pipeline.predict_proba(splits["X_test"])[:, 1]
        assert len(preds) == len(splits["X_test"])
        assert np.all((preds >= 0.0) & (preds <= 1.0))
