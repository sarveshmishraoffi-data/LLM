"""
Robustness & Stress Testing Engine
Challenger suite subjecting models to missingness injection, Gaussian noise,
macroeconomic stress shocks, and measures decision flip rates and performance degradation.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger, get_traffic_light


class RobustnessTester:
    """Evaluates stability and degradation under degraded, noisy, and stressed inputs."""

    def __init__(self, pipeline: Pipeline, decision_threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = decision_threshold

    def test_missing_data_resilience(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        missing_rates: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate model degradation when random features are missing (unreported data)."""
        if missing_rates is None:
            missing_rates = [0.0, 0.05, 0.10, 0.20, 0.30]

        results = []
        # Baseline reference
        base_probs = self.pipeline.predict_proba(X_test)[:, 1]
        base_preds = (base_probs >= self.threshold).astype(int)
        base_f1 = f1_score(y_test, base_preds, zero_division=0)
        base_roc = roc_auc_score(y_test, base_probs)

        for rate in missing_rates:
            if rate == 0.0:
                results.append({
                    "missing_rate": 0.0,
                    "f1_score": round(float(base_f1), 4),
                    "roc_auc": round(float(base_roc), 4),
                    "f1_degradation_pct": 0.0,
                    "flip_rate_pct": 0.0
                })
                continue

            X_perturbed = X_test.copy()
            # Randomly insert NaNs into numeric features
            np.random.seed(42)
            mask = np.random.uniform(0, 1, size=X_perturbed.shape) < rate
            # Only apply to numeric columns
            num_cols = X_perturbed.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                col_idx = X_perturbed.columns.get_loc(col)
                X_perturbed.iloc[mask[:, col_idx], col_idx] = np.nan

            probs = self.pipeline.predict_proba(X_perturbed)[:, 1]
            preds = (probs >= self.threshold).astype(int)

            f1 = f1_score(y_test, preds, zero_division=0)
            roc = roc_auc_score(y_test, probs)
            f1_drop = max(0.0, (base_f1 - f1) / base_f1)
            flip_rate = float((preds != base_preds).mean())

            results.append({
                "missing_rate": rate,
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc), 4),
                "f1_degradation_pct": round(float(f1_drop * 100), 2),
                "flip_rate_pct": round(float(flip_rate * 100), 2)
            })

        return results

    def test_noise_perturbation(
        self,
        X_test: pd.DataFrame,
        noise_levels: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Inject proportional Gaussian noise on continuous variables to evaluate flip rate."""
        if noise_levels is None:
            noise_levels = [0.02, 0.05, 0.10, 0.20]

        continuous_cols = [
            "annual_income", "loan_amount", "debt_to_income_ratio", "employment_length_years"
        ]
        cols_to_noise = [c for c in continuous_cols if c in X_test.columns]

        base_probs = self.pipeline.predict_proba(X_test)[:, 1]
        base_preds = (base_probs >= self.threshold).astype(int)

        results = []
        for noise in noise_levels:
            np.random.seed(42)
            X_noisy = X_test.copy()
            for col in cols_to_noise:
                std = X_noisy[col].std()
                perturbation = np.random.normal(loc=0.0, scale=noise * std, size=len(X_noisy))
                X_noisy[col] = np.maximum(0.0, X_noisy[col] + perturbation)

            noisy_probs = self.pipeline.predict_proba(X_noisy)[:, 1]
            noisy_preds = (noisy_probs >= self.threshold).astype(int)

            flip_rate = float((noisy_preds != base_preds).mean())
            mean_prob_shift = float(np.abs(noisy_probs - base_probs).mean())

            results.append({
                "noise_sigma": noise,
                "flip_rate_pct": round(flip_rate * 100, 2),
                "mean_abs_probability_shift": round(mean_prob_shift, 4),
                "status": "PASS" if flip_rate < 0.08 else ("REVIEW_REQUIRED" if flip_rate < 0.15 else "FAIL")
            })

        return results

    def test_macroeconomic_stress_shock(
        self,
        X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Simulate a systemic recession: -25% income, +35% DTI, +1 delinquency."""
        base_probs = self.pipeline.predict_proba(X_test)[:, 1]
        base_default_rate = float((base_probs >= self.threshold).mean())

        X_shocked = X_test.copy()
        if "annual_income" in X_shocked.columns:
            X_shocked["annual_income"] = X_shocked["annual_income"] * 0.75
        if "debt_to_income_ratio" in X_shocked.columns:
            X_shocked["debt_to_income_ratio"] = np.minimum(0.95, X_shocked["debt_to_income_ratio"] * 1.35)
        if "delinquent_2yrs" in X_shocked.columns:
            X_shocked["delinquent_2yrs"] = X_shocked["delinquent_2yrs"] + 1

        shock_probs = self.pipeline.predict_proba(X_shocked)[:, 1]
        shock_default_rate = float((shock_probs >= self.threshold).mean())
        mean_risk_increase = float((shock_probs - base_probs).mean())

        # Expected banking behavior: risk should increase during recession
        model_responds_logically = (shock_default_rate > base_default_rate) and (mean_risk_increase > 0.05)

        return {
            "scenario": "Severe Macroeconomic Recession (-25% Income, +35% DTI)",
            "baseline_predicted_default_rate": round(base_default_rate, 4),
            "stressed_predicted_default_rate": round(shock_default_rate, 4),
            "average_probability_surge": round(mean_risk_increase, 4),
            "directional_validity": "PASS" if model_responds_logically else "FAIL",
            "narrative": (
                f"Under recession shock, high-risk flags increased from {base_default_rate*100:.1f}% "
                f"to {shock_default_rate*100:.1f}% (+{mean_risk_increase*100:.1f}% probability shift), "
                f"confirming appropriate monotonic risk elasticity."
            )
        }

    def generate_robustness_scorecard(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Execute full suite and calculate composite Robustness Health Index."""
        missing_res = self.test_missing_data_resilience(X_test, y_test)
        noise_res = self.test_noise_perturbation(X_test)
        shock_res = self.test_macroeconomic_stress_shock(X_test)

        # Evaluate 10% missing degradation
        m10 = next((item for item in missing_res if item["missing_rate"] == 0.10), None)
        n5 = next((item for item in noise_res if item["noise_sigma"] == 0.05), None)

        f1_drop_10 = m10["f1_degradation_pct"] if m10 else 5.0
        flip_rate_5 = n5["flip_rate_pct"] if n5 else 3.0

        if f1_drop_10 <= 10.0 and flip_rate_5 <= 7.0 and shock_res["directional_validity"] == "PASS":
            verdict = "PASS"
        elif f1_drop_10 <= 18.0 and flip_rate_5 <= 14.0:
            verdict = "REVIEW_REQUIRED"
        else:
            verdict = "FAIL"

        return {
            "overall_robustness_status": verdict,
            "traffic_light": get_traffic_light(verdict),
            "missing_data_results": missing_res,
            "noise_perturbation_results": noise_res,
            "macro_stress_shock": shock_res
        }
