"""
Production Monitoring & Data Drift Engine
Implements Population Stability Index (PSI), Kolmogorov-Smirnov distribution testing,
prediction drift tracking, and longitudinal performance degradation alerting.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger, get_traffic_light


class DriftMonitor:
    """Calculates feature-level and prediction-level drift using banking standards."""

    PSI_STABLE_THRESHOLD = 0.10
    PSI_WARNING_THRESHOLD = 0.25

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        num_buckets: int = 10,
        epsilon: float = 1e-4
    ) -> float:
        """
        Compute Population Stability Index (PSI) between baseline (expected)
        and production batch (actual).
        """
        expected = np.asarray(expected).astype(float)
        actual = np.asarray(actual).astype(float)

        # Drop NaNs
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantile bins based on baseline distribution
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, quantiles)
        bins = np.unique(bins)

        # Handle zero-variance edge cases
        if len(bins) < 2:
            return 0.0

        # Adjust outer edges
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calculate counts in each bucket
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = (expected_counts / len(expected)) + epsilon
        actual_pct = (actual_counts / len(actual)) + epsilon

        # Normalize to sum to 1
        expected_pct = expected_pct / np.sum(expected_pct)
        actual_pct = actual_pct / np.sum(actual_pct)

        # PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(np.round(psi_val, 4))

    def evaluate_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate PSI and KS statistic for all numerical features."""
        if features is None:
            features = baseline_df.select_dtypes(include=[np.number]).columns.tolist()

        records = []
        for feat in features:
            if feat in ["customer_id", "risk_label", "delinquent_2yrs", "has_previous_default"]:
                continue
            if feat not in current_df.columns:
                continue

            base_vals = baseline_df[feat].dropna().values
            curr_vals = current_df[feat].dropna().values

            psi = self.calculate_psi(base_vals, curr_vals)
            ks_stat, p_val = ks_2samp(base_vals, curr_vals)

            if psi < self.PSI_STABLE_THRESHOLD:
                status = "STABLE"
            elif psi < self.PSI_WARNING_THRESHOLD:
                status = "MODERATE_DRIFT"
            else:
                status = "SIGNIFICANT_DRIFT"

            records.append({
                "Feature": feat,
                "PSI": psi,
                "KS_Statistic": round(float(ks_stat), 4),
                "KS_p_value": round(float(p_val), 5),
                "Drift_Status": status,
                "Traffic_Light": get_traffic_light(status)
            })

        return pd.DataFrame(records).sort_values(by="PSI", ascending=False).reset_index(drop=True)

    def monitor_longitudinal_batches(
        self,
        baseline_df: pd.DataFrame,
        monthly_batches_df: pd.DataFrame,
        model_pipeline: Any,
        feature_cols: List[str],
        batch_col: str = "batch_month"
    ) -> Dict[str, Any]:
        """
        Track monthly progression of:
        1. Feature PSI drift
        2. Prediction distribution shift (predicted high risk rate)
        3. Realized performance decay over time
        """
        months = sorted(monthly_batches_df[batch_col].unique())
        timeline_metrics = []
        all_alerts = []

        baseline_features_df = baseline_df[feature_cols]

        for m in months:
            month_data = monthly_batches_df[monthly_batches_df[batch_col] == m].copy()
            X_curr = month_data[feature_cols]
            y_curr = month_data["risk_label"]

            # 1. Model inference
            probs = model_pipeline.predict_proba(X_curr)[:, 1]
            preds = (probs >= 0.50).astype(int)
            predicted_default_rate = float(preds.mean())

            # 2. Key feature PSI against baseline
            dti_psi = self.calculate_psi(
                baseline_df["debt_to_income_ratio"].values,
                month_data["debt_to_income_ratio"].values
            )
            income_psi = self.calculate_psi(
                baseline_df["annual_income"].values,
                month_data["annual_income"].values
            )
            max_psi = max(dti_psi, income_psi)

            # 3. Realized performance
            f1 = float(f1_score(y_curr, preds, zero_division=0))
            roc = float(roc_auc_score(y_curr, probs))
            recall = float(recall_score(y_curr, preds, zero_division=0))
            precision = float(precision_score(y_curr, preds, zero_division=0))

            # 4. Generate alerts
            if max_psi >= self.PSI_WARNING_THRESHOLD:
                all_alerts.append(f"CRITICAL: Significant feature drift in {m} (Max PSI = {max_psi:.3f} >= 0.25).")
            elif max_psi >= self.PSI_STABLE_THRESHOLD:
                all_alerts.append(f"WARNING: Moderate feature drift in {m} (Max PSI = {max_psi:.3f} >= 0.10).")

            timeline_metrics.append({
                "Batch": m,
                "Sample_Count": len(month_data),
                "Max_Feature_PSI": round(max_psi, 4),
                "DTI_PSI": round(dti_psi, 4),
                "Income_PSI": round(income_psi, 4),
                "Predicted_Default_Rate": round(predicted_default_rate, 4),
                "Realized_Default_Rate": round(float(y_curr.mean()), 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc, 4),
                "Recall": round(recall, 4),
                "Precision": round(precision, 4)
            })

        df_timeline = pd.DataFrame(timeline_metrics)
        overall_status = "FAIL" if any("CRITICAL" in a for a in all_alerts) else (
            "REVIEW_REQUIRED" if len(all_alerts) > 0 else "PASS"
        )

        return {
            "monitoring_timeline_df": df_timeline,
            "alerts_triggered": all_alerts,
            "overall_monitoring_status": overall_status,
            "traffic_light": get_traffic_light(overall_status)
        }
