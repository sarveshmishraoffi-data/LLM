"""
Error Analysis & Diagnostic Engine
Deconstructs False Positives vs False Negatives, isolates high-confidence mistakes,
and profiles cohort slices where model vulnerabilities exist.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger


class ErrorAnalyzer:
    """Performs granular error profiling and sub-population diagnostic slicing."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def segment_errors(
        self,
        df: pd.DataFrame,
        y_true_col: str = "risk_label",
        prob_col: str = "pred_prob"
    ) -> pd.DataFrame:
        """Add error categorization flags to evaluation dataframe."""
        analyzed_df = df.copy()
        analyzed_df["pred_label"] = (analyzed_df[prob_col] >= self.threshold).astype(int)

        conditions = [
            (analyzed_df[y_true_col] == 0) & (analyzed_df["pred_label"] == 0),
            (analyzed_df[y_true_col] == 0) & (analyzed_df["pred_label"] == 1),
            (analyzed_df[y_true_col] == 1) & (analyzed_df["pred_label"] == 0),
            (analyzed_df[y_true_col] == 1) & (analyzed_df["pred_label"] == 1)
        ]
        choices = ["True Negative", "False Positive", "False Negative", "True Positive"]
        analyzed_df["error_category"] = np.select(conditions, choices, default="Unknown")
        
        # Absolute error margin
        analyzed_df["error_margin"] = np.abs(analyzed_df[y_true_col] - analyzed_df[prob_col])
        return analyzed_df

    def get_high_confidence_errors(
        self,
        analyzed_df: pd.DataFrame,
        top_n: int = 10,
        prob_col: str = "pred_prob"
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract highest confidence failures:
        1. High-confidence False Negatives (model is sure customer is safe, but defaults)
        2. High-confidence False Positives (model is sure customer is bad, but repays)
        """
        fn_df = analyzed_df[analyzed_df["error_category"] == "False Negative"]
        fp_df = analyzed_df[analyzed_df["error_category"] == "False Positive"]

        # Sort FN by lowest predicted default probability (most confident in approval)
        high_conf_fn = fn_df.sort_values(by=prob_col, ascending=True).head(top_n)
        
        # Sort FP by highest predicted default probability (most confident in rejection)
        high_conf_fp = fp_df.sort_values(by=prob_col, ascending=False).head(top_n)

        return {
            "high_confidence_false_negatives": high_conf_fn,
            "high_confidence_false_positives": high_conf_fp
        }

    def analyze_feature_slices(
        self,
        analyzed_df: pd.DataFrame,
        y_true_col: str = "risk_label"
    ) -> Dict[str, pd.DataFrame]:
        """Evaluate error rates across financial feature slices (income, DTI, credit history)."""
        df = analyzed_df.copy()
        slice_reports = {}

        # 1. Debt-to-Income (DTI) Slices
        if "debt_to_income_ratio" in df.columns:
            df["dti_bucket"] = pd.cut(
                df["debt_to_income_ratio"],
                bins=[-np.inf, 0.20, 0.35, 0.50, np.inf],
                labels=["Low (<20%)", "Moderate (20-35%)", "High (35-50%)", "Severe (>50%)"]
            )
            slice_reports["dti_slices"] = self._compute_slice_metrics(df, "dti_bucket", y_true_col)

        # 2. Income Slices
        if "annual_income" in df.columns:
            df["income_bucket"] = pd.cut(
                df["annual_income"],
                bins=[-np.inf, 35000, 60000, 100000, np.inf],
                labels=["Lower (<$35k)", "Middle ($35k-$60k)", "Upper-Middle ($60k-$100k)", "Affluent (>$100k)"]
            )
            slice_reports["income_slices"] = self._compute_slice_metrics(df, "income_bucket", y_true_col)

        # 3. Credit History Length Slices
        if "credit_history_years" in df.columns:
            df["history_bucket"] = pd.cut(
                df["credit_history_years"],
                bins=[-np.inf, 4, 10, 20, np.inf],
                labels=["Thin File (<4 yrs)", "Establishing (4-10 yrs)", "Mature (10-20 yrs)", "Established (>20 yrs)"]
            )
            slice_reports["history_slices"] = self._compute_slice_metrics(df, "history_bucket", y_true_col)

        return slice_reports

    def _compute_slice_metrics(self, df: pd.DataFrame, group_col: str, y_true_col: str) -> pd.DataFrame:
        """Helper to calculate error rates within a given category slice."""
        records = []
        for group_name, group_data in df.groupby(group_col, observed=True):
            n = len(group_data)
            if n == 0:
                continue
            fp = int((group_data["error_category"] == "False Positive").sum())
            fn = int((group_data["error_category"] == "False Negative").sum())
            total_errors = fp + fn
            error_rate = total_errors / n
            
            # Ground truth defaults in slice
            defaults = int((group_data[y_true_col] == 1).sum())
            default_rate = defaults / n if n > 0 else 0.0

            records.append({
                "Slice": str(group_name),
                "Sample Count": n,
                "Defaults": defaults,
                "Default Rate": round(default_rate, 4),
                "False Positives": fp,
                "False Negatives": fn,
                "Total Errors": total_errors,
                "Error Rate": round(error_rate, 4)
            })
        return pd.DataFrame(records)

    def generate_diagnostic_narrative(self, slice_reports: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate human-readable diagnostic conclusions for audit reports."""
        insights = []
        if "dti_slices" in slice_reports:
            dti_df = slice_reports["dti_slices"]
            highest_err_dti = dti_df.sort_values(by="Error Rate", ascending=False).iloc[0]
            insights.append(
                f"Applicants in the '{highest_err_dti['Slice']}' DTI cohort exhibit the highest error rate "
                f"({highest_err_dti['Error Rate'] * 100:.1f}%), with {highest_err_dti['False Negatives']} False Negatives."
            )

        if "history_slices" in slice_reports:
            hist_df = slice_reports["history_slices"]
            thin_file = hist_df[hist_df["Slice"].str.contains("Thin File", na=False)]
            if not thin_file.empty:
                tf_rate = thin_file.iloc[0]["Error Rate"]
                insights.append(
                    f"Thin-file applicants (<4 years history) experience an error rate of {tf_rate * 100:.1f}%, "
                    f"indicating higher epistemic uncertainty when historical repayment signals are limited."
                )

        return insights
