"""
Fairness & Responsible AI Audit Engine
Assesses demographic parity, disparate impact (EEOC Four-Fifths Rule),
equal opportunity, and equalized odds using Fairlearn and banking compliance standards.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame, selection_rate, false_positive_rate, false_negative_rate,
    demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger, get_traffic_light


class FairnessAuditor:
    """Performs algorithmic bias audits across sensitive/protected demographics."""

    def __init__(self, four_fifths_threshold: float = 0.80):
        self.four_fifths_threshold = four_fifths_threshold

    def audit_attribute(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_feature: pd.Series,
        attribute_name: str = "gender"
    ) -> Dict[str, Any]:
        """
        Audit a specific protected attribute (e.g., gender, age_group).
        Calculates selection rates, disparate impact, equal opportunity, and error disparities.
        """
        # Note: In credit risk, an 'approval' is when risk_label == 0 (Low Risk).
        # We audit both the raw default prediction (risk_label == 1) and approval rate (1 - risk_label).
        approval_true = (y_true == 0).astype(int)
        approval_pred = (y_pred == 0).astype(int)

        metrics = {
            "accuracy": accuracy_score,
            "precision": lambda y, p: precision_score(y, p, zero_division=0),
            "recall": lambda y, p: recall_score(y, p, zero_division=0),
            "f1": lambda y, p: f1_score(y, p, zero_division=0),
            "approval_rate": selection_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate
        }

        # Build Fairlearn MetricFrame on approval decisions
        frame = MetricFrame(
            metrics=metrics,
            y_true=approval_true,
            y_pred=approval_pred,
            sensitive_features=sensitive_feature
        )

        by_group_df = frame.by_group.round(4)
        
        # Disparate Impact (Four-Fifths Rule) on approvals
        dp_ratio = float(demographic_parity_ratio(
            approval_true, approval_pred, sensitive_features=sensitive_feature
        ))
        dp_diff = float(demographic_parity_difference(
            approval_true, approval_pred, sensitive_features=sensitive_feature
        ))
        
        eq_odds_diff = float(equalized_odds_difference(
            approval_true, approval_pred, sensitive_features=sensitive_feature
        ))

        # Regulatory rule evaluation
        # Four-Fifths rule: ratio must be >= 0.80
        if dp_ratio >= self.four_fifths_threshold and eq_odds_diff <= 0.10:
            status = "PASS"
            finding = f"Disparate impact ratio ({dp_ratio:.3f}) complies with the 80% Four-Fifths rule."
        elif dp_ratio >= 0.70:
            status = "REVIEW_REQUIRED"
            finding = f"Marginal disparate impact ratio ({dp_ratio:.3f}) detected. Requires qualitative review."
        else:
            status = "FAIL"
            finding = f"Disparate impact ratio ({dp_ratio:.3f}) falls below 80% statutory guidance threshold."

        group_summary = {}
        for group in by_group_df.index:
            group_summary[str(group)] = {
                "sample_count": int((sensitive_feature == group).sum()),
                "approval_rate": float(by_group_df.loc[group, "approval_rate"]),
                "accuracy": float(by_group_df.loc[group, "accuracy"]),
                "recall": float(by_group_df.loc[group, "recall"]),
                "false_positive_rate": float(by_group_df.loc[group, "false_positive_rate"]),
                "false_negative_rate": float(by_group_df.loc[group, "false_negative_rate"])
            }

        return {
            "attribute_name": attribute_name,
            "overall_governance_status": status,
            "traffic_light": get_traffic_light(status),
            "finding_summary": finding,
            "disparate_impact_ratio": round(dp_ratio, 4),
            "demographic_parity_difference": round(dp_diff, 4),
            "equalized_odds_difference": round(eq_odds_diff, 4),
            "group_metrics": group_summary,
            "by_group_dataframe": by_group_df
        }

    def run_full_bias_audit(
        self,
        df: pd.DataFrame,
        y_true_col: str = "risk_label",
        prob_col: str = "pred_prob",
        threshold: float = 0.5,
        protected_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute fairness audit across all configured protected attributes."""
        if protected_attributes is None:
            protected_attributes = ["gender", "age_group"]

        y_true = df[y_true_col]
        y_pred = (df[prob_col] >= threshold).astype(int)

        results = {}
        overall_verdict = "PASS"

        for attr in protected_attributes:
            if attr in df.columns:
                audit = self.audit_attribute(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_feature=df[attr],
                    attribute_name=attr
                )
                results[attr] = audit
                if audit["overall_governance_status"] == "FAIL":
                    overall_verdict = "FAIL"
                elif audit["overall_governance_status"] == "REVIEW_REQUIRED" and overall_verdict != "FAIL":
                    overall_verdict = "REVIEW_REQUIRED"

        return {
            "audit_type": "Fair Lending & Algorithmic Bias Audit",
            "overall_status": overall_verdict,
            "traffic_light": get_traffic_light(overall_verdict),
            "audited_attributes": results
        }
