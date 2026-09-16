"""
Model Performance & Discrimination Evaluation Engine
Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score,
and evaluates optimal decision thresholds for credit risk policy.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger, get_traffic_light


class PerformanceEvaluator:
    """Evaluates discrimination, precision-recall trade-offs, and probability calibration."""

    def __init__(self, default_threshold: float = 0.5):
        self.default_threshold = default_threshold

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compute primary credit risk discrimination metrics."""
        thresh = threshold if threshold is not None else self.default_threshold
        y_pred = (y_prob >= thresh).astype(int)

        # Basic confusion elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            roc_auc = 0.5
            
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc = 0.0

        brier = float(brier_score_loss(y_true, y_prob))

        # Regulatory / Business Governance Status
        # For Tier-1 credit models: ROC-AUC >= 0.75 is typical threshold
        if roc_auc >= 0.82 and f1 >= 0.72:
            status = "PASS"
        elif roc_auc >= 0.75 and f1 >= 0.65:
            status = "REVIEW_REQUIRED"
        else:
            status = "FAIL"

        return {
            "threshold": thresh,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "rates": {
                "approval_rate": round(float((y_pred == 0).mean()), 4),
                "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
                "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4)
            },
            "governance_status": status,
            "traffic_light": get_traffic_light(status)
        }

    def compute_curves(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Compute ROC and PR curve coordinates for visualization."""
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)

        return {
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr],
                "tpr": [round(float(x), 4) for x in tpr],
                "thresholds": [round(float(x), 4) for x in roc_thresh]
            },
            "pr_curve": {
                "precision": [round(float(x), 4) for x in prec],
                "recall": [round(float(x), 4) for x in rec],
                "thresholds": [round(float(x), 4) for x in pr_thresh]
            }
        }

    def evaluate_threshold_tradeoffs(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Sweep decision thresholds to analyze trade-offs between credit loss and approval volume."""
        if thresholds is None:
            thresholds = [round(x, 2) for x in np.arange(0.10, 0.95, 0.05)]

        results = []
        for t in thresholds:
            perf = self.evaluate_model(y_true, y_prob, threshold=t)
            results.append({
                "threshold": t,
                "precision": perf["precision"],
                "recall": perf["recall"],
                "f1_score": perf["f1_score"],
                "approval_rate": perf["rates"]["approval_rate"],
                "false_positive_rate": perf["rates"]["false_positive_rate"],
                "false_negative_rate": perf["rates"]["false_negative_rate"]
            })
        return results

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cost_false_negative: float = 5.0,
        cost_false_positive: float = 1.0
    ) -> Dict[str, Any]:
        """
        Find optimal financial cut-off threshold balancing cost of bad loan (FN)
        vs cost of lost interest/customer friction (FP).
        """
        thresholds = np.linspace(0.10, 0.90, 81)
        best_cost = float("inf")
        best_thresh = 0.50

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
            if total_cost < best_cost:
                best_cost = total_cost
                best_thresh = t

        optimal_perf = self.evaluate_model(y_true, y_prob, threshold=best_thresh)
        return {
            "optimal_threshold": round(float(best_thresh), 3),
            "minimized_financial_loss": round(float(best_cost), 2),
            "cost_assumptions": {
                "cost_false_negative": cost_false_negative,
                "cost_false_positive": cost_false_positive
            },
            "performance_at_optimal": optimal_perf
        }

    def compare_models(
        self,
        y_true: np.ndarray,
        models_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Create structured comparison table between baseline and champion models."""
        rows = []
        for name, probs in models_dict.items():
            metrics = self.evaluate_model(y_true, probs)
            rows.append({
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
                "Brier Score": metrics["brier_score"],
                "Status": metrics["governance_status"]
            })
        return pd.DataFrame(rows)
