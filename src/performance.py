"""
Model Performance & Discrimination Evaluation Engine
Computes comprehensive financial credit risk metrics:
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC, PR-AUC, Gini Coefficient
- Kolmogorov-Smirnov (KS) Statistic & Optimal KS Cutoff
- Balanced Accuracy, Matthews Correlation Coefficient (MCC)
- Log-Loss / Cross-Entropy, Brier Score
- Specificity (TNR), False Discovery Rate (FDR)
- Positive & Negative Likelihood Ratios (LR+, LR-)
- Expected Calibration Error (ECE)
- Multi-Model Leaderboard Generator
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    balanced_accuracy_score, matthews_corrcoef, log_loss
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import logger, get_traffic_light


class PerformanceEvaluator:
    """Evaluates discrimination, precision-recall trade-offs, calibration, and banking score metrics."""

    def __init__(self, default_threshold: float = 0.5):
        self.default_threshold = default_threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compute performance metrics and return dictionary."""
        metrics = self.evaluate_model(y_true, y_prob, threshold=threshold)
        metrics["gini"] = metrics.get("gini_coefficient", 0.0)
        return metrics

    @staticmethod
    def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE) across confidence bins."""
        bin_limits = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            bin_low, bin_high = bin_limits[i], bin_limits[i + 1]
            mask = (y_prob >= bin_low) & (y_prob < bin_high if i < n_bins - 1 else y_prob <= bin_high)
            bin_count = np.sum(mask)
            if bin_count > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                ece += (bin_count / n_total) * np.abs(bin_acc - bin_conf)

        return float(np.round(ece, 4))

    @staticmethod
    def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
        """Compute Kolmogorov-Smirnov (KS) statistic and optimal KS separation threshold."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        ks_diff = np.abs(tpr - fpr)
        max_idx = np.argmax(ks_diff)
        ks_stat = float(ks_diff[max_idx])
        ks_cutoff = float(thresholds[max_idx]) if max_idx < len(thresholds) else 0.50
        return round(ks_stat, 4), round(ks_cutoff, 4)

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compute complete suite of discrimination, calibration, and financial risk metrics."""
        thresh = threshold if threshold is not None else self.default_threshold
        y_pred = (y_prob >= thresh).astype(int)

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Core Metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))

        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            roc_auc = 0.5
            
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc = 0.0

        brier = float(brier_score_loss(y_true, y_prob))
        loss_val = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)))

        # Banking Specific Scorecard Metrics
        gini = float(2 * roc_auc - 1)
        ks_stat, ks_cutoff = self.compute_ks_statistic(y_true, y_prob)
        ece = self.compute_expected_calibration_error(y_true, y_prob)

        # Specificity and Diagnostic Ratios
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fdr = float(fp / (fp + tp)) if (fp + tp) > 0 else 0.0
        lr_plus = float(rec / (1 - specificity)) if (1 - specificity) > 0 else 0.0
        lr_minus = float((1 - rec) / specificity) if specificity > 0 else 0.0

        # Regulatory Governance Status
        if roc_auc >= 0.85 and f1 >= 0.75 and ks_stat >= 0.50:
            status = "PASS"
        elif roc_auc >= 0.75 and f1 >= 0.65:
            status = "REVIEW_REQUIRED"
        else:
            status = "FAIL"

        return {
            "threshold": thresh,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "mcc": round(mcc, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "gini_coefficient": round(gini, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_optimal_cutoff": round(ks_cutoff, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(loss_val, 4),
            "expected_calibration_error": round(ece, 4),
            "false_discovery_rate": round(fdr, 4),
            "likelihood_ratios": {
                "positive_lr": round(lr_plus, 3),
                "negative_lr": round(lr_minus, 3)
            },
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
        """Compute ROC, PR, and KS separation curve coordinates for visualization."""
        fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
        ks_diff = np.abs(tpr - fpr)

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
            },
            "ks_curve": {
                "fpr": [round(float(x), 4) for x in fpr],
                "tpr": [round(float(x), 4) for x in tpr],
                "ks_separation": [round(float(x), 4) for x in ks_diff],
                "thresholds": [round(float(x), 4) for x in roc_thresh]
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
                "mcc": perf["mcc"],
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
        """Find optimal financial cut-off threshold balancing bad loan cost vs lost interest."""
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
        """Create structured multi-algorithm leaderboard across all financial metrics."""
        rows = []
        for name, probs in models_dict.items():
            metrics = self.evaluate_model(y_true, probs)
            rows.append({
                "Model": name,
                "ROC-AUC": metrics["roc_auc"],
                "Gini": metrics["gini_coefficient"],
                "KS Stat": metrics["ks_statistic"],
                "F1 Score": metrics["f1_score"],
                "MCC": metrics["mcc"],
                "Accuracy": metrics["accuracy"],
                "Balanced Acc": metrics["balanced_accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Specificity": metrics["specificity"],
                "Brier": metrics["brier_score"],
                "Log Loss": metrics["log_loss"],
                "ECE": metrics["expected_calibration_error"],
                "Governance Status": metrics["governance_status"]
            })
        df_leaderboard = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
        return df_leaderboard
