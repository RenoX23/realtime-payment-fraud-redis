"""Comprehensive model evaluation metrics tailored for extreme class imbalance and financial fraud risk."""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)


class ModelEvaluator:
    """Rigorous evaluation suite rejecting accuracy in favor of PR-AUC, Precision@K, and cost utility."""

    def __init__(self, cost_fn: float = 250.0, cost_fp: float = 12.0):
        """
        Args:
            cost_fn: Dollar cost of False Negative (undetected fraud, chargeback, penalty).
            cost_fp: Dollar cost of False Positive (false decline, customer friction, verification overhead).
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Compute complete metric suite including PR-AUC, ROC-AUC, and financial cost."""
        y_true = np.asarray(y_true).astype(int)
        y_probs = np.asarray(y_probs).astype(float)
        y_pred = (y_probs >= threshold).astype(int)

        pr_auc = float(average_precision_score(y_true, y_probs))
        roc_auc = float(roc_auc_score(y_true, y_probs))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_probs))

        # Expected financial loss
        financial_loss = float((fn * self.cost_fn) + (fp * self.cost_fp))

        return {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "threshold": round(threshold, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 5),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "financial_loss_usd": round(financial_loss, 2),
            "fraud_prevalence": round(float(np.mean(y_true)), 5),
        }

    def find_optimal_threshold_f1(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray
    ) -> Tuple[float, float]:
        """Find decision threshold that maximizes F1 score."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        # Avoid zero division
        numerator = 2 * (precisions * recalls)
        denominator = precisions + recalls
        f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        return round(best_threshold, 4), round(best_f1, 4)

    def find_optimal_threshold_cost(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        steps: int = 200
    ) -> Tuple[float, float]:
        """Find decision threshold that minimizes total financial loss ($)."""
        candidate_thresholds = np.linspace(0.01, 0.99, steps)
        best_loss = float("inf")
        best_thresh = 0.50

        y_true = np.asarray(y_true).astype(int)
        y_probs = np.asarray(y_probs).astype(float)

        for thresh in candidate_thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            loss = (fn * self.cost_fn) + (fp * self.cost_fp)
            if loss < best_loss:
                best_loss = loss
                best_thresh = thresh

        return round(float(best_thresh), 4), round(float(best_loss), 2)

    def compute_precision_at_k(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        k: int = 100
    ) -> float:
        """Calculate Precision@Top-K: fraction of top K highest-risk transactions that are actual fraud."""
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)

        k = min(k, len(y_probs))
        top_k_indices = np.argsort(y_probs)[-k:]
        actual_frauds_in_top_k = np.sum(y_true[top_k_indices])
        return round(float(actual_frauds_in_top_k / k), 4)

    def compute_recall_at_precision(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        target_precision: float = 0.95
    ) -> float:
        """Calculate highest recall achievable while maintaining at least target_precision."""
        precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
        # Find indices where precision >= target_precision
        valid_indices = np.where(precisions >= target_precision)[0]
        if len(valid_indices) == 0:
            return 0.0
        return round(float(np.max(recalls[valid_indices])), 4)
