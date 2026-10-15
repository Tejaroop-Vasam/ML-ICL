"""Evaluation metrics for ICL classification tasks."""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive classification performance metrics.

    Args:
        y_true: Ground truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Ordered list of class string names.

    Returns:
        Dictionary with accuracy, macro_f1, weighted_f1, per_class metrics, and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }
