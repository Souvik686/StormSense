"""Evaluation metrics for the severe-weather classification head and the
rainfall regression head, computed per lead time (never pooled across lead
times, since skill genuinely degrades with lead time and pooling would hide
that -- see docs/EVALUATION.md).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, fast: bool = False) -> dict:
    """y_true, y_prob: flat arrays of the same shape (any number of dims)."""
    y_true = y_true.ravel()
    y_prob = y_prob.ravel()
    y_pred = (y_prob >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # == POD (probability of detection)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0     # false alarm ratio
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # critical success index

    out = {"precision": precision, "recall_pod": recall, "f1": f1, "far": far, "csi": csi,
           "confusion": c, "n": int(y_true.size), "n_positive": int(y_true.sum())}
    if not fast and y_true.sum() > 0 and y_true.sum() < y_true.size:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_per_lead_time(y_true_severe: np.ndarray, y_prob_severe: np.ndarray,
                            y_true_rain: np.ndarray, y_pred_rain: np.ndarray,
                            lead_times: list[int], threshold: float | dict = 0.5) -> dict:
    """All arrays shaped (N, n_lead, H, W)."""
    from src.training.calibration import compute_calibration_metrics
    out = {}
    for i, lh in enumerate(lead_times):
        if isinstance(threshold, dict):
            thr = float(threshold.get(lh, threshold.get(int(lh), 0.5)))
        else:
            thr = float(threshold)
        cls = classification_metrics(y_true_severe[:, i], y_prob_severe[:, i], thr)
        reg = regression_metrics(y_true_rain[:, i], y_pred_rain[:, i])
        cal = compute_calibration_metrics(y_true_severe[:, i], y_prob_severe[:, i])
        out[f"lead_{lh}h"] = {**cls, **reg, "brier_score": cal["brier_score"], "ece": cal["ece"]}
    return out
