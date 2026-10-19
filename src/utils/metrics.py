"""Metriques d'evaluation (v0.1). Valables uniquement avec des labels fiables."""
from __future__ import annotations

import numpy as np


def _div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def classification_metrics(y_true, y_score, threshold: float) -> dict:
    """Precision, rappel, F1 et taux de fausses alarmes a un seuil donne."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return {
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "false_alarm_rate": _div(fp, fp + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def segmentation_metrics(pred, gt) -> dict:
    """IoU et Dice entre deux masques binaires. Deux masques vides -> 1.0."""
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    if union == 0:
        return {"iou": 1.0, "dice": 1.0}
    return {"iou": inter / union, "dice": 2 * inter / int(p.sum() + g.sum())}
