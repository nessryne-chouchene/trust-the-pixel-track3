"""Metriques d'evaluation (v0.2). Valables uniquement avec des labels fiables."""
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


def auc_score(y_true, scores) -> float:
    """Aire sous la courbe ROC (statistique de Mann-Whitney) ; 0.5 = hasard.
    NaN si une des deux classes est absente."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # rangs moyens en cas d'egalite
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    r_pos = r[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def recall_at_false_alarm(y_true, scores, max_far: float) -> dict:
    """Rappel obtenu en fixant le seuil pour que le taux de fausses alarmes
    reste <= max_far (seuil calcule sur les memes donnees : a utiliser sur la
    validation pour choisir le seuil, sur le test seulement pour rapporter)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return {"threshold": float("nan"), "recall": float("nan"), "far": float("nan")}
    k = int(np.floor(max_far * len(neg)))
    thr = float(np.nextafter(neg[k], np.inf)) if k < len(neg) else float(neg[-1])
    m = classification_metrics(y, s, thr)
    return {"threshold": thr, "recall": m["recall"], "far": m["false_alarm_rate"]}
