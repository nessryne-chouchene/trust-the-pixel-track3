"""Evaluation commune aux detecteurs (v0.1) : global, par type d'attaque,
attaque tenue en reserve, fausses alarmes."""
from __future__ import annotations

import numpy as np

from src.utils.metrics import (auc_score, classification_metrics,
                               recall_at_false_alarm, segmentation_metrics)


def choose_threshold(y_true, scores) -> float:
    """Seuil maximisant le F1 sur la VALIDATION. En cas d'egalite (plage de
    seuils optimaux), on prend le milieu de la plage : plus de marge des deux
    cotes, donc un seuil moins fragile quand les donnees changent."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    f1 = np.array([classification_metrics(y_true, scores, float(t))["f1"] for t in grid])
    f1 = np.where(np.isnan(f1), -1.0, f1)
    best = grid[f1 >= f1.max() - 1e-12]
    return float(best[len(best) // 2])


def evaluate(labels, attacks, scores, mask_probs, gt_masks, threshold: float,
             seen_attacks, mask_threshold: float = 0.5) -> dict:
    """labels (N,), attacks (N,) ('none' = authentique), scores (N,) dans [0, 1],
    mask_probs et gt_masks (N, H, W). Le seuil vient de la VALIDATION."""
    labels = np.asarray(labels).astype(int)
    attacks = np.asarray(attacks).astype(str)
    scores = np.asarray(scores, dtype=float)
    overall = classification_metrics(labels, scores, threshold)
    overall["auc"] = auc_score(labels, scores)
    overall["recall_at_far_5pct"] = recall_at_false_alarm(labels, scores, 0.05)["recall"]

    seg_all = [segmentation_metrics(p >= mask_threshold, g)
               for p, g, y in zip(mask_probs, gt_masks, labels) if y == 1]
    overall["iou"] = float(np.mean([s["iou"] for s in seg_all])) if seg_all else float("nan")
    overall["dice"] = float(np.mean([s["dice"] for s in seg_all])) if seg_all else float("nan")
    auth = labels == 0
    overall["authentic_mask_false_pixels"] = float(
        np.mean([(p >= mask_threshold).mean() for p in np.asarray(mask_probs)[auth]])) if auth.any() else float("nan")

    per_attack = {}
    for a in sorted(set(attacks) - {"none"}):
        sel = (attacks == "none") | (attacks == a)
        pos = attacks == a
        m = classification_metrics(labels[sel], scores[sel], threshold)
        segs = [segmentation_metrics(p >= mask_threshold, g)
                for p, g in zip(np.asarray(mask_probs)[pos], np.asarray(gt_masks)[pos])]
        per_attack[a] = {
            "seen_in_training": a in tuple(seen_attacks),
            "n": int(pos.sum()),
            "auc": auc_score(labels[sel], scores[sel]),
            "recall": m["recall"],
            "f1": m["f1"],
            "iou": float(np.mean([s["iou"] for s in segs])),
            "dice": float(np.mean([s["dice"] for s in segs])),
        }
    return {"threshold": threshold, "overall": overall, "per_attack": per_attack}


def print_summary(res: dict, title: str) -> None:
    o = res["overall"]
    print(f"\n=== {title} : RESULTATS TEST (seuil choisi sur la validation) ===")
    print(f"seuil={res['threshold']:.2f}")
    print(f"AUC globale={o['auc']:.3f} (hasard = 0.5)   rappel a 5 % de fausses alarmes={o['recall_at_far_5pct']:.3f}")
    print(f"taux de fausses alarmes={o['false_alarm_rate']:.3f}   precision={o['precision']:.3f}"
          f"   rappel={o['recall']:.3f}   F1={o['f1']:.3f}")
    print(f"localisation : IoU={o['iou']:.3f}  Dice={o['dice']:.3f}"
          f"   pixels faussement marques sur images authentiques={o['authentic_mask_false_pixels']:.3f}")
    print(f"\n{'attaque':12s} {'vue ?':6s} {'AUC':>6s} {'rappel':>7s} {'IoU':>6s} {'Dice':>6s}")
    for a, m in res["per_attack"].items():
        print(f"{a:12s} {'oui' if m['seen_in_training'] else 'NON':6s} "
              f"{m['auc']:6.3f} {m['recall']:7.3f} {m['iou']:6.3f} {m['dice']:6.3f}")
    print("\nATTENTION : falsifications synthetiques, resultats limites a ce cadre.")
