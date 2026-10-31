"""Evaluation de la robustesse aux attaques d'evasion (v0.1).

Charge un ForensicUNet entraine (outputs/forensic_unet.pt), applique chaque
operation de blanchiment de src/forensics/evasion.py a TOUTES les images de
test, et mesure AUC, rappel, fausses alarmes et IoU. Le seuil reste celui
choisi sur la validation propre : c'est le cas reel, l'attaquant ne previent pas.
Ajoute aussi l'attaque adaptative 'feather' (bords de collage adoucis).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.forensics.evasion import LAUNDERING, feather, launder, random_launder
from src.forensics.signals import detector_inputs
from src.models.forensic_unet import load_detector
from src.models.evaluation import choose_threshold
from src.models.train_forensic import predict_all
from src.utils.metrics import (auc_score, classification_metrics, recall_at_false_alarm,
                               segmentation_metrics)


def load_model(path: str):
    return load_detector(path)


def _inputs(model, images):
    return detector_inputs(images, getattr(model, "input_channels", None))


def _metrics(labels, scores, probs, masks, threshold) -> dict:
    m = classification_metrics(labels, scores, threshold)
    pos = np.where(labels == 1)[0]
    iou = float(np.mean([segmentation_metrics(probs[i] >= 0.5, masks[i])["iou"] for i in pos])) if len(pos) else float("nan")
    return {"auc": auc_score(labels, scores), "recall": m["recall"],
            "false_alarm_rate": m["false_alarm_rate"], "iou": iou}


def mixed_threshold(model, data: dict, seed: int = 0, max_far: float | None = None) -> float:
    """Seuil choisi sur la VALIDATION propre + une copie blanchie au hasard.
    En exploitation, des images authentiques sont legitimement recompressees ou
    redimensionnees : calibrer sur ce melange limite les fausses alarmes.
    max_far=None : meilleur F1 ; max_far=0.05 : plus petit seuil gardant au plus
    5 % de fausses alarmes sur ce melange (politique operationnelle)."""
    va = np.where(data["split"] == "val")[0]
    rng = np.random.default_rng(seed)
    laundered = np.stack([random_launder(img, rng) for img in data["images"][va]])
    s_clean, _ = predict_all(model, _inputs(model, data["images"][va]))
    s_laund, _ = predict_all(model, _inputs(model, laundered))
    y, sc = np.concatenate([data["labels"][va]] * 2), np.concatenate([s_clean, s_laund])
    if max_far is None:
        return choose_threshold(y, sc)
    return float(recall_at_false_alarm(y, sc, max_far)["threshold"])


def run(npz: str = "data/processed/eurosat_tampered.npz", model_path: str = "outputs/forensic_unet.pt",
        out: str = "outputs/evasion_results.json", seed: int = 0, calibration: str = "clean") -> dict:
    model, threshold = load_model(model_path)
    with np.load(npz) as z:
        data = {k: z[k] for k in z.files}
    if calibration == "mixed":
        threshold = mixed_threshold(model, data, seed)
    elif calibration == "far5":
        threshold = mixed_threshold(model, data, seed, max_far=0.05)
    elif calibration != "clean":
        raise ValueError("calibration : 'clean', 'mixed' ou 'far5'")
    if calibration != "clean":   # modele pret pour l'exploitation, avec ce seuil
        ckpt = torch.load(model_path, map_location="cpu")
        ckpt.update({"threshold": threshold, "calibration": calibration})
        torch.save(ckpt, Path(model_path).with_name(f"{Path(model_path).stem}_{calibration}.pt"))
    te = np.where(data["split"] == "test")[0]
    images, labels, masks = data["images"][te], data["labels"][te], data["masks"][te]
    attacks, sources = data["attack"][te], data["source"][te]

    rows = {}
    for name in LAUNDERING:
        x = _inputs(model, launder(images, name, seed))
        s, p = predict_all(model, x)
        rows[name] = _metrics(labels, s, p, masks, threshold)

    # Attaque adaptative : collages aux bords adoucis (authentiques inchangees).
    authentic_of = {int(src): images[i] for i, src in enumerate(sources) if attacks[i] == "none"}
    sel = np.where((attacks == "none") | (attacks == "splice"))[0]
    adv = np.stack([images[i] if attacks[i] == "none"
                    else feather(images[i], authentic_of[int(sources[i])], masks[i]) for i in sel])
    s, p = predict_all(model, _inputs(model, adv))
    rows["adaptive_feathered_splice"] = _metrics(labels[sel], s, p, masks[sel], threshold)
    s0, p0 = predict_all(model, _inputs(model, images[sel]))
    rows["splice_only_reference"] = _metrics(labels[sel], s0, p0, masks[sel], threshold)

    res = {"threshold": threshold, "calibration": calibration, "n_test": int(len(te)), "results": rows,
           "warning": "Falsifications et blanchiments synthetiques : resultats limites a ce cadre."}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def print_table(res: dict) -> None:
    cal = {"clean": "validation propre", "mixed": "validation propre + blanchie (meilleur F1)",
           "far5": "validation propre + blanchie (au plus 5 % de fausses alarmes)"}[res["calibration"]]
    print(f"\nRobustesse (seuil fixe = {res['threshold']:.2f}, choisi sur {cal})")
    print(f"{'operation':28s} {'AUC':>6s} {'rappel':>7s} {'fausses al.':>11s} {'IoU':>6s}")
    for name, m in res["results"].items():
        print(f"{name:28s} {m['auc']:6.3f} {m['recall']:7.3f} {m['false_alarm_rate']:11.3f} {m['iou']:6.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/forensic_unet.pt")
    ap.add_argument("--out", default="outputs/evasion_results.json")
    ap.add_argument("--calibration", default="clean", choices=["clean", "mixed", "far5"],
                    help="mixed : meilleur F1 sur validation propre + blanchie ; "
                         "far5 : au plus 5 %% de fausses alarmes sur ce melange")
    a = ap.parse_args()
    print_table(run(model_path=a.model, out=a.out, calibration=a.calibration))
