"""Entrainement des tetes sur donnees SYNTHETIQUES (v0.1).

Verification technique du pipeline uniquement : la tache synthetique est triviale,
ses scores n'ont AUCUNE valeur forensique.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.features.efficientnet_features import EfficientNetB0Features
from src.models.classification_head import ClassificationHead
from src.models.localization_head import LocalizationHead
from src.models.losses import LocalizationLoss
from src.utils.metrics import classification_metrics, segmentation_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_group(groups, seed: int = 0, ratios=(0.6, 0.2, 0.2)):
    """Indices train/val/test sans fuite : un groupe entier va dans un seul ensemble."""
    uniq = sorted(set(groups))
    n = len(uniq)
    if n < 3:
        raise ValueError("Il faut au moins 3 groupes pour separer train/val/test.")
    random.Random(seed).shuffle(uniq)
    n_val = max(1, round(ratios[1] * n))
    n_te = max(1, round(ratios[2] * n))
    n_tr = n - n_val - n_te
    if n_tr < 1:
        raise ValueError("Pas assez de groupes pour ces proportions.")
    g_tr = set(uniq[:n_tr])
    g_va = set(uniq[n_tr:n_tr + n_val])
    g_te = set(uniq[n_tr + n_val:])
    pick = lambda gs: [i for i, g in enumerate(groups) if g in gs]
    return pick(g_tr), pick(g_va), pick(g_te)


def make_synthetic(n: int, size: int, seed: int):
    """Bruit + carre 'modifie' sur une image sur deux. Retourne images, masques, labels, groupes."""
    rng = np.random.default_rng(seed)
    images = rng.normal(0.0, 1.0, (n, 3, size, size)).astype(np.float32)
    masks = np.zeros((n, 1, size, size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    s = size // 4
    for i in range(n):
        if i % 2 == 1:
            y = int(rng.integers(0, size - s))
            x = int(rng.integers(0, size - s))
            images[i, :, y:y + s, x:x + s] += 2.0
            masks[i, 0, y:y + s, x:x + s] = 1.0
            labels[i] = 1
    groups = [i // 4 for i in range(n)]
    return (torch.from_numpy(images), torch.from_numpy(masks),
            torch.from_numpy(labels), groups)


def choose_threshold(y_true, scores) -> float:
    """Seuil maximisant le F1 (a appeler sur la VALIDATION uniquement)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = classification_metrics(y_true, scores, float(t))["f1"]
        if not np.isnan(f1) and f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def _extract(backbone, images, bs: int = 16):
    fmaps, pooled = [], []
    for i in range(0, len(images), bs):
        out = backbone(images[i:i + bs])
        fmaps.append(out.feature_maps)
        pooled.append(out.pooled)
    return torch.cat(fmaps), torch.cat(pooled)


@torch.no_grad()
def _predict(cls_head, loc_head, fmaps, pooled, idx, size):
    cls_head.eval()
    loc_head.eval()
    idx = torch.tensor(idx)
    scores = torch.sigmoid(cls_head(pooled[idx])).numpy()
    probs = torch.sigmoid(loc_head(fmaps[idx], (size, size)))[:, 0].numpy()
    return scores, probs


def run(n: int = 96, size: int = 128, epochs: int = 15, lr: float = 1e-3,
        batch: int = 16, seed: int = 0, pretrained: bool = False) -> dict:
    set_seed(seed)
    images, masks, labels, groups = make_synthetic(n, size, seed)
    tr, va, te = split_by_group(groups, seed)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))

    backbone = EfficientNetB0Features(pretrained=pretrained)
    fmaps, pooled = _extract(backbone, images)

    cls_head, loc_head = ClassificationHead(), LocalizationHead()
    loc_loss = LocalizationLoss()
    opt = torch.optim.Adam(
        list(cls_head.parameters()) + list(loc_head.parameters()), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    tr_t = torch.tensor(tr)

    train_losses = []
    for ep in range(epochs):
        cls_head.train()
        loc_head.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen)]
        total, count = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:  # BatchNorm exige au moins 2 echantillons
                continue
            loss = (
                F.binary_cross_entropy_with_logits(cls_head(pooled[b]), labels[b].float())
                + loc_loss(loc_head(fmaps[b], (size, size)), masks[b])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            count += len(b)
        train_losses.append(total / max(count, 1))
        print(f"epoch {ep + 1:2d}/{epochs}  perte train = {train_losses[-1]:.4f}")

    # Seuil choisi sur la validation, applique tel quel au test.
    s_va, _ = _predict(cls_head, loc_head, fmaps, pooled, va, size)
    threshold = choose_threshold(labels[va].numpy(), s_va)
    s_te, p_te = _predict(cls_head, loc_head, fmaps, pooled, te, size)
    cls_m = classification_metrics(labels[te].numpy(), s_te, threshold)
    seg = [segmentation_metrics(p >= 0.5, masks[j, 0].numpy())
           for p, j in zip(p_te, te)]
    seg_m = {k: float(np.mean([m[k] for m in seg])) for k in ("iou", "dice")}

    return {
        "warning": "Donnees synthetiques : aucune valeur forensique.",
        "seed": seed, "n": n, "size": size, "epochs": epochs,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "train_losses": train_losses,
        "threshold": threshold,
        "test_classification": cls_m,
        "test_segmentation": seg_m,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true")
    a = ap.parse_args()
    res = run(n=a.n, size=a.size, epochs=a.epochs, seed=a.seed, pretrained=a.pretrained)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/synthetic_run.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("split_sizes", "threshold",
                                          "test_classification", "test_segmentation")},
                     indent=2))
    print("ATTENTION : donnees synthetiques, resultats sans valeur forensique.")
