"""Entrainement du detecteur forensique (ForensicUNet) sur EuroSAT falsifie (v0.1).

Falsifications SYNTHETIQUES : les scores ne valent que dans ce cadre.
'blur' est une attaque tenue en reserve : absente du train et de la validation.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.forensics.evasion import random_launder
from src.forensics.signals import CHANNEL_NAMES, forensic_inputs
from src.models.evaluation import choose_threshold, evaluate, print_summary
from src.models.forensic_unet import ForensicUNet, count_parameters
from src.models.losses import dice_loss
from src.models.train import set_seed
from src.utils.metrics import auc_score, segmentation_metrics

SEEN_ATTACKS = ("splice", "copy_move", "color_shift")
#: Jeux de canaux pour l'ablation (indices dans CHANNEL_NAMES : R G B srm_1-3 ela).
CHANNEL_SETS = {
    "rgb": (0, 1, 2),
    "rgb+srm": (0, 1, 2, 3, 4, 5),
    "rgb+ela": (0, 1, 2, 6),
    "rgb+srm+ela": (0, 1, 2, 3, 4, 5, 6),
}


def _augment(x: torch.Tensor, m: torch.Tensor, gen: torch.Generator):
    """Symetries horizontale/verticale aleatoires, identiques pour l'entree et le masque."""
    if torch.rand(1, generator=gen).item() < 0.5:
        x, m = x.flip(-1), m.flip(-1)
    if torch.rand(1, generator=gen).item() < 0.5:
        x, m = x.flip(-2), m.flip(-2)
    return x, m


@torch.no_grad()
def predict_all(model, inputs: torch.Tensor, bs: int = 128):
    model.eval()
    scores, probs = [], []
    for i in range(0, len(inputs), bs):
        c, m = model(inputs[i:i + bs])
        scores.append(torch.sigmoid(c))
        probs.append(torch.sigmoid(m)[:, 0])
    return torch.cat(scores).numpy(), torch.cat(probs).numpy()


def _val_score(labels, scores, probs, masks) -> tuple[float, float, float]:
    auc = auc_score(labels, scores)
    pos = np.where(labels == 1)[0]
    iou = float(np.mean([segmentation_metrics(probs[i] >= 0.5, masks[i])["iou"] for i in pos])) if len(pos) else 0.0
    return auc + iou, auc, iou


def measure_latency(model, image_uint8: np.ndarray, repeats: int = 20,
                    sel: tuple = CHANNEL_SETS["rgb+srm+ela"]) -> dict:
    """Latence CPU par image (batch 1), calcul des traces forensiques INCLUS."""
    model.eval()
    sel = list(sel)
    with torch.no_grad():
        model(forensic_inputs(image_uint8)[:, sel])  # echauffement
        t0 = time.perf_counter()
        for _ in range(repeats):
            x = forensic_inputs(image_uint8)[:, sel]
        t1 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        t2 = time.perf_counter()
    return {"signals_ms": 1000 * (t1 - t0) / repeats, "model_ms": 1000 * (t2 - t1) / repeats,
            "total_ms": 1000 * (t2 - t0) / repeats,
            "image_size": list(image_uint8.shape[:2]), "threads": torch.get_num_threads()}


def run(npz: str = "data/processed/eurosat_tampered.npz", epochs: int = 30, lr: float = 1e-3,
        batch: int = 32, seed: int = 0, base: int = 24, out_dir: str = "outputs",
        robust: bool = False, robust_prob: float = 0.5, verbose: bool = True,
        channels: str = "rgb+srm+ela") -> dict:
    """robust=True : la moitie des images d'entrainement de chaque lot est
    blanchie au hasard (JPEG, reechantillonnage, bruit, flou) : c'est la
    reponse aux attaques d'evasion. Plus lent (traces recalculees par lot)."""
    if channels not in CHANNEL_SETS:
        raise ValueError(f"channels : un de {sorted(CHANNEL_SETS)}")
    sel = list(CHANNEL_SETS[channels])
    set_seed(seed)
    with np.load(npz) as z:
        data = {k: z[k] for k in z.files}
    t0 = time.perf_counter()
    inputs = forensic_inputs(data["images"])[:, sel]
    t_signals = time.perf_counter() - t0
    labels_np = data["labels"]
    labels = torch.from_numpy(labels_np).float()
    masks = torch.from_numpy(data["masks"]).float().unsqueeze(1)
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}

    model = ForensicUNet(in_channels=len(sel), base=base)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = torch.as_tensor(idx["train"])

    best, best_state, best_epoch, history = -1.0, None, 0, []
    t_train = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=gen)]
        total, count = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            if robust:
                imgs = data["images"][b.numpy()].copy()
                for j in range(len(imgs)):
                    if rng.random() < robust_prob:
                        imgs[j] = random_launder(imgs[j], rng)
                x = forensic_inputs(imgs)[:, sel]
            else:
                x = inputs[b]
            x, m = _augment(x, masks[b], gen)
            cls_logits, mask_logits = model(x)
            y = labels[b]
            loss = F.binary_cross_entropy_with_logits(cls_logits, y)
            loss = loss + F.binary_cross_entropy_with_logits(mask_logits, m)
            pos = y == 1
            if pos.any():
                loss = loss + dice_loss(mask_logits[pos], m[pos])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            count += len(b)
        sched.step()
        s_va, p_va = predict_all(model, inputs[idx["val"]])
        score, auc_va, iou_va = _val_score(labels_np[idx["val"]], s_va, p_va, data["masks"][idx["val"]])
        history.append({"epoch": ep + 1, "loss": total / max(count, 1), "val_auc": auc_va, "val_iou": iou_va})
        if verbose:
            print(f"epoch {ep + 1:2d}/{epochs}  perte={history[-1]['loss']:.4f}"
                  f"  AUC val={auc_va:.3f}  IoU val={iou_va:.3f}")
        if score > best:
            best, best_epoch, best_state = score, ep + 1, copy.deepcopy(model.state_dict())
    t_train = time.perf_counter() - t_train
    if best_state is not None:
        model.load_state_dict(best_state)

    s_va, _ = predict_all(model, inputs[idx["val"]])
    threshold = choose_threshold(labels_np[idx["val"]], s_va)
    te = idx["test"]
    s_te, p_te = predict_all(model, inputs[te])
    res = evaluate(labels_np[te], data["attack"][te], s_te, p_te, data["masks"][te],
                   threshold, SEEN_ATTACKS)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "forensic_unet_robust" if robust else "forensic_unet"
    torch.save({"state_dict": model.state_dict(), "base": base, "threshold": threshold,
                "channels": [CHANNEL_NAMES[i] for i in sel], "seed": seed, "robust": robust}, out / f"{name}.pt")
    res.update({
        "model": "ForensicUNet (RGB + SRM + ELA)" + (" + entrainement robuste" if robust else ""),
        "warning": "Falsifications synthetiques : resultats limites a ce cadre.",
        "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
        "split_sizes": {k: int(len(v)) for k, v in idx.items()},
        "parameters": count_parameters(model),
        "model_size_mb": count_parameters(model) * 4 / 1e6,
        "signals_seconds_total": t_signals, "train_seconds": t_train,
        "channels": channels,
        "latency": measure_latency(model, data["images"][te[0]], sel=tuple(sel)),
        "history": history,
    })
    (out / f"{name}_run.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--robust", action="store_true", help="entrainement avec blanchiment aleatoire")
    a = ap.parse_args()
    r = run(epochs=a.epochs, seed=a.seed, base=a.base, robust=a.robust)
    print_summary(r, r["model"])
    lat = r["latency"]
    print(f"\nparametres={r['parameters']:,}  taille={r['model_size_mb']:.1f} Mo"
          f"  latence CPU={lat['total_ms']:.1f} ms/image (traces {lat['signals_ms']:.1f} + modele {lat['model_ms']:.1f})")
    n = "forensic_unet_robust" if a.robust else "forensic_unet"
    print(f"Poids : outputs/{n}.pt   Resultats : outputs/{n}_run.json")
