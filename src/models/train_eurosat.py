"""Base de comparaison : EfficientNet-B0 GELE (ImageNet) + deux tetes, sur EuroSAT
falsifie (v0.2). Sert a montrer ce qu'apporte le detecteur forensique
(src/models/train_forensic.py) par rapport a des features generiques.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.features.efficientnet_features import EfficientNetB0Features
from src.models.classification_head import ClassificationHead
from src.models.evaluation import choose_threshold, evaluate, print_summary
from src.models.localization_head import LocalizationHead
from src.models.losses import dice_loss
from src.models.train import set_seed
from src.preprocessing.preprocess import IMAGENET_MEAN, IMAGENET_STD
from src.utils.metrics import auc_score

INPUT_SIZE = 224
SEEN_ATTACKS = ("splice", "copy_move", "color_shift")


def _to_input(batch_uint8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def _extract(images: np.ndarray, pretrained: bool, bs: int = 32):
    backbone = EfficientNetB0Features(pretrained=pretrained)
    fm, po = [], []
    for k, i in enumerate(range(0, len(images), bs)):
        out = backbone(_to_input(images[i:i + bs]))
        fm.append(out.feature_maps.half())
        po.append(out.pooled)
        if k % 10 == 0:
            print(f"  extraction : {min(i + bs, len(images))}/{len(images)}")
    return torch.cat(fm), torch.cat(po)


def _load_or_extract(images, pretrained, cache, fingerprint):
    p = Path(cache)
    if p.exists():
        c = torch.load(p)
        if c["fingerprint"] == fingerprint and c["pretrained"] == pretrained:
            print("Features lues depuis le cache.")
            return c["fmaps"], c["pooled"]
        print("Cache obsolete : recalcul des features.")
    fm, po = _extract(images, pretrained)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "pretrained": pretrained,
                "fmaps": fm, "pooled": po}, p)
    return fm, po


@torch.no_grad()
def _predict(cls_head, loc_head, fmaps, pooled, idx, size, want_masks=True):
    cls_head.eval()
    loc_head.eval()
    t = torch.as_tensor(idx)
    scores, probs = [], []
    for i in range(0, len(t), 64):
        b = t[i:i + 64]
        scores.append(torch.sigmoid(cls_head(pooled[b])))
        if want_masks:
            probs.append(torch.sigmoid(loc_head(fmaps[b].float(), size))[:, 0])
    s = torch.cat(scores).numpy()
    return s, (torch.cat(probs).numpy() if want_masks else None)


def run(npz: str = "data/processed/eurosat_tampered.npz", epochs: int = 30,
        lr: float = 1e-3, batch: int = 32, seed: int = 0, pretrained: bool = True,
        cache: str = "data/processed/eurosat_features.pt", out_dir: str = "outputs") -> dict:
    set_seed(seed)
    with np.load(npz) as z:
        data = {k: z[k] for k in z.files}
    images = data["images"]
    fp = hashlib.sha256(images.tobytes()).hexdigest()
    fmaps, pooled = _load_or_extract(images, pretrained, cache, fp)

    labels = torch.from_numpy(data["labels"])
    masks = torch.from_numpy(data["masks"]).float().unsqueeze(1)
    size = tuple(int(v) for v in images.shape[1:3])
    idx = {s: np.where(data["split"] == s)[0] for s in ("train", "val", "test")}

    cls_head, loc_head = ClassificationHead(), LocalizationHead()
    opt = torch.optim.Adam(list(cls_head.parameters()) + list(loc_head.parameters()), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    tr = torch.as_tensor(idx["train"])

    best_auc, best_state, best_epoch, losses = -1.0, None, 0, []
    for ep in range(epochs):
        cls_head.train()
        loc_head.train()
        perm = tr[torch.randperm(len(tr), generator=gen)]
        total, count = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            cls_loss = F.binary_cross_entropy_with_logits(cls_head(pooled[b]), labels[b].float())
            logits = loc_head(fmaps[b].float(), size)
            tgt = masks[b]
            loc_loss = F.binary_cross_entropy_with_logits(logits, tgt)
            pos = labels[b] == 1
            if pos.any():
                loc_loss = loc_loss + dice_loss(logits[pos], tgt[pos])
            loss = cls_loss + loc_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            count += len(b)
        losses.append(total / max(count, 1))
        s_va, _ = _predict(cls_head, loc_head, fmaps, pooled, idx["val"], size, want_masks=False)
        auc_va = auc_score(data["labels"][idx["val"]], s_va)
        print(f"epoch {ep + 1:2d}/{epochs}  perte={losses[-1]:.4f}  AUC val={auc_va:.3f}")
        if not np.isnan(auc_va) and auc_va > best_auc:
            best_auc, best_epoch = auc_va, ep + 1
            best_state = {"cls": copy.deepcopy(cls_head.state_dict()),
                          "loc": copy.deepcopy(loc_head.state_dict())}
    if best_state is not None:
        cls_head.load_state_dict(best_state["cls"])
        loc_head.load_state_dict(best_state["loc"])

    s_va, _ = _predict(cls_head, loc_head, fmaps, pooled, idx["val"], size, want_masks=False)
    threshold = choose_threshold(data["labels"][idx["val"]], s_va)
    te = idx["test"]
    s_te, p_te = _predict(cls_head, loc_head, fmaps, pooled, te, size)
    res = evaluate(data["labels"][te], data["attack"][te], s_te, p_te, data["masks"][te],
                   threshold, SEEN_ATTACKS)
    res.update({
        "model": "EfficientNet-B0 gele (ImageNet) + tetes",
        "warning": "Falsifications synthetiques : resultats limites a ce cadre.",
        "seed": seed, "pretrained": pretrained, "epochs": epochs, "best_epoch": best_epoch,
        "split_sizes": {k: int(len(v)) for k, v in idx.items()},
        "val_auc_best": best_auc, "train_losses": losses,
    })
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "eurosat_run.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_pretrained", action="store_true")
    a = ap.parse_args()
    r = run(epochs=a.epochs, seed=a.seed, pretrained=not a.no_pretrained)
    print_summary(r, "Base EfficientNet-B0 gele")
