"""Construction d'un jeu EuroSAT falsifie avec masques exacts (v0.1).

Separation par image source (sans fuite entre train/val/test).
L'attaque 'blur' est tenue en reserve : presente uniquement dans le test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.models.train import split_by_group
from src.preprocessing.tamper import tamper

SEEN_ATTACKS = ("splice", "copy_move", "color_shift")
HELD_OUT_ATTACK = "blur"


def build_from_images(imgs: np.ndarray, seed: int = 0) -> dict:
    """imgs : (N, H, W, 3) uint8. Retourne un dict de tableaux NumPy."""
    n, h, w = len(imgs), imgs.shape[1], imgs.shape[2]
    rng = np.random.default_rng(seed)
    tr, va, te = split_by_group(list(range(n)), seed)

    images, masks, labels, sources, attacks, splits = [], [], [], [], [], []

    def add(img, mask, label, src, attack, split):
        images.append(img)
        masks.append(mask)
        labels.append(label)
        sources.append(src)
        attacks.append(attack)
        splits.append(split)

    empty = np.zeros((h, w), dtype=np.uint8)
    for split_name, members in (("train", tr), ("val", va), ("test", te)):
        for s in members:
            img = imgs[s]
            add(img, empty, 0, s, "none", split_name)
            others = [m for m in members if m != s]
            donor = imgs[int(rng.choice(others))]
            atk = SEEN_ATTACKS[int(rng.integers(len(SEEN_ATTACKS)))]
            t, m = tamper(img, atk, rng, donor)
            add(t, m, 1, s, atk, split_name)
            if split_name == "test":
                t2, m2 = tamper(img, HELD_OUT_ATTACK, rng)
                add(t2, m2, 1, s, HELD_OUT_ATTACK, "test")

    return {
        "images": np.stack(images),
        "masks": np.stack(masks),
        "labels": np.array(labels, dtype=np.int64),
        "source": np.array(sources, dtype=np.int64),
        "attack": np.array(attacks),
        "split": np.array(splits),
    }


def build(n_sources: int = 1200, seed: int = 0, root: str = "data/raw",
          out: str = "data/processed/eurosat_tampered.npz") -> dict:
    from torchvision.datasets import EuroSAT

    ds = EuroSAT(root=root, download=False)
    idx = np.random.default_rng(seed).choice(len(ds), size=n_sources, replace=False)
    imgs = np.stack([np.asarray(ds[int(i)][0], dtype=np.uint8) for i in idx])
    data = build_from_images(imgs, seed)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sources", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = build(n_sources=a.n_sources, seed=a.seed)
    print("images :", d["images"].shape, d["images"].dtype)
    for sp in ("train", "val", "test"):
        sel = d["split"] == sp
        kinds = {k: int((d["attack"][sel] == k).sum()) for k in np.unique(d["attack"][sel])}
        print(f"{sp:5s}: {int(sel.sum())} images  {kinds}")
