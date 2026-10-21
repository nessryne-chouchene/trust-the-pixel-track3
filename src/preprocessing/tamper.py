"""Generation de falsifications synthetiques avec masques exacts (v0.2).

Falsifications SIMPLES (rectangles) : un detecteur qui les repere ne prouve rien
sur des falsifications realistes (images generees par IA, par exemple).
"""
from __future__ import annotations

import numpy as np
from PIL import ImageFilter, Image

ATTACK_TYPES = ("splice", "copy_move", "color_shift", "blur")


def _rand_box(rng, h: int, w: int, min_frac: float = 0.2, max_frac: float = 0.4):
    bh = int(rng.integers(max(2, int(h * min_frac)), max(3, int(h * max_frac)) + 1))
    bw = int(rng.integers(max(2, int(w * min_frac)), max(3, int(w * max_frac)) + 1))
    y = int(rng.integers(0, h - bh + 1))
    x = int(rng.integers(0, w - bw + 1))
    return y, x, bh, bw


def _overlap(y1, x1, y2, x2, bh, bw) -> bool:
    return abs(y1 - y2) < bh and abs(x1 - x2) < bw


def tamper(image: np.ndarray, attack: str, rng: np.random.Generator,
           donor: np.ndarray | None = None):
    """image : (H, W, 3) uint8. Retourne (image falsifiee, masque (H, W) uint8)."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Image attendue : (H, W, 3) en uint8.")
    if attack not in ATTACK_TYPES:
        raise ValueError(f"Falsification inconnue : {attack}. Choix : {ATTACK_TYPES}")

    h, w = image.shape[:2]
    out = image.copy()
    y, x, bh, bw = _rand_box(rng, h, w)

    if attack == "splice":
        if donor is None or donor.shape != image.shape:
            raise ValueError("splice exige une image donneuse de meme forme.")
        dy = int(rng.integers(0, h - bh + 1))
        dx = int(rng.integers(0, w - bw + 1))
        out[y:y + bh, x:x + bw] = donor[dy:dy + bh, dx:dx + bw]

    elif attack == "copy_move":
        # On retire la zone ENTIERE (taille + position) a chaque essai : une grande
        # zone au centre peut ne laisser aucune source sans chevauchement.
        for _ in range(200):
            y, x, bh, bw = _rand_box(rng, h, w)
            sy = int(rng.integers(0, h - bh + 1))
            sx = int(rng.integers(0, w - bw + 1))
            if not _overlap(y, x, sy, sx, bh, bw):
                break
        else:
            raise RuntimeError("Impossible de placer une source sans chevauchement.")
        out[y:y + bh, x:x + bw] = image[sy:sy + bh, sx:sx + bw]

    elif attack == "color_shift":
        shift = rng.integers(25, 60, size=3) * rng.choice([-1, 1], size=3)
        region = out[y:y + bh, x:x + bw].astype(np.int16) + shift
        out[y:y + bh, x:x + bw] = np.clip(region, 0, 255).astype(np.uint8)

    elif attack == "blur":
        crop = Image.fromarray(out[y:y + bh, x:x + bw]).filter(ImageFilter.GaussianBlur(radius=2))
        out[y:y + bh, x:x + bw] = np.asarray(crop)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y:y + bh, x:x + bw] = 1
    return out, mask
