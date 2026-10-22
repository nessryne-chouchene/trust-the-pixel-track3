"""Attaques d'evasion (validation adversariale, v0.1).

Un attaquant qui connait les detecteurs "blanchit" l'image apres la retouche
pour effacer les traces : recompression JPEG, redimensionnement, bruit, flou,
ou bords de collage adoucis. On applique ces operations a TOUTES les images de
test (authentiques comme falsifiees), comme le ferait une chaine de diffusion
reelle, puis on mesure la baisse de detection et la hausse des fausses alarmes.
Les masques restent valides : chaque operation conserve la taille de l'image.
"""
from __future__ import annotations

import io
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def resample(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = Image.fromarray(img).resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.BILINEAR)
    return np.asarray(small.resize((w, h), Image.BILINEAR))


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def blur(img: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))


def feather(img: np.ndarray, authentic: np.ndarray, mask: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Attaque adaptative : melange progressif aux bords de la zone retouchee
    (le collage n'a plus de bord net). Necessite l'image authentique d'origine."""
    alpha = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)),
                       dtype=np.float32)[..., None] / 255.0
    return np.clip(alpha * img + (1 - alpha) * authentic, 0, 255).astype(np.uint8)


#: Scenario d'evasion standard de l'equipe : nom -> fonction(image, rng) -> image.
LAUNDERING: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "none": lambda x, r: x,
    "jpeg_q90": lambda x, r: jpeg(x, 90),
    "jpeg_q75": lambda x, r: jpeg(x, 75),
    "jpeg_q60": lambda x, r: jpeg(x, 60),
    "resample_0.75": lambda x, r: resample(x, 0.75),
    "resample_0.5": lambda x, r: resample(x, 0.5),
    "noise_s2": lambda x, r: noise(x, 2.0, r),
    "noise_s5": lambda x, r: noise(x, 5.0, r),
    "blur_r0.6": lambda x, r: blur(x, 0.6),
    "blur_r1.0": lambda x, r: blur(x, 1.0),
    "chain_resample+jpeg75": lambda x, r: jpeg(resample(x, 0.75), 75),
}


def launder(images: np.ndarray, name: str, seed: int = 0) -> np.ndarray:
    """Applique l'operation `name` a chaque image (N, H, W, 3) uint8."""
    if name not in LAUNDERING:
        raise ValueError(f"Operation inconnue : {name}. Choix : {sorted(LAUNDERING)}")
    rng = np.random.default_rng(seed)
    fn = LAUNDERING[name]
    return np.stack([fn(img, rng) for img in images])


def random_launder(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blanchiment aleatoire leger pour l'entrainement robuste (la reponse)."""
    k = rng.integers(5)
    if k == 0:
        return jpeg(img, int(rng.integers(60, 96)))
    if k == 1:
        return resample(img, float(rng.uniform(0.5, 0.9)))
    if k == 2:
        return noise(img, float(rng.uniform(1, 5)), rng)
    if k == 3:
        return blur(img, float(rng.uniform(0.3, 1.0)))
    return jpeg(resample(img, float(rng.uniform(0.6, 0.9))), int(rng.integers(70, 95)))
