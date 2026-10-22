"""Traces forensiques du signal (traitement d'image, v0.1).

Une retouche laisse des traces que l'oeil ne voit pas mais que le signal garde :
  * residus de bruit (filtres SRM, Fridrich & Kodovsky 2012 ; utilises pour la
    detection de manipulations par Zhou et al., CVPR 2018) : le contenu est
    supprime, il reste la "texture de bruit" du capteur et de la compression ;
    une zone collee ou floutee a une texture de bruit differente ;
  * ELA (Error Level Analysis, outil cite par les organisateurs via
    Forensically) : on recompresse l'image en JPEG ; une zone deja compressee
    de la meme facon change peu, une zone retouchee change differemment.

Ces cartes ne sont PAS des preuves : ce sont des entrees pour le detecteur.
"""
from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

#: Trois noyaux SRM classiques (5x5), normalises comme dans Zhou et al. 2018.
SRM_KERNELS = np.stack([
    np.array([[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0,
    np.array([[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0,
    np.array([[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]], dtype=np.float32) / 2.0,
])
SRM_TRUNCATION = 3.0          # troncature classique des residus SRM
ELA_QUALITY = 90              # qualite JPEG de recompression
CHANNEL_NAMES = ("R", "G", "B", "srm_1", "srm_2", "srm_3", "ela")
N_CHANNELS = len(CHANNEL_NAMES)


def _check(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("Images attendues : (N, H, W, 3) ou (H, W, 3) en uint8.")
    return images


def srm_residuals(images: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> residus SRM (N, 3, H, W) float32 dans [-1, 1]."""
    images = _check(images)
    x = torch.from_numpy(images).float()
    gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).unsqueeze(1)
    k = torch.from_numpy(SRM_KERNELS).unsqueeze(1)          # (3, 1, 5, 5)
    res = F.conv2d(F.pad(gray, (2, 2, 2, 2), mode="reflect"), k)
    return res.clamp(-SRM_TRUNCATION, SRM_TRUNCATION) / SRM_TRUNCATION


def ela_map(images: np.ndarray, quality: int = ELA_QUALITY) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> carte ELA (N, 1, H, W) float32 dans [0, 1]."""
    images = _check(images)
    out = np.empty(images.shape[:3], dtype=np.float32)
    for i, img in enumerate(images):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        recomp = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.int16)
        diff = np.abs(img.astype(np.int16) - recomp).mean(axis=2)
        out[i] = np.log1p(diff) / np.log1p(255.0)
    return torch.from_numpy(out).unsqueeze(1)


def forensic_inputs(images: np.ndarray, batch: int = 256) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> entree du detecteur (N, 7, H, W) float32 :
    RGB centre-reduit, 3 residus SRM, 1 carte ELA."""
    images = _check(images)
    parts = []
    for i in range(0, len(images), batch):
        b = images[i:i + batch]
        rgb = (torch.from_numpy(b).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.25
        parts.append(torch.cat([rgb, srm_residuals(b), ela_map(b)], dim=1))
    return torch.cat(parts)
