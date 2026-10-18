"""Pretraitement des images EO -> PreprocessedSample (contrat B, v0.1)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from src.utils.contracts import PreprocessedSample

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# HYPOTHESE A VERIFIER : ordre Sentinel-2 L1C (B01..B12) -> B04, B03, B02
S2_13_RGB_INDICES = (3, 2, 1)


def load_image(path):
    """Retourne (tableau (C, H, W) dans son dtype d'origine, nom du dtype)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            return src.read(), src.dtypes[0]


def select_bands(arr: np.ndarray, band_indices: Optional[Sequence[int]] = None):
    c = arr.shape[0]
    if band_indices is not None:
        idx = list(band_indices)
        names = [f"band_{i + 1}" for i in idx]
    elif c == 1:
        idx, names = [0, 0, 0], ["gray", "gray", "gray"]
    elif c == 3:
        idx, names = [0, 1, 2], ["R", "G", "B"]
    elif c == 13:
        idx, names = list(S2_13_RGB_INDICES), ["B04", "B03", "B02"]
    else:
        raise ValueError(
            f"{c} bandes : choix automatique impossible, fournir band_indices."
        )
    return arr[idx], names


def normalize_to_unit(arr: np.ndarray, dtype: str, reflectance_max: float = 10000.0):
    x = arr.astype(np.float32)
    if dtype == "uint8":
        return x / 255.0, {"method": "uint8/255"}
    if dtype == "uint16":
        x = np.clip(x / reflectance_max, 0.0, 1.0)
        return x, {"method": "uint16/reflectance_max_clip", "reflectance_max": reflectance_max}
    if dtype.startswith("float"):
        lo, hi = float(x.min()), float(x.max())
        if lo >= 0.0 and hi <= 1.0:
            return x, {"method": "float_already_unit"}
        warnings.warn("float hors [0, 1] : min-max par image (a verifier).")
        scale = (hi - lo) or 1.0
        return (x - lo) / scale, {"method": "float_minmax_per_image", "min": lo, "max": hi}
    raise ValueError(f"dtype non gere : {dtype}")


def resize_image(x: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t[0, 0].numpy().astype(np.uint8)


def imagenet_standardize(x: np.ndarray) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return (x - mean) / std


def preprocess(
    path,
    sample_id: Optional[str] = None,
    size: int = 224,
    band_indices: Optional[Sequence[int]] = None,
    reflectance_max: float = 10000.0,
    imagenet_norm: bool = True,
    label: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> PreprocessedSample:
    arr, dtype = load_image(path)
    original_size = (arr.shape[1], arr.shape[2])
    rgb, names = select_bands(arr, band_indices)
    x, norm_info = normalize_to_unit(rgb, dtype, reflectance_max)
    x = resize_image(x, size)
    if imagenet_norm:
        x = imagenet_standardize(x)
    norm_info = {**norm_info, "imagenet_standardized": imagenet_norm, "size": size}
    return PreprocessedSample(
        sample_id=sample_id or Path(path).stem,
        image=x.astype(np.float32),
        band_names=names,
        original_size=original_size,
        original_dtype=dtype,
        normalization=norm_info,
        label=label,
        mask=resize_mask(mask, size) if mask is not None else None,
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = preprocess(p)
        print(f"{s.sample_id}: image={s.image.shape} {s.image.dtype} "
              f"min={s.image.min():.3f} max={s.image.max():.3f} "
              f"bandes={s.band_names} origine={s.original_size} {s.original_dtype}")
        print(f"  normalisation={s.normalization}")
