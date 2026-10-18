"""Extraction de features EfficientNet-B0 -> FeatureOutput (contrat C, v0.1).

ATTENTION : des poids ImageNet ne constituent pas un detecteur de manipulations.
Ce module ne produit que des caracteristiques ; leur valeur forensique doit etre
etablie par entrainement et validation specifiques.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.utils.contracts import FeatureOutput, PreprocessedSample

OUT_CHANNELS = 1280


class EfficientNetB0Features(nn.Module):
    def __init__(self, pretrained: bool = False, freeze: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = efficientnet_b0(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_channels = OUT_CHANNELS
        self.frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def train(self, mode: bool = True):
        # Un backbone gele reste en mode eval (BatchNorm fige).
        return super().train(False if self.frozen else mode)

    def forward(self, x: torch.Tensor, return_pooled: bool = True) -> FeatureOutput:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Entree attendue (N, 3, H, W), recu {tuple(x.shape)}")
        if self.frozen:
            with torch.no_grad():
                fmap = self.features(x)
        else:
            fmap = self.features(x)
        pooled = self.pool(fmap).flatten(1) if return_pooled else None
        return FeatureOutput(feature_maps=fmap, pooled=pooled)


def samples_to_batch(samples: list[PreprocessedSample]) -> torch.Tensor:
    """Empile des PreprocessedSample de meme taille en un tenseur (N, C, H, W)."""
    return torch.from_numpy(np.stack([s.image for s in samples]))


if __name__ == "__main__":
    import time

    model = EfficientNetB0Features()
    x = torch.randn(1, 3, 224, 224)
    model(x)  # echauffement
    t0 = time.perf_counter()
    for _ in range(10):
        out = model(x)
    dt = (time.perf_counter() - t0) / 10
    print(f"feature_maps={tuple(out.feature_maps.shape)} pooled={tuple(out.pooled.shape)}")
    print(f"latence CPU moyenne (batch 1, 224x224) : {dt * 1000:.1f} ms")
