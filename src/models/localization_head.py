"""Tete de localisation : masque spatial (contrat E, v0.1)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.contracts import LocalizationOutput


class LocalizationHead(nn.Module):
    def __init__(self, in_channels: int = 1280, hidden: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, feature_maps: torch.Tensor,
                output_size: tuple[int, int]) -> torch.Tensor:
        """Retourne les logits (N, 1, H, W). Pertes conseillees : BCE + Dice."""
        if feature_maps.ndim != 4 or feature_maps.shape[1] != self.in_channels:
            raise ValueError(
                f"Entree attendue (N, {self.in_channels}, h, w), "
                f"recu {tuple(feature_maps.shape)}"
            )
        x = self.decoder(feature_maps)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def predict(self, feature_maps: torch.Tensor, output_size: tuple[int, int],
                threshold: Optional[float] = None) -> list[LocalizationOutput]:
        was_training = self.training
        self.eval()
        prob = torch.sigmoid(self(feature_maps, output_size))[:, 0].cpu().numpy()
        self.train(was_training)
        prob = prob.astype(np.float32)
        return [
            LocalizationOutput(
                mask_prob=p,
                mask_binary=(p >= threshold).astype(np.uint8) if threshold is not None else None,
            )
            for p in prob
        ]
