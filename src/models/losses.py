"""Fonctions de perte (v0.1)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Perte de Dice douce. logits et target : (N, 1, H, W), target dans {0, 1}."""
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    total = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (total + eps)).mean()


class LocalizationLoss(nn.Module):
    """Combinaison ponderee BCE + Dice pour la tete de localisation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                f"Formes differentes : logits {tuple(logits.shape)}, "
                f"cible {tuple(target.shape)}"
            )
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice_loss(logits, target)
