"""Tete de classification au niveau de l'image (contrat D, v0.1)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.utils.contracts import ClassificationOutput


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int = 1280, dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Retourne les logits (N,). Perte conseillee : BCEWithLogitsLoss."""
        if pooled.ndim != 2 or pooled.shape[1] != self.fc.in_features:
            raise ValueError(
                f"Entree attendue (N, {self.fc.in_features}), recu {tuple(pooled.shape)}"
            )
        return self.fc(self.drop(pooled)).squeeze(1)

    @torch.no_grad()
    def predict(self, pooled: torch.Tensor,
                threshold: Optional[float] = None) -> list[ClassificationOutput]:
        """Score = sigmoid(logit). Sans seuil valide, aucune decision n'est prise."""
        was_training = self.training
        self.eval()
        scores = torch.sigmoid(self(pooled)).tolist()
        self.train(was_training)
        return [
            ClassificationOutput(
                score=s, threshold=threshold,
                decision=(s >= threshold) if threshold is not None else None,
            )
            for s in scores
        ]
