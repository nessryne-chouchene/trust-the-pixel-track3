"""Detecteur forensique leger a deux sorties (v0.1).

Entree : RGB + residus SRM + ELA (7 canaux, src/forensics/signals.py), a la
resolution NATIVE de l'image (pas d'agrandissement a 224x224).
Sorties : un masque de localisation pixel par pixel (encodeur-decodeur de type
U-Net, Ronneberger et al. 2015) et un score au niveau de l'image.
Environ 1,1 M de parametres : entrainable et executable sur CPU.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forensics.signals import N_CHANNELS
from src.utils.contracts import ClassificationOutput, LocalizationOutput


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class ForensicUNet(nn.Module):
    DEPTH = 3  # 3 sous-echantillonnages : H et W sont completes a un multiple de 8

    def __init__(self, in_channels: int = N_CHANNELS, base: int = 24, dropout: float = 0.2):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.in_channels = in_channels
        self.enc1, self.enc2, self.enc3 = _block(in_channels, c1), _block(c1, c2), _block(c2, c3)
        self.bottleneck = _block(c3, c4)
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), _block(c4, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), _block(c3, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), _block(c2, c1)
        self.mask_out = nn.Conv2d(c1, 1, 1)
        # Score image : contexte global (moyenne + max du goulot) + statistiques du masque.
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c4 + 2, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x : (N, 7, H, W). Retourne (logits image (N,), logits masque (N, 1, H, W))."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Entree attendue (N, {self.in_channels}, H, W), recu {tuple(x.shape)}")
        h, w = x.shape[-2:]
        m = 2 ** self.DEPTH
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        mask_logits = self.mask_out(d1)[..., :h, :w]
        flat = mask_logits.flatten(1)
        feats = torch.cat([b.mean((2, 3)), b.amax((2, 3)),
                           flat.amax(1, keepdim=True), flat.mean(1, keepdim=True)], 1)
        return self.cls(feats).squeeze(1), mask_logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: Optional[float] = None,
                mask_threshold: Optional[float] = None
                ) -> list[tuple[ClassificationOutput, LocalizationOutput]]:
        """Sorties aux formats des contrats D et E. Sans seuil valide : pas de decision."""
        was_training = self.training
        self.eval()
        cls_logits, mask_logits = self(x)
        self.train(was_training)
        scores = torch.sigmoid(cls_logits).tolist()
        probs = torch.sigmoid(mask_logits)[:, 0].cpu().numpy().astype(np.float32)
        out = []
        for s, p in zip(scores, probs):
            out.append((
                ClassificationOutput(score=s, threshold=threshold,
                                     decision=(s >= threshold) if threshold is not None else None),
                LocalizationOutput(mask_prob=p,
                                   mask_binary=(p >= mask_threshold).astype(np.uint8)
                                   if mask_threshold is not None else None),
            ))
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
