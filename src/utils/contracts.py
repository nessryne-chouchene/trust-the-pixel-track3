"""Contrats d'interface Track 3 (version 0.1, a valider avec l'equipe)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch


class ValidationStatus(str, Enum):
    ACCEPTED = "accepted"
    ACCEPTED_WITH_WARNINGS = "accepted_with_warnings"
    REJECTED = "rejected"


@dataclass
class ValidationReport:
    """A. Sortie du module validation/securite."""
    status: ValidationStatus
    sha256: Optional[str]                      # hash du fichier original
    metadata: dict = field(default_factory=dict)   # ExifTool, dimensions, format...
    provenance: dict = field(default_factory=dict)  # C2PA: "verified" | "invalid" | "missing"
    warnings: list[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


@dataclass
class PreprocessedSample:
    """B. Sortie du pretraitement."""
    sample_id: str
    image: np.ndarray                 # (C, H, W) float32, normalisee
    band_names: list[str]             # ex. ["R", "G", "B"]
    original_size: tuple[int, int]    # (H, W) avant redimensionnement
    original_dtype: str               # ex. "uint16"
    normalization: dict               # methode et parametres utilises
    label: Optional[int] = None       # 0 = authentique, 1 = manipulee (si connu)
    mask: Optional[np.ndarray] = None  # (H, W) uint8, memes dimensions que image


@dataclass
class FeatureOutput:
    """C. Sortie de l'extraction EfficientNet-B0."""
    feature_maps: torch.Tensor        # (N, 1280, H/32, W/32)
    pooled: Optional[torch.Tensor] = None  # (N, 1280)


@dataclass
class ClassificationOutput:
    """D. Sortie de la tete de classification."""
    score: float                      # score du modele, PAS une preuve
    threshold: Optional[float] = None
    decision: Optional[bool] = None   # None tant que le seuil n'est pas valide


@dataclass
class LocalizationOutput:
    """E. Sortie de la tete de localisation."""
    mask_prob: np.ndarray             # (H, W) float32 dans [0, 1], taille de l'image d'origine
    mask_binary: Optional[np.ndarray] = None  # (H, W) uint8


@dataclass
class ForensicReport:
    """F. Rapport final."""
    validation: ValidationReport
    classification: Optional[ClassificationOutput] = None
    localization: Optional[LocalizationOutput] = None
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
