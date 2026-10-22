"""Contrats d'interface Track 3 (version 0.3).

v0.3 : le rapport final porte un verdict, une explication, les temps et les fichiers produits.

v0.2 : le contrat A (validation) reprend les statuts et champs du module
src/security (PASSED / REVIEW / FAILED, codes de motifs, preuves).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch


class ValidationStatus(str, Enum):
    PASSED = "PASSED"   # format, signature, integrite, metadonnees et rejeu OK
    REVIEW = "REVIEW"   # provenance prouvee mais metadonnees douteuses : revue humaine
    FAILED = "FAILED"   # format, signature/integrite invalide ou rejeu detecte


@dataclass
class ValidationReport:
    """A. Sortie du module validation/securite (src/security/adapter.py)."""
    status: ValidationStatus
    image_id: Optional[str]
    sha256: Optional[str]                         # hash du fichier recu (None si format invalide)
    format_valid: Optional[bool]                  # None = controle non execute
    signature_valid: Optional[bool]
    metadata_valid: Optional[bool]
    replay_detected: Optional[bool]
    reason_codes: list[str] = field(default_factory=list)   # ex. ["HASH_MISMATCH"]
    reasons: list[str] = field(default_factory=list)        # messages lisibles
    evidence: dict = field(default_factory=dict)            # cle, politique, liste des controles

    @property
    def can_proceed(self) -> bool:
        """Seule une image PASSED part automatiquement vers les modeles."""
        return self.status is ValidationStatus.PASSED


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
    """F. Rapport final (src/pipeline/analyze.py)."""
    validation: ValidationReport
    classification: Optional[ClassificationOutput] = None
    localization: Optional[LocalizationOutput] = None
    verdict: str = ""                                     # REJECTED / REVIEW / SUSPECTED / NOT_DETECTED
    explanation: str = ""                                 # phrase lisible pour l'analyste
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)        # securite, traces, modele, total
    artifacts: dict = field(default_factory=dict)         # chemins des fichiers produits
