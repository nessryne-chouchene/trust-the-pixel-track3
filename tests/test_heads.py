"""Tests des tetes + passage de bout en bout sur donnees SYNTHETIQUES.
Un modele non entraine ne produit aucun resultat forensique interpretable."""
import numpy as np
import pytest
import torch
from PIL import Image

from src.features.efficientnet_features import EfficientNetB0Features, samples_to_batch
from src.models.classification_head import ClassificationHead
from src.models.localization_head import LocalizationHead
from src.preprocessing.preprocess import preprocess


def test_classification_logits_shape():
    assert tuple(ClassificationHead()(torch.randn(4, 1280)).shape) == (4,)


def test_classification_predict_without_threshold():
    outs = ClassificationHead().predict(torch.randn(3, 1280))
    assert len(outs) == 3
    assert all(0.0 <= o.score <= 1.0 and o.decision is None for o in outs)


def test_classification_predict_with_threshold():
    outs = ClassificationHead().predict(torch.randn(3, 1280), threshold=0.5)
    assert all(isinstance(o.decision, bool) and o.threshold == 0.5 for o in outs)


def test_classification_wrong_input_raises():
    with pytest.raises(ValueError):
        ClassificationHead()(torch.randn(4, 512))


def test_localization_shape_and_size():
    head = LocalizationHead()
    assert tuple(head(torch.randn(2, 1280, 7, 7), (224, 224)).shape) == (2, 1, 224, 224)
    assert tuple(head(torch.randn(1, 1280, 8, 6), (256, 192)).shape) == (1, 1, 256, 192)


def test_localization_predict():
    outs = LocalizationHead().predict(torch.randn(2, 1280, 7, 7), (64, 48))
    assert outs[0].mask_prob.shape == (64, 48)
    assert outs[0].mask_prob.dtype == np.float32
    assert 0.0 <= outs[0].mask_prob.min() and outs[0].mask_prob.max() <= 1.0
    assert outs[0].mask_binary is None
    outs_t = LocalizationHead().predict(torch.randn(1, 1280, 7, 7), (64, 48), threshold=0.5)
    assert outs_t[0].mask_binary.dtype == np.uint8


def test_localization_wrong_channels_raises():
    with pytest.raises(ValueError):
        LocalizationHead()(torch.randn(1, 512, 7, 7), (224, 224))


def test_end_to_end_technical(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (90, 120, 3), dtype=np.uint8)).save(p)
    sample = preprocess(p)
    batch = samples_to_batch([sample])
    feats = EfficientNetB0Features()(batch)
    cls = ClassificationHead().predict(feats.pooled)[0]
    loc = LocalizationHead().predict(feats.feature_maps, sample.original_size)[0]
    assert 0.0 <= cls.score <= 1.0
    assert loc.mask_prob.shape == sample.original_size  # (90, 120)
