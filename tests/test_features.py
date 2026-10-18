"""Tests unitaires de l'extraction EfficientNet-B0 (tenseurs synthetiques)."""
import numpy as np
import pytest
import torch

from src.features.efficientnet_features import EfficientNetB0Features, samples_to_batch
from src.utils.contracts import PreprocessedSample


def test_shapes_224():
    out = EfficientNetB0Features()(torch.randn(2, 3, 224, 224))
    assert tuple(out.feature_maps.shape) == (2, 1280, 7, 7)
    assert tuple(out.pooled.shape) == (2, 1280)


def test_non_square_input():
    out = EfficientNetB0Features()(torch.randn(1, 3, 256, 192))
    assert tuple(out.feature_maps.shape) == (1, 1280, 8, 6)


def test_wrong_channels_raises():
    with pytest.raises(ValueError):
        EfficientNetB0Features()(torch.randn(1, 4, 224, 224))
    with pytest.raises(ValueError):
        EfficientNetB0Features()(torch.randn(3, 224, 224))


def test_frozen_has_no_grad_and_stays_eval():
    m = EfficientNetB0Features(freeze=True)
    m.train()
    assert not m.training
    assert all(not p.requires_grad for p in m.parameters())
    out = m(torch.randn(1, 3, 224, 224))
    assert not out.feature_maps.requires_grad


def test_no_pooled_when_not_requested():
    out = EfficientNetB0Features()(torch.randn(1, 3, 224, 224), return_pooled=False)
    assert out.pooled is None


def test_samples_to_batch():
    def mk(i):
        return PreprocessedSample(
            sample_id=str(i), image=np.zeros((3, 224, 224), np.float32),
            band_names=["R", "G", "B"], original_size=(10, 10),
            original_dtype="uint8", normalization={})
    b = samples_to_batch([mk(0), mk(1), mk(2)])
    assert tuple(b.shape) == (3, 3, 224, 224)
