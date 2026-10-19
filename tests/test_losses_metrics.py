"""Tests des pertes et des metriques (donnees synthetiques)."""
import math

import numpy as np
import pytest
import torch

from src.models.losses import LocalizationLoss, dice_loss
from src.utils.metrics import classification_metrics, segmentation_metrics


def _target():
    t = torch.zeros(2, 1, 16, 16)
    t[:, :, 4:10, 4:10] = 1.0
    return t


def test_dice_loss_perfect_and_wrong():
    t = _target()
    assert dice_loss((t * 2 - 1) * 20, t).item() < 1e-3
    assert dice_loss((1 - t) * 40 - 20, t).item() > 0.9


def test_localization_loss_backward():
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    loss = LocalizationLoss()(logits, _target())
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_localization_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        LocalizationLoss()(torch.randn(2, 1, 16, 16), torch.zeros(2, 1, 8, 8))


def test_classification_metrics_known_example():
    m = classification_metrics([1, 1, 0, 0, 1, 0], [0.9, 0.4, 0.2, 0.8, 0.7, 0.1], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_alarm_rate"] == pytest.approx(1 / 3)


def test_classification_metrics_undefined_is_nan():
    m = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert math.isnan(m["precision"]) and math.isnan(m["recall"])


def test_segmentation_metrics_known_example():
    p = np.zeros((4, 4), dtype=np.uint8)
    g = np.zeros((4, 4), dtype=np.uint8)
    p[0:2, 0:2] = 1
    g[0:2, 1:3] = 1
    m = segmentation_metrics(p, g)
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["dice"] == pytest.approx(0.5)


def test_segmentation_metrics_empty_masks():
    z = np.zeros((4, 4), dtype=np.uint8)
    assert segmentation_metrics(z, z) == {"iou": 1.0, "dice": 1.0}
    o = np.ones((4, 4), dtype=np.uint8)
    assert segmentation_metrics(z, o)["iou"] == 0.0
