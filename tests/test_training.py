"""Tests du script d'entrainement (donnees SYNTHETIQUES)."""
import math

import pytest

from src.models.train import make_synthetic, run, split_by_group


def test_split_has_no_leakage_and_covers_all():
    groups = [i // 4 for i in range(40)]
    tr, va, te = split_by_group(groups, seed=1)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert sorted(tr + va + te) == list(range(40))
    g = lambda idx: {groups[i] for i in idx}
    assert not (g(tr) & g(va)) and not (g(tr) & g(te)) and not (g(va) & g(te))


def test_split_is_deterministic():
    groups = [i // 4 for i in range(40)]
    assert split_by_group(groups, seed=3) == split_by_group(groups, seed=3)


def test_split_too_few_groups_raises():
    with pytest.raises(ValueError):
        split_by_group([0, 0, 1, 1])


def test_make_synthetic_shapes_and_labels():
    images, masks, labels, groups = make_synthetic(8, 64, seed=0)
    assert tuple(images.shape) == (8, 3, 64, 64)
    assert tuple(masks.shape) == (8, 1, 64, 64)
    assert labels.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert all((masks[i].sum() > 0) == bool(labels[i]) for i in range(8))
    assert len(groups) == 8


def test_tiny_training_run():
    res = run(n=24, size=64, epochs=2, batch=8, seed=0)
    assert len(res["train_losses"]) == 2
    assert all(math.isfinite(x) for x in res["train_losses"])
    assert {"test_classification", "test_segmentation", "threshold"} <= set(res)
    assert sum(res["split_sizes"].values()) == 24
