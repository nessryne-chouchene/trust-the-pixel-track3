"""Tests de la construction du jeu falsifie (images SYNTHETIQUES)."""
import numpy as np

from src.preprocessing.build_tampered_eurosat import (
    HELD_OUT_ATTACK, SEEN_ATTACKS, build_from_images)


def _imgs(n=30):
    return np.random.default_rng(0).integers(0, 256, (n, 64, 64, 3), dtype=np.uint8)


def test_no_source_in_two_splits():
    d = build_from_images(_imgs(), seed=1)
    seen = {}
    for src, sp in zip(d["source"], d["split"]):
        seen.setdefault(int(src), set()).add(str(sp))
    assert all(len(v) == 1 for v in seen.values())


def test_held_out_attack_only_in_test():
    d = build_from_images(_imgs(), seed=1)
    blur = d["attack"] == HELD_OUT_ATTACK
    assert blur.any()
    assert set(d["split"][blur]) == {"test"}
    for sp in ("train", "val"):
        used = set(d["attack"][d["split"] == sp]) - {"none"}
        assert used <= set(SEEN_ATTACKS)


def test_labels_and_masks_consistent():
    d = build_from_images(_imgs(), seed=1)
    assert d["images"].dtype == np.uint8 and d["masks"].dtype == np.uint8
    assert d["images"].shape[1:] == (64, 64, 3) and d["masks"].shape[1:] == (64, 64)
    for lab, mask in zip(d["labels"], d["masks"]):
        assert (mask.sum() > 0) == bool(lab)


def test_each_source_has_authentic_and_tampered():
    d = build_from_images(_imgs(), seed=1)
    for s in np.unique(d["source"]):
        labs = d["labels"][d["source"] == s]
        assert (labs == 0).sum() == 1 and (labs == 1).sum() >= 1


def test_deterministic():
    a = build_from_images(_imgs(), seed=3)
    b = build_from_images(_imgs(), seed=3)
    assert np.array_equal(a["images"], b["images"]) and np.array_equal(a["masks"], b["masks"])
