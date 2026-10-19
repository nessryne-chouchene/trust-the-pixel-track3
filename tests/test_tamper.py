"""Tests du generateur de falsifications (images synthetiques)."""
import numpy as np
import pytest

from src.preprocessing.tamper import ATTACK_TYPES, tamper


def _img(seed):
    return np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.mark.parametrize("attack", ATTACK_TYPES)
def test_mask_exact_and_outside_untouched(attack):
    img = _img(0)
    out, mask = tamper(img, attack, np.random.default_rng(1), donor=_img(2))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert mask.shape == (64, 64) and mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}
    outside = mask == 0
    assert np.array_equal(out[outside], img[outside])
    assert (out[mask == 1] != img[mask == 1]).any()
    assert 0.03 <= mask.mean() <= 0.17


def test_deterministic_with_same_seed():
    a = tamper(_img(0), "color_shift", np.random.default_rng(5))
    b = tamper(_img(0), "color_shift", np.random.default_rng(5))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_splice_requires_donor():
    with pytest.raises(ValueError):
        tamper(_img(0), "splice", np.random.default_rng(0))


def test_unknown_attack_and_bad_input_raise():
    with pytest.raises(ValueError):
        tamper(_img(0), "unknown", np.random.default_rng(0))
    with pytest.raises(ValueError):
        tamper(_img(0).astype(np.float32), "blur", np.random.default_rng(0))
