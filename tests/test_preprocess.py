"""Tests unitaires du pretraitement (donnees synthetiques uniquement)."""
import numpy as np
import pytest
import rasterio
from PIL import Image

from src.preprocessing.preprocess import preprocess


def write_tif(path, arr, dtype):
    c, h, w = arr.shape
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=c, dtype=dtype) as dst:
        dst.write(arr.astype(dtype))


def test_rgb_uint8_png(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)).save(p)
    s = preprocess(p)
    assert s.image.shape == (3, 224, 224)
    assert s.image.dtype == np.float32
    assert s.band_names == ["R", "G", "B"]
    assert s.original_size == (40, 60)
    assert s.original_dtype == "uint8"


def test_s2_13_bands_mapping(tmp_path):
    p = tmp_path / "s2.tif"
    arr = np.stack([np.full((16, 16), i * 100) for i in range(13)])
    write_tif(p, arr, "uint16")
    s = preprocess(p, size=8, imagenet_norm=False)
    assert s.band_names == ["B04", "B03", "B02"]
    # indices 3, 2, 1 -> valeurs 300, 200, 100 -> /10000
    for ch, expected in enumerate([0.03, 0.02, 0.01]):
        assert s.image[ch] == pytest.approx(expected, abs=1e-6)


def test_single_band_replicated(tmp_path):
    p = tmp_path / "g.tif"
    write_tif(p, np.random.default_rng(1).random((1, 30, 20)), "float32")
    s = preprocess(p, imagenet_norm=False)
    assert s.band_names == ["gray", "gray", "gray"]
    assert np.array_equal(s.image[0], s.image[1])
    assert np.array_equal(s.image[1], s.image[2])


def test_unsupported_band_count_raises(tmp_path):
    p = tmp_path / "five.tif"
    write_tif(p, np.zeros((5, 10, 10)), "uint16")
    with pytest.raises(ValueError):
        preprocess(p)
    s = preprocess(p, band_indices=[4, 2, 0], size=8)
    assert s.band_names == ["band_5", "band_3", "band_1"]


def test_uint16_is_clipped(tmp_path):
    p = tmp_path / "hi.tif"
    write_tif(p, np.full((3, 10, 10), 20000), "uint16")
    s = preprocess(p, size=8, imagenet_norm=False)
    assert s.image.max() <= 1.0
    assert s.image.min() == pytest.approx(1.0)


def test_float_out_of_range_warns(tmp_path):
    p = tmp_path / "f.tif"
    write_tif(p, np.random.default_rng(2).random((1, 10, 10)) * 500 - 100, "float32")
    with pytest.warns(UserWarning):
        s = preprocess(p, size=8, imagenet_norm=False)
    assert 0.0 <= s.image.min() and s.image.max() <= 1.0
    assert s.normalization["method"] == "float_minmax_per_image"


def test_mask_stays_binary_and_resized(tmp_path):
    p = tmp_path / "m.tif"
    write_tif(p, np.random.default_rng(3).integers(0, 256, (3, 50, 80)), "uint8")
    mask = np.zeros((50, 80), dtype=np.uint8)
    mask[10:30, 20:60] = 1
    s = preprocess(p, mask=mask, label=1)
    assert s.mask.shape == (224, 224)
    assert s.mask.dtype == np.uint8
    assert set(np.unique(s.mask)) <= {0, 1}
    assert s.label == 1
