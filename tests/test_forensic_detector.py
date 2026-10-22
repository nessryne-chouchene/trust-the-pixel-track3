"""Tests des traces forensiques, du ForensicUNet, des metriques et des
entrainements (donnees SYNTHETIQUES uniquement)."""
import math

import numpy as np
import pytest
import torch

from src.forensics.signals import N_CHANNELS, ela_map, forensic_inputs, srm_residuals
from src.models.forensic_unet import ForensicUNet
from src.preprocessing.build_tampered_eurosat import build_from_images
from src.utils.contracts import ClassificationOutput, LocalizationOutput
from src.utils.metrics import auc_score, recall_at_false_alarm


def _imgs(n=4, h=64, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, h, w, 3), dtype=np.uint8)


# --- traces forensiques -------------------------------------------------------

def test_srm_is_zero_on_flat_image():
    flat = np.full((1, 32, 32, 3), 128, dtype=np.uint8)
    assert torch.allclose(srm_residuals(flat), torch.zeros(1, 3, 32, 32), atol=1e-4)


def test_srm_and_ela_shapes_and_ranges():
    x = _imgs(3, 40, 56)
    s, e = srm_residuals(x), ela_map(x)
    assert tuple(s.shape) == (3, 3, 40, 56) and s.abs().max() <= 1.0
    assert tuple(e.shape) == (3, 1, 40, 56) and 0.0 <= e.min() and e.max() <= 1.0


def test_forensic_inputs_stack():
    x = forensic_inputs(_imgs(2))
    assert tuple(x.shape) == (2, N_CHANNELS, 64, 64) and x.dtype == torch.float32
    assert torch.isfinite(x).all()


def test_signals_reject_bad_input():
    with pytest.raises(ValueError):
        srm_residuals(np.zeros((2, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        forensic_inputs(np.zeros((1, 8, 8, 3), dtype=np.float32))


def test_ela_reacts_to_local_edit():
    """Une zone retouchee apres compression JPEG ressort dans la carte ELA."""
    import io
    from PIL import Image
    base = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(base).save(buf, format="JPEG", quality=90)
    jpeg = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).copy()
    edited = jpeg.copy()
    edited[16:48, 16:48] = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    e = ela_map(edited[None])[0, 0].numpy()
    inside, outside = e[20:44, 20:44].mean(), np.concatenate([e[:12].ravel(), e[52:].ravel()]).mean()
    assert inside > outside


# --- modele -------------------------------------------------------------------

def test_unet_shapes_and_padding():
    m = ForensicUNet(base=8)
    c, k = m(torch.randn(2, N_CHANNELS, 64, 64))
    assert tuple(c.shape) == (2,) and tuple(k.shape) == (2, 1, 64, 64)
    c, k = m(torch.randn(1, N_CHANNELS, 50, 70))  # non multiple de 8
    assert tuple(k.shape) == (1, 1, 50, 70)


def test_unet_backward_and_bad_input():
    m = ForensicUNet(base=8)
    c, k = m(torch.randn(2, N_CHANNELS, 32, 32))
    (c.sum() + k.sum()).backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 32, 32))


def test_unet_predict_follows_contracts():
    outs = ForensicUNet(base=8).predict(torch.randn(2, N_CHANNELS, 32, 32))
    cls, loc = outs[0]
    assert isinstance(cls, ClassificationOutput) and cls.decision is None
    assert isinstance(loc, LocalizationOutput) and loc.mask_binary is None
    assert loc.mask_prob.shape == (32, 32) and loc.mask_prob.dtype == np.float32
    cls, loc = ForensicUNet(base=8).predict(torch.randn(1, N_CHANNELS, 32, 32), 0.5, 0.5)[0]
    assert isinstance(cls.decision, bool) and loc.mask_binary.dtype == np.uint8


# --- metriques ----------------------------------------------------------------

def test_auc_known_values_and_ties():
    assert auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert auc_score([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert math.isnan(auc_score([1, 1], [0.2, 0.3]))


def test_recall_at_false_alarm():
    y = [0] * 10 + [1] * 4
    s = [i / 10 for i in range(10)] + [0.95, 0.85, 0.5, 0.05]
    r = recall_at_false_alarm(y, s, 0.1)  # au plus 1 fausse alarme sur 10
    assert r["far"] <= 0.1 and r["recall"] == pytest.approx(0.5)


# --- entrainements de bout en bout (minuscules) -------------------------------

@pytest.fixture
def tiny_npz(tmp_path):
    d = build_from_images(_imgs(15, seed=3), seed=0)
    p = tmp_path / "d.npz"
    np.savez_compressed(p, **d)
    return p, d


def test_train_forensic_tiny(tiny_npz, tmp_path):
    from src.models.train_forensic import run
    p, d = tiny_npz
    res = run(npz=str(p), epochs=1, batch=8, base=8, out_dir=str(tmp_path / "out"), verbose=False)
    assert res["per_attack"]["blur"]["seen_in_training"] is False
    assert sum(res["split_sizes"].values()) == len(d["labels"])
    assert res["latency"]["total_ms"] > 0
    assert (tmp_path / "out" / "forensic_unet.pt").exists()
    assert (tmp_path / "out" / "forensic_unet_run.json").exists()


def test_train_eurosat_baseline_tiny(tiny_npz, tmp_path):
    from src.models.train_eurosat import run
    p, d = tiny_npz
    res = run(npz=str(p), epochs=1, batch=8, pretrained=False,
              cache=str(tmp_path / "f.pt"), out_dir=str(tmp_path / "out"))
    assert "blur" in res["per_attack"] and res["per_attack"]["blur"]["seen_in_training"] is False


# --- evasion ------------------------------------------------------------------

def test_laundering_keeps_shape_and_dtype():
    from src.forensics.evasion import LAUNDERING, launder
    x = _imgs(3)
    for name in LAUNDERING:
        y = launder(x, name)
        assert y.shape == x.shape and y.dtype == np.uint8
    assert np.array_equal(launder(x, "none"), x)
    with pytest.raises(ValueError):
        launder(x, "unknown")


def test_feather_blends_only_inside_mask_area():
    from src.forensics.evasion import feather
    auth = np.zeros((32, 32, 3), np.uint8)
    tampered = np.full((32, 32, 3), 200, np.uint8)
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 1
    out = feather(tampered, auth, mask)
    assert out[0, 0].max() == 0 and out[16, 16].min() > 150 and 0 < out[8, 16, 0] < 200


def test_robust_training_and_evasion_eval(tiny_npz, tmp_path):
    from src.models.evaluate_evasion import run as eval_run
    from src.models.train_forensic import run
    p, _ = tiny_npz
    run(npz=str(p), epochs=1, batch=8, base=8, out_dir=str(tmp_path), robust=True, verbose=False)
    assert (tmp_path / "forensic_unet_robust.pt").exists()
    res = eval_run(npz=str(p), model_path=str(tmp_path / "forensic_unet_robust.pt"),
                   out=str(tmp_path / "ev.json"))
    assert {"none", "jpeg_q75", "adaptive_feathered_splice"} <= set(res["results"])


def test_threshold_takes_middle_of_optimal_range():
    from src.models.evaluation import choose_threshold
    t = choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])  # tout seuil dans ]0.2, 0.8] est optimal
    assert 0.4 <= t <= 0.6
