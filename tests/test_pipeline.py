"""Chaine de bout en bout : emetteur emule, analyse a deux couches, demonstration,
mesures. Modele NON entraine : on verifie la logique, pas la qualite de detection."""
import json
from datetime import timedelta

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.forensic_unet import ForensicUNet
from src.pipeline.analyze import analyze, detect, load_rgb_uint8
from src.pipeline.sender import sign_image
from src.preprocessing.build_tampered_eurosat import build_from_images
from src.security import security_assessment as sa
from src.security.adapter import DEFAULT_CONFIG_DIR
from src.utils.contracts import ValidationStatus

NOW = sa.load_reference_time(DEFAULT_CONFIG_DIR / "reference_time.txt")
OBS = NOW - timedelta(hours=3)


@pytest.fixture
def model_file(tmp_path):
    torch.manual_seed(0)
    m = ForensicUNet(base=8)
    p = tmp_path / "m.pt"
    torch.save({"state_dict": m.state_dict(), "base": 8, "threshold": 0.5}, p)
    return p


@pytest.fixture
def tile(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    p = tmp_path / "t1.png"
    Image.fromarray(img).save(p)
    return p, img


def test_sender_output_is_accepted_by_security_module(tile):
    p, _ = tile
    m, s = sign_image(p, "t1", observed=OBS, signed_at=OBS)
    r = sa.assess(p, m, s, sa.load_trust_store(DEFAULT_CONFIG_DIR / "trusted_keys.json"),
                  sa.load_policy(DEFAULT_CONFIG_DIR / "policy.json"),
                  sa.load_ledger(DEFAULT_CONFIG_DIR / "ledger.json"), NOW)
    assert r["security_status"] == "PASSED", r["reason_codes"]


def test_sender_requires_aware_dates(tile):
    with pytest.raises(ValueError):
        sign_image(tile[0], "t1", observed=OBS.replace(tzinfo=None), signed_at=OBS)


def test_analyze_passed_then_tampered_in_transit(tile, model_file, tmp_path):
    p, img = tile
    m, _ = sign_image(p, "t1", observed=OBS, signed_at=OBS)
    rep = analyze(p, m, now=NOW, model_path=model_file, out_dir=str(tmp_path / "out"))
    assert rep.validation.status is ValidationStatus.PASSED
    assert rep.verdict in ("SUSPECTED", "NOT_DETECTED")
    assert rep.localization.mask_prob.shape == (64, 64)
    assert (tmp_path / "out" / "t1_report.html").exists()
    d = json.loads((tmp_path / "out" / "t1_report.json").read_text(encoding="utf-8"))
    assert d["validation"]["status"] == "PASSED" and d["timings_ms"]["total_ms"] > 0
    bad = img.copy()
    bad[:10, :10] = 0
    Image.fromarray(bad).save(p)                       # modifiee apres signature
    rep = analyze(p, m, now=NOW, model_path=model_file)
    assert rep.verdict == "REJECTED" and "HASH_MISMATCH" in rep.validation.reason_codes
    assert rep.classification is None                  # pas d'analyse d'un contenu non fiable


def test_detect_tiles_large_image(model_file):
    from src.pipeline.analyze import load_model
    model, _ = load_model(model_file)
    img = np.random.default_rng(1).integers(0, 256, (150, 100, 3), dtype=np.uint8)
    score, mask, t = detect(model, img)
    assert mask.shape == (150, 100) and 0 <= score <= 1 and t["tiles"] == 3 * 2


def test_load_rgb_uint8_png_and_multiband_tif(tmp_path):
    import rasterio
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "a.png")
    assert load_rgb_uint8(tmp_path / "a.png").shape == (8, 8, 3)
    arr = np.random.default_rng(0).integers(0, 10000, (13, 16, 16)).astype("uint16")
    with rasterio.open(tmp_path / "s2.tif", "w", driver="GTiff", height=16, width=16, count=13,
                       dtype="uint16") as dst:
        dst.write(arr)
    out = load_rgb_uint8(tmp_path / "s2.tif")
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8


@pytest.fixture
def tiny_npz(tmp_path):
    imgs = np.random.default_rng(5).integers(0, 256, (15, 64, 64, 3), dtype=np.uint8)
    p = tmp_path / "d.npz"
    np.savez_compressed(p, **build_from_images(imgs, seed=0))
    return p


def test_demo_layer_one_scenarios(tiny_npz, model_file, tmp_path):
    from src.pipeline.demo import run
    s = run(npz=str(tiny_npz), model_path=str(model_file), out_dir=str(tmp_path / "demo"))
    sc = s["scenarios"]
    assert sc["A_authentic"]["validation"]["status"] == "PASSED"
    assert sc["B_tampered_in_transit"]["verdict"] == "REJECTED"
    assert "HASH_MISMATCH" in sc["B_tampered_in_transit"]["validation"]["reason_codes"]
    assert sc["C_tampered_before_signing"]["validation"]["status"] == "PASSED"
    assert set(sc["D_replay"]["validation"]["reason_codes"]) >= {"REPLAY_ID", "REPLAY_CONTENT"}
    assert (tmp_path / "demo" / "index.html").exists()
    # le registre d'origine n'a pas ete modifie par la demo
    led = sa.load_ledger(DEFAULT_CONFIG_DIR / "ledger.json")
    assert all(e["image_id"] != "demo_a" for e in led["entries"])


def test_benchmark_runs(tiny_npz, model_file, tmp_path):
    from src.pipeline.benchmark import run
    r = run(npz=str(tiny_npz), model_path=str(model_file), n=5, out=str(tmp_path / "b.json"))
    assert r["n_images"] == 5 and r["warm"]["total_ms"]["mean"] > 0 and 0 < r["warm"]["security_overhead_percent"] < 100
    assert "cold_first_open" in r


def test_mixed_calibration_runs(tiny_npz, model_file, tmp_path):
    from src.models.evaluate_evasion import run
    r = run(npz=str(tiny_npz), model_path=str(model_file), out=str(tmp_path / "e.json"), calibration="mixed")
    assert r["calibration"] == "mixed" and 0 < r["threshold"] < 1


def test_far5_calibration_saves_deployable_model(tiny_npz, model_file, tmp_path):
    from src.models.evaluate_evasion import run
    r = run(npz=str(tiny_npz), model_path=str(model_file), out=str(tmp_path / "e.json"), calibration="far5")
    saved = torch.load(model_file.with_name(f"{model_file.stem}_far5.pt"))
    assert saved["calibration"] == "far5" and saved["threshold"] == pytest.approx(r["threshold"])


def test_analyze_multiband_geotiff_end_to_end(model_file, tmp_path):
    import rasterio
    p = tmp_path / "s2.tif"
    arr = np.random.default_rng(0).integers(0, 10000, (13, 64, 64)).astype("uint16")
    with rasterio.open(p, "w", driver="GTiff", height=64, width=64, count=13, dtype="uint16") as dst:
        dst.write(arr)
    m, _ = sign_image(p, "s2", observed=OBS, signed_at=OBS)
    rep = analyze(p, m, now=NOW, model_path=model_file)
    assert rep.validation.status is ValidationStatus.PASSED, rep.validation.reason_codes
    assert rep.classification is not None and rep.localization.mask_prob.shape == (64, 64)
