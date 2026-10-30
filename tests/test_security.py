"""Module securite : la suite de 53 controles et l'adaptateur vers le contrat A.
Tests de provenance et d'integrite sur donnees SYNTHETIQUES uniquement."""
from pathlib import Path

from src.security import run_security_cases
from src.security.adapter import assess_file
from src.utils.contracts import ValidationStatus

DATA = Path(run_security_cases.__file__).resolve().parent / "test_data"


def test_all_security_cases_pass(tmp_path):
    assert run_security_cases.main(["--data-dir", str(DATA), "--results-dir", str(tmp_path)]) == 0


def test_adapter_clean_image_passes():
    d = DATA / "scenes" / "01_clean_pass"
    r = assess_file(d / "scene_001.png", d / "scene_001.manifest.json")
    assert r.status is ValidationStatus.PASSED and r.can_proceed
    assert r.reason_codes == [] and len(r.sha256) == 64


def test_adapter_tampered_image_fails():
    d = DATA / "scenes" / "03_image_tampered"
    r = assess_file(d / "scene_003.png", d / "scene_003.manifest.json")
    assert r.status is ValidationStatus.FAILED and not r.can_proceed
    assert "HASH_MISMATCH" in r.reason_codes


def test_adapter_bad_metadata_goes_to_review():
    d = DATA / "scenes" / "10_bad_metadata_review"
    manifest = next(d.glob("*.manifest.json"))
    image = next(p for p in d.iterdir() if p.suffix in {".png", ".jpg", ".tif"})
    r = assess_file(image, manifest)
    assert r.status is ValidationStatus.REVIEW and not r.can_proceed


# --- GeoTIFF multibande (Sentinel-2) : decode par rasterio quand Pillow ne sait pas ---

def _s2_tif(path, bands=13, size=32):
    import numpy as np
    import rasterio
    arr = np.random.default_rng(0).integers(0, 10000, (bands, size, size)).astype("uint16")
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands,
                       dtype="uint16") as dst:
        dst.write(arr)


def _assess(img, manifest, sig, policy=None):
    from src.security import security_assessment as sa
    return sa.assess(img, manifest, sig, sa.load_trust_store(DATA / "trusted_keys.json"),
                     policy or sa.load_policy(DATA / "policy.json"), sa.load_ledger(DATA / "ledger.json"),
                     sa.load_reference_time(DATA / "reference_time.txt"))


def _signed_s2(tmp_path, bands=13):
    from datetime import timedelta
    from src.pipeline.sender import sign_image
    from src.security import security_assessment as sa
    now = sa.load_reference_time(DATA / "reference_time.txt")
    p = tmp_path / "s2.tif"
    _s2_tif(p, bands)
    m, s = sign_image(p, "s2", observed=now - timedelta(hours=2), signed_at=now - timedelta(hours=2))
    return p, m, s


def test_multiband_geotiff_passes_layer_one(tmp_path):
    p, m, s = _signed_s2(tmp_path, 13)
    r = _assess(p, m, s)
    assert r["security_status"] == "PASSED", r["reasons"]
    assert any("rasterio" in c.get("detail", "") for c in r["evidence"]["checks"])


def test_truncated_multiband_geotiff_fails(tmp_path):
    from datetime import timedelta
    from src.pipeline.sender import sign_image
    from src.security import security_assessment as sa
    now = sa.load_reference_time(DATA / "reference_time.txt")
    p = tmp_path / "cut.tif"
    _s2_tif(p, 13, 64)
    p.write_bytes(p.read_bytes()[:3000])               # tronque avant de signer
    m, s = sign_image(p, "cut", observed=now - timedelta(hours=2), signed_at=now - timedelta(hours=2))
    r = _assess(p, m, s)
    assert r["security_status"] == "FAILED" and r["reason_codes"][0] in ("FORMAT_DECODE",)


def test_multiband_geotiff_pixel_limit(tmp_path):
    from src.security import security_assessment as sa
    p, m, s = _signed_s2(tmp_path, 4)
    policy = dict(sa.load_policy(DATA / "policy.json"), max_pixels=100)   # 32x32 = 1024 px > 100
    r = _assess(p, m, s, policy)
    assert r["reason_codes"] == ["FORMAT_DIMENSIONS"]
