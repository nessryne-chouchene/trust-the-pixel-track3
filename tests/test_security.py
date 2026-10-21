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
