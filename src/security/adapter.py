"""Pont entre le module securite et le contrat A (ValidationReport).

    from src.security.adapter import assess_file
    report = assess_file(image, manifest)          # config : src/security/test_data
    if report.can_proceed: ...                     # seule une image PASSED continue

La verification de provenance ne detecte PAS les manipulations visuelles :
une image falsifiee avant d'etre signee par un emetteur de confiance passe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from src.security import security_assessment as sa
from src.utils.contracts import ValidationReport, ValidationStatus

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "test_data"
PathLike = Union[str, Path]


def to_validation_report(result: Mapping[str, Any]) -> ValidationReport:
    """Convertit le dict renvoye par security_assessment.assess()."""
    evidence = dict(result.get("evidence") or {})
    return ValidationReport(
        status=ValidationStatus(result["security_status"]),
        image_id=result.get("image_id"),
        sha256=evidence.get("image_sha256"),
        format_valid=result.get("format_valid"),
        signature_valid=result.get("signature_valid"),
        metadata_valid=result.get("metadata_valid"),
        replay_detected=result.get("replay_detected"),
        reason_codes=list(result.get("reason_codes") or []),
        reasons=list(result.get("reasons") or []),
        evidence=evidence,
    )


def assess_file(image_path: PathLike, manifest_path: PathLike,
                sig_path: Optional[PathLike] = None,
                config_dir: PathLike = DEFAULT_CONFIG_DIR,
                now: Optional[str] = None) -> ValidationReport:
    """Evalue une image avec la configuration du recepteur stockee dans config_dir
    (trusted_keys.json, policy.json, ledger.json, reference_time.txt)."""
    cfg = Path(config_dir)
    result = sa.assess(
        image_path, manifest_path,
        sig_path if sig_path is not None else sa.default_sig_path(manifest_path),
        trust_store=sa.load_trust_store(cfg / "trusted_keys.json"),
        policy=sa.load_policy(cfg / "policy.json"),
        ledger=sa.load_ledger(cfg / "ledger.json"),
        now=now or sa.load_reference_time(cfg / "reference_time.txt"),
    )
    return to_validation_report(result)
