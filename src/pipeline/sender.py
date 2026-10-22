"""Emetteur emule (segment sol) : signe une image avant son envoi (v0.1).

C'est l'ancre de confiance LOGICIELLE de la demonstration (le cahier des charges
accepte une ancre logicielle ou emulee). Elle produit, a cote de l'image :
  <image_id>.manifest.json  (hash SHA-256, taille, metadonnees, cle, nonce, date)
  <image_id>.manifest.sig   (signature Ed25519 detachee, en base64)

ATTENTION : les cles utilisees ici sont les cles de TEST JETABLES du module de
securite (derivees d'un libelle public). Elles ne protegent rien en reel ; un
deploiement utiliserait une cle privee gardee dans un module materiel (HSM/TPM).
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.security.generate_test_dataset import derive_test_key
from src.security.security_assessment import format_utc, signing_input

DEFAULT_KEY = "sat-alpha-2026a"
DEFAULT_SENSOR = "SAT-ALPHA-01"


def build_manifest(image_path: Path, image_id: str, *, observed: datetime, signed_at: datetime,
                   key_id: str, sensor_id: str, latitude: float, longitude: float,
                   nonce: Optional[str] = None) -> dict:
    data = Path(image_path).read_bytes()
    return {
        "manifest_version": 1,
        "image_id": image_id,
        "file_name": Path(image_path).name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "metadata": {"timestamp": format_utc(observed), "sensor_id": sensor_id,
                     "latitude": latitude, "longitude": longitude},
        "key_id": key_id,
        "nonce": nonce or secrets.token_hex(16),
        "signed_at": format_utc(signed_at),
    }


def sign_image(image_path, image_id: str, *, observed: datetime, signed_at: datetime,
               key_id: str = DEFAULT_KEY, sensor_id: str = DEFAULT_SENSOR,
               latitude: float = 48.137, longitude: float = 11.575,
               nonce: Optional[str] = None) -> tuple[Path, Path]:
    """Ecrit le manifeste et sa signature a cote de l'image ; retourne leurs chemins."""
    image_path = Path(image_path)
    for dt in (observed, signed_at):
        if dt.tzinfo is None:
            raise ValueError("Les dates doivent avoir un fuseau horaire (UTC).")
    manifest = build_manifest(image_path, image_id, observed=observed.astimezone(timezone.utc),
                              signed_at=signed_at.astimezone(timezone.utc), key_id=key_id,
                              sensor_id=sensor_id, latitude=latitude, longitude=longitude, nonce=nonce)
    signature = derive_test_key(key_id).sign(signing_input(manifest))
    m_path = image_path.with_name(f"{image_id}.manifest.json")
    s_path = image_path.with_name(f"{image_id}.manifest.sig")
    m_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s_path.write_text(base64.b64encode(signature).decode("ascii") + "\n", encoding="ascii")
    return m_path, s_path
