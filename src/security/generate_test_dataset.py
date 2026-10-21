"""Generate the synthetic TSYP14 security-assessment test dataset.

Creates ``test_data/`` next to this script:

    test_data/
      trusted_keys.json    public keys only (receiver trust store)
      policy.json          thresholds and rules (guide 3.3)
      ledger.json          replay memory (guide 3.4)
      reference_time.txt   fixed "now" = 2026-10-13T12:00:00Z
      CASES.md             human-readable table of every case
      scenes/              reproductions of the 13 scenes of guide section 7
      edge_cases/          extra cases for the checks marked "no scene"

Every case folder holds the image, ``<image_id>.manifest.json``,
``<image_id>.manifest.sig`` (unless the case is about a missing file),
``case.json`` (which files to feed to assess(), plus an optional policy
override) and ``expected.json`` (the answer key, derived from the guide).

TEST KEYS ARE DISPOSABLE. Each Ed25519 private key is derived from a public,
fixed label so the dataset is reproducible. Anyone can recompute them, so they
must never be used for anything real. They exist only inside this script and
are never written to disk. The production module never sees them.

This generator deliberately does NOT import security_assessment.py: the
signing input is built by its own implementation (checked against a
hard-coded byte string below), so a serialisation bug in the production code
cannot be silently mirrored here.

The images are harmless synthetic pictures (blurred fields, a road and a
river). They are not real satellite imagery, and their hashes and sizes do not
match the official sample_input/ files described in the guide.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import shutil
import struct
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "test_data"
MARKER = ".generated_by_generate_test_dataset"

REFERENCE_TIME = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)
DOMAIN_PREFIX = b"TSYP14-IMG-MANIFEST-v1\n"

ALPHA, BRAVO, ALPHA_OLD = "sat-alpha-2026a", "sat-bravo-2026a", "sat-alpha-2025x"
ROGUE, ATTACKER = "sat-rogue-99", "attacker"
CHARLIE_EXPIRED, DELTA_FUTURE = "sat-charlie-2024", "sat-delta-2027"


# --------------------------------------------------------------------------
# Signing input - independent implementation of guide section 4.2
# --------------------------------------------------------------------------

def canonical_signing_bytes(manifest: dict[str, Any]) -> bytes:
    """Prefix line + JSON with sorted keys, no whitespace, raw UTF-8."""
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return DOMAIN_PREFIX + text.encode("utf-8")


def _selftest_canonical() -> None:
    """Pin the exact byte layout so a wrong serialisation cannot go unnoticed."""
    sample = {"b": 1, "a": "été", "c": [1.5, {"z": None, "y": True}]}
    expected = (
        b"TSYP14-IMG-MANIFEST-v1\n"
        b'{"a":"\xc3\xa9t\xc3\xa9","b":1,"c":[1.5,{"y":true,"z":null}]}'
    )
    got = canonical_signing_bytes(sample)
    if got != expected:
        raise SystemExit(f"canonical signing input self-test failed: {got!r}")


# --------------------------------------------------------------------------
# Disposable test keys
# --------------------------------------------------------------------------

def derive_test_key(label: str) -> Ed25519PrivateKey:
    """DISPOSABLE TEST KEY derived from a public label. Never use for real."""
    seed = hashlib.sha256(b"TSYP14-DISPOSABLE-TEST-KEY:" + label.encode("utf-8")).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


def public_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


KEYS = {label: derive_test_key(label)
        for label in (ALPHA, BRAVO, ALPHA_OLD, ROGUE, ATTACKER, CHARLIE_EXPIRED, DELTA_FUTURE)}


def sign(manifest: dict[str, Any], key_label: str) -> str:
    """Base64 detached Ed25519 signature over the signing input."""
    return base64.b64encode(KEYS[key_label].sign(canonical_signing_bytes(manifest))).decode("ascii") + "\n"


# --------------------------------------------------------------------------
# Synthetic images
# --------------------------------------------------------------------------

_FIELD_COLOURS = [(86, 125, 52), (110, 140, 60), (70, 105, 45), (140, 115, 75),
                  (155, 130, 90), (120, 95, 60), (95, 130, 70), (165, 150, 100)]


def make_scene_image(seed: int, size: int = 256) -> Image.Image:
    """Blurred green/brown fields, a grey road and a blue river."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), rng.choice(_FIELD_COLOURS))
    draw = ImageDraw.Draw(img)
    for _ in range(18):
        x0, y0 = rng.randrange(-20, size), rng.randrange(-20, size)
        w, h = rng.randrange(30, 120), rng.randrange(30, 120)
        r, g, b = rng.choice(_FIELD_COLOURS)
        jitter = rng.randrange(-12, 13)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(r + jitter, g + jitter, b + jitter))
    img = img.filter(ImageFilter.GaussianBlur(3))
    draw = ImageDraw.Draw(img)
    y_road = rng.randrange(40, size - 40)
    draw.line([(0, y_road), (size, y_road + rng.randrange(-60, 61))], fill=(150, 148, 140), width=3)
    base, amp = rng.randrange(60, size - 60), rng.randrange(10, 35)
    period, phase = rng.uniform(25, 60), rng.uniform(0, 6.28)
    river = [(x, base + amp * math.sin(x / period + phase)) for x in range(-8, size + 9, 4)]
    draw.line(river, fill=(45, 85, 170), width=6, joint="curve")
    return img.filter(ImageFilter.GaussianBlur(0.8))


def encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG")
    elif fmt == "TIFF":
        img.save(buf, format="TIFF", compression="raw")
    elif fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
    elif fmt == "BMP":
        img.save(buf, format="BMP")
    else:
        raise ValueError(fmt)
    return buf.getvalue()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def bomb_png_header(width: int, height: int) -> bytes:
    """A tiny PNG whose header claims a gigantic canvas (decompression bomb)."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(b"\x00" * 1024)) + _png_chunk(b"IEND", b""))


def huge_bilevel_png(width: int, height: int) -> bytes:
    """A genuinely decodable but over-sized (width*height) PNG, small on disk."""
    img = Image.new("1", (width, height), 0)
    ImageDraw.Draw(img).rectangle([10, 10, 200, 200], fill=1)
    return encode(img, "PNG")


# --------------------------------------------------------------------------
# Manifests and cases
# --------------------------------------------------------------------------

def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def hours_ago(hours: float) -> datetime:
    return REFERENCE_TIME - timedelta(hours=hours)


def build_manifest(image_id: str, file_name: str, image_bytes: bytes, *, observed: datetime,
                   sensor: Any, lat: Any, lon: Any, key_id: str,
                   signed_at: Optional[datetime] = None, timestamp: Optional[str] = None) -> dict[str, Any]:
    """Manifest in the layout of guide 3.6. signed_at defaults to observed + 5 min."""
    signed = signed_at or observed + timedelta(minutes=5)
    return {
        "manifest_version": 1,
        "image_id": image_id,
        "file_name": file_name,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "size_bytes": len(image_bytes),
        "metadata": {
            "timestamp": timestamp if timestamp is not None else ts(observed),
            "sensor_id": sensor,
            "latitude": lat,
            "longitude": lon,
        },
        "key_id": key_id,
        "nonce": hashlib.sha256(f"nonce:{key_id}:{image_id}".encode()).hexdigest()[:32],
        "signed_at": ts(signed),
    }


def expected(image_id: Optional[str], fmt: Optional[bool], sig: Optional[bool], meta: Optional[bool],
             replay: Optional[bool], status: str, codes: list[str]) -> dict[str, Any]:
    return {"image_id": image_id, "format_valid": fmt, "signature_valid": sig, "metadata_valid": meta,
            "replay_detected": replay, "security_status": status, "reason_codes": codes}


CASES: list[dict[str, Any]] = []


def write_case(group: str, folder: str, *, image_name: str, image_bytes: Optional[bytes],
               manifest: Any, signature: Optional[str], exp: dict[str, Any],
               description: str, spec: str, manifest_name: Optional[str] = None,
               policy_override: Optional[dict[str, Any]] = None) -> None:
    """Write one case folder.

    ``manifest`` is a dict (written as indented JSON - deliberately NOT the
    canonical form, so the checker must re-canonicalise), a raw string (for
    malformed manifests) or None (no manifest file). ``signature`` None means
    no .sig file. ``image_bytes`` None means no image file.
    """
    d = OUT / group / folder
    d.mkdir(parents=True)
    stem = image_name.rsplit(".", 1)[0]
    manifest_name = manifest_name or f"{stem}.manifest.json"
    sig_name = manifest_name[: -len(".json")] + ".sig"
    if image_bytes is not None:
        (d / image_name).write_bytes(image_bytes)
    if isinstance(manifest, dict):
        (d / manifest_name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif isinstance(manifest, str):
        (d / manifest_name).write_text(manifest, encoding="utf-8")
    if signature is not None:
        (d / sig_name).write_text(signature, encoding="ascii")
    case = {"case_id": folder, "group": group, "description": description, "spec_reference": spec,
            "image": image_name, "manifest": manifest_name, "signature": sig_name}
    if policy_override:
        case["policy_override"] = policy_override
    (d / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    (d / "expected.json").write_text(json.dumps(exp, indent=2) + "\n", encoding="utf-8")
    CASES.append({**case, "expected": exp})


def signed_case(group: str, folder: str, n: int, *, exp_builder, description: str, spec: str,
                fmt: str = "PNG", ext: str = "png", key: str = ALPHA, sign_with: Optional[str] = None,
                sensor: Any = "SAT-ALPHA-01", lat: Any = 36.8, lon: Any = 10.8, age_h: float = 9,
                image_id: Optional[str] = None, image_bytes: Optional[bytes] = None,
                policy_override: Optional[dict[str, Any]] = None, **manifest_kw: Any) -> None:
    """The common case: a fresh image, a correct manifest, a correct signature."""
    image_id = image_id or f"scene_{n:03d}"
    data = image_bytes if image_bytes is not None else encode(make_scene_image(1000 + n), fmt)
    name = f"{image_id}.{ext}"
    m = build_manifest(image_id, name, data, observed=hours_ago(age_h), sensor=sensor, lat=lat, lon=lon,
                       key_id=key, **manifest_kw)
    write_case(group, folder, image_name=name, image_bytes=data, manifest=m,
               signature=sign(m, sign_with or key), exp=exp_builder(image_id),
               description=description, spec=spec, policy_override=policy_override)


# --------------------------------------------------------------------------
# Shared receiver-side files
# --------------------------------------------------------------------------

LEDGER_PICTURE = encode(make_scene_image(99), "PNG")  # the picture accepted as scene_099


def write_shared_files() -> None:
    def key_entry(label: str, sensors: list[str], valid_from: str, valid_until: str,
                  status: str = "active", **extra: str) -> dict[str, Any]:
        return {"key_id": label, "public_key_hex": public_hex(KEYS[label]), "status": status,
                **extra, "allowed_sensors": sensors, "valid_from": valid_from, "valid_until": valid_until}

    trust = {"keys": [
        key_entry(ALPHA, ["SAT-ALPHA-01", "SAT-ALPHA-02"], "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z"),
        key_entry(BRAVO, ["SAT-BRAVO-01"], "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z"),
        key_entry(ALPHA_OLD, ["SAT-ALPHA-01"], "2025-01-01T00:00:00Z", "2027-01-01T00:00:00Z",
                  status="revoked", revoked_at="2026-09-01T00:00:00Z"),
        # Extra keys (not in the guide) for the KEY_NOT_VALID edge cases.
        key_entry(CHARLIE_EXPIRED, ["SAT-CHARLIE-01"], "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
        key_entry(DELTA_FUTURE, ["SAT-DELTA-01"], "2027-01-01T00:00:00Z", "2028-01-01T00:00:00Z"),
    ]}
    policy = {"version": "2026.10-demo", "allowed_formats": ["tif", "png", "jpg"],
              "max_file_size_mb": 100, "max_pixels": 40000000, "max_age_hours": 72,
              "clock_skew_seconds": 300, "metadata_failure_action": "REVIEW",
              "min_file_size_bytes": 1}
    ledger = {"entries": [
        {"image_id": "scene_000", "sha256": hashlib.sha256(b"previously accepted image").hexdigest(),
         "accepted_at": "2026-10-10T08:00:00Z"},
        {"image_id": "scene_099", "sha256": hashlib.sha256(LEDGER_PICTURE).hexdigest(),
         "accepted_at": "2026-10-12T21:00:00Z"},
    ]}
    for name, obj in (("trusted_keys.json", trust), ("policy.json", policy), ("ledger.json", ledger)):
        (OUT / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    (OUT / "reference_time.txt").write_text(ts(REFERENCE_TIME) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# The 13 scenes of guide section 7
# --------------------------------------------------------------------------

def ok(i): return expected(i, True, True, True, False, "PASSED", [])
def sig_fail(code): return lambda i: expected(i, True, False, True, False, "FAILED", [code])
def meta_review(*codes): return lambda i: expected(i, True, True, False, False, "REVIEW", list(codes))
def fmt_fail(code): return lambda i: expected(i, False, None, None, None, "FAILED", [code])


def write_official_scenes() -> None:
    g = "scenes"
    signed_case(g, "01_clean_pass", 1, exp_builder=ok, sensor="SAT-ALPHA-02", lat=36.8065, lon=10.1815, age_h=6,
                description="Clean PNG signed by sat-alpha-2026a.", spec="guide 7.3 scene 01")
    signed_case(g, "02_clean_pass_bravo_tif", 2, exp_builder=ok, fmt="TIFF", ext="tif", key=BRAVO,
                sensor="SAT-BRAVO-01", lat=34.7406, lon=10.7603, age_h=10,
                description="Clean uncompressed TIFF signed by a second key for its own sensor.", spec="guide 7.3 scene 02")

    # 03: sign the original, then draw a red rectangle on the picture.
    img = make_scene_image(1003)
    original = encode(img, "PNG")
    m = build_manifest("scene_003", "scene_003.png", original, observed=hours_ago(8),
                       sensor="SAT-ALPHA-01", lat=35.8256, lon=10.6084, key_id=ALPHA)
    ImageDraw.Draw(img).rectangle([100, 100, 140, 130], fill=(220, 20, 20))
    write_case(g, "03_image_tampered", image_name="scene_003.png", image_bytes=encode(img, "PNG"),
               manifest=m, signature=sign(m, ALPHA), exp=sig_fail("HASH_MISMATCH")("scene_003"),
               description="Image pixels changed after signing; manifest and signature intact.", spec="guide 7.3 scene 03")

    # 04: sign with latitude 36.40, then edit it to 36.90.
    data = encode(make_scene_image(1004), "PNG")
    m = build_manifest("scene_004", "scene_004.png", data, observed=hours_ago(7),
                       sensor="SAT-ALPHA-02", lat=36.40, lon=10.50, key_id=ALPHA)
    sig = sign(m, ALPHA)
    m["metadata"]["latitude"] = 36.90
    write_case(g, "04_manifest_tampered", image_name="scene_004.png", image_bytes=data, manifest=m,
               signature=sig, exp=sig_fail("SIG_INVALID")("scene_004"),
               description="Manifest latitude edited after signing.", spec="guide 7.3 scene 04")

    signed_case(g, "05_forged_signature", 5, exp_builder=sig_fail("SIG_INVALID"), sign_with=ATTACKER, lat=36.0, lon=10.0,
                description="Claims sat-alpha-2026a but signed with the attacker's key.", spec="guide 7.3 scene 05")
    signed_case(g, "06_unknown_key", 6, exp_builder=sig_fail("KEY_UNKNOWN"), key=ROGUE, lat=36.1, lon=10.1,
                description="Correctly signed by sat-rogue-99, which is not in the trust store. "
                            "Sensor exists elsewhere in the store, so metadata is valid.", spec="guide 7.3 scene 06")
    signed_case(g, "07_revoked_key", 7, exp_builder=sig_fail("KEY_REVOKED"), key=ALPHA_OLD, lat=36.2, lon=10.2,
                description="Mathematically valid signature from the revoked key sat-alpha-2025x.", spec="guide 7.3 scene 07")
    signed_case(g, "08_replay_same_id", 8, image_id="scene_099", lat=36.3, lon=10.3,
                exp_builder=lambda i: expected(i, True, True, True, True, "FAILED", ["REPLAY_ID"]),
                description="New picture reusing the already-accepted image_id scene_099.", spec="guide 7.3 scene 08")
    signed_case(g, "09_replay_same_content", 9, image_bytes=LEDGER_PICTURE, lat=36.4, lon=10.4,
                exp_builder=lambda i: expected(i, True, True, True, True, "FAILED", ["REPLAY_CONTENT"]),
                description="Fresh image_id, but the bytes of the picture accepted as scene_099.", spec="guide 7.3 scene 09")
    signed_case(g, "10_bad_metadata_review", 10, sensor="SAT-UNKNOWN", lat=123.4, lon=10.18,
                exp_builder=meta_review("META_LATITUDE", "META_SENSOR"),
                description="Valid signature; latitude out of range and sensor not allowed for the key.", spec="guide 7.3 scene 10")
    signed_case(g, "11_stale_timestamp_review", 11, sensor="SAT-ALPHA-02", lat=36.5, lon=10.5, age_h=100,
                exp_builder=meta_review("META_TIME_STALE"),
                description="Observation 100 h old (limit 72 h).", spec="guide 7.3 scene 11")

    text = b"Quarterly report draft. This is a text file renamed to .png."
    assert len(text) == 60
    signed_case(g, "12_fake_format", 12, image_bytes=text, lat=36.6, lon=10.6, exp_builder=fmt_fail("FORMAT_MAGIC"),
                description="60-byte text file renamed to .png; manifest and signature consistent with it.",
                spec="guide 7.3 scene 12")

    data = encode(make_scene_image(1013), "PNG")
    m = build_manifest("scene_013", "scene_013.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.7, lon=10.7, key_id=ALPHA)
    write_case(g, "13_missing_signature", image_name="scene_013.png", image_bytes=data, manifest=m,
               signature=None, exp=sig_fail("SIG_MISSING")("scene_013"),
               description="Image and manifest present, no .manifest.sig file.", spec="guide 7.3 scene 13")


# --------------------------------------------------------------------------
# Extra edge cases (guide 4.x checks marked "no scene", guide section 10)
# --------------------------------------------------------------------------

def write_edge_cases() -> None:
    g = "edge_cases"
    base = make_scene_image(2000)

    # ---- Step 1: format -------------------------------------------------
    signed_case(g, "14_empty_file", 14, image_bytes=b"", exp_builder=fmt_fail("FORMAT_SIZE"),
                description="Zero-byte .png (below policy.min_file_size_bytes).",
                spec="guide 10 stretch goal: reject empty files -> FORMAT_SIZE (documented choice)")
    full = encode(base, "PNG")
    signed_case(g, "15_truncated_png", 15, image_bytes=full[: len(full) * 6 // 10], exp_builder=fmt_fail("FORMAT_DECODE"),
                description="PNG cut to 60 % of its length; header parses, full decode fails.",
                spec="guide 4.1 'full decode succeeds' -> FORMAT_DECODE")
    signed_case(g, "16_disallowed_extension", 16, fmt="BMP", ext="bmp", exp_builder=fmt_fail("FORMAT_EXTENSION"),
                description="Valid BMP image; bmp is not in allowed_formats.", spec="guide 4.1 FORMAT_EXTENSION")
    signed_case(g, "17_png_magic_garbage", 17,
                image_bytes=b"\x89PNG\r\n\x1a\n" + random.Random(17).randbytes(4000),
                exp_builder=fmt_fail("FORMAT_DECODE"),
                description="Correct PNG magic followed by random bytes; the image library cannot open it.",
                spec="guide 4.1 'image library reports the same format' -> FORMAT_DECODE")
    signed_case(g, "18_oversized_file", 18, exp_builder=fmt_fail("FORMAT_SIZE"),
                policy_override={"max_file_size_mb": 0.001},
                description="Normal PNG (several KB) assessed with max_file_size_mb = 0.001 (~1 KB) to avoid a 100 MB fixture.",
                spec="guide 4.1 FORMAT_SIZE")
    signed_case(g, "19_excessive_dimensions", 19, image_bytes=huge_bilevel_png(7000, 6000),
                exp_builder=fmt_fail("FORMAT_DIMENSIONS"),
                description="Decodable 7000x6000 (42 Mpx) 1-bit PNG, a few KB on disk; limit is 40 Mpx.",
                spec="guide 4.1 FORMAT_DIMENSIONS")
    signed_case(g, "20_decompression_bomb_header", 20, image_bytes=bomb_png_header(100000, 100000),
                exp_builder=fmt_fail("FORMAT_DIMENSIONS"),
                description="Tiny PNG whose IHDR claims 100000x100000 px; must be rejected from the header.",
                spec="guide 4.1 FORMAT_DIMENSIONS (decompression bomb guard)")
    signed_case(g, "21_jpeg_renamed_png", 21, image_bytes=encode(base, "JPEG"), exp_builder=fmt_fail("FORMAT_MAGIC"),
                description="Real JPEG saved with a .png extension.", spec="guide 4.1 FORMAT_MAGIC")

    # ---- Step 2: manifest / signature -----------------------------------
    data = encode(make_scene_image(1022), "PNG")
    write_case(g, "22_malformed_manifest_json", image_name="scene_022.png", image_bytes=data,
               manifest='{"manifest_version": 1, "image_id": "scene_022", "file_name": ',
               signature=base64.b64encode(b"\x00" * 64).decode() + "\n",
               exp=expected(None, True, False, None, False, "FAILED", ["MANIFEST_INVALID"]),
               description="Truncated, unparseable manifest JSON. No manifest object, so metadata cannot run (null).",
               spec="guide 4.2 MANIFEST_INVALID; null-when-not-run rule (documented choice)")

    data = encode(make_scene_image(1023), "PNG")
    m = build_manifest("scene_023", "scene_023.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    del m["nonce"]
    write_case(g, "23_manifest_missing_field", image_name="scene_023.png", image_bytes=data, manifest=m,
               signature=sign(m, ALPHA), exp=sig_fail("MANIFEST_INVALID")("scene_023"),
               description="Validly signed manifest without the required 'nonce' field.", spec="guide 4.2 MANIFEST_INVALID")

    data = encode(make_scene_image(1024), "PNG")
    m = build_manifest("scene_024", "scene_024.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    m["manifest_version"] = 2
    write_case(g, "24_manifest_bad_version", image_name="scene_024.png", image_bytes=data, manifest=m,
               signature=sign(m, ALPHA), exp=sig_fail("MANIFEST_INVALID")("scene_024"),
               description="Validly signed manifest with manifest_version 2.", spec="guide 3.6 'Must be 1' -> MANIFEST_INVALID")

    data = encode(make_scene_image(1025), "PNG")
    m = build_manifest("scene_025", "scene_025.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    raw = json.dumps(m, indent=2)
    raw = raw.replace('"image_id": "scene_025",', '"image_id": "scene_025",\n  "image_id": "scene_125",', 1)
    write_case(g, "25_manifest_duplicate_keys", image_name="scene_025.png", image_bytes=data, manifest=raw,
               signature=sign(m, ALPHA), exp=expected(None, True, False, None, False, "FAILED", ["MANIFEST_INVALID"]),
               description="Manifest text repeats the image_id key with two values; ambiguous, so rejected.",
               spec="implementation hardening (documented): duplicate JSON keys -> MANIFEST_INVALID")

    data = encode(make_scene_image(1026), "PNG")
    write_case(g, "26_missing_manifest_file", image_name="scene_026.png", image_bytes=data, manifest=None,
               signature=base64.b64encode(b"\x00" * 64).decode() + "\n",
               exp=expected(None, True, False, None, False, "FAILED", ["MANIFEST_INVALID"]),
               description="No manifest file at all.", spec="guide 4.2 'manifest unreadable' -> MANIFEST_INVALID")

    data = encode(make_scene_image(1027), "PNG")
    m = build_manifest("scene_027", "scene_027.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    write_case(g, "27_invalid_base64_signature", image_name="scene_027.png", image_bytes=data, manifest=m,
               signature="this is *not* base64!!\n", exp=sig_fail("SIG_INVALID")("scene_027"),
               description=".sig file exists but is not valid base64.",
               spec="guide 10 stretch goal; decoded in the verify step -> SIG_INVALID (documented choice)")

    data = encode(make_scene_image(1028), "PNG")
    m = build_manifest("scene_028", "scene_028.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    write_case(g, "28_signature_wrong_length", image_name="scene_028.png", image_bytes=data, manifest=m,
               signature=base64.b64encode(KEYS[ALPHA].sign(canonical_signing_bytes(m))[:32]).decode() + "\n",
               exp=sig_fail("SIG_INVALID")("scene_028"),
               description="Valid base64 but only 32 bytes (an Ed25519 signature is 64).", spec="guide 3.6 / 4.2 SIG_INVALID")

    signed_case(g, "29_key_expired", 29, key=CHARLIE_EXPIRED, sensor="SAT-CHARLIE-01",
                exp_builder=sig_fail("KEY_NOT_VALID"),
                description="Active key valid 2024-01-01..2025-01-01 used to sign on 2026-10-13.",
                spec="guide 4.2 KEY_NOT_VALID")
    signed_case(g, "30_key_not_yet_valid", 30, key=DELTA_FUTURE, sensor="SAT-DELTA-01",
                exp_builder=sig_fail("KEY_NOT_VALID"),
                description="Active key valid from 2027-01-01 used to sign on 2026-10-13.",
                spec="guide 4.2 KEY_NOT_VALID")
    data = encode(make_scene_image(1031), "PNG")
    m = build_manifest("scene_031", "scene_031.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    m["signed_at"] = "13/10/2026 03:05"
    write_case(g, "31_signed_at_unparseable", image_name="scene_031.png", image_bytes=data, manifest=m,
               signature=sign(m, ALPHA), exp=sig_fail("KEY_NOT_VALID")("scene_031"),
               description="signed_at is not ISO-8601, so it cannot be shown to lie in the key's validity period.",
               spec="guide 4.2 KEY_NOT_VALID (documented choice)")

    # ---- Step 3: metadata ------------------------------------------------
    data = encode(make_scene_image(1032), "PNG")
    m = build_manifest("scene_032", "scene_032.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    del m["metadata"]["timestamp"]
    write_case(g, "32_metadata_missing_timestamp", image_name="scene_032.png", image_bytes=data, manifest=m,
               signature=sign(m, ALPHA), exp=meta_review("META_MISSING_FIELD")("scene_032"),
               description="Validly signed manifest whose metadata lacks 'timestamp'.", spec="guide 4.3 META_MISSING_FIELD")
    signed_case(g, "33_invalid_longitude", 33, lon=200.5, exp_builder=meta_review("META_LONGITUDE"),
                description="Longitude 200.5 (range is +-180).", spec="guide 4.3 META_LONGITUDE")
    signed_case(g, "34_non_numeric_latitude", 34, lat="36.8", exp_builder=meta_review("META_LATITUDE"),
                description="Latitude given as the string \"36.8\" instead of a number.", spec="guide 4.3 'is a number'")
    signed_case(g, "35_future_timestamp", 35, age_h=-1, exp_builder=meta_review("META_TIME_FUTURE"),
                description="Observation 1 h after the reference time (skew tolerance 300 s).", spec="guide 4.3 META_TIME_FUTURE")
    signed_case(g, "36_timestamp_within_clock_skew", 36, age_h=-4 / 60, exp_builder=ok,
                description="Observation 4 min after the reference time; inside the 300 s skew, so accepted.",
                spec="guide 3.3 clock_skew_seconds")
    signed_case(g, "37_invalid_timestamp_format", 37, timestamp="2026-10-13 03:00:00",
                exp_builder=meta_review("META_TIME_INVALID"),
                description="Timestamp uses a space and no 'Z'.", spec="guide 4.3 META_TIME_INVALID")
    data = encode(make_scene_image(1038), "PNG")
    m = build_manifest("scene_038", "scene_038_original.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    write_case(g, "38_filename_mismatch", image_name="scene_038.png", image_bytes=data, manifest=m,
               signature=sign(m, ALPHA), exp=meta_review("META_FILENAME")("scene_038"),
               description="Manifest names scene_038_original.png; the received file is scene_038.png.",
               spec="guide 4.3 META_FILENAME")
    signed_case(g, "39_sensor_of_other_key", 39, sensor="SAT-BRAVO-01", exp_builder=meta_review("META_SENSOR"),
                description="sat-alpha-2026a signs for SAT-BRAVO-01, which only the bravo key may speak for.",
                spec="guide 3.2 allowed_sensors / 4.3 META_SENSOR")
    signed_case(g, "40_all_metadata_checks_fail", 40, sensor="SAT-NOBODY", lat=-91, lon=-180.5, age_h=200,
                exp_builder=meta_review("META_LATITUDE", "META_LONGITUDE", "META_SENSOR", "META_TIME_STALE"),
                description="Several metadata failures at once; codes must follow guide 4.3 order.",
                spec="guide 4.3 / 5.1 reason-code order")

    # ---- Combinations, replay, policy -----------------------------------
    signed_case(g, "41_unknown_key_unknown_sensor", 41, key=ROGUE, sensor="SAT-ROGUE-X",
                exp_builder=lambda i: expected(i, True, False, False, False, "FAILED", ["KEY_UNKNOWN", "META_SENSOR"]),
                description="Unknown key and a sensor that exists nowhere in the trust store.",
                spec="guide 7.3 scene 06 rule + 4.3 META_SENSOR")
    data = encode(make_scene_image(1042), "PNG")
    m = build_manifest("scene_042", "scene_042.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.4, lon=10.8, key_id=ALPHA)
    s = sign(m, ALPHA)
    m["metadata"]["latitude"] = 95.0
    write_case(g, "42_tampered_manifest_bad_latitude", image_name="scene_042.png", image_bytes=data, manifest=m,
               signature=s, exp=expected("scene_042", True, False, False, False, "FAILED", ["SIG_INVALID", "META_LATITUDE"]),
               description="Manifest edited after signing to an invalid latitude: FAILED dominates REVIEW.",
               spec="guide 5 decision order")
    signed_case(g, "43_replay_id_and_content", 43, image_id="scene_099", image_bytes=LEDGER_PICTURE,
                exp_builder=lambda i: expected(i, True, True, True, True, "FAILED", ["REPLAY_ID", "REPLAY_CONTENT"]),
                description="Exact resend of the accepted scene_099 (same ID and same bytes).", spec="guide 4.4")
    signed_case(g, "44_metadata_policy_failed", 44, age_h=100, policy_override={"metadata_failure_action": "FAILED"},
                exp_builder=lambda i: expected(i, True, True, False, False, "FAILED", ["META_TIME_STALE"]),
                description="Stale observation under a policy whose metadata_failure_action is FAILED.",
                spec="guide 3.3 metadata_failure_action")
    signed_case(g, "45_clean_jpeg_pass", 45, fmt="JPEG", ext="jpeg", exp_builder=ok,
                description="Clean JPEG with the .jpeg spelling (treated as jpg).", spec="guide 4.1 'treat jpeg as jpg'")
    data = encode(make_scene_image(1046), "PNG")
    m = build_manifest("scene_046", "scene_046.png", data, observed=hours_ago(9),
                       sensor="SAT-ALPHA-01", lat=36.8, lon=10.8, key_id=ALPHA)
    write_case(g, "46_missing_image_file", image_name="scene_046.png", image_bytes=None, manifest=m,
               signature=sign(m, ALPHA), exp=fmt_fail("FORMAT_MISSING")("scene_046"),
               description="Manifest and signature present, image file absent.",
               spec="not covered by the guide: extension code FORMAT_MISSING (documented)")

    # ---- Non-ASCII manifest content (exercises ensure_ascii=False) -------
    for n, escaped in ((47, False), (48, True)):
        data = encode(make_scene_image(1000 + n), "PNG")
        m = build_manifest(f"scene_{n:03d}", f"scene_{n:03d}.png", data, observed=hours_ago(9),
                           sensor="SAT-ALPHA-01", lat=34.74, lon=10.76, key_id=ALPHA)
        m["metadata"]["site_name"] = "Site d’essai – zone côtière, منطقة اختبار"
        on_disk = json.dumps(m, indent=2, ensure_ascii=escaped) + "\n"
        write_case(g, f"{n}_non_ascii_manifest_{'escaped' if escaped else 'utf8'}", image_name=f"scene_{n:03d}.png",
                   image_bytes=data, manifest=on_disk, signature=sign(m, ALPHA), exp=ok(f"scene_{n:03d}"),
                   description=("Signed manifest with non-ASCII text in an extra metadata field, stored on disk as "
                                + ("\\u escapes" if escaped else "raw UTF-8")
                                + ". Verification must re-canonicalise with ensure_ascii=False."),
                   spec="guide 4.2 signing input (ensure_ascii=False, UTF-8)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def write_cases_md() -> None:
    lines = ["# Synthetic test cases", "",
             "Generated by `generate_test_dataset.py`. Expected values are derived from the TSYP14 teammate guide.", "",
             "| Case | Expected status | Flags (format / signature / metadata / replay) | Reason codes | Description |",
             "|---|---|---|---|---|"]
    j = lambda v: json.dumps(v)
    for c in CASES:
        e = c["expected"]
        flags = " / ".join(j(e[k]) for k in ("format_valid", "signature_valid", "metadata_valid", "replay_detected"))
        lines.append(f"| `{c['group']}/{c['case_id']}` | {e['security_status']} | {flags} | "
                     f"{', '.join(e['reason_codes']) or '-'} | {c['description']} |")
    (OUT / "CASES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    _selftest_canonical()
    if OUT.exists():
        if not (OUT / MARKER).exists():
            print(f"refusing to overwrite {OUT}: it was not created by this script", file=sys.stderr)
            return 1
        shutil.rmtree(OUT)
    OUT.mkdir()
    (OUT / MARKER).write_text("Generated test data. Safe to delete and regenerate.\n", encoding="utf-8")
    write_shared_files()
    write_official_scenes()
    write_edge_cases()
    write_cases_md()
    n_off = sum(c["group"] == "scenes" for c in CASES)
    print(f"Wrote {len(CASES)} cases ({n_off} guide scenes, {len(CASES) - n_off} edge cases) to {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
