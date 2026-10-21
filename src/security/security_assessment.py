"""TSYP14 Track 3 ("Trust the Pixel") - Security Assessment module.

Decides whether an incoming satellite image can be trusted *before* it is
handed to the AI image-forensics pipeline. The behaviour follows the
"TSYP14 Security Assessment - Teammate guide" (sections 3-5):

    1. Format     - extension, size, magic bytes, decoder format,
                    header dimensions, full decode.  A failure stops here.
    2. Signature  - manifest schema, detached .sig, trusted key lookup,
                    revocation, key validity period, Ed25519 verify,
                    SHA-256 image hash comparison.  Stops at first failure.
    3. Metadata   - required fields, latitude, longitude, sensor,
                    timestamp format, future, staleness, file name.
    4. Replay     - image_id and image content already in the ledger.

Final status: FAILED if format / signature / integrity failed or a replay was
detected; otherwise the policy's ``metadata_failure_action`` (REVIEW in the
demo) if metadata failed; otherwise PASSED.

This module verifies *provenance and integrity* of a file. It does NOT detect
visual manipulation: an image that was manipulated before it was signed by a
trusted sender will pass. That is the job of the downstream forensics models.

The module only ever handles *public* keys. It never reads, needs or exposes a
private signing key, and it never reads ``expected.json`` test answer keys.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import io
import json
import math
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from PIL import Image

__all__ = [
    "assess",
    "signing_input",
    "parse_utc",
    "load_policy",
    "load_trust_store",
    "load_ledger",
    "load_reference_time",
    "record_accepted",
    "ConfigurationError",
]

MODULE_VERSION = "1.0.1"
PathLike = Union[str, "os.PathLike[str]"]

# --------------------------------------------------------------------------
# Constants from the guide
# --------------------------------------------------------------------------

#: Domain-separation prefix of the signing input (guide section 4.2).
SIGNING_PREFIX = b"TSYP14-IMG-MANIFEST-v1\n"

PASSED, REVIEW, FAILED = "PASSED", "REVIEW", "FAILED"

# Step 1 - format (guide 4.1)
FORMAT_EXTENSION = "FORMAT_EXTENSION"
FORMAT_SIZE = "FORMAT_SIZE"
FORMAT_MAGIC = "FORMAT_MAGIC"
FORMAT_DECODE = "FORMAT_DECODE"
FORMAT_DIMENSIONS = "FORMAT_DIMENSIONS"
#: Extension code, NOT in the guide: the image file does not exist at all.
FORMAT_MISSING = "FORMAT_MISSING"

# Step 2 - signature and integrity (guide 4.2)
MANIFEST_INVALID = "MANIFEST_INVALID"
SIG_MISSING = "SIG_MISSING"
KEY_UNKNOWN = "KEY_UNKNOWN"
KEY_REVOKED = "KEY_REVOKED"
KEY_NOT_VALID = "KEY_NOT_VALID"
SIG_INVALID = "SIG_INVALID"
HASH_MISMATCH = "HASH_MISMATCH"

# Step 3 - metadata (guide 4.3), listed in check order
META_MISSING_FIELD = "META_MISSING_FIELD"
META_LATITUDE = "META_LATITUDE"
META_LONGITUDE = "META_LONGITUDE"
META_SENSOR = "META_SENSOR"
META_TIME_INVALID = "META_TIME_INVALID"
META_TIME_FUTURE = "META_TIME_FUTURE"
META_TIME_STALE = "META_TIME_STALE"
META_FILENAME = "META_FILENAME"

# Step 4 - replay (guide 4.4)
REPLAY_ID = "REPLAY_ID"
REPLAY_CONTENT = "REPLAY_CONTENT"

#: Manifest fields that must be present (guide 3.6 field table).
MANIFEST_REQUIRED_FIELDS = (
    "manifest_version", "image_id", "file_name", "sha256", "size_bytes",
    "metadata", "key_id", "nonce", "signed_at",
)
METADATA_REQUIRED_FIELDS = ("timestamp", "sensor_id", "latitude", "longitude")

#: Magic bytes per normalised extension (guide 4.1).
_MAGIC: dict[str, tuple[bytes, ...]] = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpg": (b"\xff\xd8\xff",),
    "tif": (b"II*\x00", b"MM\x00*"),
}
#: Pillow format names accepted for each extension. MPO is Pillow's name for
#: a multi-picture JPEG, which is still a JPEG container.
_PIL_FORMATS: dict[str, tuple[str, ...]] = {
    "png": ("PNG",),
    "jpg": ("JPEG", "MPO"),
    "tif": ("TIFF",),
}
_EXT_ALIASES = {"jpeg": "jpg", "tiff": "tif"}

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_PUBKEY_RE = re.compile(r"[0-9a-fA-F]{64}")

#: Hard caps on the small text inputs, so a hostile manifest or .sig file
#: cannot exhaust memory. Real manifests are a few hundred bytes.
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SIGNATURE_FILE_BYTES = 4096
ED25519_SIGNATURE_LEN = 64


class ConfigurationError(ValueError):
    """Raised for malformed *receiver-side* configuration (policy, trust
    store, ledger, reference time). Bad *inputs* (image, manifest, signature)
    never raise: they produce a FAILED or REVIEW result instead."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def parse_utc(value: Any) -> Optional[datetime]:
    """Parse a strict ``YYYY-MM-DDTHH:MM:SSZ`` UTC timestamp.

    Returns an aware UTC datetime, or None if ``value`` is not a string in
    exactly that form (guide 4.3, META_TIME_INVALID).
    """
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def format_utc(value: datetime) -> str:
    """Format an aware datetime as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def signing_input(manifest: Mapping[str, Any]) -> bytes:
    """Return the exact bytes covered by the manifest signature (guide 4.2).

    ``b"TSYP14-IMG-MANIFEST-v1\\n"`` followed by the UTF-8 encoding of the
    manifest serialised with sorted keys, compact separators and
    ``ensure_ascii=False``.
    """
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return SIGNING_PREFIX + body.encode("utf-8")


def _normalize_ext(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    return _EXT_ALIASES.get(ext, ext)


def _is_number(value: Any) -> bool:
    """True for a finite JSON number (bool is excluded on purpose)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_nonneg_number(value: Any) -> bool:
    return _is_number(value) and value >= 0


_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+>")


def _exc_text(exc: BaseException) -> str:
    """Exception text without memory addresses (e.g. ``<_io.BytesIO object
    at 0x7f...>``), so that identical inputs always give identical results."""
    return f"{type(exc).__name__}: {_ADDRESS_RE.sub('>', str(exc))}"


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name!r} is not allowed")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def _strict_json_loads(text: str) -> Any:
    """json.loads that rejects duplicate keys and NaN/Infinity.

    Duplicate keys are ambiguous (which value did the signer mean?), so a
    manifest that contains them is treated as invalid rather than guessed at.
    """
    return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant)


def _read_json_file(path: PathLike) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return _strict_json_loads(fh.read())
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot load {path}: {exc}") from exc


# --------------------------------------------------------------------------
# Configuration loaders (receiver side)
# --------------------------------------------------------------------------

def load_policy(path: PathLike) -> dict[str, Any]:
    """Load and validate ``policy.json``."""
    policy = _read_json_file(path)
    validate_policy(policy)
    return policy


def validate_policy(policy: Any) -> None:
    """Raise ConfigurationError if the policy is malformed."""
    if not isinstance(policy, dict):
        raise ConfigurationError("policy must be a JSON object")
    formats = policy.get("allowed_formats")
    if not isinstance(formats, list) or not formats or not all(isinstance(f, str) for f in formats):
        raise ConfigurationError("policy.allowed_formats must be a non-empty list of strings")
    unsupported = {_normalize_ext(f) for f in formats} - set(_MAGIC)
    if unsupported:
        raise ConfigurationError(f"policy.allowed_formats has unsupported formats: {sorted(unsupported)}")
    for name in ("max_file_size_mb", "max_age_hours", "clock_skew_seconds"):
        if not _is_nonneg_number(policy.get(name)):
            raise ConfigurationError(f"policy.{name} must be a non-negative number")
    max_pixels = policy.get("max_pixels")
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
        raise ConfigurationError("policy.max_pixels must be a positive integer")
    min_size = policy.get("min_file_size_bytes", 1)
    if isinstance(min_size, bool) or not isinstance(min_size, int) or min_size < 0:
        raise ConfigurationError("policy.min_file_size_bytes must be a non-negative integer")
    if policy.get("metadata_failure_action") not in (REVIEW, FAILED):
        raise ConfigurationError("policy.metadata_failure_action must be 'REVIEW' or 'FAILED'")


def load_trust_store(path: PathLike) -> dict[str, Any]:
    """Load and validate ``trusted_keys.json`` (public keys only)."""
    store = _read_json_file(path)
    _index_trust_store(store)  # validates
    return store


def _index_trust_store(store: Any) -> dict[str, dict[str, Any]]:
    """Validate the trust store and index it by key_id.

    Each indexed entry carries the parsed validity datetimes. A private key
    field in the store is treated as a configuration error: the receiver
    must never hold private keys.
    """
    if not isinstance(store, Mapping) or not isinstance(store.get("keys"), list):
        raise ConfigurationError("trust store must be an object with a 'keys' list")
    index: dict[str, dict[str, Any]] = {}
    for entry in store["keys"]:
        if not isinstance(entry, Mapping):
            raise ConfigurationError("trust store entries must be objects")
        key_id = entry.get("key_id")
        if not isinstance(key_id, str) or not key_id:
            raise ConfigurationError("trust store entry without a valid key_id")
        if key_id in index:
            raise ConfigurationError(f"duplicate key_id in trust store: {key_id}")
        if any("private" in str(name).lower() for name in entry):
            raise ConfigurationError(f"trust store entry {key_id} contains private key material")
        pub_hex = entry.get("public_key_hex")
        if not isinstance(pub_hex, str) or not _PUBKEY_RE.fullmatch(pub_hex):
            raise ConfigurationError(f"key {key_id}: public_key_hex must be 64 hex characters")
        if not isinstance(entry.get("status"), str):
            raise ConfigurationError(f"key {key_id}: status must be a string")
        sensors = entry.get("allowed_sensors")
        if not isinstance(sensors, list) or not all(isinstance(s, str) for s in sensors):
            raise ConfigurationError(f"key {key_id}: allowed_sensors must be a list of strings")
        valid_from = parse_utc(entry.get("valid_from"))
        valid_until = parse_utc(entry.get("valid_until"))
        if valid_from is None or valid_until is None:
            raise ConfigurationError(f"key {key_id}: valid_from/valid_until must be YYYY-MM-DDTHH:MM:SSZ")
        index[key_id] = {
            "key_id": key_id,
            "public_key": bytes.fromhex(pub_hex),
            "status": entry["status"],
            "allowed_sensors": frozenset(sensors),
            "valid_from": valid_from,
            "valid_until": valid_until,
        }
    return index


def load_ledger(path: PathLike) -> dict[str, Any]:
    """Load and validate ``ledger.json`` (the replay memory)."""
    ledger = _read_json_file(path)
    _index_ledger(ledger)  # validates
    return ledger


def _index_ledger(ledger: Any) -> tuple[frozenset[str], frozenset[str]]:
    """Return (accepted image_ids, accepted lowercase sha256 hashes)."""
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("entries"), list):
        raise ConfigurationError("ledger must be an object with an 'entries' list")
    ids, hashes = set(), set()
    for entry in ledger["entries"]:
        if not isinstance(entry, Mapping):
            raise ConfigurationError("ledger entries must be objects")
        image_id, digest = entry.get("image_id"), entry.get("sha256")
        if not isinstance(image_id, str) or not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ConfigurationError(f"malformed ledger entry: {dict(entry)!r}")
        ids.add(image_id)
        hashes.add(digest.lower())
    return frozenset(ids), frozenset(hashes)


def load_reference_time(path: PathLike) -> datetime:
    """Load the fixed assessment clock from ``reference_time.txt``."""
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigurationError(f"cannot read reference time {path}: {exc}") from exc
    when = parse_utc(text)
    if when is None:
        raise ConfigurationError(f"reference time must be YYYY-MM-DDTHH:MM:SSZ, got {text!r}")
    return when


def _coerce_now(now: Union[datetime, str]) -> datetime:
    if isinstance(now, datetime):
        if now.tzinfo is None:
            raise ConfigurationError("'now' must be timezone-aware")
        return now.astimezone(timezone.utc)
    parsed = parse_utc(now)
    if parsed is None:
        raise ConfigurationError(f"'now' must be an aware datetime or YYYY-MM-DDTHH:MM:SSZ, got {now!r}")
    return parsed


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------

class _Trail:
    """Collects reason codes, human-readable reasons and the evidence trail."""

    def __init__(self) -> None:
        self.reason_codes: list[str] = []
        self.reasons: list[str] = []
        self.checks: list[dict[str, Any]] = []

    def ok(self, step: str, check: str, detail: str = "") -> None:
        self.checks.append({"step": step, "check": check, "passed": True, "detail": detail})

    def fail(self, step: str, check: str, code: str, detail: str) -> None:
        self.checks.append({"step": step, "check": check, "passed": False, "code": code, "detail": detail})
        self.reason_codes.append(code)
        self.reasons.append(f"{code}: {detail}")


def _load_manifest(manifest_path: Optional[PathLike]) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Read the manifest.

    Returns ``(manifest_or_None, problems)``. The manifest object is returned
    whenever it parses as a JSON object, even if its schema is incomplete, so
    the metadata and replay checks can still run on it (guide 4.3: "run this
    even if the signature failed"). ``problems`` is empty only for a manifest
    that satisfies the schema.
    """
    if manifest_path is None:
        return None, ["no manifest path given"]
    try:
        with open(manifest_path, "rb") as fh:
            raw = fh.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        return None, [f"manifest unreadable: {exc.strerror or exc}"]
    if len(raw) > MAX_MANIFEST_BYTES:
        return None, [f"manifest larger than {MAX_MANIFEST_BYTES} bytes"]
    try:
        manifest = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        return None, [f"manifest is not valid JSON: {_exc_text(exc)}"]
    if not isinstance(manifest, dict):
        return None, ["manifest is not a JSON object"]

    problems = [f"missing field '{name}'" for name in MANIFEST_REQUIRED_FIELDS if name not in manifest]
    version = manifest.get("manifest_version")
    if "manifest_version" in manifest and (isinstance(version, bool) or version != 1 or not isinstance(version, int)):
        problems.append(f"manifest_version must be 1, got {version!r}")
    for name in ("image_id", "key_id"):
        if name in manifest and (not isinstance(manifest[name], str) or not manifest[name]):
            problems.append(f"'{name}' must be a non-empty string")
    for name in ("file_name", "signed_at"):
        if name in manifest and not isinstance(manifest[name], str):
            problems.append(f"'{name}' must be a string")
    if "sha256" in manifest and not (isinstance(manifest["sha256"], str) and _SHA256_RE.fullmatch(manifest["sha256"])):
        problems.append("'sha256' must be 64 hex characters")
    if "metadata" in manifest and not isinstance(manifest["metadata"], dict):
        problems.append("'metadata' must be an object")
    return manifest, problems


def _check_format(image_path: PathLike, policy: Mapping[str, Any], trail: _Trail) -> Optional[bytes]:
    """Step 1. Returns the image bytes when every format check passes.

    The file is read into memory once (it is bounded by max_file_size_mb) and
    the same bytes are decoded and hashed, so the image cannot be swapped
    between the format check and the integrity check.
    """
    step = "format"
    path = Path(image_path)
    try:
        st = os.stat(path)
    except OSError as exc:
        trail.fail(step, "exists", FORMAT_MISSING, f"image file not found: {exc.strerror or exc}")
        return None
    if not stat.S_ISREG(st.st_mode):
        trail.fail(step, "exists", FORMAT_MISSING, "image path is not a regular file")
        return None
    trail.ok(step, "exists")

    ext = _normalize_ext(path.suffix)
    allowed = {_normalize_ext(f) for f in policy["allowed_formats"]}
    if not ext or ext not in allowed:
        trail.fail(step, "extension", FORMAT_EXTENSION, f"extension '{path.suffix}' not in allowed_formats {sorted(allowed)}")
        return None
    trail.ok(step, "extension", ext)

    max_bytes = int(policy["max_file_size_mb"] * 1024 * 1024)
    min_bytes = int(policy.get("min_file_size_bytes", 1))
    if st.st_size < min_bytes or st.st_size > max_bytes:
        trail.fail(step, "size", FORMAT_SIZE, f"file size {st.st_size} B outside [{min_bytes}, {max_bytes}] B")
        return None
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes + 1)
    except OSError as exc:
        trail.fail(step, "size", FORMAT_MISSING, f"image file unreadable: {exc.strerror or exc}")
        return None
    if len(data) < min_bytes or len(data) > max_bytes:  # file changed after stat
        trail.fail(step, "size", FORMAT_SIZE, f"file size {len(data)} B outside [{min_bytes}, {max_bytes}] B")
        return None
    trail.ok(step, "size", f"{len(data)} B")

    if not any(data.startswith(magic) for magic in _MAGIC[ext]):
        trail.fail(step, "magic", FORMAT_MAGIC, f"magic bytes {data[:8].hex()} do not match .{ext}")
        return None
    trail.ok(step, "magic")

    expected_formats = _PIL_FORMATS[ext]
    try:
        # Restrict Pillow to the decoders for the claimed format only.
        with Image.open(io.BytesIO(data), formats=list(expected_formats)) as img:
            if img.format not in expected_formats:
                trail.fail(step, "decoder_format", FORMAT_DECODE, f"decoder reports {img.format}, expected {expected_formats}")
                return None
            trail.ok(step, "decoder_format", str(img.format))

            # Header dimensions, checked BEFORE any pixel is decoded.
            width, height = img.size
            if width <= 0 or height <= 0:
                trail.fail(step, "dimensions", FORMAT_DIMENSIONS, f"empty canvas {width}x{height}")
                return None
            if width * height > policy["max_pixels"]:
                trail.fail(step, "dimensions", FORMAT_DIMENSIONS,
                           f"{width}x{height} = {width * height} px exceeds max_pixels {policy['max_pixels']}")
                return None
            trail.ok(step, "dimensions", f"{width}x{height}")

            try:
                img.load()
            except Image.DecompressionBombError as exc:
                trail.fail(step, "decode", FORMAT_DIMENSIONS, f"decompression bomb guard: {_exc_text(exc)}")
                return None
            except Exception as exc:  # any decoder failure means a malformed image
                trail.fail(step, "decode", FORMAT_DECODE, f"full decode failed: {_exc_text(exc)}")
                return None
            trail.ok(step, "decode", f"mode {img.mode}")
    except Image.DecompressionBombError as exc:
        # Pillow's own header guard fired while opening (far above max_pixels).
        trail.fail(step, "dimensions", FORMAT_DIMENSIONS, f"decompression bomb guard: {_exc_text(exc)}")
        return None
    except Exception as exc:  # UnidentifiedImageError, SyntaxError, struct.error, ...
        trail.fail(step, "decoder_format", FORMAT_DECODE, f"image library cannot open file as {expected_formats}: {_exc_text(exc)}")
        return None
    return data


def _check_signature(
    manifest: Optional[dict[str, Any]],
    manifest_problems: list[str],
    sig_path: Optional[PathLike],
    keys: Mapping[str, dict[str, Any]],
    image_sha256: str,
    trail: _Trail,
) -> bool:
    """Step 2: signature and integrity, in the order of guide figure 7.
    Stops at the first failure."""
    step = "signature"
    if manifest is None or manifest_problems:
        trail.fail(step, "manifest", MANIFEST_INVALID, "; ".join(manifest_problems))
        return False
    trail.ok(step, "manifest")

    if sig_path is None or not Path(sig_path).is_file():
        trail.fail(step, "sig_present", SIG_MISSING, "detached signature file not found")
        return False
    trail.ok(step, "sig_present")

    key = keys.get(manifest["key_id"])
    if key is None:
        trail.fail(step, "key_known", KEY_UNKNOWN, f"key_id '{manifest['key_id']}' is not in the trust store")
        return False
    trail.ok(step, "key_known", key["key_id"])

    # Revocation is checked BEFORE any cryptography (guide 4.2 "Order matters").
    if key["status"] != "active":
        trail.fail(step, "key_status", KEY_REVOKED, f"key '{key['key_id']}' has status '{key['status']}'")
        return False
    trail.ok(step, "key_status")

    signed_at = parse_utc(manifest["signed_at"])
    if signed_at is None or not (key["valid_from"] <= signed_at <= key["valid_until"]):
        trail.fail(step, "key_validity", KEY_NOT_VALID,
                   f"signed_at {manifest['signed_at']!r} not inside {format_utc(key['valid_from'])} .. {format_utc(key['valid_until'])}")
        return False
    trail.ok(step, "key_validity")

    try:
        with open(sig_path, "rb") as fh:
            raw = fh.read(MAX_SIGNATURE_FILE_BYTES + 1)
        if len(raw) > MAX_SIGNATURE_FILE_BYTES:
            raise ValueError("signature file too large")
        signature = base64.b64decode(raw.decode("ascii").strip(), validate=True)
        if len(signature) != ED25519_SIGNATURE_LEN:
            raise ValueError(f"decoded signature is {len(signature)} bytes, expected {ED25519_SIGNATURE_LEN}")
        message = signing_input(manifest)
        Ed25519PublicKey.from_public_bytes(key["public_key"]).verify(signature, message)
    except InvalidSignature:
        trail.fail(step, "verify", SIG_INVALID, "Ed25519 signature does not verify against the manifest")
        return False
    except (OSError, UnicodeError, binascii.Error, ValueError) as exc:
        trail.fail(step, "verify", SIG_INVALID, f"signature unusable: {_exc_text(exc)}")
        return False
    trail.ok(step, "verify")

    # Only now is manifest.sha256 trustworthy enough to compare against.
    if not hmac.compare_digest(image_sha256, manifest["sha256"].lower()):
        trail.fail(step, "hash", HASH_MISMATCH, f"image SHA-256 {image_sha256} != signed {manifest['sha256'].lower()}")
        return False
    trail.ok(step, "hash")
    return True


def _check_metadata(
    manifest: dict[str, Any],
    image_path: PathLike,
    keys: Mapping[str, dict[str, Any]],
    policy: Mapping[str, Any],
    now: datetime,
    trail: _Trail,
) -> bool:
    """Step 3: every metadata check runs; all failures are reported in the
    order of guide section 4.3. A field that is missing is reported once as
    META_MISSING_FIELD and its specific check is skipped."""
    step = "metadata"
    before = len(trail.reason_codes)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    missing = [f for f in METADATA_REQUIRED_FIELDS if metadata.get(f) is None]
    if missing:
        trail.fail(step, "required_fields", META_MISSING_FIELD, f"missing metadata fields: {missing}")
    else:
        trail.ok(step, "required_fields")

    for name, limit, code in (("latitude", 90, META_LATITUDE), ("longitude", 180, META_LONGITUDE)):
        if name in missing:
            continue
        value = metadata[name]
        if _is_number(value) and -limit <= value <= limit:
            trail.ok(step, name, str(value))
        else:
            trail.fail(step, name, code, f"{name} {value!r} is not a number in [-{limit}, {limit}]")

    if "sensor_id" not in missing:
        sensor = metadata["sensor_id"]
        key_id = manifest.get("key_id")
        key = keys.get(key_id) if isinstance(key_id, str) else None
        if key is not None:
            allowed, scope = key["allowed_sensors"], f"key '{key['key_id']}'"
        else:
            # Guide 7.3 scene 06: with no key to consult, accept a sensor that
            # exists anywhere in the trust store.
            allowed = frozenset().union(*(k["allowed_sensors"] for k in keys.values()))
            scope = "any trusted key"
        if isinstance(sensor, str) and sensor in allowed:
            trail.ok(step, "sensor", sensor)
        else:
            trail.fail(step, "sensor", META_SENSOR, f"sensor {sensor!r} not allowed for {scope}")

    if "timestamp" not in missing:
        observed = parse_utc(metadata["timestamp"])
        if observed is None:
            trail.fail(step, "timestamp", META_TIME_INVALID, f"timestamp {metadata['timestamp']!r} is not YYYY-MM-DDTHH:MM:SSZ")
        elif observed > now + timedelta(seconds=policy["clock_skew_seconds"]):
            trail.fail(step, "timestamp", META_TIME_FUTURE,
                       f"timestamp {format_utc(observed)} is later than now {format_utc(now)} + {policy['clock_skew_seconds']} s")
        elif now - observed > timedelta(hours=policy["max_age_hours"]):
            age_h = (now - observed).total_seconds() / 3600
            trail.fail(step, "timestamp", META_TIME_STALE, f"observation is {age_h:.1f} h old, limit {policy['max_age_hours']} h")
        else:
            trail.ok(step, "timestamp", format_utc(observed))

    actual_name = Path(image_path).name
    if manifest.get("file_name") == actual_name:
        trail.ok(step, "file_name", actual_name)
    else:
        trail.fail(step, "file_name", META_FILENAME, f"manifest file_name {manifest.get('file_name')!r} != actual {actual_name!r}")

    return len(trail.reason_codes) == before


def _check_replay(
    image_id: Optional[str],
    image_sha256: str,
    ledger: Mapping[str, Any],
    trail: _Trail,
) -> bool:
    """Step 4. Returns True when a replay is detected (the bad outcome)."""
    step = "replay"
    ids, hashes = _index_ledger(ledger)
    detected = False
    if image_id is not None and image_id in ids:
        trail.fail(step, "image_id", REPLAY_ID, f"image_id '{image_id}' was already accepted")
        detected = True
    else:
        trail.ok(step, "image_id")
    if image_sha256 in hashes:
        trail.fail(step, "content", REPLAY_CONTENT, "identical image bytes were already accepted")
        detected = True
    else:
        trail.ok(step, "content")
    return detected


def assess(
    image_path: PathLike,
    manifest_path: Optional[PathLike],
    sig_path: Optional[PathLike],
    trust_store: Mapping[str, Any],
    policy: Mapping[str, Any],
    ledger: Mapping[str, Any],
    now: Union[datetime, str],
) -> dict[str, Any]:
    """Assess one incoming image.

    Args:
        image_path: the received image file.
        manifest_path: its ``<image_id>.manifest.json``.
        sig_path: its detached ``<image_id>.manifest.sig`` (may not exist).
        trust_store: parsed ``trusted_keys.json`` (see :func:`load_trust_store`).
        policy: parsed ``policy.json`` (see :func:`load_policy`).
        ledger: parsed ``ledger.json`` (see :func:`load_ledger`). Read only.
        now: the assessment clock - an aware datetime or a
            ``YYYY-MM-DDTHH:MM:SSZ`` string (normally reference_time.txt).
            The real system clock is never consulted.

    Returns:
        A dict with ``image_id``, ``format_valid``, ``signature_valid``,
        ``metadata_valid``, ``replay_detected`` (True = replay found),
        ``security_status``, ``reasons``, ``reason_codes`` and ``evidence``.
        A flag is None when its check never ran.

    Raises:
        ConfigurationError: only for malformed receiver configuration.
    """
    validate_policy(policy)
    keys = _index_trust_store(trust_store)
    _index_ledger(ledger)
    now_utc = _coerce_now(now)

    trail = _Trail()
    manifest, manifest_problems = _load_manifest(manifest_path)
    image_id = manifest.get("image_id") if manifest is not None else None
    if not isinstance(image_id, str):
        image_id = None

    result: dict[str, Any] = {
        "image_id": image_id,
        "format_valid": None,
        "signature_valid": None,
        "metadata_valid": None,
        "replay_detected": None,
        "security_status": FAILED,
        "reasons": trail.reasons,
        "reason_codes": trail.reason_codes,
    }
    evidence: dict[str, Any] = {
        "image_file": Path(image_path).name,
        "image_sha256": None,
        "key_id": manifest.get("key_id") if manifest is not None else None,
        "policy_version": policy.get("version"),
        "assessed_at": format_utc(now_utc),
        "module_version": MODULE_VERSION,
        "checks": trail.checks,
    }
    result["evidence"] = evidence

    # 1. Format - a failure stops everything; the other flags stay None.
    data = _check_format(image_path, policy, trail)
    result["format_valid"] = data is not None
    if data is None:
        return result
    image_sha256 = hashlib.sha256(data).hexdigest()
    evidence["image_sha256"] = image_sha256
    evidence["image_size_bytes"] = len(data)

    # 2. Signature and integrity.
    result["signature_valid"] = _check_signature(manifest, manifest_problems, sig_path, keys, image_sha256, trail)

    # 3. Metadata - runs even if the signature failed. It needs a manifest
    #    object; if the manifest could not be parsed at all it stays None.
    if manifest is not None:
        result["metadata_valid"] = _check_metadata(manifest, image_path, keys, policy, now_utc, trail)

    # 4. Replay - the content check always runs; the ID check needs an ID.
    result["replay_detected"] = _check_replay(image_id, image_sha256, ledger, trail)

    # Decision (guide section 5).
    if not result["signature_valid"] or result["replay_detected"]:
        result["security_status"] = FAILED
    elif result["metadata_valid"] is not True:
        result["security_status"] = policy["metadata_failure_action"]
    else:
        result["security_status"] = PASSED
    return result


# --------------------------------------------------------------------------
# Ledger update (only after PASSED)
# --------------------------------------------------------------------------

def record_accepted(ledger_path: PathLike, result: Mapping[str, Any], accepted_at: Union[datetime, str]) -> bool:
    """Append a PASSED image to the ledger file, atomically.

    Does nothing and returns False unless ``result['security_status']`` is
    PASSED: recording rejected images would let an attacker burn legitimate
    IDs (guide 3.4). The ledger is re-read, re-checked for the ID and hash,
    and replaced via a temporary file + ``os.replace`` so a crash never
    leaves a half-written ledger.
    """
    if result.get("security_status") != PASSED:
        return False
    image_id = result.get("image_id")
    digest = (result.get("evidence") or {}).get("image_sha256")
    if not isinstance(image_id, str) or not isinstance(digest, str):
        raise ValueError("result lacks image_id or evidence.image_sha256")
    ledger_path = Path(ledger_path)
    ledger = load_ledger(ledger_path)
    ids, hashes = _index_ledger(ledger)
    if image_id in ids or digest in hashes:
        raise ValueError(f"ledger already contains image_id '{image_id}' or its content")
    ledger["entries"].append({"image_id": image_id, "sha256": digest, "accepted_at": format_utc(_coerce_now(accepted_at))})
    fd, tmp = tempfile.mkstemp(prefix=".ledger-", suffix=".tmp", dir=ledger_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, ledger_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return True


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def default_sig_path(manifest_path: PathLike) -> Path:
    """``x.manifest.json`` -> ``x.manifest.sig`` (guide 3.6 naming)."""
    return Path(manifest_path).with_suffix(".sig")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: assess one image and print the result as JSON.

    Exit code: 0 PASSED, 2 REVIEW, 3 FAILED, 1 configuration error.
    """
    parser = argparse.ArgumentParser(description="TSYP14 security assessment of one image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sig", help="detached signature (default: <manifest>.sig)")
    parser.add_argument("--config-dir", default="test_data",
                        help="folder with trusted_keys.json, policy.json, ledger.json, reference_time.txt")
    parser.add_argument("--now", help="override the reference time (YYYY-MM-DDTHH:MM:SSZ)")
    parser.add_argument("--record", action="store_true", help="append the image to the ledger if it PASSED")
    args = parser.parse_args(argv)

    cfg = Path(args.config_dir)
    try:
        trust_store = load_trust_store(cfg / "trusted_keys.json")
        policy = load_policy(cfg / "policy.json")
        ledger = load_ledger(cfg / "ledger.json")
        now = args.now or load_reference_time(cfg / "reference_time.txt")
        sig = args.sig or default_sig_path(args.manifest)
        result = assess(args.image, args.manifest, sig, trust_store, policy, ledger, now)
        if args.record and record_accepted(cfg / "ledger.json", result, now):
            result["ledger_updated"] = True
    except (ConfigurationError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return {PASSED: 0, REVIEW: 2}.get(result["security_status"], 3)


if __name__ == "__main__":
    sys.exit(main())
