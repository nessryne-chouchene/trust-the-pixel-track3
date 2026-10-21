# TSYP14 Track 3: Security Assessment module

This is the **Security Assessment** stage of the *Trust the Pixel* pipeline. It decides whether an incoming satellite image can be trusted **before** the AI image-forensics models see it. The implementation follows the *TSYP14 Security Assessment teammate guide* (kept outside the repository). The signed-manifest scheme is a design choice of this guide; it is not stated as a requirement in the official challenge instructions. Where the guide was silent, the choices made are listed in the [ambiguities](#documented-ambiguities-and-implementation-choices) section below.

```
image file ──► 1. Format ──► 2. Signature & integrity ──► 3. Metadata ──► 4. Replay ──► decision
               (stop on fail)   (stop at 1st failure)      (all checks)    (ID, content)  PASSED / REVIEW / FAILED
                                                                                             │
                                          PASSED (or REVIEW after a human) ──► preprocessing ► EfficientNet ► heads ► report
```

In the architecture diagram (`docs/reference/architecture_diagram.jpg`), this module covers the **File Validation** and **Security Assessment** boxes: hash, signature, metadata and replay. The preprocessing, EfficientNet feature extractor, localization head and classification head are **not** implemented here.

## Beginner guide (web page)

`web/index.html` is a plain-English guide to the project. It covers what each check does, how the files fit together, step-by-step Windows/Linux commands, and how to read the test reports. It needs no installation: **double-click `web/index.html`** to open it in any browser. It uses only `web/style.css` and `web/script.js`, with no internet and no external libraries.

When the page is opened by double-click, the browser does not let it read other local files, so its **Load latest results** button cannot work that way. To use that button, run this from the project folder, then open http://localhost:8000/web/:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

## Provenance is not manipulation detection

This module verifies **provenance and integrity**:

* the file is a well-formed PNG, JPEG or TIFF;
* a trusted, unrevoked key signed the manifest;
* the received bytes are exactly the signed bytes;
* the metadata is plausible;
* the image has not been seen before.

It does **not** look at pixels for signs of editing. If a trusted sender signs an image that was already manipulated, or if a sender's key is stolen, the image will PASS. Catching visual manipulation is the job of the downstream forensics models. The synthetic tests here only show that the module follows the specification. They say **nothing** about performance on real manipulated satellite imagery.

## Project layout

```
src/security/
├── README.md                 this file
├── __init__.py
├── security_assessment.py    production module (public keys only, no test knowledge)
├── adapter.py                assess() result -> shared ValidationReport contract
├── generate_test_dataset.py  builds test_data/ (disposable test keys live only here)
├── run_security_cases.py     independent test runner, writes test_results/
├── tools/mutation_check.py   optional: checks that the tests catch injected bugs
├── web/                      beginner guide page: index.html, style.css, script.js
├── test_data/                generated; safe to delete and regenerate
└── test_results/             reports of the latest test run: results.json, results.csv
```

The runner was renamed from `test_security_assessment.py` to `run_security_cases.py` so that pytest does not collect it as a test module; `tests/test_security.py` calls it instead. The architecture diagram is in `docs/reference/architecture_diagram.jpg`.

Each case folder holds:

* the image;
* `<image_id>.manifest.json`;
* `<image_id>.manifest.sig`, except where a case is about a missing file;
* `case.json`, which tells the runner which files to pass and gives any `policy_override`;
* `expected.json`, the answer key. **Only the test runner reads it.**

## Commands

This module now lives in `src/security/` of the shared repository and uses the repository's Python 3.12 environment (`.venv312`) and root `requirements.txt` (`cryptography` and `Pillow` are included there). Activate the environment at the repository root, then run the commands **from this folder** (`src/security`):

```powershell
.\.venv312\Scripts\Activate.ps1
cd src\security
python generate_test_dataset.py          # optional: regenerates test_data/
python run_security_cases.py             # 53 checks, writes test_results/
python tools\mutation_check.py           # optional: 17 injected bugs must be caught
python security_assessment.py --image test_data\scenes\01_clean_pass\scene_001.png --manifest test_data\scenes\01_clean_pass\scene_001.manifest.json
```

From the repository root, `python -m pytest -q` also runs this suite (`tests/test_security.py`).

The CLI prints the full result as JSON and exits with **0** for PASSED, **2** for REVIEW, **3** for FAILED, or **1** for a configuration error. `--sig` defaults to `<manifest>.sig`. `--config-dir` defaults to `test_data`. `--record` appends a PASSED image to that folder's ledger.

**Running against an official `sample_input/` folder**, if one is provided: the runner understands the layout where scene folders sit directly under `sample_input/` without `case.json`.

```powershell
python run_security_cases.py --data-dir <path>\sample_input --results-dir test_results\official
```

Tested with Python 3.12 (integration) and 3.14.5 (original development), cryptography 50.0.2 and Pillow 13.0.0.

**Integration with the pipeline:** `src/security/adapter.py` converts the result of `assess()` into the shared `ValidationReport` contract (`src/utils/contracts.py`), so the downstream stages only see one format.

## The interface

```python
from security_assessment import assess, load_trust_store, load_policy, load_ledger, load_reference_time

result = assess(image_path, manifest_path, sig_path,
                trust_store=load_trust_store("test_data/trusted_keys.json"),
                policy=load_policy("test_data/policy.json"),
                ledger=load_ledger("test_data/ledger.json"),
                now=load_reference_time("test_data/reference_time.txt"))
```

Differences from the guide's signature, all documented here:

* `trust_store`, `policy` and `ledger` are the **parsed** JSON objects. Loaders that validate them are provided.
* `now` is an aware `datetime` or a `YYYY-MM-DDTHH:MM:SSZ` string. The real clock is never consulted.
* `sig_path` may point to a file that does not exist. That gives SIG_MISSING.
* `assess()` raises `ConfigurationError` **only** for malformed receiver configuration. Bad images, manifests and signatures never raise; they produce a result.
* The result contains the guide's fields plus `reasons` (human-readable messages) and `evidence`. The evidence holds the image SHA-256, key_id, policy version, assessment time, module version, and every check with pass or fail (guide §10 stretch goal).
* `record_accepted(ledger_path, result, accepted_at)` updates the ledger. It writes **only** if the status is PASSED, re-checks for duplicates, and writes atomically (a temp file then `os.replace`). `assess()` itself never writes anything.

## What is checked, in order

| Step | Check | Code |
|---|---|---|
| 1 Format (stops at the first failure; other flags `null`) | image file exists *(extension, see below)* | `FORMAT_MISSING` |
| | extension in `allowed_formats` (`jpeg`→`jpg`, `tiff`→`tif`) | `FORMAT_EXTENSION` |
| | `min_file_size_bytes` ≤ size ≤ `max_file_size_mb` | `FORMAT_SIZE` |
| | magic bytes match the extension (PNG `89 50 4E 47 0D 0A 1A 0A`, JPEG `FF D8 FF`, TIFF `II*\0` / `MM\0*`) | `FORMAT_MAGIC` |
| | Pillow, restricted to the claimed format's decoder, opens the file and reports that format | `FORMAT_DECODE` |
| | width × height ≤ `max_pixels`, read from the header **before** decoding | `FORMAT_DIMENSIONS` |
| | full decode succeeds | `FORMAT_DECODE` |
| 2 Signature (stops at the first failure) | manifest is valid JSON with all fields | `MANIFEST_INVALID` |
| | `.sig` exists | `SIG_MISSING` |
| | `key_id` in the trust store | `KEY_UNKNOWN` |
| | key status `active` (checked **before** the cryptography) | `KEY_REVOKED` |
| | `valid_from ≤ signed_at ≤ valid_until` | `KEY_NOT_VALID` |
| | base64-decode, then Ed25519 verify over the signing input | `SIG_INVALID` |
| | SHA-256(image) = `manifest.sha256` (only **after** verify) | `HASH_MISMATCH` |
| 3 Metadata (every check runs) | timestamp, sensor_id, latitude, longitude present | `META_MISSING_FIELD` |
| | latitude is a number in [−90, 90] | `META_LATITUDE` |
| | longitude is a number in [−180, 180] | `META_LONGITUDE` |
| | sensor in the key's `allowed_sensors` (unknown key: in any key's list) | `META_SENSOR` |
| | timestamp is `YYYY-MM-DDTHH:MM:SSZ` | `META_TIME_INVALID` |
| | timestamp ≤ now + `clock_skew_seconds` | `META_TIME_FUTURE` |
| | now − timestamp ≤ `max_age_hours` | `META_TIME_STALE` |
| | `manifest.file_name` = the real file name | `META_FILENAME` |
| 4 Replay | `image_id` not in the ledger | `REPLAY_ID` |
| | SHA-256 of the image not in the ledger | `REPLAY_CONTENT` |

**Signing input** (guide §4.2), built in `signing_input()`:

```python
b"TSYP14-IMG-MANIFEST-v1\n" + json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
```

The input is built from the *parsed* manifest, so the formatting of the file on disk does not matter. The generated manifests are written indented, and one test stores non-ASCII text as `\u` escapes.

**Decision** (guide §5):

* **FAILED** if the format failed, the signature or integrity failed, or a replay was detected.
* Otherwise, if the metadata failed, the policy's `metadata_failure_action` applies (REVIEW in the demo; FAILED is also allowed).
* Otherwise **PASSED**.

`reason_codes` lists every failed check, in the order of the table above.

## Documented ambiguities and implementation choices

These points are not fully specified by the guide. Each one is a deliberate, tested choice.

1. **Empty file → `FORMAT_SIZE`.** The guide suggests rejecting empty files (§10) but gives no code for it. The new limit is the policy field `min_file_size_bytes` (default 1).
2. **Missing image file → `FORMAT_MISSING`.** This is a new code that is **not** in the guide. No guide code fits a file that does not exist.
3. **File size unit.** `max_file_size_mb` is read as MiB (×1024×1024).
4. **One format code at most.** Format checks run in the guide's table order and stop at the first failure. The image is read into memory once, then the same bytes are decoded and hashed, so the file cannot be swapped between the checks.
5. **Pillow safety.** Pillow is restricted to the claimed format's decoder. Pillow's own decompression-bomb error (about 179 Mpx and up) is mapped to `FORMAT_DIMENSIONS`. Pillow's *warning* zone (89–179 Mpx) is left to `max_pixels`. Pillow's `MPO` (multi-picture JPEG) counts as JPEG.
6. **No bit-depth check.** The diagram's "bit depth = 8" and the guide's §10 stretch goal mention one, but no reason code exists for it, so it is not enforced.
7. **Manifest schema.** These fields are checked for type:
   * `manifest_version` must be the integer 1;
   * `image_id` and `key_id` must be non-empty strings;
   * `file_name` and `signed_at` must be strings;
   * `sha256` must be 64 hex characters;
   * `metadata` must be an object.

   `size_bytes` and `nonce` only need to exist, as the guide says. Extra fields are allowed, and the signature covers them.
8. **Strict JSON.** Duplicate keys and `NaN`/`Infinity` in the manifest give `MANIFEST_INVALID`. This is hardening beyond the guide, because a duplicate key makes it unclear what was signed.
9. **Unparseable manifest.** `image_id` and `metadata_valid` are `null`, because there is no object to check. Replay still runs the content check, so `replay_detected` reflects it. If the manifest parses but is incomplete, the metadata and replay checks run on what is there.
10. **Unusable `.sig`.** Invalid base64, a length other than 64 bytes, or an unreadable file gives `SIG_INVALID`. This is evaluated at the verify step, *after* the key checks. A garbage signature under an unknown key therefore reports `KEY_UNKNOWN`.
11. **Key status.** Any status other than `active` is treated as `KEY_REVOKED` (fail closed).
12. **Key validity window.** Both ends are inclusive. An unparseable `signed_at` gives `KEY_NOT_VALID`, because it cannot be shown to lie in the window. As the guide states, the window is checked against `signed_at`, not against `now`.
13. **Metadata details.**
    * A JSON `null` counts as a missing field.
    * A missing field is reported once, as `META_MISSING_FIELD`, and its specific check is skipped.
    * An invalid timestamp skips the future and stale checks.
    * Future and stale cannot both fire.
    * The boundaries are accepted: an age of exactly `max_age_hours`, or exactly now + skew.
14. **Sensor check.** It uses the claimed key's `allowed_sensors` even if that key is revoked or expired. It falls back to the union of all keys' sensors only when the key is unknown (guide scene 06).
15. **`META_FILENAME`** compares exactly and case-sensitively against the base name of the received file.
16. **Configuration errors raise.** Malformed policy, trust store, ledger or reference time raises `ConfigurationError`. A trust-store entry with any `private*` field is refused.

## Synthetic dataset

`generate_test_dataset.py` writes all of `test_data/`. It deliberately **does not import** the production module. Its own `canonical_signing_bytes()` is pinned against a hard-coded byte string, so a serialisation bug in one implementation cannot hide in the other.

**Keys are disposable test keys.** Each one is derived from a public label, for example `sha256("TSYP14-DISPOSABLE-TEST-KEY:sat-alpha-2026a")`, as the guide's `make_samples.py` does. Anyone can recompute them. They are never written to disk and must never be used for anything real.

**The trust store** contains the guide's three keys:

* `sat-alpha-2026a`, active;
* `sat-bravo-2026a`, active;
* `sat-alpha-2025x`, revoked.

It also has two extra active keys for the `KEY_NOT_VALID` tests: `sat-charlie-2024`, which has expired, and `sat-delta-2027`, which is not yet valid. `sat-rogue-99` and `attacker` are not in it.

**Images** are 256×256 blurred fields with a road and a river, made with Pillow. Their sizes and hashes differ from the official samples.

**The 13 guide scenes** reproduce the setups and expected results of guide §7: same keys, sensors, ages, coordinates, tampering steps and ledger relations. The ledger holds `scene_000` (the hash of the text "previously accepted image") and `scene_099` (the picture that scene 09 reuses).

**35 edge cases** cover every check the guide marks "no scene", plus the guide's §10 extras and a few hardening cases:

* **Format:**
  * empty file, truncated PNG, `.bmp`;
  * PNG magic followed by garbage;
  * oversized file (policy override `max_file_size_mb: 0.001`, to avoid a 100 MB fixture);
  * a real 7000×6000 PNG;
  * a 100000×100000 "bomb" header;
  * a JPEG renamed to `.png`;
  * a missing image.
* **Manifest and signature:**
  * malformed JSON, a missing field, version 2, duplicate keys, a missing manifest;
  * invalid base64, a 32-byte signature;
  * an expired key, a not-yet-valid key, an unparseable `signed_at`.
* **Metadata:**
  * missing timestamp;
  * longitude 200.5, latitude as a string;
  * a timestamp 1 h in the future, and one 4 min ahead (inside the skew, so PASSED);
  * a malformed timestamp, a file name mismatch, a sensor that belongs to another key;
  * four metadata failures at once (tests code order).
* **Combinations:**
  * unknown key plus unknown sensor;
  * a tampered manifest with a bad latitude (FAILED overrides REVIEW);
  * replay by both ID and content;
  * `metadata_failure_action: FAILED`;
  * a clean `.jpeg`;
  * two clean manifests containing non-ASCII text (these exercise `ensure_ascii=False`).

The full table, with expected flags and codes, is in `test_data/CASES.md`.

## Test runner

`run_security_cases.py` calls `assess()` for every case. For each one it compares:

* `image_id`;
* the four flags, with strict types so that `true` ≠ `1` and `null` ≠ `false`;
* `security_status`;
* the **ordered** `reason_codes`.

It prints PASS or FAIL per case, with one line of explanation for each wrong field. It also runs five invariant tests:

* `assess()` never opens `expected.json`;
* the ledger file and dict are unchanged;
* `record_accepted()` records only PASSED results, and a recorded image then triggers REPLAY_ID + REPLAY_CONTENT;
* results are deterministic;
* a trust store holding private keys is refused.

It writes `test_results/results.json` and `results.csv`, and exits non-zero on any failure. Example output from a deliberately broken module that skips the revocation check:

```
[FAIL] 07_revoked_key                       PASSED  -
         -> signature_valid: expected false, got true
         -> security_status: expected "FAILED", got "PASSED"
         -> reason_codes: expected ["KEY_REVOKED"], got []
```

`tools/mutation_check.py` runs the suite against 17 such single-bug copies of the module, for example unsorted keys, a missing prefix, `ensure_ascii=True`, no base64 decode, revocation ignored, hash check skipped, no early format stop, and the real clock. The check passes only if all 17 are caught.

## Results (this environment)

* `run_security_cases.py`: **53/53 passed (100%)**. That is 13/13 guide scenes, 35/35 edge cases and 5/5 invariants.
* `tools/mutation_check.py`: **17/17 mutants killed**.
* A flattened copy of the 13 scenes in the official `sample_input/` layout (no `case.json`): 13/13 plus 5/5 invariants.

**Not executed:** the official organiser `sample_input/` was not available. Its `expected.json` files, real hashes and signatures could not be tested. The command above runs them once you have the folder.

## The guide's self-check questions

1. **Why compare hashes when the signature is valid?** The signature covers only the manifest. Swapping the image file leaves the manifest and signature intact (scene 03), so only the hash comparison links the bytes to what was signed.
2. **Why is bad metadata REVIEW but a bad signature FAILED?** A bad signature means the origin is unproven, so nothing can be trusted. Bad metadata from a proven, trusted sender may be an honest mistake and needs a human look.
3. **Why does the receiver need no key pair?** It only *verifies*. Verification needs the sender's public key, and the receiver signs nothing and decrypts nothing.
4. **Why check key status before verifying?** A revoked key's signatures are still mathematically valid. Checking first ensures that a compromised key can never pass, and it avoids treating "math OK" as "trusted".
5. **Why add to the ledger only after PASSED?** Recording rejected images would let an attacker send garbage under legitimate IDs or hashes and "burn" them, blocking the genuine image later.
