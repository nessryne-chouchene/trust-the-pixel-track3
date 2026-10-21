"""Mutation check for the test suite (optional quality tool).

Creates temporary copies of the project in which security_assessment.py has
ONE deliberate spec violation (wrong signing bytes, revocation ignored, ...),
runs test_security_assessment.py against each copy and reports whether the
suite noticed ("KILLED"). A mutant that SURVIVES points at a gap in the test
data. The real project files are never modified.

Usage (from the project directory, after generate_test_dataset.py):
    python tools/mutation_check.py
Exit status 1 if any mutant survives.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = sys.executable
SRC = (PROJ / "security_assessment.py").read_text(encoding="utf-8")
MUTANTS = {
 "unsorted keys":        ('sort_keys=True, separators', 'sort_keys=False, separators'),
 "spaced separators":    ('separators=(",", ":"), ensure_ascii=False)\n    return', 'separators=(", ", ": "), ensure_ascii=False)\n    return'),
 "ensure_ascii=True":    ('ensure_ascii=False)\n    return SIGNING_PREFIX', 'ensure_ascii=True)\n    return SIGNING_PREFIX'),
 "no domain prefix":     ('return SIGNING_PREFIX + body', 'return body'),
 "no base64 decode":     ('signature = base64.b64decode(raw.decode("ascii").strip(), validate=True)', 'signature = raw.strip()'),
 "revocation ignored":   ('if key["status"] != "active":', 'if False:'),
 "hash check skipped":   ('if not hmac.compare_digest(image_sha256, manifest["sha256"].lower()):', 'if False:'),
 "no early format stop": ('    if data is None:\n        return result\n', '    if data is None:\n        data = b""\n'),
 "metadata -> FAILED":   ('result["security_status"] = policy["metadata_failure_action"]', 'result["security_status"] = FAILED'),
 "no content replay":    ('if image_sha256 in hashes:', 'if False:'),
 "skew ignored":         ('now + timedelta(seconds=policy["clock_skew_seconds"])', 'now'),
 "sensor union always":  ('if key is not None:\n            allowed', 'if False:\n            allowed'),
 "metadata codes reversed": ('    return len(trail.reason_codes) == before', '    trail.reason_codes[before:] = trail.reason_codes[before:][::-1]\n    return len(trail.reason_codes) == before'),
 "no magic check":       ('if not any(data.startswith(magic) for magic in _MAGIC[ext]):', 'if False:'),
 "no pixel limit":       ('width * height > policy["max_pixels"]', 'False'),
 "real clock":           ('    now_utc = _coerce_now(now)\n', '    now_utc = datetime.now(timezone.utc)\n'),
 "key validity ignored": ('if signed_at is None or not (key["valid_from"] <= signed_at <= key["valid_until"]):', 'if False:'),
}


def main() -> int:
    if not (PROJ / "test_data" / "policy.json").is_file():
        print("run generate_test_dataset.py first", file=sys.stderr)
        return 2
    survivors = []
    for name, (old, new) in MUTANTS.items():
        if SRC.count(old) != 1:
            print(f"SKIPPED  {name:<24} pattern no longer matches the source")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            shutil.copy(PROJ / "run_security_cases.py", t)
            shutil.copytree(PROJ / "test_data", t / "test_data")
            (t / "security_assessment.py").write_text(SRC.replace(old, new), encoding="utf-8")
            p = subprocess.run([PY, "run_security_cases.py"], cwd=t, capture_output=True, text=True)
            summary = next((line for line in p.stdout.splitlines() if line.startswith("Summary")), p.stderr[-200:])
            killed = p.returncode != 0
            print(f"{'KILLED  ' if killed else 'SURVIVED'} {name:<24} {summary}")
            if not killed:
                survivors.append(name)
    print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed; survivors: {survivors or 'none'}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
