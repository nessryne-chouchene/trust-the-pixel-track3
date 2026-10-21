"""Independent test runner for security_assessment.assess().

Runs every case of a dataset, compares the result with the case's
expected.json (image_id, the four flags, security_status and the ordered
reason_codes), and runs invariant tests (ledger untouched, ledger update only
after PASSED, assess() never reads expected.json, determinism, no private
keys in the trust store).

Two dataset layouts are understood:

* the synthetic one made by generate_test_dataset.py: shared files at the top,
  cases under ``scenes/`` and ``edge_cases/``, each with a ``case.json``;
* the official ``sample_input/`` layout described in the guide (section 3.1):
  shared files at the top and scene folders directly below it, without
  case.json. The inputs are then inferred from the folder contents
  (``*.manifest.json``, ``*.manifest.sig`` and the one remaining file).

Writes <results-dir>/results.json and results.csv. Exit status: 0 all
passed, 1 at least one failure, 2 dataset missing.

Usage (from the project directory):
    python generate_test_dataset.py
    python run_security_cases.py
    python run_security_cases.py --data-dir ../sample_input --results-dir test_results/official
"""

from __future__ import annotations

import argparse
import builtins
import copy
import csv
import hashlib
import io
import json
import platform
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

try:  # run as a script from this folder, or as a mutant copy (tools/mutation_check.py)
    import security_assessment as sa
except ImportError:  # imported from the repository root (pytest)
    from src.security import security_assessment as sa

ROOT = Path(__file__).resolve().parent
GROUPS = ("scenes", "edge_cases")
COMPARED_FIELDS = ("image_id", "format_valid", "signature_valid", "metadata_valid",
                   "replay_detected", "security_status", "reason_codes")
NOT_INPUTS = {"expected.json", "case.json"}


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _same(a: Any, b: Any) -> bool:
    """Strict equality: True must not equal 1, None must not equal False."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return one human-readable explanation per mismatching field."""
    problems = []
    for name in COMPARED_FIELDS:
        exp, act = expected.get(name), actual.get(name)
        if _same(exp, act):
            continue
        if name == "reason_codes" and isinstance(act, list) and isinstance(exp, list) \
                and sorted(map(str, act)) == sorted(map(str, exp)):
            problems.append(f"reason_codes order differs: expected {exp}, got {act}")
        else:
            problems.append(f"{name}: expected {json.dumps(exp)}, got {json.dumps(act)}")
    return problems


def merged_policy(base: dict[str, Any], override: Optional[dict[str, Any]]) -> dict[str, Any]:
    policy = copy.deepcopy(base)
    policy.update(override or {})
    return policy


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------

class Dataset:
    """Shared receiver files plus the list of case folders."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.trust = sa.load_trust_store(root / "trusted_keys.json")
        self.policy = sa.load_policy(root / "policy.json")
        self.ledger = sa.load_ledger(root / "ledger.json")
        self.now = sa.load_reference_time(root / "reference_time.txt")
        if any((root / g).is_dir() for g in GROUPS):
            self.case_dirs = [d for g in GROUPS if (root / g).is_dir()
                              for d in sorted((root / g).iterdir()) if (d / "expected.json").is_file()]
        else:  # official sample_input/ layout
            self.case_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and (d / "expected.json").is_file()]

    def case(self, case_dir: Path) -> dict[str, Any]:
        """Read case.json, or infer the inputs from the folder contents."""
        if (case_dir / "case.json").is_file():
            return json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        manifests = sorted(p.name for p in case_dir.glob("*.manifest.json"))
        images = sorted(p.name for p in case_dir.iterdir() if p.is_file() and p.name not in NOT_INPUTS
                        and not p.name.endswith((".manifest.json", ".manifest.sig")))
        if len(manifests) != 1 or len(images) != 1:
            raise ValueError(f"cannot infer inputs in {case_dir}: manifests={manifests}, images={images}")
        group = case_dir.parent.name if case_dir.parent != self.root else "sample"
        return {"case_id": case_dir.name, "group": group, "description": "(inputs inferred from folder contents)",
                "image": images[0], "manifest": manifests[0], "signature": manifests[0][: -len(".json")] + ".sig"}

    def assess_case(self, case_dir: Path, ledger: Optional[dict] = None) -> dict[str, Any]:
        c = self.case(case_dir)
        return sa.assess(case_dir / c["image"], case_dir / c["manifest"], case_dir / c["signature"],
                         self.trust, merged_policy(self.policy, c.get("policy_override")),
                         self.ledger if ledger is None else ledger, self.now)

    def find(self, case_id: str) -> Optional[Path]:
        return next((d for d in self.case_dirs if d.name == case_id), None)


def run_case(ds: Dataset, case_dir: Path) -> dict[str, Any]:
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    record: dict[str, Any] = {"case_id": case_dir.name, "group": "?", "description": "", "expected": expected}
    try:
        case = ds.case(case_dir)
        record.update(group=case["group"], description=case["description"],
                      spec_reference=case.get("spec_reference"), policy_override=case.get("policy_override"))
        actual = ds.assess_case(case_dir)
    except Exception as exc:  # a crash is a failed test, not a crashed runner
        record.update(passed=False, actual=None, reasons=[],
                      mismatches=[f"assess() raised {type(exc).__name__}: {exc}"], traceback=traceback.format_exc())
        return record
    record["actual"] = {k: actual.get(k) for k in COMPARED_FIELDS}
    record["reasons"] = actual.get("reasons", [])
    record["mismatches"] = compare(expected, actual)
    record["passed"] = not record["mismatches"]
    return record


# --------------------------------------------------------------------------
# Invariant tests (beyond the per-case comparison)
# --------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inv_never_reads_expected(ds: Dataset, snapshot: dict) -> list[str]:
    """assess() must not open any expected.json while assessing."""
    opened: list[str] = []
    real_open = builtins.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    cases = [(d, ds.case(d)) for d in ds.case_dirs]  # read case.json before spying
    builtins.open = spy
    try:
        for d, c in cases:
            sa.assess(d / c["image"], d / c["manifest"], d / c["signature"], ds.trust,
                      merged_policy(ds.policy, c.get("policy_override")), ds.ledger, ds.now)
    finally:
        builtins.open = real_open
    return [f"assess() opened {p}" for p in opened if Path(p).name == "expected.json"]


def inv_ledger_untouched(ds: Dataset, snapshot: dict) -> list[str]:
    """The whole run must not modify ledger.json or the ledger dict."""
    problems = []
    if _sha(ds.root / "ledger.json") != snapshot["file"]:
        problems.append("ledger.json changed on disk during the test run")
    if json.dumps(ds.ledger, sort_keys=True) != snapshot["dict"]:
        problems.append("the in-memory ledger dict was mutated by assess()")
    return problems


def inv_record_only_after_pass(ds: Dataset, snapshot: dict) -> list[str]:
    """record_accepted(): non-PASSED results are never recorded, a PASSED one
    is, and re-sending the recorded image is then caught as a replay."""
    clean = ds.find("01_clean_pass")
    rejected = [d for d in (ds.find(n) for n in ("03_image_tampered", "10_bad_metadata_review", "08_replay_same_id")) if d]
    if clean is None or not rejected:
        return ["required scenes (01, 03, 08, 10) not found in dataset"]
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_ledger = Path(tmp) / "ledger.json"
        shutil.copyfile(ds.root / "ledger.json", tmp_ledger)
        before = _sha(tmp_ledger)
        for d in rejected:
            if sa.record_accepted(tmp_ledger, ds.assess_case(d, sa.load_ledger(tmp_ledger)), ds.now):
                problems.append(f"record_accepted() recorded non-PASSED case {d.name}")
        if _sha(tmp_ledger) != before:
            problems.append("ledger file changed after non-PASSED results")
        if not sa.record_accepted(tmp_ledger, ds.assess_case(clean, sa.load_ledger(tmp_ledger)), ds.now):
            problems.append("record_accepted() did not record a PASSED result")
        again = ds.assess_case(clean, sa.load_ledger(tmp_ledger))
        if again["security_status"] != "FAILED" or again["reason_codes"] != ["REPLAY_ID", "REPLAY_CONTENT"]:
            problems.append(f"re-sent accepted image not caught as replay: {again['reason_codes']}")
        leftovers = [p.name for p in Path(tmp).iterdir() if p.name != "ledger.json"]
        if leftovers:
            problems.append(f"temporary files left behind: {leftovers}")
    return problems


def inv_deterministic(ds: Dataset, snapshot: dict) -> list[str]:
    """Same inputs twice -> identical result (no dependence on the real clock)."""
    for d in ds.case_dirs:
        if json.dumps(ds.assess_case(d), sort_keys=True) != json.dumps(ds.assess_case(d), sort_keys=True):
            return [f"results differ between runs for {d.name}"]
    return []


def inv_no_private_keys(ds: Dataset, snapshot: dict) -> list[str]:
    """The receiver must never hold private keys: such a trust store is refused."""
    bad = copy.deepcopy(ds.trust)
    bad["keys"][0]["private_key_hex"] = "00" * 32
    d = ds.case_dirs[0]
    c = ds.case(d)
    try:
        sa.assess(d / c["image"], d / c["manifest"], d / c["signature"], bad, ds.policy, ds.ledger, ds.now)
    except sa.ConfigurationError:
        return []
    return ["a trust store containing a private key was accepted"]


INVARIANTS: list[tuple[str, str, Callable[[Dataset, dict], list[str]]]] = [
    ("INV_never_reads_expected", "assess() never opens expected.json", inv_never_reads_expected),
    ("INV_ledger_untouched", "assess() never modifies the ledger (file or dict)", inv_ledger_untouched),
    ("INV_record_only_after_pass", "ledger update only after PASSED; recorded image then replays", inv_record_only_after_pass),
    ("INV_deterministic", "identical inputs give identical results", inv_deterministic),
    ("INV_no_private_keys", "trust store with private key material is refused", inv_no_private_keys),
]


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def write_reports(results_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(
        json.dumps({"summary": summary, "results": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    columns = ["case_id", "group", "result", "expected_status", "actual_status",
               "expected_flags", "actual_flags", "expected_reason_codes", "actual_reason_codes",
               "mismatches", "description"]
    flag_names = ("format_valid", "signature_valid", "metadata_valid", "replay_detected")

    def flags(d: Optional[dict]) -> str:
        return "" if not d else "/".join(json.dumps(d.get(f)) for f in flag_names)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for r in records:
        exp, act = r.get("expected") or {}, r.get("actual") or {}
        writer.writerow({
            "case_id": r["case_id"], "group": r["group"], "result": "PASS" if r["passed"] else "FAIL",
            "expected_status": exp.get("security_status", ""), "actual_status": act.get("security_status", ""),
            "expected_flags": flags(exp), "actual_flags": flags(act),
            "expected_reason_codes": " ".join(exp.get("reason_codes") or []),
            "actual_reason_codes": " ".join(act.get("reason_codes") or []),
            "mismatches": " | ".join(r["mismatches"]), "description": r["description"],
        })
    (results_dir / "results.csv").write_text(buf.getvalue(), encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run security_assessment.assess() against a test dataset")
    parser.add_argument("--data-dir", default=str(ROOT / "test_data"), help="dataset folder (default: test_data)")
    parser.add_argument("--results-dir", default=str(ROOT / "test_results"), help="report folder (default: test_results)")
    args = parser.parse_args(argv)
    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)

    if not (data_dir / "policy.json").is_file():
        print(f"{data_dir} has no policy.json. Generate the synthetic data with:  python generate_test_dataset.py",
              file=sys.stderr)
        return 2
    ds = Dataset(data_dir)
    snapshot = {"file": _sha(data_dir / "ledger.json"), "dict": json.dumps(ds.ledger, sort_keys=True)}

    print(f"TSYP14 security assessment tests on {_rel(data_dir)}  "
          f"(reference time {sa.format_utc(ds.now)}, policy {ds.policy.get('version')})\n")
    records: list[dict[str, Any]] = []
    current_group = None
    for case_dir in ds.case_dirs:
        r = run_case(ds, case_dir)
        if r["group"] != current_group:
            current_group = r["group"]
            print(f"--- {current_group} ---")
        records.append(r)
        status = (r["actual"] or {}).get("security_status", "ERROR")
        codes = ",".join((r["actual"] or {}).get("reason_codes") or []) or "-"
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['case_id']:<36} {status:<7} {codes}")
        for m in r["mismatches"]:
            print(f"         -> {m}")

    print("--- invariants ---")
    for inv_id, description, fn in INVARIANTS:
        try:
            problems = fn(ds, snapshot)
        except Exception as exc:
            problems = [f"raised {type(exc).__name__}: {exc}"]
        records.append({"case_id": inv_id, "group": "invariants", "description": description,
                        "expected": None, "actual": None, "reasons": [], "mismatches": problems,
                        "passed": not problems})
        print(f"[{'PASS' if not problems else 'FAIL'}] {inv_id:<36} {description}")
        for m in problems:
            print(f"         -> {m}")

    total = len(records)
    passed = sum(r["passed"] for r in records)
    by_group: dict[str, dict[str, int]] = {}
    for r in records:
        g = by_group.setdefault(r["group"], {"total": 0, "passed": 0})
        g["total"] += 1
        g["passed"] += r["passed"]
    summary = {
        "dataset": _rel(data_dir), "total": total, "passed": passed, "failed": total - passed,
        "success_rate_percent": round(100.0 * passed / total, 2) if total else 0.0,
        "by_group": by_group, "failed_cases": [r["case_id"] for r in records if not r["passed"]],
        "reference_time": sa.format_utc(ds.now), "policy_version": ds.policy.get("version"),
        "module_version": sa.MODULE_VERSION, "python": platform.python_version(),
        "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Synthetic provenance/integrity tests only; they say nothing about detecting "
                "visual manipulation in real satellite imagery.",
    }
    write_reports(results_dir, records, summary)

    print(f"\nSummary: {passed}/{total} passed, {total - passed} failed ({summary['success_rate_percent']}%)")
    for g, c in by_group.items():
        print(f"  {g:<11} {c['passed']}/{c['total']}")
    if total - passed:
        print("Failed: " + ", ".join(summary["failed_cases"]))
    print(f"Reports: {_rel(results_dir / 'results.json')}, {_rel(results_dir / 'results.csv')}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
