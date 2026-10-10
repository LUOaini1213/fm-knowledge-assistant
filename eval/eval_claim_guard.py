"""Reproduce a synthetic checker comparison without search, a corpus or an LLM.

These authored challenge cases are regression evidence, not model accuracy or a
random sample of building-code questions. Boundary cases are recorded but unscored.
Source and suite fingerprints use UTF-8 with LF line endings for Windows/Linux parity.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "8b8591fcb3bf9382f01ff49299de2de574a080f5"
SUITE = ROOT / "eval" / "claim_guard_cases.jsonl"
REPORT = ROOT / "eval" / "results" / "claim_guard.json"


def fingerprint(text):
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def baseline_checker():
    sources = {}
    for name in ("numeric", "answer"):
        path = f"src/fmka/{name}.py"
        sources[path] = subprocess.run(
            ["git", "show", f"{BASELINE}:{path}"], cwd=ROOT, check=True,
            capture_output=True, text=True, encoding="utf-8").stdout
    # Only the inspected, stdlib numeric scanner and the old verify function
    # execute. Neither imports retrieval, the API, a chat client or a model.
    numeric = {"__name__": "frozen_numeric"}
    exec(compile(sources["src/fmka/numeric.py"], "frozen_numeric", "exec"), numeric)
    tree = ast.parse(sources["src/fmka/answer.py"])
    nodes = [node for node in tree.body if
             (isinstance(node, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == "CITE" for t in node.targets)) or
             (isinstance(node, ast.FunctionDef) and node.name == "verify")]
    namespace = {"re": re, "numbers": numeric["numbers"], "scan_numbers": numeric["scan_numbers"]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen_verify", "exec"), namespace)
    return namespace["verify"], {p: fingerprint(s) for p, s in sources.items()}


def read_cases():
    text = SUITE.read_text(encoding="utf-8")
    cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    seen = set()
    for case in cases:
        if set(case) != {"id", "expected", "draft", "passages", "reason"}:
            raise ValueError("case fields do not match the declared schema")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen:
            raise ValueError("case IDs must be unique nonempty strings")
        seen.add(case["id"])
        if case["expected"] not in {"accept", "reject", "boundary"}:
            raise ValueError("unknown expectation")
        if not all(isinstance(case[k], str) and case[k] for k in ("draft", "reason")):
            raise ValueError("drafts and reasons must be nonempty text")
        if not isinstance(case["passages"], list) or not case["passages"] or not all(
                isinstance(p, str) and p for p in case["passages"]):
            raise ValueError("passages must be a nonempty text list")
    if not cases:
        raise ValueError("challenge suite is empty")
    return cases, fingerprint(text)


def evaluate(checker, cases):
    outcomes = []
    summary = {"correct_examples": 0, "correct_accepted": 0, "correct_refused": 0,
               "incorrect_examples": 0, "incorrect_rejected": 0, "incorrect_accepted": 0,
               "unscored_boundaries": 0}
    for case in cases:
        cited, problems = checker(case["draft"], case["passages"])
        accepted = not problems
        outcomes.append({"id": case["id"], "expected": case["expected"],
                         "accepted": accepted, "cited": cited, "problems": problems})
        if case["expected"] == "accept":
            summary["correct_examples"] += 1
            summary["correct_accepted" if accepted else "correct_refused"] += 1
        elif case["expected"] == "reject":
            summary["incorrect_examples"] += 1
            summary["incorrect_accepted" if accepted else "incorrect_rejected"] += 1
        else:
            summary["unscored_boundaries"] += 1
    return {"summary": summary, "outcomes": outcomes}


def build_report():
    sys.path.insert(0, str(ROOT / "src"))
    from fmka.answer import verify
    cases, suite_hash = read_cases()
    previous, previous_hashes = baseline_checker()
    current_hashes = {p.relative_to(ROOT).as_posix(): fingerprint(p.read_text(encoding="utf-8"))
                      for p in sorted((ROOT / "src" / "fmka").glob("*.py"))}
    return {"schema_version": 1, "evidence_kind": "authored_synthetic_checker_regressions",
            "model_or_corpus_calls": 0, "fingerprint_format": "sha256_utf8_lf",
            "evaluator_sha256": fingerprint(Path(__file__).read_text(encoding="utf-8")),
            "suite": "eval/claim_guard_cases.jsonl", "suite_sha256": suite_hash,
            "case_count": len(cases), "baseline_commit": BASELINE,
            "baseline_source_sha256": previous_hashes, "current_source_sha256": current_hashes,
            "baseline": evaluate(previous, cases), "current": evaluate(verify, cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the stored report without writing it")
    args = parser.parse_args()
    report = build_report()
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not REPORT.exists() or REPORT.read_text(encoding="utf-8").replace("\r\n", "\n") != encoded:
            raise SystemExit("Stored claim-guard report is missing or stale; regenerate it with this script.")
        if report["current"]["summary"]["incorrect_accepted"]:
            raise SystemExit("The current checker accepted an incorrect challenge case; see the stored report.")
    else:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(encoded, encoding="utf-8", newline="\n")
    print(json.dumps({"baseline": report["baseline"]["summary"],
                      "current": report["current"]["summary"]}, ensure_ascii=False))
    print("Stored report verified." if args.check else f"Wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
