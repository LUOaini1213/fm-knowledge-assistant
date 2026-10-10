"""Mutation check: compilable defects must cause real pytest test failures.

Run without arguments for the full clean baseline and mutation suite, or use --self-test
to check the harness in temporary toy projects without touching repository sources.
Collection/setup errors, missing reports, crashes and timeouts are infrastructure errors,
not mutation kills. Each subprocess gets a fresh bytecode location and every source write
is covered by restoration, including compilation and test-launch failures.
"""
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = str(ROOT / "tests")
TEST_TIMEOUT_S = 300
MUTANTS = [
    ('src/fmka/answer.py', 'problems.append(f"numbers not found in the cited passages: {missing}")', 'pass'),
    ('src/fmka/answer.py', 'source_nums = set().union(*(numbers(passages[c - 1]) for c in good))',
     'source_nums = set().union(*(numbers(passages[c - 1]) for c in range(1, len(passages) + 1)))'),
    ('src/fmka/answer.py', 'bad = [c for c in cited if c < 1 or c > len(passages)]', 'bad = []'),
    ('src/fmka/answer.py', 'if top_sim is None or top_sim < min_sim:', 'if top_sim is None:'),
    ('src/fmka/answer.py', 'status = "answered" if not problems else "unverified"', 'status = "answered"'),
    ('src/fmka/answer.py', 'if not draft.upper().startswith("NOT_FOUND") and not CITE.search(draft):', 'if False:'),
    ('src/fmka/answer.py', 'if strict and not problems:', 'if False:'),
    ('src/fmka/extract.py', '("min", re.compile(MIN_WORDS + r"\\s*" + QTY, re.I)),', '("max", re.compile(MIN_WORDS + r"\\s*" + QTY, re.I)),'),
    ('src/fmka/extract.py', '("max", re.compile(r"\\bno\\b[^.;]{0,40}?\\bmore than\\s+" + QTY, re.I)),', ''),
    ('src/fmka/extract.py', 'out.append(Requirement("range", _num(lo), _unit(u2 or u1), subj, sentence, _num(hi)))', 'out.append(Requirement("range", _num(lo), _unit(u2 or u1), subj, sentence, None))'),
    ('src/fmka/extract.py', 'if MEASURE.search(subj) and len(subj.split()) <= 3 and len(parts) > 1:', 'if False:'),
    ('src/fmka/retrieval.py', 'self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}', 'self.idf = {t: 1.0 for t, f in df.items()}'),
    ('src/fmka/retrieval.py', 'agg[i] = agg.get(i, 0.0) + 1.0 / (k + rank + 1)', 'agg[i] = 1.0 / (k + rank + 1)'),
    ('src/fmka/retrieval.py', 'TOKEN = re.compile(r"[a-z]+|\\d+(?:\\.\\d+)*")', 'TOKEN = re.compile(r"[a-z0-9]+(?:\\.[0-9]+)*")'),
    ('src/fmka/parse.py', '(last_major is not None and major == last_major + 1 and starts_chapter)', '(last_major is not None)'),
    ('src/fmka/classify.py', 'return " ".join(CLAUSE_REF.sub(" ", text).split())', 'return " ".join(text.split())'),
    ('src/fmka/search.py', 'allowed = None if not docs else set(index.chunks.index[index.chunks.doc_id.isin(docs)])', 'allowed = None'),
    # Classification must not encode source/chapter metadata or train on held-out rows.
    ("src/fmka/classify.py", "index.model.encode(texts,",
     'index.model.encode([f"{index.titles[r.doc_id]} {r.heading} {r.text}" for r in ch.itertuples()],'),
    ("src/fmka/classify.py", "fit([texts[i] for i in tr], labels[tr])", "fit(texts, labels)"),
    ("src/fmka/classify.py", "fit(vectors[tr], labels[tr])", "fit(vectors, labels)"),
    # Candidate restrictions must apply before top-k selection in both retrieval branches.
    ("src/fmka/retrieval.py", "candidates = np.array([i for i in candidates if i in allowed_ids], dtype=int)",
     "candidates = np.arange(len(s))"),
    ("src/fmka/retrieval.py", "if allowed_ids is not None:\n            ids = sorted({int(i) for i in allowed_ids})",
     "if False:\n            ids = sorted({int(i) for i in allowed_ids})"),
    # Public interfaces withhold invalid drafts and return readable extractive evidence.
    ("src/fmka/api.py", 'a.text if a.status == "answered" else None',
     'a.text if a.status != "extractive" else None'),
    ("src/fmka/api.py", '"text": ix.row(c)["text"]', '"text": ""'),
    ("src/fmka/cli.py", 'text = "The generated answer failed validation and has been withheld." if a.status == "unverified" else a.text',
     "text = a.text"),
    # Numeric verification must include worded values, unsupported forms and signs.
    ("src/fmka/numeric.py", "for match in WORD_SPAN.finditer(text):", "for match in ():"),
    ("src/fmka/answer.py", 'problems.append(f"unsupported numeric expressions: {unsupported}")', "pass"),
    ("src/fmka/numeric.py", 'return ("-" if sign else "") + result', "return result"),
    # Failed rebuilds preserve the old database, while successful rebuilds publish the replacement.
    ("src/fmka/index.py", "staged = Path(tmp) / db.name",
     "staged = Path(tmp) / db.name; db.unlink(missing_ok=True)"),
    ("src/fmka/index.py", "os.replace(staged, db)", "if not db.exists(): os.replace(staged, db)"),
    # Quantity claims bind local citations, units, exact bounds and explicit subjects.
    ("src/fmka/answer.py", "problems.extend(verify_claims(draft, passages))", "pass"),
    ("src/fmka/claims.py", "ALIASES[unit[1].lower()]", '"any-unit"'),
    ("src/fmka/claims.py", 'comparator = operator.lastgroup if operator else "eq"', 'comparator = "eq"'),
    ("src/fmka/claims.py", "return (subject, tuple(quantities)), None", 'return (("any-subject",), tuple(quantities)), None'),
    ("src/fmka/claims.py", "for c in refs for piece in _clauses", "for c in range(1, len(passages) + 1) for piece in _clauses"),
    ("src/fmka/claims.py", "local_numbers = set().union(*(scan_numbers(passages[c - 1])[0] for c in refs))",
     "local_numbers = set().union(*(scan_numbers(p)[0] for p in passages))"),
    ("src/fmka/claims.py", "if not subject:", "if False:"),
    ("src/fmka/answer.py", "for n, claim in enumerate(claims(draft), 1):", "for n, claim in enumerate(claims(draft)[:1], 1):"),
    ("src/fmka/answer.py", 'return reply.upper() == "YES"', 'return reply.upper().startswith("YES")'),
    ("src/fmka/claims.py", "if any(c.isalpha() and not c.isascii() for c in text):", "if False:"),
]

class HarnessError(RuntimeError):
    """The run supplies no valid evidence about a mutant."""


def replacement(root, mutant):
    fname, old, new = mutant
    path = root / fname
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    # LF-authored anchors also work in Windows CRLF checkouts; restoration still uses raw bytes.
    newline = "\r\n" if "\r\n" in text else "\n"
    old = old.replace("\r\n", "\n").replace("\n", newline)
    new = new.replace("\r\n", "\n").replace("\n", newline)
    if text.count(old) != 1 or old == new:
        raise HarnessError(f"mutation target must occur once and change: {fname}: {old[:70]}")
    return path, raw, text.replace(old, new).encode("utf-8")


def compile_source(raw, path):
    # Compile without importing the module or creating a .pyc next to product sources.
    try:
        compile(raw, str(path), "exec")
    except (SyntaxError, ValueError) as exc:
        raise HarnessError(f"mutant does not compile; not a kill: {path}: {exc}") from exc


def run_tests(root, tests, output_dir, *, fail_fast=False, timeout=TEST_TIMEOUT_S):
    output_dir.mkdir(parents=True)
    report = output_dir / "pytest.xml"
    env = dict(os.environ)
    env.pop("PYTEST_ADDOPTS", None)  # always run the declared suite, not a shell's -k filter
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), str(root), env.get("PYTHONPATH")]))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(output_dir / "pycache")
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
           f"--junitxml={report}", str(tests)]
    if fail_fast:
        cmd.append("-x")
    try:
        result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"pytest timed out after {timeout}s; not a kill") from exc
    except OSError as exc:
        raise HarnessError(f"pytest could not start; not a kill: {exc}") from exc
    output = result.stdout + result.stderr
    (output_dir / "pytest.log").write_text(output, encoding="utf-8")
    try:
        document = ET.parse(report)
    except (OSError, ET.ParseError) as exc:
        raise HarnessError(f"pytest exited {result.returncode} without a valid report; not a kill:\n"
                           + output[-3000:]) from exc
    cases = document.findall(".//testcase")
    errors = document.findall(".//error")
    skipped = sum(case.find("skipped") is not None for case in cases)
    failures = []
    for case in cases:
        failure = case.find("failure")
        if failure is not None:
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            message = " ".join((failure.get("message") or failure.text or "test failure").split())
            failures.append(f"{name}: {message[:180]}")
    if errors or not cases or len(cases) == skipped or result.returncode not in (0, 1):
        raise HarnessError(f"pytest exited {result.returncode}, {len(errors)} errors, "
                           f"{len(cases)} cases/{skipped} skipped; not a kill:\n" + output[-3000:])
    if (result.returncode == 1) != bool(failures):
        raise HarnessError(f"pytest exit/report disagree; not a kill:\n" + output[-3000:])
    return {"killed": bool(failures), "tests": len(cases), "skipped": skipped, "failures": failures}


def run_mutant(root, tests, mutant, output_dir, *, timeout=TEST_TIMEOUT_S):
    path, raw, changed = replacement(root, mutant)
    try:
        path.write_bytes(changed)
        compile_source(path.read_bytes(), path)
        return run_tests(root, tests, output_dir, fail_fast=True, timeout=timeout)
    finally:
        path.write_bytes(raw)
        if path.read_bytes() != raw:
            raise HarnessError(f"source restoration failed: {path}")


def main():
    # Check every anchor and the syntax of every proposed mutant before any source write.
    originals = {}
    for mutant in MUTANTS:
        path, raw, changed = replacement(ROOT, mutant)
        originals[path] = raw
        compile_source(changed, path)
    started = time.monotonic()
    escaped = 0
    try:
        with tempfile.TemporaryDirectory(prefix="fmka-mutation-") as scratch:
            output_dir = Path(scratch)
            base = run_tests(ROOT, TESTS, output_dir / "baseline")
            if base["killed"]:
                raise HarnessError("unmodified tests fail; fix the baseline first:\n" + "\n".join(base["failures"]))
            print(f"baseline: {base['tests']} cases, {base['skipped']} skipped, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
            for i, mutant in enumerate(MUTANTS, 1):
                result = run_mutant(ROOT, TESTS, mutant, output_dir / f"mutant-{i:03d}")
                escaped += not result["killed"]
                evidence = result["failures"][0] if result["killed"] else "no test failed"
                anchor = " ".join(mutant[1].split())[:70]
                print(f"{'killed ' if result['killed'] else 'ESCAPED'} [{i}/{len(MUTANTS)}] "
                      f"{mutant[0]}: {anchor} -- {evidence}", flush=True)
                if not result["killed"]:
                    raise HarnessError(f"mutant {i}/{len(MUTANTS)} escaped; stopped for review after restoring source")
    finally:
        unexpected = [path for path, raw in originals.items() if not path.exists() or path.read_bytes() != raw]
        for path in unexpected:
            path.write_bytes(originals[path])
        if unexpected:
            raise HarnessError("unexpected source changes restored: " + ", ".join(map(str, unexpected)))
    print(f"{len(MUTANTS) - escaped}/{len(MUTANTS)} mutants killed "
          f"({time.monotonic() - started:.1f}s); all source bytes restored", flush=True)
    return 1 if escaped else 0


def self_test():
    """Exercise failure classification/restoration without importing product code."""
    import py_compile
    import unittest
    from unittest.mock import patch

    class HarnessTests(unittest.TestCase):
        def setUp(self):
            self.scratch = tempfile.TemporaryDirectory(prefix="fmka-harness-test-")
            self.addCleanup(self.scratch.cleanup)
            self.root = Path(self.scratch.name)
            self.source = self.root / "subject.py"
            self.original = b"VALUE = 1\n"
            self.source.write_bytes(self.original)
            self.tests = self.root / "test_subject.py"
            self.tests.write_text("from subject import VALUE\ndef test_value():\n    assert VALUE == 1\n")

        def mutate(self, new, **kwargs):
            try:
                return run_mutant(self.root, self.tests, ("subject.py", "VALUE = 1", new),
                                  self.root / "run", **kwargs)
            finally:
                self.assertEqual(self.source.read_bytes(), self.original)

        def test_assertion_failure_ignores_stale_bytecode(self):
            py_compile.compile(str(self.source), doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            result = self.mutate("VALUE = 2")
            self.assertTrue(result["killed"])
            self.assertIn("test_value", result["failures"][0])

        def test_multiline_anchor_preserves_windows_line_endings(self):
            raw = b"VALUE = 1\r\nUNUSED = 0\r\n"
            self.source.write_bytes(raw)
            path, original, changed = replacement(self.root,
                ("subject.py", "VALUE = 1\nUNUSED = 0", "VALUE = 2\nUNUSED = 0"))
            self.assertEqual(original, raw)
            self.assertEqual(changed, b"VALUE = 2\r\nUNUSED = 0\r\n")
            self.assertEqual(path.read_bytes(), raw)

        def test_survivor_is_not_a_kill(self):
            self.assertFalse(self.mutate("VALUE = 1 # unchanged behavior")["killed"])

        def test_compile_failure_restores_source(self):
            with self.assertRaisesRegex(HarnessError, "does not compile"):
                self.mutate("VALUE = (")

        def test_collection_failure_is_not_a_kill(self):
            with self.assertRaisesRegex(HarnessError, "not a kill"):
                self.mutate("raise ImportError('synthetic collection failure')")

        def test_setup_failure_is_not_a_kill(self):
            self.tests.write_text("import pytest\n@pytest.fixture(autouse=True)\n"
                                  "def broken():\n    raise RuntimeError('synthetic fixture failure')\n"
                                  "def test_value():\n    assert False\n")
            with self.assertRaisesRegex(HarnessError, "not a kill"):
                self.mutate("VALUE = 2")

        def test_timeout_is_not_a_kill_and_restores_source(self):
            with self.assertRaisesRegex(HarnessError, "timed out"):
                self.mutate("import time; time.sleep(10); VALUE = 1", timeout=1)

        def test_launch_failure_is_not_a_kill_and_restores_source(self):
            with patch.object(subprocess, "run", side_effect=OSError("synthetic launch failure")):
                with self.assertRaisesRegex(HarnessError, "could not start; not a kill"):
                    self.mutate("VALUE = 2")

        def test_crash_without_report_is_not_a_kill(self):
            with self.assertRaisesRegex(HarnessError, "without a valid report; not a kill"):
                self.mutate("import os; os._exit(3)")

        def test_all_skipped_is_not_a_kill(self):
            self.tests.write_text("import pytest\n@pytest.mark.skip(reason='synthetic skip')\n"
                                  "def test_value():\n    assert False\n")
            with self.assertRaisesRegex(HarnessError, "not a kill"):
                self.mutate("VALUE = 2")

        def test_exit_and_failure_report_must_agree(self):
            def fake_run(*args, **kwargs):
                (self.root / "run" / "pytest.xml").write_text(
                    '<testsuites><testsuite><testcase name="fake">'
                    '<failure message="synthetic"/></testcase></testsuite></testsuites>')
                return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

            with patch.object(subprocess, "run", side_effect=fake_run):
                with self.assertRaisesRegex(HarnessError, "exit/report disagree; not a kill"):
                    self.mutate("VALUE = 2")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HarnessTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if sys.argv[1:]:
        sys.exit("usage: python tests/mutate.py [--self-test]")
    try:
        sys.exit(main())
    except HarnessError as exc:
        sys.exit(f"MUTATION ERROR: {exc}")
