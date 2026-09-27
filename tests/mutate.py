"""Mutation check: each deliberate bug below must make at least one unit test fail."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "fmka"
MUTANTS = [
    ("answer.py", 'problems.append(f"numbers not found in the cited passages: {missing}")', "pass"),
    ("answer.py", 'source = " ".join(passages[c - 1] for c in good)', 'source = " ".join(passages)'),
    ("answer.py", "bad = [c for c in cited if c < 1 or c > len(passages)]", "bad = []"),
    ("answer.py", "if top_sim is None or top_sim < min_sim:", "if top_sim is None:"),
    ("answer.py", 'status = "answered" if not problems else "unverified"', 'status = "answered"'),
    ("answer.py", 'if not draft.upper().startswith("NOT_FOUND") and not CITE.search(draft):', "if False:"),
    ("answer.py", "if strict and not problems and not any(", "if False and not any("),
    ("extract.py", '("min", re.compile(MIN_WORDS + r"\\s*" + QTY, re.I)),', '("max", re.compile(MIN_WORDS + r"\\s*" + QTY, re.I)),'),
    ("extract.py", '("max", re.compile(r"\\bno\\b[^.;]{0,40}?\\bmore than\\s+" + QTY, re.I)),', ""),
    ("extract.py", "out.append(Requirement(\"range\", _num(lo), _unit(u2 or u1), subj, sentence, _num(hi)))",
     "out.append(Requirement(\"range\", _num(lo), _unit(u2 or u1), subj, sentence, None))"),
    ("extract.py", 'if MEASURE.search(subj) and len(subj.split()) <= 3 and len(parts) > 1:', "if False:"),
    ("retrieval.py", "self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}",
     "self.idf = {t: 1.0 for t, f in df.items()}"),
    ("retrieval.py", "agg[i] = agg.get(i, 0.0) + 1.0 / (k + rank + 1)", "agg[i] = 1.0 / (k + rank + 1)"),
    ("retrieval.py", 'TOKEN = re.compile(r"[a-z]+|\\d+(?:\\.\\d+)*")', 'TOKEN = re.compile(r"[a-z0-9]+(?:\\.[0-9]+)*")'),
    ("parse.py", "(last_major is not None and major == last_major + 1 and starts_chapter)", "(last_major is not None)"),
    ("classify.py", 'return " ".join(CLAUSE_REF.sub(" ", text).split())', 'return " ".join(text.split())'),
    ("search.py", "allowed = None if not docs else set(index.chunks.index[index.chunks.doc_id.isin(docs)])", "allowed = None"),
]

# The unmutated code must pass, otherwise every mutant would look "killed".
base = subprocess.run([sys.executable, "-m", "pytest", "-q", str(SRC.parents[1] / "tests" / "test_core.py")], capture_output=True)
assert base.returncode == 0, "tests fail on the unmutated code; fix them before running the mutation check"

escaped = 0
for fname, old, new in MUTANTS:
    f = SRC / fname
    text = f.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"mutation target not unique in {fname}: {old[:70]}"
    f.write_text(text.replace(old, new), encoding="utf-8")
    compiled = subprocess.run([sys.executable, "-m", "py_compile", str(f)], capture_output=True)
    assert compiled.returncode == 0, f"mutant does not compile, so it proves nothing: {old[:70]}"
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(SRC.parents[1] / "tests" / "test_core.py")],
                           capture_output=True, text=True)
    finally:
        f.write_text(text, encoding="utf-8")
    killed = r.returncode != 0
    escaped += not killed
    print(f"{'killed ' if killed else 'ESCAPED'}  {fname}: {old[:70]}", flush=True)
print(f"{len(MUTANTS) - escaped}/{len(MUTANTS)} mutants killed")
sys.exit(1 if escaped else 0)
