"""The README number check agrees with the committed results and catches a changed cell or phrase."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_readme", ROOT / "eval" / "check_readme.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_matches_results():
    assert cr.check(README) == []


@pytest.mark.parametrize("old,new", [
    ("| **Hybrid (RRF)** | **0.900** |", "| **Hybrid (RRF)** | **0.925** |"),                       # retrieval table
    ("| 1 | 0 / 0 | 0 | 18 / 19 | 3 | 17 / 16 | 8 |", "| 1 | 0 / 0 | 0 | 18 / 19 | 3 | 18 / 16 | 8 |"),  # answer runs
    ("| `e091712` (measured 2026-10-14) | 1 |", "| `f00ba12` (measured 2026-10-14) | 1 |"),       # run label
    ("| v3 |", "| v5 |"),                                                                         # archived label
    ("| 18 | 2 | 8 |", "| 19 | 2 | 8 |"),                                                         # archived answers
    ("| Current quantity/citation guard | 34 | 24 | 0 |", "| Current quantity/citation guard | 33 | 24 | 0 |"),
    ("| **0.943** | **0.717** | **0.815** |", "| **0.943** | **0.717** | **0.851** |"),            # extraction
    ("| 0.836 / 0.809 |", "| 0.836 / 0.890 |"),                                                   # corrected classes
    ("acc 0.996 / F1 0.975", "acc 0.996 / F1 0.957"),                                             # archived classes
    ("33 and 34 of the 40 drafts (two runs)", "34 and 34 of the 40 drafts (two runs)"),           # status line
    ("0 of 40 answerable questions got a shown answer", "2 of 40 answerable questions got a shown answer"),
    ("F1 0.815 on a 20-passage", "F1 0.85 on a 20-passage"),
    ("(0.689–0.893 against 0.683–0.741)", "(0.689–0.893 against 0.693–0.741)"),                  # Limits
    ("giving 2,180 chunks", "giving 2,200 chunks"),
])
def test_changed_number_is_reported(old, new):
    assert README.count(old) >= 1, old
    assert cr.check(README.replace(old, new, 1)), f"not caught: {old!r} -> {new!r}"


def test_extra_table_row_is_reported():
    row = "| v4 | tokenizer"
    i = README.index(row)
    line = README[i:README.index("\n", i) + 1]
    assert cr.check(README.replace(line, line + line.replace("v4", "v5"), 1))
