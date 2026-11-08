"""Check every number in the README's result tables and status lines against the committed eval/results files.

No model, corpus or network is needed. Exit 1 lists each README cell or phrase that differs from the JSON.
Run as `python eval/check_readme.py` (CI runs it on every push).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
NUMBER = re.compile(r"(?<![\w@.])\d[\d,]*(?:\.\d+)?")  # skips the digit in labels such as F1 or Hit@1
WORDS = {2: "two", 3: "three", 4: "four"}


def load(name, results):
    return json.loads((results / name).read_text(encoding="utf-8"))


def tables(text):
    """All markdown tables as (header cells, body rows of cells)."""
    out, block = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            if len(block) >= 2 and set("".join(block[1])) <= set("-: "):
                out.append((block[0], block[2:]))
            block = []
    return out


def numbers(cell):
    cell = re.sub(r"`[^`]*`", " ", cell)
    return [n.replace(",", "") for n in NUMBER.findall(cell)]


def code(cell):
    m = re.search(r"`([^`]*)`", cell)
    return m.group(1) if m else None


def f3(x):
    return f"{x:.3f}"


def draft_correct(run, split):
    return sum(1 for r in run["rows"]
               if r["split"] == split and not r["unanswerable"] and r["correct"] and r["cited_gold"])


def no_local_citation(run):
    return sum(1 for r in run["rows"] if any("no valid local citation" in p for p in r["problems"]))


def expected(results):
    """(table specs, required phrases) computed from the result files."""
    qa = load("qa_runs_2026-10-14.json", results)["runs"]
    runs = [qa["pre_fix"], qa["fix_run1"], qa["fix_run2"]]
    current = runs[1:]
    archived = load("qa_runs.json", results)
    guard = load("claim_guard.json", results)
    ext, ext_v1 = load("extraction.json", results), load("extraction_v1.json", results)
    cls_old = load("classification.json", results)
    cls_new = load("classification_corrected.json", results)["results"]

    retrieval = runs[0]["retrieval"]
    assert all(r["retrieval"] == retrieval for r in runs), "retrieval differs between the 2026-10-14 runs"
    specs = []
    specs.append(("Mode", None, [[f3(retrieval[m][k]) for k in ("hit@1", "hit@5", "mrr@20")]
                                 for m in ("bm25", "dense", "hybrid")]))
    answer_rows = []
    for run in runs:
        s = run["summary"]
        answer_rows.append([
            str(s["dev"]["answered_correct_and_cites_gold"]), str(s["test"]["answered_correct_and_cites_gold"]),
            str(s["all"]["answered_but_wrong"]),
            str(s["dev"]["withheld_unverified"]), str(s["test"]["withheld_unverified"]),
            str(s["all"]["abstained_on_answerable"]),
            str(draft_correct(run, "dev")), str(draft_correct(run, "test")),
            str(s["all"]["unanswerable_abstained"])])
    specs.append(("Guard", [r["commit"] for r in runs], answer_rows, 2))
    specs.append(("Run", [k for k in ("v1", "v2", "v3", "v4")],
                  [[str(archived[k]["summary"]["test"]["answered_correct_and_cites_gold"]),
                    str(archived[k]["summary"]["test"]["answered_but_wrong"]),
                    str(archived[k]["summary"]["all"]["unanswerable_abstained"])] for k in ("v1", "v2", "v3", "v4")], 2))
    specs.append(("Checker", None, [[str(guard[k]["summary"][f]) for f in ("incorrect_rejected", "correct_accepted",
                                                                            "correct_refused")]
                                    for k in ("baseline", "current")]))
    specs.append(("Split", None, [[str(e["passages"]), f3(e["precision"]), f3(e["recall"]), f3(e["f1"])]
                                  for e in (ext_v1["dev"], ext["dev"], ext["test"])]))
    tasks = ("source_document", "firecode_chapter", "accessibility_chapter")
    specs.append(("Task|TF-IDF + LR acc / macro-F1", None,
                  [[str(cls_new[t]["classes"]),
                    f3(cls_new[t]["tfidf_logreg"]["accuracy"]), f3(cls_new[t]["tfidf_logreg"]["macro_f1"]),
                    f3(cls_new[t]["embedding_logreg"]["accuracy"]), f3(cls_new[t]["embedding_logreg"]["macro_f1"])]
                   for t in tasks], 0))
    specs.append(("Task|Passages", None,
                  [[str(cls_old[t]["n"]), str(cls_old[t]["classes"]), f3(cls_old[t]["majority"]["accuracy"]),
                    f3(cls_old[t]["tfidf_logreg"]["accuracy"]), f3(cls_old[t]["tfidf_logreg"]["macro_f1"]),
                    f3(cls_old[t]["embedding_logreg"]["accuracy"]), f3(cls_old[t]["embedding_logreg"]["macro_f1"])]
                   for t in tasks]))

    # Phrases in the status line, the prose and Limits.
    alls = [r["summary"]["all"] for r in current]
    shown = {a["answered"] for a in alls}
    assert len(shown) == 1, "the current-guard runs disagree on shown answers"
    n_ans = alls[0]["answerable"]
    n_un = alls[0]["unanswerable"]
    refused = {a["unanswerable_abstained"] for a in alls}
    assert len(refused) == 1
    drafts = [draft_correct(r, "dev") + draft_correct(r, "test") for r in current]
    dev = [r for r in runs[0]["rows"] if r["split"] == "dev"]
    a_sim = [r["top_sim"] for r in dev if not r["unanswerable"]]
    u_sim = [r["top_sim"] for r in dev if r["unanswerable"]]
    hit1 = retrieval["hybrid"]["hit@1"]
    gsum = guard["current"]["summary"]
    phrases = [
        f"{shown.pop()} of {n_ans} answerable questions got a shown answer",
        f"{drafts[0]} and {drafts[1]} of the {n_ans} drafts (two runs) stated the expected values",
        f"{drafts[0]} and {drafts[1]} of the {n_ans} drafts state the expected values",
        f"of the {drafts[0]} or {drafts[1]} correct drafts passes",
        f"{refused.pop()} of {n_un} unanswerable questions were refused",
        f"for {round(hit1 * n_ans)} of the {n_ans} answerable questions (Hit@1 {f3(hit1)}, "
        f"MRR@20 {f3(retrieval['hybrid']['mrr@20'])})",
        f"F1 {f3(ext['test']['f1'])} on a {ext['test']['passages']}-passage test set",
        f"{n_ans + n_un} questions written by me",
        f"### Grounded answering ({n_ans + n_un} questions)",
        f"The set has {n_ans} answerable questions and {n_un} that the corpus does not cover",
        f"### Retrieval ({n_ans} answerable questions)",
        f"#### Guard at `{current[0]['commit']}`, measured 2026-10-14",
        f"Before the fix, {no_local_citation(runs[0])} of {n_ans} drafts had a statement with no valid local citation",
        f"After the fix, {no_local_citation(current[0])} and {no_local_citation(current[1])} of {n_ans} drafts "
        "still have an uncited statement",
        f"({f3(min(a_sim))}–{f3(max(a_sim))} against {f3(min(u_sim))}–{f3(max(u_sim))})",
        f"contains {gsum['incorrect_examples']} incorrect bindings, {gsum['correct_examples']} correct examples and "
        f"{WORDS.get(gsum['unscored_boundaries'], gsum['unscored_boundaries'])} unscored policy boundaries",
        f"Incorrect examples rejected / {gsum['incorrect_examples']} | Correct examples accepted / "
        f"{gsum['correct_examples']} | Correct examples refused / {gsum['correct_examples']}",
        f"giving {cls_old['source_document']['n']:,} chunks",
        f"existing {cls_new['source_document']['n']:,}-chunk local index",
    ]
    return specs, phrases


def find_table(all_tables, key):
    want = key.split("|")
    hits = [t for t in all_tables if t[0][:len(want)] == want]
    return hits


def check(readme_text, results=RESULTS):
    """Return a list of differences between the README and the result files (empty when they agree)."""
    errors = []
    specs, phrases = expected(results)
    all_tables = tables(readme_text)
    for spec in specs:
        key, labels, rows = spec[:3]
        first = spec[3] if len(spec) > 3 else 1
        hits = find_table(all_tables, key)
        if len(hits) != 1:
            errors.append(f"table '{key}': found {len(hits)} tables, expected 1")
            continue
        _, body = hits[0]
        if len(body) != len(rows):
            errors.append(f"table '{key}': {len(body)} rows in README, {len(rows)} expected")
            continue
        for i, (cells, want) in enumerate(zip(body, rows)):
            got = [n for c in cells[first:] for n in numbers(c)]
            if first == 0:  # the task label carries the class count, e.g. "(8 classes)"
                got = numbers(cells[0]) + [n for c in cells[1:] for n in numbers(c)]
            if got != want:
                errors.append(f"table '{key}' row {i + 1} ({cells[0]}): README {got}, results {want}")
            if labels is not None:
                label = code(cells[0]) or cells[0]
                if not label.startswith(labels[i]) and not labels[i].startswith(label):
                    errors.append(f"table '{key}' row {i + 1}: label {cells[0]!r}, results {labels[i]!r}")
    flat = " ".join(readme_text.split())
    for p in phrases:
        if p not in flat:
            errors.append(f"phrase not found: {p!r}")
    return errors


def main():
    errors = check((ROOT / "README.md").read_text(encoding="utf-8"))
    if errors:
        print("README numbers differ from eval/results:")
        for e in errors:
            print("  " + e)
        return 1
    print("README numbers match eval/results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
