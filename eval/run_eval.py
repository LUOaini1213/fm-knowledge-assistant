"""Evaluate retrieval and grounded answering on eval/questions.jsonl.

Questions q01-q20 and u01-u04 are the development split (the abstention threshold is chosen there);
q21-q40 and u05-u08 are the test split. Gold references are (document, clause) or (document, "p<page>").
Results go to eval/results/*.json and a summary table on stdout.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmka.answer import ask, llm_client, llm_model  # noqa: E402
from fmka.index import Index  # noqa: E402
from fmka.search import MODES, search  # noqa: E402

OUT = ROOT / "eval" / "results"
DEV = {f"q{i:02d}" for i in range(1, 21)} | {f"u0{i}" for i in range(1, 5)}


def load():
    return [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def is_gold(row, gold):
    for doc, ref in gold:
        if row["doc_id"] != doc:
            continue
        if ref.startswith("p") and ref[1:].isdigit():
            if int(row["page"]) == int(ref[1:]):
                return True
        elif row["clause"] == ref:
            return True
    return False


def retrieval(ix, qs, k=5):
    res = {}
    for mode in MODES:
        ranks = []
        for q in qs:
            hits = search(ix, q["q"], k=20, mode=mode)
            r = next((n for n, (i, _, _) in enumerate(hits, start=1) if is_gold(ix.row(i), q["gold"])), None)
            ranks.append(r)
        res[mode] = {"hit@1": np.mean([r == 1 for r in ranks]), f"hit@{k}": np.mean([r is not None and r <= k for r in ranks]),
                     "mrr@20": np.mean([1 / r if r else 0 for r in ranks]), "ranks": ranks}
    return res


def choose_threshold(sims_ans, sims_un):
    """Similarity cut-off that maximises dev accuracy (answerable kept, unanswerable abstained); ties -> lower cut."""
    grid = np.round(np.arange(0.60, 0.90, 0.005), 3)
    acc = [(np.sum(np.array(sims_ans) >= t) + np.sum(np.array(sims_un) < t)) / (len(sims_ans) + len(sims_un)) for t in grid]
    return float(grid[int(np.argmax(acc))]), float(max(acc))


def correct(answer_text, expect):
    t = answer_text.lower()
    return all(e.lower() in t for e in expect)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ix = Index()
    qs = load()
    ans_q = [q for q in qs if not q.get("unanswerable")]

    r = retrieval(ix, ans_q)
    for mode, m in r.items():
        print(f"retrieval {mode:6s} hit@1 {m['hit@1']:.3f}  hit@5 {m['hit@5']:.3f}  mrr@20 {m['mrr@20']:.3f}")

    top_sim = {}
    for q in qs:
        hits = search(ix, q["q"], k=6)
        top_sim[q["id"]] = max(h[2] for h in hits)
    dev_a = [top_sim[q["id"]] for q in qs if q["id"] in DEV and not q.get("unanswerable")]
    dev_u = [top_sim[q["id"]] for q in qs if q["id"] in DEV and q.get("unanswerable")]
    thr, dev_acc = choose_threshold(dev_a, dev_u)
    print(f"abstention threshold (dev) {thr:.3f}, dev accuracy {dev_acc:.3f}; "
          f"answerable sims {min(dev_a):.3f}-{max(dev_a):.3f}, unanswerable {min(dev_u):.3f}-{max(dev_u):.3f}")

    client = llm_client()
    rows = []
    for q in qs:
        t0 = time.time()
        a = ask(ix, q["q"], min_sim=thr, client=client)
        rows.append({"id": q["id"], "split": "dev" if q["id"] in DEV else "test", "unanswerable": bool(q.get("unanswerable")),
                     "status": a.status, "answer": a.text if a.status in ("answered", "unverified") else "",
                     "top_sim": a.top_sim, "problems": a.problems,
                     "cited_gold": any(is_gold(ix.row(c), q["gold"]) for c in a.citations),
                     "correct": correct(a.text, q["expect"]) if not q.get("unanswerable") else None,
                     "seconds": round(time.time() - t0, 2)})
    summary = {}
    for split in ("dev", "test", "all"):
        s = [x for x in rows if split == "all" or x["split"] == split]
        a = [x for x in s if not x["unanswerable"]]
        u = [x for x in s if x["unanswerable"]]
        shown_a = [x for x in a if x["status"] == "answered"]
        summary[split] = {
            "answerable": len(a), "answered": len(shown_a),
            "answered_correct_and_cites_gold": sum(x["correct"] and x["cited_gold"] for x in shown_a),
            "answered_but_wrong": sum(not (x["correct"] and x["cited_gold"]) for x in shown_a),
            "withheld_unverified": sum(x["status"] == "unverified" for x in a),
            "abstained_on_answerable": sum(x["status"] in ("abstained_low_similarity", "not_found") for x in a),
            "unanswerable": len(u),
            "unanswerable_abstained": sum(x["status"] in ("abstained_low_similarity", "not_found") for x in u),
            "unanswerable_answered": sum(x["status"] == "answered" for x in u),
        }
    json.dump({"threshold": thr, "retrieval": {m: {k: v for k, v in d.items() if k != "ranks"} for m, d in r.items()},
               "summary": summary, "rows": rows, "llm": llm_model() if client else None},
              open(OUT / "eval_results.json", "w", encoding="utf-8"), indent=1, default=float)
    for split, s in summary.items():
        print(split, s)


if __name__ == "__main__":
    main()
