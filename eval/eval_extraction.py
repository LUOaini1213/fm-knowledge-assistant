"""Score the requirement extractor against hand-labelled passages (eval/extraction_gold.jsonl).

A predicted requirement matches a gold one when comparator, value (and upper value for ranges) and unit are equal.
Matching is one-to-one, so a requirement stated twice must be extracted twice. Subjects are not scored.
The dev split was labelled first and used while writing rules; the test split was labelled afterwards and only scored.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmka.extract import extract  # noqa: E402


def key(comp, value, unit, vmax=None):
    unit = {"m²": "m2", "°c": "°C"}.get(unit, unit)
    return (comp, round(float(value), 6), unit, None if vmax is None else round(float(vmax), 6))


def score(rows, texts):
    tp = fp = fn = 0
    per = []
    for r in rows:
        gold = Counter(key(*g[:3], g[3] if len(g) > 3 else None) for g in r["gold"])
        pred = Counter(key(p.comparator, p.value, p.unit, p.value_max) for p in extract(texts[r["chunk_id"]]))
        hit = sum((gold & pred).values())
        tp += hit
        fp += sum(pred.values()) - hit
        fn += sum(gold.values()) - hit
        per.append({"chunk_id": r["chunk_id"], "missed": list((gold - pred).elements()), "spurious": list((pred - gold).elements())})
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return {"precision": p, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}, per


def main():
    rows = [json.loads(line) for line in (ROOT / "eval" / "extraction_gold.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    con = duckdb.connect(str(ROOT / "data" / "processed" / "fmka.duckdb"), read_only=True)
    texts = dict(con.execute("select chunk_id, text from chunks").fetchall())
    out = {}
    for split in ("dev", "test"):
        part = [r for r in rows if r["split"] == split]
        if part:
            out[split], per = score(part, texts)
            out[split]["passages"] = len(part)
            out[split + "_errors"] = [p for p in per if p["missed"] or p["spurious"]]
    (ROOT / "eval" / "results").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "eval" / "results" / "extraction.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for split in ("dev", "test"):
        if split in out:
            print(split, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[split].items()})
    return out


if __name__ == "__main__":
    main()
