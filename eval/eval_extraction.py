"""Score the requirement extractor against hand-labelled passages (eval/extraction_gold.jsonl).

A predicted requirement matches a gold one when comparator, value (and upper value for ranges) and unit are equal.
Matching is one-to-one, so a requirement stated twice must be extracted twice. Subjects are not scored.
The dev split was labelled first and used while writing rules; the test split was labelled afterwards and only scored.

Gold rows are keyed by the integer chunk_id of a locally rebuilt index, so a different PDF edition (or parser change)
would silently shift them onto other passages. Before scoring, the script therefore aborts when
  * a PDF in data/raw does not have the SHA-256 of the tested edition in sources.json, or
  * a gold chunk_id is missing from the index, or
  * a gold row carries a text_sha256 that does not match the indexed chunk text.
`--pin-gold` writes text_sha256 into every gold row from the current (verified) index, so later runs check the passage
text itself, not only the edition. The result file records the commit, index, gold and source fingerprints and the date.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
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


GOLD = ROOT / "eval" / "extraction_gold.jsonl"
SOURCES = ROOT / "sources.json"
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "processed" / "fmka.duckdb"
OUT = ROOT / "eval" / "results" / "extraction.json"


NL = "\n"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def text_sha256(text):
    return sha256_bytes(text.replace("\r\n", NL).encode("utf-8"))


def edition_mismatches(sources, raw):
    """Return the sources whose local PDF is missing or differs from the tested edition."""
    bad = []
    for s in sources:
        path = Path(raw) / s["file"]
        if not path.exists():
            bad.append(f"{s['id']}: {s['file']} missing")
        elif s.get("sha256") and sha256_bytes(path.read_bytes()) != s["sha256"]:
            bad.append(f"{s['id']}: {s['file']} differs from the tested edition")
    return bad


def gold_problems(rows, texts):
    """Gold rows whose chunk is missing from the index or whose pinned text hash does not match."""
    bad = []
    for r in rows:
        cid = r["chunk_id"]
        if cid not in texts:
            bad.append(f"chunk {cid}: not in the index")
        elif "text_sha256" in r and text_sha256(texts[cid]) != r["text_sha256"]:
            bad.append(f"chunk {cid}: text differs from the labelled passage")
    return bad


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "eval"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-other-edition", action="store_true",
                        help="score even if a PDF differs from the tested edition (result is then not comparable)")
    parser.add_argument("--pin-gold", action="store_true",
                        help="write text_sha256 of each gold chunk into extraction_gold.jsonl, then score")
    args = parser.parse_args(argv)

    gold_text = GOLD.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in gold_text.splitlines() if line.strip()]
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    mismatched = edition_mismatches(sources, RAW)
    if mismatched and not args.allow_other_edition:
        raise SystemExit("Refusing to score: gold chunk ids were labelled on the tested editions."
                         + "".join(NL + "  " + m for m in mismatched))
    con = duckdb.connect(str(DB), read_only=True)
    texts = dict(con.execute("select chunk_id, text from chunks").fetchall())
    con.close()
    if args.pin_gold:
        if mismatched:
            raise SystemExit("--pin-gold needs the tested editions; refusing with --allow-other-edition")
        missing = [r["chunk_id"] for r in rows if r["chunk_id"] not in texts]
        if missing:
            raise SystemExit(f"--pin-gold: chunk ids not in the index: {missing}")
        for r in rows:
            r["text_sha256"] = text_sha256(texts[r["chunk_id"]])
        gold_text = "".join(json.dumps(r, ensure_ascii=False) + NL for r in rows)
        GOLD.write_text(gold_text, encoding="utf-8", newline=NL)
    problems = gold_problems(rows, texts)
    if problems:
        raise SystemExit("Refusing to score: the index does not match the labelled passages."
                         + "".join(NL + "  " + p for p in problems))
    out = {}
    for split in ("dev", "test"):
        part = [r for r in rows if r["split"] == split]
        if part:
            out[split], per = score(part, texts)
            out[split]["passages"] = len(part)
            out[split + "_errors"] = [p for p in per if p["missed"] or p["spurious"]]
    out["provenance"] = {
        "run_date": datetime.date.today().isoformat(),
        "commit": git_commit(),
        "index_sha256": sha256_bytes(DB.read_bytes()),
        "gold_sha256": text_sha256(gold_text),
        "gold_rows_with_text_sha256": sum("text_sha256" in r for r in rows),
        "sources_match_tested_editions": not mismatched,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for split in ("dev", "test"):
        if split in out:
            print(split, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[split].items()})
    return out


if __name__ == "__main__":
    main()
