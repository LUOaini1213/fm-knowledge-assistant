"""The extraction evaluation refuses to score gold chunk ids against a different edition or passage."""
import hashlib
import importlib.util
import json
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_extraction", ROOT / "eval" / "eval_extraction.py")
ee = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ee)


def test_edition_mismatch_reports_missing_and_changed_pdfs(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF tested")
    (tmp_path / "b.pdf").write_bytes(b"%PDF other edition")
    sources = [
        {"id": "a", "file": "a.pdf", "sha256": hashlib.sha256(b"%PDF tested").hexdigest()},
        {"id": "b", "file": "b.pdf", "sha256": hashlib.sha256(b"%PDF tested").hexdigest()},
        {"id": "c", "file": "c.pdf", "sha256": "0" * 64},
    ]
    bad = ee.edition_mismatches(sources, tmp_path)
    assert bad == ["b: b.pdf differs from the tested edition", "c: c.pdf missing"]
    assert ee.edition_mismatches(sources[:1], tmp_path) == []


def test_gold_rows_must_exist_and_match_pinned_text():
    texts = {1: "Exit doors not less than 850 mm.", 2: "Other text."}
    rows = [
        {"chunk_id": 1, "text_sha256": ee.text_sha256(texts[1])},
        {"chunk_id": 2, "text_sha256": ee.text_sha256("A different passage.")},
        {"chunk_id": 3},
        {"chunk_id": 2},  # unpinned rows only need to exist
    ]
    assert ee.gold_problems(rows, texts) == ["chunk 2: text differs from the labelled passage",
                                             "chunk 3: not in the index"]
    assert ee.text_sha256("a\r\nb") == ee.text_sha256("a\nb")


def test_main_aborts_before_scoring_on_other_edition_and_records_provenance(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "d.pdf").write_bytes(b"%PDF tested")
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([{"id": "d", "file": "d.pdf", "sha256": hashlib.sha256(b"%PDF tested").hexdigest()}]),
                       encoding="utf-8")
    db = tmp_path / "fmka.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create table chunks (chunk_id integer, text varchar)")
    con.execute("insert into chunks values (0, 'The clear width shall be not less than 850 mm.')")
    con.close()
    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"chunk_id": 0, "split": "test", "gold": [["min", 850, "mm"]]}) + "\n", encoding="utf-8")
    out = tmp_path / "extraction.json"
    for name, value in {"RAW": raw, "SOURCES": sources, "DB": db, "GOLD": gold, "OUT": out}.items():
        monkeypatch.setattr(ee, name, value)

    result = ee.main(["--pin-gold"])
    assert result["test"]["tp"] == 1 and result["test"]["fn"] == 0
    prov = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert prov["sources_match_tested_editions"] is True and prov["gold_rows_with_text_sha256"] == 1
    assert prov["index_sha256"] == hashlib.sha256(db.read_bytes()).hexdigest()
    assert "text_sha256" in json.loads(gold.read_text(encoding="utf-8"))

    # The passage behind the pinned chunk id changes: scoring stops instead of producing a wrong number.
    con = duckdb.connect(str(db))
    con.execute("update chunks set text = 'Some unrelated clause.' where chunk_id = 0")
    con.close()
    with pytest.raises(SystemExit, match="does not match the labelled passages"):
        ee.main([])

    # A different edition is refused unless explicitly allowed.
    (raw / "d.pdf").write_bytes(b"%PDF revised")
    with pytest.raises(SystemExit, match="labelled on the tested editions"):
        ee.main([])
