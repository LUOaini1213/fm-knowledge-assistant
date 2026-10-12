"""Failed rebuilds must not destroy the last usable local index."""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka import index


@pytest.fixture
def build_inputs(tmp_path, monkeypatch):
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps([{"id": "mini", "title": "Mini Code"}]), encoding="utf-8")
    monkeypatch.setattr(index, "parse_all", lambda *a: [{"doc_id": "mini", "page": 1,
        "clause": "1.1", "chapter": "1", "heading": "Width", "text": "A synthetic width clause."}])

    class Model:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), index.DIM), dtype=np.float32)

    return sources, tmp_path / "index.duckdb", Model()


@pytest.mark.parametrize("existing", [True, False])
def test_vector_failure_does_not_publish_partial_database(build_inputs, monkeypatch, existing):
    sources, db, model = build_inputs
    if existing:
        with duckdb.connect(str(db)) as con:
            con.execute("create table previous_version as select 42 as value")
    before = db.read_bytes() if existing else None

    class BrokenStore:
        def __init__(self, con, dim):
            self.con = con

        def build(self, ids, vectors):
            self.con.execute("create table partially_written as select 1")
            raise RuntimeError("synthetic vector build failure")

    monkeypatch.setattr(index, "VectorStore", BrokenStore)
    with pytest.raises(RuntimeError, match="synthetic vector"):
        index.build(sources=sources, raw=db.parent, db=db, model=model)
    assert (db.read_bytes() if db.exists() else None) == before
    assert not list(db.parent.glob(".index.duckdb.build-*"))
    if existing:
        with duckdb.connect(str(db), read_only=True) as con:
            assert con.execute("select value from previous_version").fetchone() == (42,)


def test_success_replaces_index_after_tables_are_complete(build_inputs, monkeypatch):
    sources, db, model = build_inputs
    db.write_bytes(b"old unusable database")

    class Store:
        def __init__(self, con, dim):
            self.con = con

        def build(self, ids, vectors):
            self.con.execute("create table vectors as select 0 as chunk_id")
            assert db.read_bytes() == b"old unusable database"

    monkeypatch.setattr(index, "VectorStore", Store)
    assert index.build(sources=sources, raw=db.parent, db=db, model=model) == 1
    with duckdb.connect(str(db), read_only=True) as con:
        assert con.execute("select doc_id from chunks").fetchone() == ("mini",)
        assert con.execute("select id from sources").fetchone() == ("mini",)
        assert con.execute("select count(*) from vectors").fetchone() == (1,)
    assert not list(db.parent.glob(".index.duckdb.build-*"))
