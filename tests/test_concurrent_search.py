"""Force overlapping HTTP reads on real DuckDB without a model or VSS download."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import duckdb
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka import api, index  # noqa: E402
from fmka.retrieval import VectorStore  # noqa: E402


class InterleavedConnection:
    """Pause both SELECTs after execute, before fetch, to expose a shared result.

    SQL still runs on real DuckDB. Non-target statements are serialized so each
    case isolates one production query site; no sleep or scheduling luck is used.
    """

    def __init__(self, con, phase, owner=None):
        self.con, self.phase = con, phase
        self.owner = owner or self
        self.closed = self.locked = False
        if owner is None:
            self.barrier, self.lock, self.children = Barrier(2), Lock(), []

    def cursor(self):
        child = InterleavedConnection(self.con.cursor(), self.phase, self.owner)
        self.owner.children.append(child)
        return child

    def execute(self, sql, params=None):
        targeted = ("array_cosine_distance" in sql) == (self.phase == "dense")
        if not targeted:
            self.owner.lock.acquire()
            self.locked = True
        try:
            self.con.execute(sql, params)
            if targeted:
                self.owner.barrier.wait(timeout=10)
            return self
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        if self.locked:
            self.locked = False
            self.owner.lock.release()

    def fetchall(self):
        try:
            return self.con.fetchall()
        finally:
            self._unlock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._unlock()
        self.con.close()
        self.closed = True


@pytest.fixture
def shared_index(tmp_path, monkeypatch):
    db = tmp_path / "concurrent.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("create table chunks (chunk_id INTEGER, doc_id VARCHAR, heading VARCHAR, text VARCHAR, "
                    "page INTEGER, clause VARCHAR)")
        con.execute("insert into chunks values "
                    "(0, 'doors', '', 'Door width is 850 mm.', 1, '1.1'), "
                    "(1, 'pumps', '', 'Pump cleaning occurs every 12 months.', 2, '2.1')")
        con.execute("create table sources as select 'doors' as id, 'Door code' as title "
                    "union all select 'pumps', 'Pump code'")
        con.execute("create table vectors (chunk_id INTEGER, v FLOAT[2])")
        con.execute("insert into vectors values (0, [1, 0]), (1, [0, 1])")

    def store_without_extension(con, dim):
        store = VectorStore.__new__(VectorStore)
        store.con, store.dim = con, 2
        return store

    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array([[1., 0.] if "access" in text else [0., 1.] for text in texts], dtype=np.float32)

    monkeypatch.setattr(index, "VectorStore", store_without_extension)
    ix = index.Index(db, model=Embedder())
    owner = ix.con
    monkeypatch.setattr(api, "index", lambda: ix)
    monkeypatch.setattr(api, "llm_client", lambda: None)
    try:
        yield ix, owner
    finally:
        owner.close()


@pytest.mark.parametrize("phase", ["dense", "similarity"])
@pytest.mark.parametrize("endpoint,mode", [("search", "dense"), ("search", "hybrid"), ("ask", "hybrid")])
def test_parallel_requests_keep_document_filters_scores_and_sources(shared_index, phase, endpoint, mode):
    ix, owner = shared_index
    guarded = InterleavedConnection(owner, phase)
    ix.con = ix.store.con = guarded

    def request(document, query):
        body = {"docs": [document]}
        body.update({"query": query, "mode": mode, "k": 1} if endpoint == "search" else {"question": query})
        with TestClient(api.app) as client:
            response = client.post("/" + endpoint, json=body)
        assert response.status_code == 200, response.text
        return response.json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        # Paraphrases have no lexical matches: BM25 must not accidentally mask
        # lost dense candidates in the hybrid case.
        door = pool.submit(request, "doors", "access route")
        pump = pool.submit(request, "pumps", "service interval")
        results = [door.result(timeout=20), pump.result(timeout=20)]

    for result, document in zip(results, ["doors", "pumps"]):
        if endpoint == "search":
            assert len(result) == 1
            assert result[0]["document"] == document
            assert result[0]["similarity"] == pytest.approx(1.0)
        else:
            assert result["status"] == "extractive"
            assert [r["document"] for r in result["extracts"]] == [document]
            assert [r["document"] for r in result["sources_considered"]] == [document]
    assert all(child.closed for child in guarded.children)
    assert owner.execute("select count(*) from chunks").fetchone() == (2,)


def test_failed_read_closes_its_cursor_and_preserves_owner():
    from fmka.retrieval import query_rows

    with duckdb.connect(":memory:") as con:
        class Owner:
            children = []

            def cursor(self):
                child = con.cursor()
                self.children.append(child)
                return child

        tracked = Owner()
        # No file reopening: duplicate cursors must also see the same memory DB.
        con.execute("create table saved as select 42 as value")
        with pytest.raises(duckdb.CatalogException):
            query_rows(tracked, "select * from missing_table", [])
        assert query_rows(tracked, "select value from saved", []) == [(42,)]
        for child in tracked.children:
            with pytest.raises(duckdb.ConnectionException):
                child.execute("select 1")
        assert con.execute("select value from saved").fetchone() == (42,)
