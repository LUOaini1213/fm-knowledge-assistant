"""Document restrictions must apply before either retrieval branch truncates its pool."""
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka.answer import ask  # noqa: E402
from fmka.retrieval import BM25, VectorStore  # noqa: E402
from fmka.search import search  # noqa: E402


@pytest.fixture
def filtered_index(tmp_path):
    """Real DuckDB vectors; no embedding model or downloadable HNSW extension is needed."""
    con = duckdb.connect(str(tmp_path / "filtered.duckdb"))
    # Keep real disk-backed SQL, but avoid one durability flush for every fixture row.
    con.execute("BEGIN TRANSACTION")
    con.execute("create table chunks (chunk_id INTEGER, doc_id VARCHAR, heading VARCHAR, text VARCHAR, "
                "page INTEGER, clause VARCHAR)")
    con.execute("create table vectors (chunk_id INTEGER, v FLOAT[2])")
    # Both selected passages fall below the old global 50 BM25 / 200 dense candidates.
    n_decoys = 205
    texts = ["exit exit width width"] * n_decoys + ["exit width " + "padding " * 25, "grease trap"]
    con.executemany("insert into chunks values (?, ?, ?, ?, ?, ?)", [
        (i, "other" if i < n_decoys else "selected", "", text, 1, str(i))
        for i, text in enumerate(texts)])
    con.executemany("insert into vectors values (?, ?)", [
        (i, [1.0, 0.0] if i < n_decoys else ([0.8, 0.6] if i == n_decoys else [0.6, 0.8]))
        for i in range(len(texts))])
    con.execute("COMMIT")
    # Exercise the production SQL retrieval method against DuckDB without installing vss.
    store = VectorStore.__new__(VectorStore)
    store.con, store.dim = con, 2
    ix = SimpleNamespace(con=con, store=store, bm25=BM25(texts),
                         chunks=con.execute("select * from chunks order by chunk_id").df(),
                         embed_query=lambda query: np.array([1.0, 0.0], dtype=np.float32),
                         titles={"selected": "Selected code"}, target=n_decoys, texts=texts)
    ix.row = lambda chunk_id: ix.chunks.iloc[int(chunk_id)].to_dict()
    try:
        yield ix
    finally:
        con.close()


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_matching_document_is_found_beyond_global_candidate_pool(filtered_index, mode):
    ix = filtered_index
    assert ix.target not in dict(ix.bm25.top("exit width", 50))
    assert ix.target not in dict(ix.store.top(ix.embed_query("exit width"), 200))
    hits = search(ix, "exit width", k=2, mode=mode, docs=["selected"])
    assert hits and hits[0][0] == ix.target
    assert all(ix.row(chunk_id)["doc_id"] == "selected" for chunk_id, _, _ in hits)


def test_filtered_bm25_keeps_full_corpus_idf_and_length_normalisation(filtered_index):
    ix = filtered_index
    expected = ix.bm25.scores("exit width")[ix.target]
    subset_score = BM25(ix.texts[-2:]).scores("exit width")[0]
    assert expected != pytest.approx(subset_score)
    hits = search(ix, "exit width", mode="bm25", docs=["selected"])
    assert len(hits) == 1
    assert hits[0] == (ix.target, pytest.approx(expected), None)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_empty_docs_preserves_the_existing_unfiltered_semantics(filtered_index, mode):
    ix = filtered_index
    unfiltered = search(ix, "exit width", k=3, mode=mode)
    assert unfiltered
    assert search(ix, "exit width", k=3, mode=mode, docs=[]) == unfiltered


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_no_matching_document_returns_without_embedding_or_search(filtered_index, monkeypatch, mode):
    def forbidden(*args, **kwargs):
        pytest.fail("a document filter with no matching chunks must not run retrieval")

    ix = filtered_index
    monkeypatch.setattr(ix, "embed_query", forbidden)
    monkeypatch.setattr(ix.bm25, "top", forbidden)
    monkeypatch.setattr(ix.store, "top", forbidden)
    assert search(ix, "exit width", mode=mode, docs=["missing-document"]) == []


def test_hybrid_keeps_cosine_separate_from_rrf_for_abstention(filtered_index):
    ix = filtered_index
    hits = search(ix, "exit width", k=2, mode="hybrid", docs=["selected"])
    assert hits and hits[0][0] == ix.target
    assert hits[0][1] == pytest.approx(2 / 61)
    assert hits[0][2] == pytest.approx(0.8)
    assert ask(ix, "exit width", docs=["selected"], min_sim=0.79).status == "extractive"
    answer = ask(ix, "exit width", docs=["selected"], min_sim=0.81)
    assert answer.status == "abstained_low_similarity"
    assert answer.top_sim == pytest.approx(0.8)


def test_bm25_allowed_ids_limit_candidates_before_top_k(filtered_index):
    ix = filtered_index
    hits = ix.bm25.top("exit width", k=1, allowed_ids={ix.target, ix.target + 1})
    assert hits == [(ix.target, pytest.approx(ix.bm25.scores("exit width")[ix.target]))]
    assert ix.bm25.top("exit width", allowed_ids=set()) == []


def test_dense_allowed_ids_are_applied_before_sql_limit(filtered_index):
    ix = filtered_index
    hits = ix.store.top(ix.embed_query("exit width"), k=1, allowed_ids={ix.target, ix.target + 1})
    assert hits == [(ix.target, pytest.approx(0.8))]
    assert ix.store.top(ix.embed_query("exit width"), allowed_ids=set()) == []
