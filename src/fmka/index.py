"""Build and load the knowledge index: chunks table, dense vectors with an HNSW index, and an in-memory BM25.

Nothing built here is committed: the chunk text and vectors are derived from documents whose terms do not allow
redistribution, so each user rebuilds them locally from the official URLs in sources.json (python -m fmka.cli build).
"""
import json
import os
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .parse import parse_all
from .retrieval import BM25, VectorStore

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "processed" / "fmka.duckdb"
MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge retrieval instruction


def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)


def passage_text(row, titles):
    """What gets embedded: document title and clause heading give context the clause body often lacks."""
    head = f"{titles.get(row['doc_id'], row['doc_id'])}. Clause {row['clause']} {row['heading']}".strip()
    return f"{head}. {row['text']}"


def build(sources=ROOT / "sources.json", raw=ROOT / "data" / "raw", db=DB, model=None):
    db = Path(db)
    chunks = pd.DataFrame(parse_all(sources, raw))
    chunks.insert(0, "chunk_id", range(len(chunks)))
    titles = {s["id"]: s["title"] for s in json.loads(Path(sources).read_text(encoding="utf-8"))}
    model = model or embedder()
    vecs = model.encode([passage_text(r, titles) for r in chunks.to_dict("records")], batch_size=64,
                        normalize_embeddings=True, show_progress_bar=False)
    db.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination, then publish only the fully closed database.
    # An extension/vector/write failure must leave the previous index usable.
    with tempfile.TemporaryDirectory(prefix=f".{db.name}.build-", dir=db.parent) as tmp:
        staged = Path(tmp) / db.name
        con = duckdb.connect(str(staged))
        try:
            # executemany otherwise commits every row, making first-start builds
            # unnecessarily slow on container filesystems.
            con.execute("BEGIN TRANSACTION")
            con.execute("create table chunks as select * from chunks")
            con.execute("create table sources as select * from read_json_auto(?)", [str(sources)])
            VectorStore(con, DIM).build(chunks.chunk_id, vecs)
            con.execute("COMMIT")
            con.execute("CHECKPOINT")
        finally:
            con.close()
        os.replace(staged, db)
    return len(chunks)


class Index:
    """Read-side handle used by search, answer and the API."""

    def __init__(self, db=DB, model=None):
        self.con = duckdb.connect(str(db), read_only=True)
        self.store = VectorStore(self.con, DIM)
        self.chunks = self.con.execute("select * from chunks order by chunk_id").df()
        self.titles = dict(self.con.execute("select id, title from sources").fetchall())
        self.bm25 = BM25([f"{h} {t}" for h, t in zip(self.chunks.heading, self.chunks.text)])
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = embedder()
        return self._model

    def embed_query(self, q):
        return self.model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]

    def row(self, chunk_id):
        return self.chunks.iloc[int(chunk_id)].to_dict()
