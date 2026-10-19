"""Retrieval: BM25 (keyword), dense vectors in DuckDB with an HNSW index (semantic), and reciprocal-rank fusion."""
import math
import re
from collections import Counter
from dataclasses import dataclass

import duckdb
import numpy as np

TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)*")  # words and clause numbers ("cl.2.3.9" -> "cl", "2.3.9")
STOP = set("a an and are as at be by for from has have in is it its of on or shall that the this to was were which with "
           "not no any all be been being can may must should".split())


def query_rows(con, sql, params):
    """Read through a private cursor, including result consumption and cleanup.

    The API shares one Index between worker threads. DuckDB execute() replaces
    its connection's pending result, so sharing that cursor can mix requests.
    A short-lived duplicate retains the same database handle (including an
    in-memory index), without reopening a possibly rebuilt database by path.
    """
    with con.cursor() as cursor:
        return cursor.execute(sql, params).fetchall()


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    """Okapi BM25 over a list of documents (k1=1.5, b=0.75)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokenize(query)
        s = np.zeros(len(self.tf))
        for i, c in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s[i] = sum(self.idf.get(t, 0.0) * c[t] * (self.k1 + 1) / (c[t] + norm) for t in q if t in c)
        return s

    def top(self, query, k=10, allowed_ids=None):
        """Rank allowed chunks using the original corpus's IDF and length statistics."""
        s = self.scores(query)
        candidates = np.arange(len(s))
        if allowed_ids is not None:
            allowed_ids = set(allowed_ids)
            candidates = np.array([i for i in candidates if i in allowed_ids], dtype=int)
        idx = candidates[np.argsort(-s[candidates])[:k]]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]


class VectorStore:
    """Chunk vectors in DuckDB with an HNSW cosine index (vss extension)."""

    def __init__(self, con: duckdb.DuckDBPyConnection, dim: int):
        self.con, self.dim = con, dim
        con.execute("INSTALL vss; LOAD vss; SET hnsw_enable_experimental_persistence = true;")

    def build(self, ids, vectors):
        self.con.execute(f"create or replace table vectors (chunk_id INTEGER, v FLOAT[{self.dim}])")
        self.con.executemany("insert into vectors values (?, ?)", [(int(i), list(map(float, v))) for i, v in zip(ids, vectors)])
        self.con.execute("create index vec_hnsw on vectors using hnsw (v) with (metric = 'cosine')")

    def top(self, qvec, k=10, allowed_ids=None):
        """Apply any document restriction before ordering and limiting vector candidates."""
        vector = list(map(float, qvec))
        params = [vector]
        where = ""
        if allowed_ids is not None:
            ids = sorted({int(i) for i in allowed_ids})
            if not ids:
                return []
            where = "where chunk_id in (select unnest(?)) "
            params.append(ids)
        params.append(vector)
        rows = query_rows(self.con,
            f"select chunk_id, 1 - array_cosine_distance(v, ?::FLOAT[{self.dim}]) as sim from vectors "
            + where +
            f"order by array_cosine_distance(v, ?::FLOAT[{self.dim}]) limit {int(k)}",
            params)
        return [(int(i), float(s)) for i, s in rows]


def rrf(rankings, k=60, n=10):
    """Reciprocal-rank fusion of several ranked lists of (id, score)."""
    agg = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking):
            agg[i] = agg.get(i, 0.0) + 1.0 / (k + rank + 1)
    return sorted(agg.items(), key=lambda x: -x[1])[:n]


@dataclass
class Hit:
    chunk_id: int
    score: float
    dense_sim: float | None = None
