"""Search modes over the index: bm25, dense, hybrid (reciprocal-rank fusion), with optional document filters."""
from .retrieval import rrf

MODES = ("bm25", "dense", "hybrid")


def search(index, query, k=5, mode="hybrid", docs=None, pool=50):
    """Return [(chunk_id, score, dense_sim)]. dense_sim (cosine to the query) is kept for the abstention rule."""
    allowed = None if not docs else set(index.chunks.index[index.chunks.doc_id.isin(docs)])
    keep = (lambda i: True) if allowed is None else (lambda i: i in allowed)
    qv = index.embed_query(query) if mode in ("dense", "hybrid") else None
    lex = [(i, s) for i, s in index.bm25.top(query, pool) if keep(i)] if mode in ("bm25", "hybrid") else []
    den = [(i, s) for i, s in index.store.top(qv, pool if allowed is None else 4 * pool) if keep(i)] \
        if mode in ("dense", "hybrid") else []
    ranked = {"bm25": lex, "dense": den, "hybrid": rrf([lex, den], n=pool)}[mode][:k]
    if qv is None:
        return [(i, s, None) for i, s in ranked]
    vecs = index.con.execute(
        "select chunk_id, v from vectors where chunk_id in (select unnest(?))", [[int(i) for i, _ in ranked]]).fetchall()
    import numpy as np
    sim = {int(i): float(np.dot(np.array(v), qv)) for i, v in vecs}
    return [(i, s, sim.get(int(i))) for i, s in ranked]
