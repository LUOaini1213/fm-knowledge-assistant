"""Search modes over the index: bm25, dense, hybrid (reciprocal-rank fusion), with optional document filters."""
from .retrieval import query_rows, rrf

MODES = ("bm25", "dense", "hybrid")


def search(index, query, k=5, mode="hybrid", docs=None, pool=50):
    """Return [(chunk_id, score, dense_sim)], retaining cosine for the abstention rule.

    docs=None and docs=[] both mean no restriction. A nonempty filter with no
    matching chunks returns immediately, before embedding or selecting candidates.
    """
    allowed = None if not docs else set(index.chunks.index[index.chunks.doc_id.isin(docs)])
    if allowed is not None and not allowed:
        return []
    qv = index.embed_query(query) if mode in ("dense", "hybrid") else None
    lex = index.bm25.top(query, pool, allowed_ids=allowed) if mode in ("bm25", "hybrid") else []
    den = index.store.top(qv, pool, allowed_ids=allowed) if mode in ("dense", "hybrid") else []
    ranked = {"bm25": lex, "dense": den, "hybrid": rrf([lex, den], n=pool)}[mode][:k]
    if qv is None:
        return [(i, s, None) for i, s in ranked]
    vecs = query_rows(index.con,
        "select chunk_id, v from vectors where chunk_id in (select unnest(?))", [[int(i) for i, _ in ranked]])
    import numpy as np
    sim = {int(i): float(np.dot(np.array(v), qv)) for i, v in vecs}
    return [(i, s, sim.get(int(i))) for i, s in ranked]
