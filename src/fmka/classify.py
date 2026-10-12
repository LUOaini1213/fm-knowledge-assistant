"""Document intelligence: route a passage to its topic.

Task: given the text of a clause, predict which chapter of the Fire Code (11 chapters) or the Accessibility Code
(9 chapters) it belongs to, plus which of the 8 source documents it comes from. The labels come from the documents'
own structure, so no hand labelling is involved. Clause numbers and cross-references ("Cl.2.2.7", "4.2.1") are
removed from the text first, because they would give the chapter away.

Two models are compared with stratified 5-fold cross-validation: TF-IDF + logistic regression, and logistic
regression on frozen bge-small sentence embeddings of the same cleaned content. Retrieval vectors are not used:
their document titles and clause numbers expose the labels being predicted.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

CLAUSE_REF = re.compile(r"\b(?:cl(?:ause)?s?\.?\s*)?\d+(?:\.\d+)+[a-z]?(?:\.\(\d+\))?", re.I)


def clean(text):
    return " ".join(CLAUSE_REF.sub(" ", text).split())


def tfidf_model():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=10, class_weight="balanced"))


def cross_validate(texts, vectors, labels, folds=5, seed=0):
    """Returns {model: {accuracy, macro_f1}} and the per-row out-of-fold predictions."""
    labels = np.asarray(labels)
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pred = {"tfidf_logreg": np.empty(len(labels), dtype=object), "embedding_logreg": np.empty(len(labels), dtype=object),
            "majority": np.empty(len(labels), dtype=object)}
    for tr, te in skf.split(texts, labels):
        m = tfidf_model().fit([texts[i] for i in tr], labels[tr])
        pred["tfidf_logreg"][te] = m.predict([texts[i] for i in te])
        e = LogisticRegression(max_iter=3000, C=10, class_weight="balanced").fit(vectors[tr], labels[tr])
        pred["embedding_logreg"][te] = e.predict(vectors[te])
        pred["majority"][te] = pd.Series(labels[tr]).mode()[0]
    scores = {k: {"accuracy": accuracy_score(labels, p), "macro_f1": f1_score(labels, p, average="macro")}
              for k, p in pred.items()}
    return scores, pred


def tasks(chunks):
    """(name, row mask, label column) for the three routing tasks."""
    return [("source_document", np.ones(len(chunks), bool), chunks.doc_id.values),
            ("firecode_chapter", (chunks.doc_id == "firecode2023").values, chunks.chapter.values),
            ("accessibility_chapter", ((chunks.doc_id == "accessibility2025") & (chunks.chapter != "")).values,
             chunks.chapter.values)]


def run(index):
    ch = index.chunks
    texts = [clean(f"{h} {t}") for h, t in zip(ch.heading, ch.text)]
    # Re-encode content rather than reusing retrieval features containing source/chapter metadata.
    # The pretrained encoder is frozen; fitted TF-IDF and classifiers remain inside each training fold.
    V = np.asarray(index.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False))
    results = {}
    for name, mask, labels in tasks(ch):
        idx = np.flatnonzero(mask)
        lab = labels[idx]
        keep = pd.Series(lab).map(pd.Series(lab).value_counts()) >= 5  # classes with at least 5 examples
        idx, lab = idx[keep.values], lab[keep.values]
        scores, _ = cross_validate([texts[i] for i in idx], V[idx], lab)
        results[name] = {"n": int(len(idx)), "classes": int(len(set(lab))), **scores}
    return results


def main(argv=None):
    """Evaluate corrected features without overwriting the archived historical run."""
    import argparse
    import json
    from pathlib import Path
    from .index import DB, MODEL, ROOT, Index

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--output", type=Path, default=ROOT / "eval/results/classification_corrected.json")
    args = parser.parse_args(argv)
    ix = Index(db=args.db)
    try:
        report = {"features": {"text": "cleaned heading and body", "encoder": MODEL,
                               "retrieval_vectors_reused": False, "folds": 5, "seed": 0},
                  "results": run(ix)}
    finally:
        ix.con.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
