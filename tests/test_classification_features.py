"""Classification uses content features, with genuinely held-out predictions."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka import classify


class ContentEncoder:
    """Frozen, deterministic content features; no model download or fitting."""

    def __init__(self):
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.append(list(texts))
        return np.array([
            [1.0 if "fire" in text else -1.0, 1.0 if "exit" in text else -1.0]
            for text in texts
        ])


class NoRetrievalVectors:
    def execute(self, *args, **kwargs):
        raise AssertionError("classification must not reuse retrieval vectors containing label metadata")


def test_classification_encodes_clean_content_and_runs_all_tasks():
    records = []
    expected = []
    for doc_id, subject in (("firecode2023", "fire"), ("accessibility2025", "accessibility")):
        for chapter, topic in (("1", "exit"), ("2", "lighting")):
            for sample in range(5):
                heading = f"{chapter}.2.3 {subject} {topic}"
                body = f"{topic} requirements sample_{sample} comply with Cl.9.8.7"
                records.append({"chunk_id": 10 + len(records) * 3, "doc_id": doc_id,
                                "chapter": chapter, "heading": heading, "text": body})
                expected.append(f"{subject} {topic} {topic} requirements sample_{sample} comply with")
    encoder = ContentEncoder()
    index = SimpleNamespace(chunks=pd.DataFrame(records), model=encoder, con=NoRetrievalVectors(),
                            titles={"firecode2023": "SECRET_SOURCE_FIRE", "accessibility2025": "SECRET_SOURCE_ACCESS"})

    results = classify.run(index)

    assert encoder.inputs == [expected]
    assert set(results) == {"source_document", "firecode_chapter", "accessibility_chapter"}
    assert results["source_document"]["n"] == 20
    for name, result in results.items():
        assert result["classes"] == 2
        if name != "source_document":
            assert result["n"] == 10
        for model in ("tfidf_logreg", "embedding_logreg"):
            assert result[model]["accuracy"] == 1.0
            assert result[model]["macro_f1"] == 1.0


def test_cross_validation_fits_only_training_rows_and_is_reproducible(monkeypatch):
    labels = np.array(["exit"] * 10 + ["lighting"] * 10)
    texts = [f"{label} requirements unique_{i}" for i, label in enumerate(labels)]
    vectors = np.array([[1.0 if label == "exit" else -1.0, i / 1000.0]
                        for i, label in enumerate(labels)])
    real_tfidf_model = classify.tfidf_model
    real_logistic = classify.LogisticRegression
    text_predictions, vector_predictions = [], []

    class CheckedTextModel:
        def __init__(self):
            self.model = real_tfidf_model()

        def fit(self, rows, targets):
            self.training = set(rows)
            assert len(rows) == 16
            self.model.fit(rows, targets)
            return self

        def predict(self, rows):
            assert self.training.isdisjoint(rows)
            text_predictions.extend(rows)
            return self.model.predict(rows)

    class CheckedLogistic(real_logistic):
        def fit(self, rows, targets, *args, **kwargs):
            if isinstance(rows, np.ndarray):
                self.training_rows = set(map(tuple, rows))
                assert len(rows) == 16
            return super().fit(rows, targets, *args, **kwargs)

        def predict(self, rows):
            if isinstance(rows, np.ndarray):
                assert self.training_rows.isdisjoint(map(tuple, rows))
                vector_predictions.extend(map(tuple, rows))
            return super().predict(rows)

    monkeypatch.setattr(classify, "tfidf_model", CheckedTextModel)
    monkeypatch.setattr(classify, "LogisticRegression", CheckedLogistic)
    scores, predictions = classify.cross_validate(texts, vectors, labels, seed=17)
    again, repeated = classify.cross_validate(texts, vectors, labels, seed=17)

    assert scores == again
    assert sorted(text_predictions) == sorted(texts * 2)
    assert sorted(vector_predictions) == sorted(list(map(tuple, vectors)) * 2)
    for name in predictions:
        np.testing.assert_array_equal(predictions[name], repeated[name])
    for name in ("tfidf_logreg", "embedding_logreg"):
        assert scores[name]["accuracy"] == 1.0
        np.testing.assert_array_equal(predictions[name], labels)
