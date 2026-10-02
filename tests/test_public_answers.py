"""Public transports must withhold drafts while retaining internal diagnostics."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka import api, answer, cli, index
from fmka.answer import Answer


class TinyIndex:
    titles = {"mini": "Mini Code"}

    def row(self, chunk_id):
        assert chunk_id == 0
        return {"doc_id": "mini", "clause": "2.2", "page": 1,
                "heading": "Exit doors", "text": "The clear width is 850 mm."}


@pytest.fixture
def transport(monkeypatch):
    ix = TinyIndex()
    monkeypatch.setattr(api, "index", lambda: ix)
    monkeypatch.setattr(index, "Index", lambda: ix)
    monkeypatch.setattr(api, "llm_client", lambda: None)
    monkeypatch.setattr(answer, "llm_client", lambda: None)

    def use(a):
        monkeypatch.setattr(api, "ask", lambda *args, **kwargs: a)
        monkeypatch.setattr(answer, "ask", lambda *args, **kwargs: a)
        return TestClient(api.app)

    return use


def test_api_withholds_unverified_draft_and_preserves_diagnostics(transport):
    a = Answer("Door width?", "unverified", "Secret rejected draft: 900 mm [1].",
               citations=[0], passages=[0], problems=["unsupported numeric value"])
    response = transport(a).post("/ask", json={"question": a.question})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unverified" and body["answer"] is None
    assert "Secret rejected draft" not in response.text
    assert body["message"] and body["problems"] == a.problems
    assert body["sources_considered"][0]["clause"] == "2.2"
    assert a.text == "Secret rejected draft: 900 mm [1]."  # evaluator still sees draft


def test_cli_withholds_unverified_draft(transport, capsys):
    a = Answer("Door width?", "unverified", "Secret rejected draft: 900 mm [1].",
               citations=[0], passages=[0], problems=["unsupported numeric value"])
    transport(a)
    cli.main(["ask", a.question])
    out = capsys.readouterr().out
    assert "Secret rejected draft" not in out
    assert "unverified" in out and "unsupported numeric value" in out


def test_api_extractive_mode_provides_readable_passages(transport):
    a = Answer("Door width?", "extractive", "[1] The clear width is 850 mm.",
               citations=[0], passages=[0])
    body = transport(a).post("/ask", json={"question": a.question}).json()
    assert body["answer"] is None
    assert body["extracts"] == [{"chunk_id": 0, "document": "mini", "title": "Mini Code",
        "clause": "2.2", "page": 1, "heading": "Exit doors", "text": "The clear width is 850 mm."}]


@pytest.mark.parametrize("status", ["not_found", "abstained_low_similarity"])
def test_abstention_is_a_message_not_an_answer(transport, status):
    a = Answer("Door width?", status, "The indexed codes do not appear to cover this question.")
    body = transport(a).post("/ask", json={"question": a.question}).json()
    assert body["answer"] is None and body["message"] == a.text


def test_answered_and_cli_extractive_content_remain_visible(transport, capsys):
    a = Answer("Door width?", "answered", "At least 850 mm [1].", citations=[0], passages=[0])
    body = transport(a).post("/ask", json={"question": a.question}).json()
    assert body["answer"] == a.text
    cli.main(["ask", a.question])
    assert a.text in capsys.readouterr().out
    a.status = "extractive"
    a.text = "[1] The clear width is 850 mm."
    cli.main(["ask", a.question])
    assert a.text in capsys.readouterr().out
