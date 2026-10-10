"""Offline adversarial quantity/citation tests; these are scripted drafts, not model outputs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka.answer import ask, verify  # noqa: E402
from fmka.claims import claims  # noqa: E402


@pytest.mark.parametrize("draft,source", [
    ("Pump cleaning occurs every 12 days [1].", "Pump cleaning occurs every 12 months."),
    ("Door width is at most 850 mm [1].", "Door width is at least 850 mm."),
    ("Door width is greater than 850 mm [1].", "Door width is at least 850 mm."),
    ("Door width is less than 850 mm [1].", "Door width is at most 850 mm."),
    ("Door width is 850 mm [1].", "Door width is at least 850 mm."),
    ("Door width is >= 850 mm [1].", "Door width is > 850 mm."),
    ("Cooling tower cleaning occurs every 12 months [1].", "Kitchen exhaust duct cleaning occurs every 12 months."),
    ("Pump replacement occurs every 12 months [1].", "Pump cleaning occurs every 12 months."),
    ("Pump cleaning does not occur every 12 months [1].", "Pump cleaning occurs every 12 months."),
    ("Every 12 months [1].", "Kitchen exhaust duct cleaning occurs every 12 months."),
    ("It requires 850 mm clear width [1].", "Exit doors require 850 mm clear width."),
    ("Door width is 35 mm [1].", "(2023 Code, clause 2.2.7, page 35) Door width is 850 mm."),
    ("Door width is about 850 mm [1].", "Door width is 850 mm."),
    ("Door width is 850 feet [1].", "Door width is 850 mm."),
    ("Two dozen fans are required [1].", "Two fans are required."),
    ("冷却塔每十二个月清洁 [1]。", "Kitchen exhaust duct cleaning occurs every 12 months."),
    ("Door width is between 850 mm and 950 mm [1].", "Door width is between 850 mm and 900 mm; door height is 950 mm."),
    ("Door width is 850-900 mm [1].", "Door width is between 850 mm and 900 mm."),
    ("Door width is 900 mm, height is 850 mm [1].", "Door width is 850 mm, height is 900 mm."),
    ("Door width is at least 900 mm, handrail height is at least 850 mm [1].", "Door width is at least 850 mm, handrail height is at least 900 mm."),
    ("Technician inspection by pump occurs every 12 months [1].", "Pump inspection by technician occurs every 12 months."),
    ("Door width != 850 mm [1].", "Door width is 850 mm."),
    ("Door width shall be 850 mm [1].", "Door width may be 850 mm."),
    ("Every 12 months [1].", "Every 12 months."),
    ("Pump A cleaning occurs every 12 months [1].", "Pump cleaning occurs every 12 months."),
    ("Pump A cleaning occurs every 12 months [1].", "Pump B cleaning occurs every 12 months."),
    ("Pump α cleaning occurs every 12 months [1].", "Pump β cleaning occurs every 12 months."),
])
def test_quantity_binding_rejects_lexically_sourced_but_unsupported_claims(draft, source):
    cited, problems = verify(draft, [source])
    assert cited == [1]
    assert any("claim 1:" in p for p in problems)


@pytest.mark.parametrize("draft,source", [
    ("Pump cleaning occurs every twelve months [1].", "Pump cleaning occurs every 12 months."),
    ("Door width is at least 850 millimetres [1].", "Door width is not less than 850 mm."),
    ("Door width is no greater than 850 mm [1].", "Door width shall not exceed 850 mm."),
    ("Door width is > 850 mm [1].", "Door width is greater than 850 mm."),
    ("Door width is <= 850 mm [1].", "Door width is at most 850 mm."),
    ("Pump inspection interval is 12 months [1].", "Pump inspection interval is twelve months."),
    ("Door width is one point five metres [1].", "Door width is 1.5 m."),
    ("Lighting is 50 lx [1].", "Lighting is 50 lux."),
    ("Floor area is 5,000 m² [1].", "Floor area is 5000 square metres."),
    ("Door width is between 850 mm and 900 mm [1].", "Door width is between 850 mm and 900 mm."),
    ("Door width is at least 850 mm and at most 900 mm [1].", "Door width is not less than 850 mm and not more than 900 mm."),
    ("Exit doors shall have a clear width of not less than 850 mm [1].", "Exit doors shall have a clear width of not less than 850 mm."),
    ("Pump A cleaning occurs every 12 months [1].", "Pump A cleaning occurs every twelve months."),
    ("A pump inspection interval is 12 months [1].", "A pump inspection interval is twelve months."),
])
def test_bound_quantity_positive_pairs(draft, source):
    assert verify(draft, [source]) == ([1], [])


@pytest.mark.parametrize("draft", [
    "Door width is 850 mm [2]. Handrail height is 900 mm [1].",
    "Door width is 850 mm [2] and handrail height is 900 mm [1].",
    "Door width is 850 mm. Handrail height is 900 mm [1][2].",
    "Door width is 850 mm [1]. Handrail height is 900 mm.",
    "Door width is 850 mm and handrail height is 900 mm [1].",
])
def test_sentences_and_local_citations_cannot_borrow_evidence(draft):
    sources = ["Door width is 850 mm.", "Handrail height is 900 mm."]
    assert verify(draft, sources)[1]


def test_same_value_in_different_local_citations_cannot_authorize_wrong_subject():
    sources = ["Door width is 850 mm.", "Handrail height is 850 mm."]
    assert verify("Door width is 850 mm [2]. Handrail height is 850 mm [1].", sources)[1]


@pytest.mark.parametrize("draft", [
    "Door width is 850 mm [1]. Handrail height is 900 mm [2].",
    "Door width is 850 mm [1] and handrail height is 900 mm [2].",
    "Door width is 850 mm and handrail height is 900 mm [1][2].",
    "Door width is 850 mm. [1] Handrail height is 900 mm. [2]",
])
def test_distinct_claims_keep_their_local_positive_citations(draft):
    assert verify(draft, ["Door width is 850 mm.", "Handrail height is 900 mm."]) == ([1, 2], [])


def test_decimal_and_citation_sentence_boundaries():
    assert verify("Door width is 0.85 m. [1]", ["Door width is .85 m."]) == ([1], [])
    assert [(c.text, c.citations) for c in claims("Door width is 0.85 m [1]. Pump interval is 12 months [2].")] == [
        ("Door width is 0.85 m", (1,)), ("Pump interval is 12 months", (2,))]


def test_reference_numbers_also_use_only_their_local_citations():
    passages = ["(Code, clause 2.2.7, page 35) Door clear width is 850 mm.",
                "(Code, clause 9.9.9, page 12) Handrail height is 900 mm."]
    _, problems = verify("See clause 2.2.7 on page 35 [2]. Door clear width is 850 mm [1].", passages)
    assert any("claim 1: numbers absent from its local citations" in p for p in problems)
    assert verify("See page 35 [2]. See page 12 [1].", passages)[1]
    assert verify("See clause 2.2.7 on page 35 [1]. See page 12 [2].", passages) == ([1, 2], [])
    assert verify("Clause 2.2.7 requires 35 [1].", passages)[1]


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self.replies, "unexpected extra model call"
        message = type("M", (), {"content": self.replies.pop(0)})
        return type("R", (), {"choices": [type("C", (), {"message": message})]})


@pytest.fixture
def mock_index(monkeypatch):
    from fmka import answer
    sources = ["Kitchen exhaust duct cleaning occurs every 12 months.", "Door width is at least 850 mm."]
    class Index:
        titles = {"test": "Test code"}
        def row(self, n):
            return {"doc_id": "test", "clause": str(n + 1), "page": 1, "heading": "", "text": sources[n]}
    monkeypatch.setattr(answer, "search", lambda *args, **kwargs: [(0, 1.0, 1.0), (1, .9, .9)])
    return Index()


@pytest.mark.parametrize("strict", [False, True])
def test_ask_withholds_missing_subject_without_judge_override(mock_index, strict):
    client = ScriptedClient(["Every 12 months [1]."])
    result = ask(mock_index, "How often must cooling towers be cleaned?", client=client, strict=strict)
    assert result.status == "unverified" and "missing explicit subject" in result.problems[0]
    assert result.text == "Every 12 months [1]."  # internal diagnostic retains the raw draft
    assert len(client.calls) == 1


def test_strict_all_claims_judged_only_against_local_citations(mock_index):
    draft = "Kitchen exhaust duct cleaning occurs every 12 months [1]. Door width is at least 850 mm [2]."
    client = ScriptedClient([draft, "YES", "NO"])
    result = ask(mock_index, "What are the requirements?", client=client, strict=True)
    assert result.status == "unverified" and any("claim 2:" in p for p in result.problems)
    assert len(client.calls) == 3
    judge_messages = [c["messages"][1]["content"] for c in client.calls[1:]]
    assert "Door width" not in judge_messages[0]
    assert "Kitchen exhaust" not in judge_messages[1]
    assert "CLAIM: Door width is at least 850 mm" in judge_messages[1]


def test_strict_requires_an_exact_yes_and_covers_every_claim(mock_index):
    draft = "Kitchen exhaust duct cleaning occurs every 12 months [1]. Door width is at least 850 mm [2]."
    for replies, status in [([draft, "YES", "YES"], "answered"), ([draft, "YES, but unsupported", "YES"], "unverified")]:
        client = ScriptedClient(replies)
        assert ask(mock_index, "What are the requirements?", client=client, strict=True).status == status


@pytest.mark.parametrize("reply,status", [
    ("Cooling tower cleaning occurs every 12 months [1].", "unverified"),
    ("Kitchen exhaust duct cleaning occurs every 12 days [1].", "unverified"),
    ("Door width is at most 850 mm [2].", "unverified"),
    ("Every 12 months [1].", "unverified"),
    ("Kitchen exhaust duct cleaning occurs every twelve months [1].", "answered"),
])
def test_api_never_exposes_bad_bound_drafts(mock_index, monkeypatch, reply, status):
    from fastapi.testclient import TestClient
    from fmka import api
    monkeypatch.setattr(api, "index", lambda: mock_index)
    monkeypatch.setattr(api, "llm_client", lambda: ScriptedClient([reply]))
    body = TestClient(api.app).post("/ask", json={"question": "How often must cooling towers be cleaned?"}).json()
    assert body["status"] == status
    assert body["answer"] == (reply if status == "answered" else None)
    assert body["sources_considered"]
    assert body["problems"] if status == "unverified" else not body["problems"]


def test_extractive_evidence_still_available_without_generation(mock_index):
    result = ask(mock_index, "How often must cooling towers be cleaned?", client=None, strict=True)
    assert result.status == "extractive" and "Kitchen exhaust duct" in result.text
