import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmka.answer import Answer, ask, numbers, verify  # noqa: E402
from fmka.classify import clean  # noqa: E402
from fmka.extract import extract  # noqa: E402
from fmka.retrieval import BM25, rrf, tokenize  # noqa: E402


# ---------- extraction ----------
@pytest.mark.parametrize("text, expected", [
    ("Exit doors shall have a clear width of not less than 850 mm.", [("min", 850, "mm")]),
    ("The travel distance shall not exceed 60 m.", [("max", 60, "m")]),
    ("Seats must be between 450 mm and 480 mm high.", [("range", 450, "mm")]),
    ("Floors of 5,000 m2 or more in gross floor area.", [("min", 5000, "m2")]),
    ("The rim must be mounted no higher than 400 mm.", [("max", 400, "mm")]),
    ("A route with a minimum clear width of 1800 mm.", [("min", 1800, "mm")]),
    ("There shall be no change in level of more than 25 mm at the doorway.", [("max", 25, "mm")]),
    ("Storeys with a habitable height more than 24m.", [("min", 24, "m")]),
    ("Access within a travel distance of 18m of the stair.", [("max", 18, "m")]),
    ("Protection of at least 1-hr fire resistance.", [("min", 1, "hours")]),
    ("Openings of 50% or more.", [("min", 50, "%")]),
    ("A gradient of not more than 1:12.", [("max", 1 / 12, "gradient (1:N)")]),
    ("Lighting of 5 lux is recommended.", []),                      # no comparator, nothing extracted
])
def test_extract(text, expected):
    got = [(r.comparator, round(r.value, 6), r.unit) for r in extract(text)]
    assert got == [(c, round(v, 6), u) for c, v, u in expected]


def test_extract_range_keeps_upper_value():
    r = extract("The switch must be located at a height of between 450 mm and 1200 mm from the floor.")[0]
    assert (r.value, r.value_max) == (450, 1200)


def test_extract_subject():
    r = extract("Exit doors shall have a clear width of not less than 850 mm.")[0]
    assert "Exit doors" in r.subject and "clear width" in r.subject


# ---------- retrieval ----------
def test_bm25_ranks_the_matching_document_first():
    docs = ["exit door clear width 850 mm", "grease trap cleaning frequency", "exit staircase pressurisation"]
    top = BM25(docs).top("minimum width of exit door", 3)
    assert top[0][0] == 0


def test_bm25_idf_downweights_common_terms():
    docs = ["exit exit exit"] * 5 + ["exit sprinkler"]
    s = BM25(docs).scores("sprinkler")
    assert s[5] > 0 and np.all(s[:5] == 0)


def test_tokenize_keeps_clause_numbers_and_drops_stopwords():
    assert tokenize("The exit shall comply with Cl.2.3.9") == ["exit", "comply", "cl", "2.3.9"]


def test_rrf_rewards_agreement():
    fused = rrf([[(1, 9), (2, 8), (3, 7)], [(3, 9), (1, 8), (4, 7)]], n=4)
    assert [i for i, _ in fused][:2] == [1, 3]


# ---------- answer checks ----------
PASSAGES = ["(Fire Code, clause 2.2.7, page 35) Exit doors shall have a clear width of not less than 850mm.",
            "(Fire Code, clause 2.2.6, page 35) Travel distance shall not exceed 60 m where sprinklered."]


def test_verify_accepts_cited_numbers():
    cited, problems = verify("Exit doors need at least 850 mm clear width [1].", PASSAGES)
    assert cited == [1] and problems == []


def test_verify_rejects_numbers_not_in_cited_passage():
    _, problems = verify("Exit doors need 900 mm [1].", PASSAGES)
    assert any("900" in p for p in problems)


def test_verify_rejects_number_from_an_uncited_passage():
    _, problems = verify("The limit is 60 m [1].", PASSAGES)       # 60 is in passage 2, not the cited passage 1
    assert any("60" in p for p in problems)


def test_verify_rejects_missing_and_invalid_citations():
    assert "no citation" in verify("Exit doors need 850 mm.", PASSAGES)[1]
    assert any("not supplied" in p for p in verify("850 mm [7].", PASSAGES)[1])


def test_verify_allows_header_numbers():
    assert verify("See clause 2.2.7 on page 35 [1].", PASSAGES)[1] == []


def test_numbers_ignores_thousands_separator():
    assert numbers("5,000 m2 and 1.5 m") == {"5000", "1.5"}


# ---------- classification preprocessing ----------
def test_clean_removes_clause_references():
    assert clean("comply with Cl.2.3.9 and clause 4.2.1.2 here") == "comply with and here"


# ---------- end to end on a generated PDF with a fake embedder ----------
class FakeEmbedder:
    """Deterministic bag-of-words hashing into 384 dims, so tests need no model download."""

    def encode(self, texts, **kw):
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokenize(t):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


@pytest.fixture(scope="module")
def mini_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("mini")
    pdf = pymupdf.open()
    page = pdf.new_page()
    y = 72
    for line in ["2.1 General", "Exits shall be provided in every storey.",
                 "2.2 Exit doors", "Exit doors shall have a clear width of not less than 850 mm.",
                 "2.3 Travel distance", "The travel distance shall not exceed 60 m where sprinklers are provided.",
                 "3.1 Fire resistance", "Compartment walls shall have fire resistance of at least 2 hours."]:
        page.insert_text((72, y), line)
        y += 18
    pdf.save(d / "mini.pdf")
    src = d / "sources.json"
    src.write_text(json.dumps([{"id": "mini", "file": "mini.pdf", "numbered": True, "title": "Mini Code"}]))
    from fmka.index import Index, build
    db = d / "mini.duckdb"
    n = build(sources=src, raw=d, db=db, model=FakeEmbedder())
    return Index(db=db, model=FakeEmbedder()), n


def test_build_cuts_at_clause_numbers(mini_index):
    ix, n = mini_index
    assert n == 4 and list(ix.chunks.clause) == ["2.1", "2.2", "2.3", "3.1"]
    assert list(ix.chunks.chapter) == ["2", "2", "2", "3"]


def test_hybrid_search_finds_the_clause(mini_index):
    from fmka.search import search
    ix, _ = mini_index
    top = search(ix, "clear width of exit doors", k=2)
    assert ix.row(top[0][0])["clause"] == "2.2"


def test_search_document_filter(mini_index):
    from fmka.search import search
    ix, _ = mini_index
    assert search(ix, "exit doors", k=3, docs=["other"]) == []


class FakeLLM:
    """Stands in for the chat client: returns a canned reply."""

    def __init__(self, reply):
        self.reply = reply
        self.chat = self
        self.completions = self

    def create(self, **kw):
        msg = type("M", (), {"content": self.reply})
        return type("R", (), {"choices": [type("C", (), {"message": msg})]})


def test_ask_answers_with_verified_citation(mini_index):
    ix, _ = mini_index
    a = ask(ix, "clear width of exit doors", min_sim=0.0, client=FakeLLM("Exit doors need at least 850 mm clear width [1]."))
    assert a.status == "answered" and ix.row(a.citations[0])["clause"] == "2.2"


def test_ask_withholds_unsupported_number(mini_index):
    ix, _ = mini_index
    a = ask(ix, "clear width of exit doors", min_sim=0.0, client=FakeLLM("At least 900 mm [1]."))
    assert a.status == "unverified"


def test_ask_abstains_below_similarity(mini_index):
    ix, _ = mini_index
    a = ask(ix, "grease trap cleaning", min_sim=0.99, client=FakeLLM("anything [1]"))
    assert a.status == "abstained_low_similarity"


def test_ask_not_found(mini_index):
    ix, _ = mini_index
    assert ask(ix, "exit doors", min_sim=0.0, client=FakeLLM("NOT_FOUND")).status == "not_found"


def test_api_endpoints(mini_index, monkeypatch):
    from fastapi.testclient import TestClient
    from fmka import api
    ix, _ = mini_index
    api.index.cache_clear()
    monkeypatch.setattr(api, "index", lambda: ix)
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    hits = c.post("/search", json={"query": "exit door width", "k": 2}).json()
    assert hits[0]["clause"] == "2.2" and hits[0]["page"] == 1
    assert c.post("/search", json={"query": "exit door", "mode": "nope"}).status_code == 422
    ex = c.post("/extract", json={"text": "Doors of not less than 850 mm."}).json()
    assert ex[0]["value"] == 850 and ex[0]["comparator"] == "min"


def test_bm25_rare_term_outweighs_common_term():
    # "exit" is in every document, "sprinkler" in one: a match on the rare term must score higher.
    docs = ["exit route", "exit door", "exit sign", "exit lobby", "sprinkler head"]
    s = BM25(docs).scores("exit sprinkler")
    assert s[4] > s[0]


def test_parse_ignores_numbers_that_jump_chapters():
    from fmka.parse import numbered
    pages = [(1, ["2.1 General", "Exits shall be provided.", "12.5 Gypsum plasterboard lining", "2.2 Exit doors",
                  "Doors shall swing outwards.", "4.1 Wrong jump", "3.1 Fire resistance", "Walls shall resist fire."])]
    chunks = numbered("t", pages)
    assert [c.clause for c in chunks] == ["2.1", "2.2", "3.1"]
    assert "Gypsum" in chunks[0].text and "Wrong jump" in chunks[1].text


class ScriptedLLM(FakeLLM):
    """Replies from a list, one per call."""

    def __init__(self, replies):
        super().__init__(None)
        self.replies = list(replies)

    def create(self, **kw):
        self.reply = self.replies.pop(0)
        return super().create(**kw)


def test_strict_mode_withholds_when_the_judge_disagrees(mini_index):
    ix, _ = mini_index
    a = ask(ix, "clear width of exit doors", min_sim=0.0, client=ScriptedLLM(["Exit doors need at least 850 mm clear width [1].", "NO"]), strict=True)
    assert a.status == "unverified"
    b = ask(ix, "clear width of exit doors", min_sim=0.0, client=ScriptedLLM(["Exit doors need at least 850 mm clear width [1].", "YES"]), strict=True)
    assert b.status == "answered"


def test_retry_when_the_model_forgets_to_cite(mini_index):
    ix, _ = mini_index
    a = ask(ix, "clear width of exit doors", min_sim=0.0, client=ScriptedLLM(["Exit doors need at least 850 mm clear width.", "Exit doors need at least 850 mm clear width [1]."]))
    assert a.status == "answered"


@pytest.mark.parametrize("reply, expected", [
    ("At least 900 mm [1].", "unverified"),
    ("At least nine hundred mm [1].", "unverified"),
    ("Exit doors need at least 850 mm clear width [1].", "answered"),
])
def test_api_exposes_only_checked_generated_answers(mini_index, monkeypatch, reply, expected):
    from fastapi.testclient import TestClient
    from fmka import api
    ix, _ = mini_index
    monkeypatch.setattr(api, "index", lambda: ix)
    monkeypatch.setattr(api, "llm_client", lambda: FakeLLM(reply))
    monkeypatch.setattr(api, "ask", lambda *args, **kwargs: ask(*args, min_sim=0.0, **kwargs))
    body = TestClient(api.app).post("/ask", json={"question": "clear width of exit doors"}).json()
    assert body["status"] == expected
    assert body["answer"] == (reply if expected == "answered" else None)
    assert body["sources_considered"][0]["clause"] == "2.2"
