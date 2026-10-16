"""HTTP API (FastAPI): /health, /search, /ask, /extract.

Run: uvicorn fmka.api:app --port 8000   (after building the index with python -m fmka.cli build)
"""
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .answer import ask, llm_client
from .extract import extract, to_dicts
from .search import MODES, search

app = FastAPI(title="FM Knowledge Assistant", version=__version__,
              description="Cited search and question answering over Singapore building and facilities-management codes.")


# Every response except the docs pages is JSON and needs no content at all. Swagger UI (/docs) and ReDoc load their
# script and stylesheet from cdn.jsdelivr.net and run an inline bootstrap script, so they get a policy that allows that.
_DOC_PAGES = ("/docs", "/redoc")
_JSON_CSP = "default-src 'none'; frame-ancestors 'none'"
_DOCS_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
             "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
             "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https://fastapi.tiangolo.com; "
             "worker-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    docs = request.url.path.startswith(_DOC_PAGES)
    response.headers.setdefault("Content-Security-Policy", _DOCS_CSP if docs else _JSON_CSP)
    return response


@lru_cache(maxsize=1)
def index():
    from .index import Index
    return Index()


class SearchIn(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=20)
    mode: str = "hybrid"
    docs: list[str] | None = None


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    docs: list[str] | None = None


class TextIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=20000)


def _ref(ix, chunk_id):
    r = ix.row(chunk_id)
    return {"chunk_id": int(chunk_id), "document": r["doc_id"], "title": ix.titles.get(r["doc_id"]),
            "clause": r["clause"] or None, "page": int(r["page"]), "heading": r["heading"]}


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/search")
def search_ep(body: SearchIn):
    if body.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    ix = index()
    return [{**_ref(ix, i), "score": s, "similarity": sim}
            for i, s, sim in search(ix, body.query, k=body.k, mode=body.mode, docs=body.docs)]


@app.post("/ask")
def ask_ep(body: AskIn):
    ix = index()
    a = ask(ix, body.question, client=llm_client(), docs=body.docs)
    message = None
    if a.status == "unverified":
        message = "The generated answer failed validation and has been withheld."
    elif a.status in ("not_found", "abstained_low_similarity"):
        message = a.text
    elif a.status == "extractive":
        message = "No language model configured; showing source passages."
    return {"status": a.status, "answer": a.text if a.status == "answered" else None,
            "message": message,
            "extracts": [{**_ref(ix, c), "text": ix.row(c)["text"]} for c in a.passages]
                        if a.status == "extractive" else [],
            "citations": [_ref(ix, c) for c in a.citations], "sources_considered": [_ref(ix, c) for c in a.passages],
            "problems": a.problems}


@app.post("/extract")
def extract_ep(body: TextIn):
    return to_dicts(extract(body.text))
