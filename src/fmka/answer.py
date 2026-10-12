"""Grounded answers with citations, abstention and automatic checks.

1. Retrieve the top passages (hybrid search).
2. Abstain before generation when even the best passage is not close to the question (dense cosine below a
   threshold chosen on the development questions only).
3. Ask the LLM to answer only from the numbered passages and cite them as [n], or reply NOT_FOUND.
4. Check supplied citations and numeric membership, then bind each generated quantity statement to its local
   citations using a bounded value/unit/comparator/explicit subject grammar. Unsupported or ambiguous detected
   quantities fail closed. This lexical guard does not prove arbitrary semantic entailment or non-quantity facts.
   A failed draft is retained for internal diagnostics and withheld from public answers as unverified.

The LLM is any OpenAI-compatible chat endpoint set by FMKA_LLM_BASE_URL / FMKA_LLM_MODEL / FMKA_LLM_KEY: a local
Ollama model (the default used in the evaluation, qwen2.5:3b), DeepSeek, or Azure OpenAI through the same client.
With no endpoint configured the assistant returns the cited passages only (extractive mode).
"""
import os
import re
from dataclasses import dataclass, field

from .numeric import numbers, scan_numbers
from .claims import claims, verify_claims
from .search import search

SYSTEM = (
    "You answer questions about Singapore building and facilities-management codes using ONLY the numbered passages "
    "provided. Cite every statement with the passage number in square brackets, e.g. [2]. Quote numbers and units "
    "exactly as written. If the passages do not contain the answer, reply exactly NOT_FOUND. Be concise (at most 3 "
    "sentences).")
MIN_SIM = 0.725  # chosen on the development questions (eval/run_eval.py)
JUDGE = ("You check citations. Given a PASSAGE and a CLAIM, answer YES only if the passage itself states the claim about "
         "the same subject (the same equipment, space or element). A number that the passage gives for a different "
         "subject does not count. Answer YES or NO.")
CITE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    status: str                      # answered | abstained_low_similarity | not_found | unverified | extractive
    text: str
    citations: list = field(default_factory=list)   # chunk_ids cited
    passages: list = field(default_factory=list)    # chunk_ids supplied
    top_sim: float | None = None
    problems: list = field(default_factory=list)


def verify(draft, passages):
    """passages: list of passage texts in the order they were numbered (1-based). Returns (cited indices, problems)."""
    cited = sorted({int(c) for c in CITE.findall(draft)})
    problems = []
    if not cited:
        problems.append("no citation")
    bad = [c for c in cited if c < 1 or c > len(passages)]
    if bad:
        problems.append(f"citation to a passage that was not supplied: {bad}")
    good = [c for c in cited if 1 <= c <= len(passages)]
    # Normalize each passage separately: words on opposite sides of a passage
    # boundary must not accidentally form a compound quantity.
    source_nums = set().union(*(numbers(passages[c - 1]) for c in good))
    body = CITE.sub(" ", draft)
    draft_nums, unsupported = scan_numbers(body)
    if unsupported:
        problems.append(f"unsupported numeric expressions: {unsupported}")
    missing = sorted(n for n in draft_nums if n not in source_nums)
    if missing:
        problems.append(f"numbers not found in the cited passages: {missing}")
    problems.extend(verify_claims(draft, passages))
    return good, problems


def llm_client():
    base = os.environ.get("FMKA_LLM_BASE_URL")
    if not base:
        return None
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("FMKA_LLM_KEY", "none"), base_url=base)


def llm_model():
    return os.environ.get("FMKA_LLM_MODEL", "qwen2.5:3b-16k")


def _chat(client, model, messages):
    resp = client.chat.completions.create(model=model or llm_model(), temperature=0, messages=messages)
    return resp.choices[0].message.content.strip()


def judged(client, model, passage, claim):
    reply = _chat(client, model, [{"role": "system", "content": JUDGE},
                                  {"role": "user", "content": f"PASSAGE: {passage}\n\nCLAIM: {claim}"}])
    return reply.upper() == "YES"


def ask(index, question, k=6, min_sim=MIN_SIM, client=None, model=None, docs=None, strict=False):
    """strict=True additionally judges every statement against its own citations.

    This optional model judgment cannot override the deterministic quantity guard;
    its historical whole-answer evaluation does not measure this per-claim policy.
    """
    hits = search(index, question, k=k, mode="hybrid", docs=docs)
    ids = [h[0] for h in hits]
    top_sim = max((h[2] for h in hits if h[2] is not None), default=None)
    if top_sim is None or top_sim < min_sim:
        return Answer(question, "abstained_low_similarity", "The indexed codes do not appear to cover this question.",
                      passages=ids, top_sim=top_sim)
    rows = [index.row(i) for i in ids]
    # The model sees metadata as well as the passage. Header values can support
    # reference-only statements; the quantity guard strips this header so a page
    # number or code year cannot authorize a physical measurement.
    texts = [f"({index.titles.get(r['doc_id'], r['doc_id'])}, clause {r['clause'] or '-'}, page {r['page']}) "
             f"{r['heading']} {r['text']}".strip() for r in rows]
    numbered = "\n\n".join(f"[{n}] {t}" for n, t in enumerate(texts, start=1))
    if client is None:
        return Answer(question, "extractive", numbered, citations=ids[:1], passages=ids, top_sim=top_sim)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {question}"}]
    draft = _chat(client, model, messages)
    if not draft.upper().startswith("NOT_FOUND") and not CITE.search(draft):
        # One retry when the model forgot to cite; the checks below still decide whether the answer is shown.
        messages += [{"role": "assistant", "content": draft},
                     {"role": "user", "content": "Rewrite the answer and put the passage number in square brackets after every statement, e.g. [1] or [2]."}]
        draft = _chat(client, model, messages)
    if draft.upper().startswith("NOT_FOUND"):
        return Answer(question, "not_found", "The indexed codes do not appear to cover this question.",
                      passages=ids, top_sim=top_sim)
    cited, problems = verify(draft, texts)
    if strict and not problems:
        for n, claim in enumerate(claims(draft), 1):
            if not any(judged(client, model, texts[c - 1], claim.text) for c in claim.citations):
                problems.append(f"claim {n}: no local cited passage states the claim for the same subject")
    status = "answered" if not problems else "unverified"
    return Answer(question, status, draft, citations=[ids[c - 1] for c in cited], passages=ids, top_sim=top_sim,
                  problems=problems)
