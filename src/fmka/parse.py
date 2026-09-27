"""PDF -> clause-level chunks with provenance (document, page, clause number, chapter).

Documents with numbered clauses (Fire Code, Accessibility Code, PSI guidelines) are cut at clause numbers such as
"2.3.9", "CLAUSE 1.1" or "4.2.1.2" at the start of a line. Documents without numbering are cut into paragraph groups of about
CHUNK_CHARS characters. Lines that repeat on many pages (running headers and footers) and bare page numbers are dropped.
"""
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

CHUNK_CHARS = 1200
CLAUSE = re.compile(r"^\s*(?:CLAUSE\s+)?(\d{1,2}(?:\.\d{1,3}){1,4})\.?\s*(.*)$")
CHAPTER = re.compile(r"^\s*C\s*H\s*A\s*P\s*T\s*E\s*R\s*(\d{1,2})\b", re.I)


@dataclass
class Chunk:
    doc_id: str
    page: int          # 1-based page of the chunk's first line
    clause: str        # "2.3.9", or "" for unnumbered documents
    chapter: str       # first number of the clause ("2"), or the latest CHAPTER heading, or ""
    heading: str       # text on the clause line (often its title)
    text: str


def page_lines(pdf_path):
    doc = pymupdf.open(pdf_path)
    pages = [[ln.strip() for ln in p.get_text().splitlines()] for p in doc]
    counts = Counter(ln for lines in pages for ln in set(lines) if ln)
    repeat = {ln for ln, c in counts.items() if c >= max(3, 0.3 * len(pages)) and len(ln) < 120}
    out = []
    for i, lines in enumerate(pages, start=1):
        keep = [ln for ln in lines if ln and ln not in repeat and not re.fullmatch(r"\d{1,3}(\s*\|\s*P\s*a\s*g\s*e)?", ln)]
        out.append((i, keep))
    return out


def numbered(doc_id, pages, min_depth=2):
    chunks, cur, chapter, last_major = [], None, "", None
    for page, lines in pages:
        for ln in lines:
            ch = CHAPTER.match(ln)
            if ch:
                chapter = ch.group(1)
            m = CLAUSE.match(ln)
            ok = m and m.group(1).count(".") + 1 >= min_depth and not re.fullmatch(r"[\d.\s]+(mm|m|%)?", m.group(2) or "x")
            if ok:
                # Clause numbers only move forward one chapter at a time, and a new chapter starts at N.1; a number that
                # jumps (a table value such as "12.5" inside chapter 3, "4.5", or "0.5") is text, not a new clause.
                parts = m.group(1).split(".")
                major = int(parts[0])
                starts_chapter = parts[1] == "1"
                ok = (last_major is None and major <= 2) or major == last_major or                     (last_major is not None and major == last_major + 1 and starts_chapter)
            if ok:
                last_major = major
                if cur and cur.text.strip():
                    chunks.append(cur)
                num = m.group(1)
                cur = Chunk(doc_id, page, num, num.split(".")[0] or chapter, m.group(2)[:120], m.group(2))
            elif cur is None:
                cur = Chunk(doc_id, page, "", chapter, "", ln)
            else:
                cur.text += " " + ln
    if cur and cur.text.strip():
        chunks.append(cur)
    return split_long(chunks)


def paragraphs(doc_id, pages):
    chunks, buf, start = [], [], None
    for page, lines in pages:
        for ln in lines:
            if start is None:
                start = page
            buf.append(ln)
            if sum(len(x) for x in buf) >= CHUNK_CHARS and ln.endswith((".", ":", ";")):
                chunks.append(Chunk(doc_id, start, "", "", buf[0][:120], " ".join(buf)))
                buf, start = [], None
    if buf:
        chunks.append(Chunk(doc_id, start or 1, "", "", buf[0][:120], " ".join(buf)))
    return chunks


def split_long(chunks, limit=2 * CHUNK_CHARS):
    out = []
    for c in chunks:
        c.text = " ".join(c.text.split())
        if len(c.text) <= limit:
            out.append(c)
            continue
        words, part = c.text.split(), []
        for w in words:
            part.append(w)
            if sum(len(x) + 1 for x in part) >= CHUNK_CHARS:
                out.append(Chunk(c.doc_id, c.page, c.clause, c.chapter, c.heading, " ".join(part)))
                part = part[-30:]  # small overlap so a sentence cut at the boundary is still retrievable
        if len(part) > 30:
            out.append(Chunk(c.doc_id, c.page, c.clause, c.chapter, c.heading, " ".join(part)))
    return out


def parse(source, raw_dir):
    # Contents pages list every clause number once before the text starts; they are skipped by page number.
    pages = [p for p in page_lines(Path(raw_dir) / source["file"]) if p[0] > source.get("skip_pages", 0)]
    chunks = numbered(source["id"], pages) if source.get("numbered") else paragraphs(source["id"], pages)
    return [c for c in chunks if len(c.text) >= 40]


def parse_all(sources_path, raw_dir):
    sources = json.loads(Path(sources_path).read_text(encoding="utf-8"))
    chunks = []
    for s in sources:
        chunks += parse(s, raw_dir)
    return [asdict(c) for c in chunks]
