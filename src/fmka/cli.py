"""Command line: fetch, build, search, ask.

  python -m fmka.cli fetch          download the official PDFs listed in sources.json into data/raw
  python -m fmka.cli build          parse, embed and index them into data/processed/fmka.duckdb
  python -m fmka.cli search "..."   hybrid search with clause and page references
  python -m fmka.cli ask "..."      cited answer (set FMKA_LLM_BASE_URL for generation; otherwise passages only)
"""
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "fm-knowledge-assistant/0.1 (research; github.com/LUOaini1213)"}


def fetch():
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for s in sources:
        path = raw / s["file"]
        if not path.exists():
            r = requests.get(s["url"], headers=UA, timeout=300)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise RuntimeError(f"{s['id']}: not a PDF")
            path.write_bytes(r.content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        note = "" if not s.get("sha256") or s["sha256"] == digest else "  (differs from the tested edition)"
        print(f"{s['id']:24s} {path.stat().st_size / 1e6:6.1f} MB  sha256 {digest[:12]}{note}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "help"
    if cmd == "fetch":
        fetch()
    elif cmd == "build":
        from .index import build
        print("chunks indexed:", build())
    elif cmd == "search":
        from .index import Index
        from .search import search
        ix = Index()
        for i, s, sim in search(ix, " ".join(argv[1:]), k=5):
            r = ix.row(i)
            print(f"{r['doc_id']} clause {r['clause'] or '-'} p.{r['page']}  sim {sim:.3f}  {r['text'][:120]}")
    elif cmd == "ask":
        from .answer import ask, llm_client
        from .index import Index
        ix = Index()
        a = ask(ix, " ".join(argv[1:]), client=llm_client())
        text = "The generated answer failed validation and has been withheld." if a.status == "unverified" else a.text
        print(a.status, "\n", text)
        for c in a.citations:
            r = ix.row(c)
            print(f"  [{r['doc_id']} clause {r['clause'] or '-'} p.{r['page']}]")
        if a.problems:
            print("  checks failed:", a.problems)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
