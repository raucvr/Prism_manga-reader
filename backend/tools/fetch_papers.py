"""
Download the evaluation papers (eval/papers.json) into eval/papers/<name>.pdf and check their SHA-256.

usage: .venv/bin/python -I backend/tools/fetch_papers.py [name ...]

Each paper is a pinned arXiv version, so everyone evaluates the same PDF. arxiv.org first, then export.arxiv.org (its
mirror for automated downloads). A file that is already there with the right checksum is kept.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = ROOT / "eval"
HOSTS = ("https://arxiv.org/pdf/", "https://export.arxiv.org/pdf/")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(paper: dict) -> str:
    out = EVAL / "papers" / f"{paper['name']}.pdf"
    if out.exists() and sha256(out.read_bytes()) == paper["sha256"]:
        return "kept"
    last = ""
    for host in HOSTS:
        try:
            r = httpx.get(host + paper["arxiv"], follow_redirects=True, timeout=120)
        except httpx.HTTPError as e:
            last = f"{type(e).__name__}: {e}"
            continue
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            last = f"HTTP {r.status_code}"
            continue
        if sha256(r.content) != paper["sha256"]:
            last = "checksum mismatch (arXiv served a different file)"
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        return "downloaded"
    return f"FAILED ({last}): download https://arxiv.org/pdf/{paper['arxiv']} by hand into {out}"


def main() -> None:
    papers = json.loads((EVAL / "papers.json").read_text(encoding="utf-8"))
    wanted = set(sys.argv[1:])
    for paper in papers:
        if not wanted or paper["name"] in wanted:
            print(f"{paper['name']:16s} {paper['arxiv']:14s} {fetch(paper)}")


if __name__ == "__main__":
    main()
