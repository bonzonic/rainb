"""Turn files into (heading, text) chunks and load them into a brain. Read-only on the source."""
from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

from .brain import Brain, content_hash
from .embedder import Embedder

SUPPORTED = {".md", ".txt", ".json", ".pdf"}
MAX_CHARS = 700


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split on headings, then pack paragraphs up to MAX_CHARS; keep the heading path."""
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)  # front matter
    path: list[str] = []
    out: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        heading = " > ".join(path)
        paras = []
        for p in re.split(r"\n\s*\n", body):
            items = re.split(r"\n(?=\s*(?:[-*+]|\d+\.)\s)", p)
            if len(items) > 1 or re.match(r"\s*(?:[-*+]|\d+\.)\s", p):
                # a list of atomic notes: one chunk per item, so search can point at the exact one
                for it in items:
                    it = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+(\[.\]\s*)?", "", it).strip()
                    if it:
                        out.append((heading, it))
            else:
                paras.append(p)
        cur = ""
        for p in paras:
            if cur and len(cur) + len(p) > MAX_CHARS:
                out.append((heading, cur.strip()))
                cur = ""
            cur += p + "\n\n"
        if cur.strip():
            out.append((heading, cur.strip()))

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m[1])
            path[:] = path[:level - 1] + [m[2].strip()]
        else:
            buf.append(line)
    flush()
    return out


def chunk_notes_json(data: list[dict]) -> list[tuple[str, str]]:
    return [(n.get("category", ""), n["text"]) for n in data if n.get("text")]


def chunk_pdf(path: Path) -> list[tuple[str, str]]:
    from pypdf import PdfReader  # optional dependency

    out = []
    for i, page in enumerate(PdfReader(path).pages, 1):
        for _, t in chunk_markdown(page.extract_text() or ""):
            out.append((f"page {i}", t))
    return out


def load_file(path: Path, data: bytes | None = None) -> tuple[str, list[tuple[str, str]]]:
    data = data if data is not None else path.read_bytes()
    ext = path.suffix.lower()
    title = path.stem
    if ext == ".json":
        return title, chunk_notes_json(json.loads(data.decode("utf-8")))
    if ext == ".pdf":
        return title, chunk_pdf(path)
    return title, chunk_markdown(data.decode("utf-8", errors="replace"))


async def ingest_file(brain: Brain, emb: Embedder, path: Path, source: str | None = None,
                      data: bytes | None = None) -> int:
    """Returns the number of chunks written; 0 if the file is unchanged."""
    data = data if data is not None else path.read_bytes()
    source = source or str(path)
    h = content_hash(data + emb.id.encode())
    if brain.doc_hash(source) == h:
        return 0
    title, chunks = load_file(path, data)
    if not chunks:
        return 0
    # embed with the title in front so a chunk like "Uokatsu on Jalan Telawi" knows its context
    vectors = await emb.passages([f"{title} - {h_}\n{t}" if h_ else f"{title}\n{t}" for h_, t in chunks])
    brain.put_document(source, title, h, chunks, vectors)
    return len(chunks)


def read_ignore(root: Path) -> list[str]:
    p = root / ".rainbignore"
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]


def walk(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    ignore = read_ignore(root) + [".obsidian/*", ".trash/*", ".git/*"]
    files = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and p.suffix.lower() in SUPPORTED and not any(fnmatch.fnmatch(rel, g) for g in ignore):
            files.append(p)
    return files


async def sync(brain: Brain, emb: Embedder, root: Path, log=print) -> dict:
    """Hash-based incremental sync of a folder (or single file). Never writes to the source."""
    files = walk(root)
    seen, written = set(), 0
    for p in files:
        src = str(p.resolve())
        seen.add(src)
        n = await ingest_file(brain, emb, p, source=src)
        if n:
            log(f"  + {p.name}: {n} chunks")
            written += n
    base = str(root.resolve())
    removed = 0
    for d in brain.documents():
        if d["source"].startswith(base) and d["source"] not in seen:
            brain.remove_document(d["source"])
            removed += 1
    return {"files": len(files), "chunks_written": written, "removed": removed}
