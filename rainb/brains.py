"""Which brain a request gets, and keeping the demo persona brain built.

No brain picker in the UI: paired phone (owner-key cookie) -> "me"; uploaded notes ->
that judge's temporary brain; anyone else -> the demo persona.
"""
from __future__ import annotations

import secrets
import shutil
import time
from pathlib import Path

from . import config
from .brain import Brain
from .ingest import sync
from .embedder import Embedder

LABELS = {"demo": "demo brain (Maya, fictional)", "me": "your brain"}
_open: dict[str, Brain] = {}


def path_for(brain_id: str) -> Path:
    return config.DATA_DIR / "brains" / f"{brain_id}.db"


def get(brain_id: str) -> Brain:
    if brain_id not in _open:
        label = LABELS.get(brain_id, "your uploaded notes (deleted after 24 h)")
        _open[brain_id] = Brain(path_for(brain_id), brain_id, label, read_only=brain_id == "demo")
    return _open[brain_id]


def exists(brain_id: str) -> bool:
    return brain_id in _open or path_for(brain_id).exists()


def new_judge_brain() -> Brain:
    return get("judge-" + secrets.token_hex(8))


def valid_id(brain_id: str | None) -> bool:
    return bool(brain_id) and (brain_id in ("demo", "me") or (
        brain_id.startswith("judge-") and len(brain_id) == 22 and brain_id[6:].isalnum()))


async def build_demo(emb: Embedder, log=print) -> dict:
    """(Re)index the persona folder; unchanged files are skipped by hash."""
    return await sync(get("demo"), emb, config.PERSONA_DIR / "demo", log=log)


def reset_demo_memory():
    """Nightly: the persona brain forgets whatever judges said to it."""
    get("demo").forget()


def purge_judge_brains(now: float | None = None) -> int:
    now = now or time.time()
    n = 0
    for p in (config.DATA_DIR / "brains").glob("judge-*.db"):
        if now - p.stat().st_mtime > config.JUDGE_BRAIN_TTL_H * 3600:
            bid = p.stem
            if bid in _open:
                _open.pop(bid).close()
            p.unlink(missing_ok=True)
            n += 1
    return n


def purge_expired_all() -> int:
    total = 0
    for p in (config.DATA_DIR / "brains").glob("*.db"):
        total += get(p.stem).purge_expired()
    return total


def wipe(brain_id: str):
    if brain_id in _open:
        _open.pop(brain_id).close()
    path_for(brain_id).unlink(missing_ok=True)


__all__ = ["get", "exists", "new_judge_brain", "valid_id", "build_demo", "wipe", "shutil"]
