"""Brain store: one SQLite file per brain (persona / me / judge-<random>).

Documents, conversation utterances and extracted facts all become rows in `chunks`,
so there is a single search path: vectors (cosine) + FTS5 keywords, fused by rank.
Keyword search matters for proper nouns that embeddings blur ("Uokatsu").
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY, source TEXT UNIQUE, title TEXT, hash TEXT, added_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,            -- doc | utterance | fact
    heading TEXT DEFAULT '',
    text TEXT NOT NULL,
    speaker INTEGER,
    created_at REAL NOT NULL,
    expires_at REAL,
    embedding BLOB
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, old.heading);
END;
CREATE TABLE IF NOT EXISTS moments (
    id INTEGER PRIMARY KEY,
    created_at REAL, reason TEXT, context TEXT, reaching_for TEXT,
    answer TEXT, source_chunk_id INTEGER, confidence REAL,
    latency_ms INTEGER, late INTEGER, outcome TEXT
);
"""

STOPWORDS = set("""a an the and or but of to in on at by for with from that this these those is was are were be
been it its i me my we our you your he she his her they them their what what's whats which who
uh um er erm ah eh hmm mm so like just one called name again know let's lets go went
do did does not no yes there here near place thing""".split())

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    text: str
    heading: str
    title: str
    kind: str
    created_at: float
    cosine: float
    score: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Brain:
    def __init__(self, path: Path, brain_id: str, label: str, read_only: bool = False):
        self.id, self.label, self.read_only = brain_id, label, read_only
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self._matrix: tuple[np.ndarray, np.ndarray] | None = None  # (ids, vectors)

    def close(self):
        self.db.close()

    # -- writes ---------------------------------------------------------------

    def doc_hash(self, source: str) -> str | None:
        row = self.db.execute("SELECT hash FROM documents WHERE source = ?", (source,)).fetchone()
        return row["hash"] if row else None

    def put_document(self, source: str, title: str, content_hash: str,
                     chunks: list[tuple[str, str]], vectors: np.ndarray) -> int:
        """Replace a document and its chunks. `chunks` = [(heading, text)]."""
        now = time.time()
        with self.lock, self.db:
            self.db.execute("DELETE FROM chunks WHERE doc_id IN (SELECT id FROM documents WHERE source = ?)", (source,))
            self.db.execute("DELETE FROM documents WHERE source = ?", (source,))
            doc_id = self.db.execute(
                "INSERT INTO documents (source, title, hash, added_at) VALUES (?, ?, ?, ?)",
                (source, title, content_hash, now)).lastrowid
            self.db.executemany(
                "INSERT INTO chunks (doc_id, kind, heading, text, created_at, embedding) VALUES (?, 'doc', ?, ?, ?, ?)",
                [(doc_id, h, t, now, v.astype(np.float32).tobytes()) for (h, t), v in zip(chunks, vectors)])
            self._matrix = None
        return doc_id

    def remove_document(self, source: str):
        with self.lock, self.db:
            self.db.execute("DELETE FROM chunks WHERE doc_id IN (SELECT id FROM documents WHERE source = ?)", (source,))
            self.db.execute("DELETE FROM documents WHERE source = ?", (source,))
            self._matrix = None

    def add_memory(self, kind: str, text: str, vector: np.ndarray, speaker: int | None,
                   expires_at: float | None, heading: str = "", created_at: float | None = None) -> int:
        with self.lock, self.db:
            cid = self.db.execute(
                "INSERT INTO chunks (kind, heading, text, speaker, created_at, expires_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (kind, heading, text, speaker, created_at or time.time(), expires_at,
                 vector.astype(np.float32).tobytes())).lastrowid
            self._matrix = None
        return cid

    def purge_expired(self, now: float | None = None) -> int:
        """Delete expired memory from chunks, FTS (via trigger) and the vector cache."""
        with self.lock, self.db:
            n = self.db.execute("DELETE FROM chunks WHERE expires_at IS NOT NULL AND expires_at <= ?",
                                (now or time.time(),)).rowcount
            if n:
                self._matrix = None
        return n

    def forget(self, kinds: tuple[str, ...] = ("utterance", "fact")) -> int:
        with self.lock, self.db:
            q = f"DELETE FROM chunks WHERE kind IN ({','.join('?' * len(kinds))})"
            n = self.db.execute(q, kinds).rowcount
            self.db.execute("DELETE FROM moments")
            self._matrix = None
        return n

    def delete_chunk(self, chunk_id: int):
        with self.lock, self.db:
            self.db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            self._matrix = None

    # -- moments --------------------------------------------------------------

    def log_moment(self, **m) -> int:
        cols = ", ".join(m)
        with self.lock, self.db:
            return self.db.execute(f"INSERT INTO moments ({cols}) VALUES ({', '.join('?' * len(m))})",
                                   tuple(m.values())).lastrowid

    def set_outcome(self, moment_id: int, outcome: str):
        with self.lock, self.db:
            self.db.execute("UPDATE moments SET outcome = ? WHERE id = ?", (outcome, moment_id))

    def moments(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT m.*, c.text AS source_text, d.title AS source_title FROM moments m "
            "LEFT JOIN chunks c ON c.id = m.source_chunk_id LEFT JOIN documents d ON d.id = c.doc_id "
            "ORDER BY m.id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- reads ----------------------------------------------------------------

    def stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {
            "documents": q("SELECT COUNT(*) FROM documents"),
            "chunks": q("SELECT COUNT(*) FROM chunks WHERE kind = 'doc'"),
            "utterances": q("SELECT COUNT(*) FROM chunks WHERE kind = 'utterance'"),
            "facts": q("SELECT COUNT(*) FROM chunks WHERE kind = 'fact'"),
            "moments": q("SELECT COUNT(*) FROM moments"),
        }

    def vocabulary(self) -> list[str]:
        """Names and terms from this brain's notes, for fixing misheard words."""
        from .correct import vocabulary

        key = self.db.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM documents").fetchone()
        if getattr(self, "_vocab_key", None) != tuple(key):
            texts = [r[0] for r in self.db.execute("SELECT text FROM chunks WHERE kind = 'doc'")]
            self._vocab, self._vocab_key = vocabulary(texts), tuple(key)
            self._aliases = None
        return self._vocab

    def aliases(self):
        """Sounds-like fixer for this brain's names (rebuilt with the vocabulary)."""
        from .aliases import Aliases
        from .speech import boost_terms

        vocab = self.vocabulary()
        if getattr(self, "_aliases", None) is None:
            notes = [r[0] for r in self.db.execute("SELECT text FROM chunks WHERE kind = 'doc'")]
            self._aliases = Aliases(boost_terms(vocab), notes)
        return self._aliases

    def documents(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT d.source, d.title, d.added_at, COUNT(c.id) AS chunks FROM documents d "
            "LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.title").fetchall()
        return [dict(r) for r in rows]

    def memories(self, limit: int = 100) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, kind, text, speaker, created_at, expires_at FROM chunks "
            "WHERE kind IN ('utterance', 'fact') ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def _vectors(self) -> tuple[np.ndarray, np.ndarray]:
        if self._matrix is None:
            with self.lock:
                rows = self.db.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
            if rows:
                ids = np.array([r["id"] for r in rows])
                mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
            else:
                ids, mat = np.array([], dtype=int), np.zeros((0, 1), dtype=np.float32)
            self._matrix = (ids, mat)
        return self._matrix

    def keyword_ranks(self, text: str, k: int = 20) -> list[int]:
        terms = [t for t in re.findall(r"[\w']+", text.lower()) if t not in STOPWORDS and len(t) > 1]
        if not terms:
            return []
        q = " OR ".join('"' + t.replace('"', "") + '"' for t in terms)
        with self.lock:
            rows = self.db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (q, k)).fetchall()
        return [r[0] for r in rows]

    def search(self, qvec: np.ndarray | None, qtext: str, k: int = 5) -> list[Hit]:
        ids, mat = self._vectors()
        scores: dict[int, float] = {}
        cos: dict[int, float] = {}
        if qvec is not None and len(ids):
            sims = mat @ qvec
            order = np.argsort(-sims)[:20]
            for rank, i in enumerate(order):
                cid = int(ids[i])
                cos[cid] = float(sims[i])
                scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        for rank, cid in enumerate(self.keyword_ranks(qtext)):
            scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank)
        top = sorted(scores, key=lambda c: -scores[c])[:k]
        if not top:
            return []
        with self.lock:
            rows = {r["id"]: r for r in self.db.execute(
                f"SELECT c.id, c.text, c.heading, c.kind, c.created_at, COALESCE(d.title, '') AS title "
                f"FROM chunks c LEFT JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({','.join('?' * len(top))})",
                top).fetchall()}
        if qvec is not None and len(ids):
            missing = [c for c in top if c not in cos]
            for c in missing:
                cos[c] = float(mat[np.where(ids == c)[0][0]] @ qvec)
        return [Hit(c, rows[c]["text"], rows[c]["heading"], rows[c]["title"], rows[c]["kind"],
                    rows[c]["created_at"], cos.get(c, 0.0), scores[c]) for c in top if c in rows]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
