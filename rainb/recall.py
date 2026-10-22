"""Recall agent: StuckCandidate -> grounded one-line answer, or silence.

Latency-first variant of "confirm ∥ speculative search -> answer": search runs locally on
the raw words the moment the candidate fires (~10 ms), and a single Nemotron call both
confirms the stall and picks the answer from the hits. One model round-trip, not two.
The answer must literally come from the cited note, or RainB stays silent.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .brain import Brain, Hit
from .embedder import Embedder
from .events import StuckCandidate
from .tokenfactory import TokenFactory

QUERY_WORDS = 18  # the tail of the sentence carries what you're reaching for
K = 5

SYSTEM = """You help someone who froze mid-sentence. Given their words and numbered notes from their own memory, return the word(s) they are reaching for, but ONLY if one note is clearly about the very thing they are describing.

Reply with JSON only: {"answer": "<the missing words, max 8>" or null, "note": <note number> or null}

Answer null when:
- no note is about the same thing (a different kind of place, a friend when they said brother, another company),
- it would need general knowledge or a guess rather than a note,
- they are just pausing, not reaching for a specific name, place, date, number or term,
- the answer is something they already said.

Examples:
Notes: [1] The Thai place by the office is Baan Rim Nam.
Their words: "lunch at that Korean barbecue place by the office, the …" -> {"answer": null, "note": null}
(the note is a Thai place; they want a Korean one)
Notes: [1] Dad's flight lands on 14 June.
Their words: "Christmas is always on, uh …" -> {"answer": null, "note": null}
(general knowledge, not in a note; a date in a note is not the date they want)
Notes: [1] My cousin Ben works at Shopee.
Their words: "my uncle works at, uh …" -> {"answer": null, "note": null}
(the note is about a cousin, not the uncle)
Notes: [1] The Thai place by the office is Baan Rim Nam.
Their words: "let's get Thai by the office, at uh …" -> {"answer": "Baan Rim Nam", "note": 1}"""


@dataclass
class Recall:
    candidate_id: int
    answer: str | None
    source: Hit | None
    hits: list[Hit] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.answer and self.source)


def query_text(context: str) -> str:
    return " ".join(context.split()[-QUERY_WORDS:])


def format_notes(hits: list[Hit]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        src = h.title or ("past conversation" if h.kind == "utterance" else h.kind)
        head = f" / {h.heading}" if h.heading else ""
        out.append(f"[{i}] ({src}{head}) {h.text}")
    return "\n".join(out)


def search(brain: Brain, emb: Embedder, text: str, k: int = K) -> list[Hit]:
    hits = brain.search(emb.query(text), text, k=k + 3)
    # the sentence you're in the middle of is already in conversation memory; skip it
    said = _norm(text)
    hits = [h for h in hits if not (h.kind == "utterance" and _norm(h.text) in said)]
    return hits[:k]


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def grounded(answer: str, note: str) -> bool:
    """Every content word of the answer must appear in the note."""
    note_n = " " + _norm(note) + " "
    if _norm(answer).replace(" ", "") in note_n.replace(" ", "") and len(_norm(answer)) > 3:
        return True  # "Camp 5" vs "Camp5"
    words = [w for w in _norm(answer).split() if len(w) > 1 and w not in ("the", "a", "an", "dr", "on", "at", "in")]
    return bool(words) and all(f" {w} " in note_n for w in words)


def already_said(answer: str, context: str) -> bool:
    a = _norm(answer)
    return len(a) > 3 and a in _norm(context)


async def recall(brain: Brain, emb: Embedder, tf: TokenFactory, c: StuckCandidate,
                 hits: list[Hit] | None = None) -> Recall:
    t0 = time.perf_counter()
    hits = hits if hits is not None else search(brain, emb, query_text(c.context))
    t1 = time.perf_counter()
    if not hits:
        return Recall(c.id, None, None, [], {"search": round((t1 - t0) * 1000)})
    user = f"Notes:\n{format_notes(hits)}\n\nTheir words: \"{c.context} …\""
    data = await tf.chat_json([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": user}], max_tokens=40)
    t2 = time.perf_counter()
    timings = {"search": round((t1 - t0) * 1000), "model": round((t2 - t1) * 1000)}

    answer, src = data.get("answer"), None
    idx = data.get("note")
    if isinstance(idx, str) and idx.strip().isdigit():
        idx = int(idx)
    if isinstance(idx, int) and 1 <= idx <= len(hits):
        src = hits[idx - 1]
    if not isinstance(answer, str) or answer.strip().lower() in ("", "null", "none"):
        answer = None
    else:
        answer = answer.strip().strip('"').rstrip(".")
    if answer and (not src or not grounded(answer, src.text) or already_said(answer, c.context)):
        answer = None
    return Recall(c.id, answer, src if answer else None, hits, timings, data)
