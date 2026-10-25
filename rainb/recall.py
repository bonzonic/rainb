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
MAX_SPOKEN = 2

SYSTEM = """You help someone who froze mid-sentence. Given their words and numbered notes from their own memory, return the word(s) they are reaching for, but ONLY if one note is clearly about the very thing they are describing.

Their words come from speech recognition, which often mis-hears names ("Control" for "Trontal"), so match on meaning and sound, not exact spelling.

Reply with JSON only: {"answer": "<the missing words, max 8>" or null, "note": <note number> or null}

Answer null when:
- no note is about the same thing (a different kind of place, a friend when they said brother, another company),
- it would need general knowledge or a guess rather than a note,
- they are just pausing, not reaching for a specific name, place, date, number or term,
- the answer is something they already said.
Prefer their written notes over "something they said earlier" when both could answer.

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
        src = h.title or ("something they said earlier, may be misheard" if h.kind == "utterance" else h.kind)
        head = f" / {h.heading}" if h.heading else ""
        out.append(f"[{i}] ({src}{head}) {h.text}")
    return "\n".join(out)


def search(brain: Brain, emb: Embedder, text: str, k: int = K) -> list[Hit]:
    hits = brain.search(emb.query(text), text, k=k + 30)
    # the sentence you're in the middle of is already in conversation memory, and an
    # earlier question you asked is not an answer to it: skip both
    said = _norm(text)
    hits = [h for h in hits if not (h.kind == "utterance" and
                                    (_norm(h.text) in said or h.text.rstrip().endswith("?")))]
    # your notes first, then at most MAX_SPOKEN things you said: repeated talk about a topic
    # must not crowd the note out, and speech in memory may itself be misheard
    # ("Mum is the Japanese restaurant…" must not beat the real Uokatsu note)
    notes = [h for h in hits if h.kind != "utterance"]
    spoken = [h for h in hits if h.kind == "utterance"][:MAX_SPOKEN]
    return (notes[:k - len(spoken[:1])] + spoken)[:k]


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def grounded(answer: str, note: str) -> bool:
    """Every content word of the answer must appear in the note."""
    note_n = " " + _norm(note) + " "
    if _norm(answer).replace(" ", "") in note_n.replace(" ", "") and len(_norm(answer)) > 3:
        return True  # "Camp 5" vs "Camp5"
    words = [w for w in _norm(answer).split() if len(w) > 1 and w not in ("the", "a", "an", "dr", "on", "at", "in")]
    return bool(words) and all(f" {w} " in note_n for w in words)


KIN = {"mum", "mom", "mother", "dad", "father", "brother", "sister", "son", "daughter", "uncle", "aunt",
       "cousin", "grandma", "grandmother", "grandpa", "grandfather", "wife", "husband", "boss", "manager",
       "friend", "neighbour", "neighbor"}


def _kin(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower().replace("'s", ""))) & KIN


def wrong_person(context: str, note: str, all_notes: list[str]) -> bool:
    """You said "my brother", the note is about someone else: not your answer.

    The note itself may only name the person ("Aisha's birthday is 9 December"), so follow
    the name: some note must link that person to the relation you said ("My sister Aisha…"
    doesn't make Aisha your brother).
    """
    said = _kin(" ".join(context.split()[-10:]))
    if not said:
        return False
    in_note = _kin(note)
    if in_note & said:
        return False
    if in_note:
        return True
    names = {w for w in re.findall(r"\b[A-Z][a-z]+\b", note)} - {"The", "My", "Our", "I"}
    if not names:
        return False
    return not any(name in n and _kin(n) & said for name in names for n in all_notes)


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
    if answer and (answer.endswith("?") or len(answer.split()) > 8):
        answer = None  # an answer is the missing words, not a sentence or a question
    if answer and (not src or not grounded(answer, src.text) or already_said(answer, c.context)
                   or wrong_person(c.context, src.text, brain.note_texts())):
        answer = None
    return Recall(c.id, answer, src if answer else None, hits, timings, data)
