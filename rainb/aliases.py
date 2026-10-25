"""Sounds-like fixes for the wearer's own names, applied to every word as it arrives (~0 ms).

Word boosting nudges Parakeet towards "Trontal", but a real voice can still land on a
real word that sounds the same ("Toronto"). A heard word is turned into a brain name when
it has the same consonant skeleton, never if the heard word itself appears in the notes,
and with one more condition for real English words:

- not a real word ("Trontel", "Yuakatsu", "Readies"): fix it.
- a real word ("Toronto"): fix it only if the words around it also appear next to that
  name in the notes ("my manager in Toronto" -> Trontal, because the note says "My
  manager at Trontal"). "I flew to Toronto last year" stays Toronto.

"Real word" = a whole token in the BGE embedding model's 30k vocabulary.
"""
from __future__ import annotations

import glob
import json
import re
from functools import lru_cache

from . import config

_VOWELS = set("aeiouy")
_MAP = str.maketrans({"c": "k", "q": "k", "z": "s", "v": "f", "d": "t", "b": "p", "g": "k"})
_STOP = set("""a an the and or but of to in on at by for with from is was are were be been my me i i'm we our
you your he she his her they their it its this that what what's who which when where how there here
do did not no so like just one go going get got uh um""".split())


def skeleton(word: str) -> str:
    """Consonant skeleton: 'Toronto' -> 'trnt', 'Trontal' -> 'trntl', 'Uokatsu' -> 'ukts'."""
    w = re.sub(r"[^a-z]", "", word.lower()).replace("ph", "f").replace("ck", "k").replace("th", "t")
    if not w:
        return ""
    head = "u" if w[0] in _VOWELS else w[0]  # any leading vowel sound (U-, Yu-, O-) is one class
    body = [ch for ch in w[1:] if ch not in _VOWELS]
    out = (head + "".join(body)).translate(_MAP)
    return re.sub(r"(.)\1+", r"\1", out)


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


@lru_cache(maxsize=1)
def known_words() -> frozenset[str]:
    """Whole-word tokens of the local embedding model's vocabulary (common English + big places)."""
    paths = glob.glob(str(config.ROOT / "models" / "fastembed" / "*bge-base*" / "snapshots" / "*" / "tokenizer.json"))
    if not paths:
        return frozenset()
    vocab = json.load(open(paths[0], encoding="utf-8"))["model"]["vocab"]
    return frozenset(w for w in vocab if w.isalpha())


class Aliases:
    def __init__(self, names: list[str], notes: list[str], known: frozenset[str] | None = None):
        self.known = known if known is not None else known_words()
        self.names: dict[str, str] = {}
        self.context: dict[str, set[str]] = {}
        for n in names:
            # single words without digits ("Camp5" would swallow every "camp")
            if " " in n or len(n) < 4 or any(ch.isdigit() for ch in n):
                continue
            k = skeleton(n)
            if len(k) < 3 or k in self.names:
                continue
            self.names[k] = n
            near = set()
            for note in notes:
                if n.lower() in note.lower():
                    near |= {w for w in _words(note) if w not in _STOP and w != n.lower()}
            self.context[n] = near
        self.note_words = {w for note in notes for w in _words(note)}

    def fix(self, word: str, context: list[str] = ()) -> str | None:
        """The brain name this heard word should be, or None. `context` = nearby heard words."""
        bare = re.sub(r"[^\w']", "", word)
        low = bare.lower()
        # contractions ("It's", "what's") are real English, never a misheard name
        if len(bare) < 4 or "'" in bare or low in self.note_words:
            return None
        k = skeleton(bare)
        if len(k) < 3:
            return None
        best = None
        for nk, name in self.names.items():
            if nk[0] != k[0] or name.lower() == low:
                continue
            d = _distance(k, nk)
            allowed = 0 if len(nk) <= 3 else 1 if len(nk) <= 5 else 2  # short skeletons match too much
            if d <= allowed and (best is None or d < best[0]):
                best = (d, name)
        if not best:
            return None
        name = best[1]
        if low in self.known:  # a real word: only with supporting context from the name's notes
            nearby = {w for c in context for w in _words(c)}
            if not (nearby & self.context.get(name, set())):
                return None
        # keep the heard word's trailing punctuation: "Toronto?" -> "Trontal?"
        return name + word[len(word.rstrip(".,?!;:")):]
