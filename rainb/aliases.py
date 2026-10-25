"""Sounds-like fixes for the wearer's own names, applied to every word as it arrives (~0 ms).

Word boosting nudges Parakeet towards "Trontal", but a real voice can still land on a
common word that sounds the same ("Toronto"). If a heard word has the same consonant
skeleton as a name from the brain, and that heard word never appears in the brain's
notes, it is almost certainly the name. "control" or "council" start differently and
stay as they are. Trade-off: in this brain a genuine "Toronto" also becomes "Trontal"
until a note mentions Toronto.
"""
from __future__ import annotations

import re
from functools import lru_cache

_VOWELS = set("aeiouy")
_MAP = str.maketrans({"c": "k", "q": "k", "z": "s", "v": "f", "d": "t", "b": "p", "g": "k"})


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


class Aliases:
    def __init__(self, names: list[str], note_text: str):
        # single-word names only; multi-word names are left to boosting
        self.names = {}
        for n in names:
            # single words without digits ("Camp5" would swallow every "camp")
            if " " not in n and len(n) >= 4 and not any(ch.isdigit() for ch in n):
                k = skeleton(n)
                if len(k) >= 3:
                    self.names.setdefault(k, n)
        self.note_words = set(re.findall(r"[a-z0-9']+", note_text.lower()))

    @lru_cache(maxsize=4096)
    def fix(self, word: str) -> str | None:
        """The brain name this heard word should be, or None."""
        bare = re.sub(r"[^\w']", "", word)
        if len(bare) < 4 or bare.lower() in self.note_words:
            return None
        k = skeleton(bare)
        if len(k) < 3:
            return None
        best = None
        for nk, name in self.names.items():
            if nk[0] != k[0] or name.lower() == bare.lower():
                continue
            d = _distance(k, nk)
            allowed = 0 if len(nk) <= 3 else 1 if len(nk) <= 5 else 2  # short skeletons match too much
            if d <= allowed and (best is None or d < best[0]):
                best = (d, name)
        if not best:
            return None
        # keep the heard word's trailing punctuation: "Toronto?" -> "Trontal?"
        return best[1] + word[len(word.rstrip(".,?!;:")):]
