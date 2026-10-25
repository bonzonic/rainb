"""Fix misheard names in finished sentences, using the names in the wearer's own brain.

Parakeet can't be biased towards custom words, so "Trontal" comes out as "Control" and
"hackathon" as "Fakaton". After each sentence (off the hot path) Nemotron rewrites only
words that look like mis-hearings of a term from the brain. The transcript on screen and
conversation memory get the corrected text; detection and recall never wait for this.
"""
from __future__ import annotations

import re
from collections import Counter

from .tokenfactory import TokenFactory

MAX_TERMS = 150
COMMON = set("""the a an i i'm i've my me we our you your he she his her they their it its this that
these those and or but if so of to in on at by for with from is was are were be been being do did
does have has had not no yes what what's which who when where why how there here then than just
also very really one two three let's lets go going get got monday tuesday wednesday thursday friday
saturday sunday january february march april may june july august september october november
december mum dad""".split())

PROMPT = """You fix speech-recognition errors in one sentence someone said. They often say these names and terms:
{terms}

Rewrite the sentence, changing ONLY words that are clearly a mis-hearing of one of those terms
(e.g. "Control" for "Trontal", "Fakaton" for "hackathon"). Keep every other word exactly as given,
including fillers like "uh". If nothing needs fixing, return the sentence unchanged.
Reply with the sentence only."""


def vocabulary(texts: list[str]) -> list[str]:
    """Distinctive terms: capitalised phrases and words with digits, minus common words."""
    counts: Counter[str] = Counter()
    for text in texts:
        for m in re.finditer(r"(?:[A-Z][\w'&-]*\d*|\w*\d\w*)(?:\s+[A-Z][\w'&-]*)*", text):
            phrase = m.group(0).strip()
            words = [w for w in phrase.split() if w.lower() not in COMMON]
            if not words:
                continue
            phrase = " ".join(words)
            if len(phrase) > 2:
                counts[phrase] += 1
        for w in re.findall(r"\b[a-z]{6,}\b", text):  # distinctive lowercase words (hackathon)
            if w not in COMMON:
                counts[w] += 0.5
    return [t for t, _ in counts.most_common(MAX_TERMS)]


def _plain_words(s: str) -> list[str]:
    return re.findall(r"[\w']+", s.lower())


async def correct(tf: TokenFactory, sentence: str, terms: list[str]) -> str:
    if not terms or len(sentence.split()) < 2:
        return sentence
    out = await tf.chat([{"role": "system", "content": PROMPT.format(terms=", ".join(terms))},
                         {"role": "user", "content": sentence}], max_tokens=120)
    out = out.strip().strip('"').strip()
    a, b = _plain_words(sentence), _plain_words(out)
    # guardrail: a fix swaps a few words, it never rewrites the sentence
    if not out or abs(len(a) - len(b)) > max(2, len(a) // 4):
        return sentence
    changed = sum(1 for x, y in zip(a, b) if x != y)
    if changed > max(2, len(a) // 3):
        return sentence
    return out
