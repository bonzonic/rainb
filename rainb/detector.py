r"""Hesitation detector: a pure state machine from WordEvents to StuckCandidates.

No network, no models, no wall clock. `feed()` takes words as the speech service
finalises them; `tick(now)` is called on every audio frame, because a pause is the
*absence* of events. Rules and thresholds come from docs/notes/RainB - Glossary.md.

    FLUENT --pause/explicit--> SENT --cool-down 20 s--> FLUENT
                                 \--you resume talking--> Cancel (self-recovery)
"""
from __future__ import annotations

import re
import statistics
from collections import deque
from dataclasses import dataclass

from .events import ME, Cancel, StuckCandidate, WordEvent

FILLERS = {"uh", "um", "er", "erm", "ah", "eh", "hmm", "mm", "uhm"}

# Last content word before the pause leaves the phrase hanging.
UNFINISHED = {
    "the", "a", "an", "that", "this", "those", "these", "his", "her", "their", "my", "our", "your",
    "with", "by", "from", "of", "to", "at", "in", "on", "for", "about", "near", "called", "named",
    "is", "was", "are", "were", "and", "or", "like", "what's", "whats", "what", "which", "who",
}

# Explicit retrieval phrases, matched on the last few words.
EXPLICIT = [
    r"what'?s (it|that|the \w+( \w+)?|his|her|their) (called|name)",
    r"what (is|was) (it|that|the \w+( \w+)?) called",
    r"\bcalled again\b",
    r"what'?s (his|her|their|its) name",
    r"what was (his|her|their|its) name",
    r"\bwhat was it\b",
    r"i (forgot|forget|can'?t remember) (the|his|her|its|their|what)",
    r"\bthe thingy\b",
    r"\byou know the\b",
    r"\bthe name of\b",
]
_EXPLICIT_RE = re.compile("|".join(f"(?:{p})" for p in EXPLICIT))

PAUSE_FLOOR_S = 0.6  # never call a gap shorter than this a pause...
BOOSTED_FLOOR_S = 0.4  # ...unless weak signals lower it, but never below this
BOOST_S = 0.15  # each weak signal (filler, repetition) lowers the threshold this much
MAD_K = 3.0  # personal threshold = median gap + k * MAD
WEAK_WINDOW_S = 2.0  # fillers / repetitions count only if this recent
COOLDOWN_S = 20.0
RECOVERY_GRACE_S = 3.0
CONTEXT_WORDS = 25


def norm(text: str) -> str:
    return re.sub(r"[^\w']", "", text.lower())


def ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!"))


@dataclass
class Status:
    """What the UI's pause meter shows."""
    pause_s: float
    threshold_s: float
    armed: bool  # an unfinished phrase is hanging, so the pause will count


class HesitationDetector:
    def __init__(self, me: int = ME):
        self.me = me
        self.words: deque[WordEvent] = deque(maxlen=200)
        self.gaps: deque[float] = deque(maxlen=200)  # wearer's fluent inter-word gaps
        self.next_id = 1
        self.sent: StuckCandidate | None = None  # awaiting self-recovery
        self.cooldown_until = float("-inf")
        self.grace_until = float("-inf")
        self.other_spoke_at = float("-inf")

    # -- inputs ---------------------------------------------------------------

    def feed(self, w: WordEvent) -> list[StuckCandidate | Cancel]:
        if w.speaker != self.me:
            self.other_spoke_at = max(self.other_spoke_at, w.end)
            return []
        out: list[StuckCandidate | Cancel] = []
        if self.words:
            gap = w.start - self.words[-1].end
            if 0 <= gap < PAUSE_FLOOR_S:
                self.gaps.append(gap)
        self.words.append(w)

        token = norm(w.text)
        # recovery = a content word; "uh", "the", "a" (often a misheard "uh") are still stalling
        recovered = token and token not in FILLERS and token not in UNFINISHED
        if self.sent and recovered and w.start > self.sent.pause_start:
            out.append(Cancel(self.sent.id, "self_recovery"))
            self.sent = None
            # you got going again: a short grace period for pauses instead of the full
            # cool-down, so a following "what's it called" can still fire
            self.cooldown_until = float("-inf")
            self.grace_until = w.end + RECOVERY_GRACE_S

        tail = " ".join(norm(x.text) for x in list(self.words)[-8:])
        m = None
        for m in _EXPLICIT_RE.finditer(tail):
            pass
        if m and m.end() == len(tail):  # the phrase was just completed by this word
            c = self._fire("explicit", at=w.end, pause_start=w.end, signals=("retrieval_phrase",))
            if c:
                out.append(c)
        return out

    def manual(self, now: float) -> StuckCandidate:
        """The "I'm stuck" button: always fires, ignores cool-down."""
        c = self._make("manual", now, self.words[-1].end if self.words else now, ())
        self.sent = c
        return c

    def tick(self, now: float) -> list[StuckCandidate]:
        if not self.words or now < self.cooldown_until or now < self.grace_until:
            return []
        last = self.words[-1]
        if self.other_spoke_at > last.end:  # turn handover, not a stall
            return []
        st = self.status(now)
        if st.armed and st.pause_s > st.threshold_s:
            c = self._fire("structural", at=now, pause_start=last.end, signals=self._weak_signals(now))
            return [c] if c else []
        return []

    # -- state ----------------------------------------------------------------

    def threshold(self, now: float) -> float:
        base = PAUSE_FLOOR_S
        if len(self.gaps) >= 10:
            med = statistics.median(self.gaps)
            mad = statistics.median(abs(g - med) for g in self.gaps)
            base = max(base, med + MAD_K * mad)
        boosted = base - BOOST_S * len(self._weak_signals(now))
        return max(BOOSTED_FLOOR_S, boosted)

    def status(self, now: float) -> Status:
        if not self.words:
            return Status(0.0, PAUSE_FLOOR_S, False)
        last = self.words[-1]
        return Status(max(0.0, now - last.end), self.threshold(now), self._unfinished())

    def _unfinished(self) -> bool:
        """Either a hanging word or trailing off into a filler.

        Parakeet punctuates at the end of whatever audio it was given, so "It was called."
        often just means the window ended there: a period after a hanging word doesn't
        finish the phrase. A question mark does (explicit phrases are handled in feed).
        """
        last = self.words[-1]
        tok = norm(last.text)
        if tok in FILLERS:
            return True
        return tok in UNFINISHED and not last.text.rstrip().endswith(("?", "!"))

    def _weak_signals(self, now: float) -> tuple[str, ...]:
        recent = [w for w in self.words if now - w.end <= WEAK_WINDOW_S + PAUSE_FLOOR_S]
        toks = [norm(w.text) for w in recent]
        sig = []
        if any(t in FILLERS for t in toks):
            sig.append("filler")
        content = [t for t in toks if t and t not in FILLERS]
        if any(a == b for a, b in zip(content, content[1:])):
            sig.append("repetition")
        return tuple(sig)

    def _fire(self, reason, at, pause_start, signals) -> StuckCandidate | None:
        if at < self.cooldown_until:
            return None
        c = self._make(reason, at, pause_start, signals)
        self.sent = c
        self.cooldown_until = at + COOLDOWN_S
        return c

    def _make(self, reason, at, pause_start, signals) -> StuckCandidate:
        context = " ".join(w.text for w in list(self.words)[-CONTEXT_WORDS:])
        c = StuckCandidate(self.next_id, reason, at, pause_start, context, tuple(signals))
        self.next_id += 1
        return c
