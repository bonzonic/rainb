"""Messages passed between the pipeline units. Times are seconds on the audio clock."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

ME = 0  # speaker id of the wearer (diarization speaker #1 / lapel-loudness gate)


@dataclass(frozen=True)
class WordEvent:
    text: str
    start: float
    end: float
    speaker: int = ME

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StuckCandidate:
    id: int
    reason: str  # "explicit" | "structural" | "manual"
    at: float  # audio time the candidate fired
    pause_start: float  # end of the last word before the pause
    context: str  # the wearer's recent words, oldest first
    signals: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Cancel:
    id: int
    reason: str  # "self_recovery"
