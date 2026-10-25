"""Speech service: 16 kHz PCM in, WordEvents out. Knows nothing about hesitation.

Parakeet TDT 0.6B (NVIDIA) runs through ONNX Runtime on CPU, so no GPU is required.
Parakeet is not a streaming model, so we stream it the cheap way: an energy VAD watches
the audio, and every time you go quiet for 250 ms the current window is transcribed and
only the new words are emitted. Pauses are exactly where the detector needs fresh words.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import config
from .events import ME, WordEvent

SR = 16000
FRAME = 320  # 20 ms
TOKEN_S = 0.08  # Parakeet encoder frame
SILENCE_TRIGGER_S = 0.25
FORCE_EVERY_S = 3.0
MAX_WINDOW_S = 10.0
CONTEXT_S = 4.0  # already-emitted audio kept in front of new speech


def _plain(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


class Parakeet:
    _lock = threading.Lock()
    _model = None

    @classmethod
    def load(cls):
        with cls._lock:
            if cls._model is None:
                import onnx_asr

                cls._model = onnx_asr.load_model(
                    "nemo-parakeet-tdt-0.6b-v2", str(config.ASR_MODEL_DIR), quantization="int8"
                ).with_timestamps()
        return cls._model

    @classmethod
    def words(cls, audio: np.ndarray) -> list[tuple[str, float, float]]:
        """Transcribe float32 mono 16 kHz audio -> [(word, start, end)] relative to its start."""
        if len(audio) < SR * 0.2:
            return []
        model = cls.load()
        with cls._lock:  # one inference at a time; the box is small
            res = model.recognize(audio, sample_rate=SR)
        return group_tokens(res.tokens, res.timestamps, len(audio) / SR)


def group_tokens(tokens, stamps, duration: float) -> list[tuple[str, float, float]]:
    words: list[list] = []  # [text, start, last_token_ts]
    for tok, ts in zip(tokens, stamps):
        if tok.startswith(" ") or not words:
            if tok.strip() and all(not ch.isalnum() for ch in tok.strip()) and words:
                words[-1][0] += tok.strip()  # punctuation token after a space
                continue
            words.append([tok.strip(), float(ts), float(ts)])
        else:
            words[-1][0] += tok
            words[-1][2] = float(ts)
    out = []
    for i, (text, start, last) in enumerate(words):
        nxt = words[i + 1][1] if i + 1 < len(words) else duration
        end = min(last + 2.5 * TOKEN_S, max(nxt - 0.02, last + TOKEN_S), duration)
        if text:
            out.append((text, start, end))
    return out


@dataclass
class Level:
    rms: float
    speaking: bool


class SpeechStream:
    """Per-connection audio window + VAD. Call push() with PCM; run transcribe() when it says so."""

    def __init__(self, speaker: int = ME):
        self.speaker = speaker
        self.window = np.zeros(0, dtype=np.float32)
        self.window_start = 0.0
        self.t = 0.0  # audio clock: seconds received
        self.emitted_until = 0.0
        self.emitted_last_start = -1.0
        self.emitted_last_text = ""
        self.noise = 0.003
        self.silence_run = 0.0
        self.speech_pending = False
        self.last_tx_at = 0.0
        self._carry = np.zeros(0, dtype=np.float32)
        self.speaking = False

    def push(self, pcm16: bytes) -> tuple[bool, Level]:
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.window = np.concatenate([self.window, x])
        x = np.concatenate([self._carry, x])
        n = len(x) // FRAME * FRAME
        self._carry = x[n:]
        rms_max = 0.0
        for f in x[:n].reshape(-1, FRAME):
            rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
            rms_max = max(rms_max, rms)
            speaking = rms > max(0.012, self.noise * 3.5)
            if not speaking:  # track the noise floor only in quiet frames
                self.noise = 0.95 * self.noise + 0.05 * rms
                self.silence_run += FRAME / SR
            else:
                self.silence_run = 0.0
                self.speech_pending = True
            self.speaking = speaking
            self.t += FRAME / SR
        due = self.speech_pending and (
            self.silence_run >= SILENCE_TRIGGER_S or self.t - self.last_tx_at >= FORCE_EVERY_S)
        return due, Level(rms_max, self.speaking)

    def snapshot(self) -> tuple[np.ndarray, float]:
        self.speech_pending = False
        self.last_tx_at = self.t
        return self.window.copy(), self.window_start

    def accept(self, words: list[tuple[str, float, float]], window_start: float,
               final_until: float) -> list[WordEvent]:
        """Keep words not yet emitted. Words ending within 0.15 s of the snapshot end may still
        be growing (forced mid-speech transcription), so they wait for the next pass."""
        hyp = [(text, s + window_start, e + window_start) for text, s, e in words]
        start_at = self._new_words_from(hyp)
        out = []
        for text, s, e in hyp[start_at:]:
            if e > final_until - 0.15 and self.silence_run < SILENCE_TRIGGER_S:
                break
            out.append(WordEvent(text, round(s, 3), round(e, 3), self.speaker))
        if out:
            self.emitted_until = out[-1].end
            self.emitted_last_start = out[-1].start
            self.emitted_last_text = _plain(out[-1].text)
        self._trim()
        return out

    def _new_words_from(self, hyp: list[tuple[str, float, float]]) -> int:
        """Index of the first word in this pass that hasn't been emitted yet.

        The window keeps a few seconds of already-emitted audio as context, so each pass
        re-transcribes those words with slightly shifted stamps. Align on the last emitted
        word by text near its old position; if the re-run heard it differently, fall back
        to timing.
        """
        if self.emitted_last_start < 0:
            return 0
        same = [i for i, (t, s, _) in enumerate(hyp)
                if _plain(t) == self.emitted_last_text and abs(s - self.emitted_last_start) < 0.6]
        if same:
            return same[-1] + 1
        for i, (_, s, e) in enumerate(hyp):
            if (s + e) / 2 >= self.emitted_until and s > self.emitted_last_start + 0.16:
                return i
        return len(hyp)

    def _trim(self):
        # keep CONTEXT_S of already-emitted audio: Parakeet hears "the place" far better
        # with the sentence before it than as an isolated two-word clip
        keep_from = self.emitted_until - CONTEXT_S
        if len(self.window) / SR > MAX_WINDOW_S:
            keep_from = max(keep_from, self.window_start + len(self.window) / SR - MAX_WINDOW_S)
        cut = keep_from - self.window_start
        if cut > 0:
            cut_n = min(len(self.window), int(cut * SR))
            self.window = self.window[cut_n:]
            self.window_start += cut_n / SR


def transcribe_file(path: str) -> list[WordEvent]:
    """Offline: whole file -> WordEvents (for fixtures and eval)."""
    import wave

    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, "need 16 kHz mono int16"
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return [WordEvent(t, round(s, 3), round(e, 3)) for t, s, e in Parakeet.words(audio)]
