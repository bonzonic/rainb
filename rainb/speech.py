"""Speech service: 16 kHz PCM in, WordEvents out. Knows nothing about hesitation.

Engines by the wearer's English ("Which English do you speak?", chosen per session):

- US / UK: NVIDIA Parakeet TDT 0.6B v2 in NeMo with word boosting of the brain's names
  (~240-350 ms on an RTX 2060); without NeMo or a GPU, the same model via ONNX on CPU.
- Malaysia: Mesolitica's Malaysian Whisper small, primed with the names. On the owner's
  phone recordings: 0% WER vs Parakeet's 12%, ~190 ms per clip. GPU only.

Neither is a streaming model, so we stream the cheap way: an energy VAD watches the
audio, and every time you go quiet for 250 ms the current window is transcribed and only
the new words are emitted. Pauses are exactly where the detector needs fresh words.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

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
TAIL_CHECK_S = 0.8  # after this much quiet, one more pass catches soft last words


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


def _cuda_dlls():
    """pip's nvidia-cublas/cudnn wheels put their DLLs where Windows won't look."""
    import site

    for sp in site.getsitepackages():
        # torch/lib: PyTorch ships its own cuBLAS/cuDNN, which the NeMo environment relies on
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin", "torch/lib"):
            d = Path(sp) / sub
            if d.is_dir():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


# Whisper invents text from silence or noise; these are its classic inventions
HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.", "."}


class Whisper:
    _lock = threading.Lock()
    _model = None

    @classmethod
    def load(cls):
        with cls._lock:
            if cls._model is None:
                _cuda_dlls()
                from faster_whisper import WhisperModel

                cls._model = WhisperModel(config.WHISPER_MODEL, device="cuda", compute_type="float16",
                                          download_root=str(config.ROOT / "models" / "whisper"))
        return cls._model

    @classmethod
    def words(cls, audio: np.ndarray, terms: list[str] | None = None) -> list[tuple[str, float, float]]:
        if len(audio) < SR * 0.2:
            return []
        model = cls.load()
        prompt = "Um, uh, so I was saying."
        if terms:
            prompt += " Names: " + ", ".join(terms[:60]) + "."
        with cls._lock:
            segments, _ = model.transcribe(audio, language="en", beam_size=1, word_timestamps=True,
                                           initial_prompt=prompt, condition_on_previous_text=False,
                                           vad_filter=False)
            out = []
            for seg in segments:
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8:
                    continue
                if seg.text.strip().lower() in HALLUCINATIONS:
                    continue
                for w in seg.words or []:
                    text = w.word.strip()
                    if text:
                        out.append((text, float(w.start), float(w.end)))
        return out


def boost_terms(terms: list[str] | None) -> list[str]:
    """Names worth boosting: capitalised or alphanumeric terms, not plain numbers or common words."""
    out = []
    for t in terms or []:
        if any(ch.isupper() for ch in t) or (any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t)):
            out.append(t.replace("'s", ""))
    return out[:100]


class ParakeetBoosted:
    """Parakeet TDT 0.6B v2 in NeMo on GPU with NVIDIA's word boosting (GPU-PB boosting tree).

    Measured on name clips (spikes/nemo_boost.py): names 4/16 -> 14/16 with context_score 3,
    alpha 0.5; real "Toronto"/"control"/"camping" untouched; ~240 ms per clip on an RTX 2060.
    Score 4 starts inventing names ("Rezolve AIDS"), so stay at 3.
    """
    _lock = threading.Lock()
    _model = None
    _terms: tuple[str, ...] | None = None
    CONTEXT_SCORE, ALPHA = 3.0, 0.5

    @classmethod
    def load(cls):
        with cls._lock:
            if cls._model is None:
                import torch  # noqa: F401
                from nemo.collections.asr.models import ASRModel
                from nemo.utils import logging as nemo_logging

                nemo_logging.setLevel(logging.WARNING)
                if not torch.cuda.is_available():
                    raise RuntimeError("no CUDA GPU")
                cls._model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()
                cls._set_terms(())
                cls._transcribe(np.zeros(SR, dtype=np.float32))  # first call is ~6 s of CUDA warm-up
        return cls._model

    @classmethod
    def _set_terms(cls, terms: tuple[str, ...]):
        """Rebuild the boosting tree (~300-600 ms) only when the brain's names change."""
        if terms == cls._terms:
            return
        from omegaconf import open_dict

        cfg = cls._model.cfg.decoding
        with open_dict(cfg):
            cfg.strategy = "greedy_batch"
            cfg.compute_timestamps = True  # set up front: asking per call rebuilds the decoder
            cfg.greedy.boosting_tree = {"key_phrases_list": list(terms) or None, "context_score": cls.CONTEXT_SCORE,
                                        "depth_scaling": 2.0, "use_triton": False}  # no Triton on Windows
            cfg.greedy.boosting_tree_alpha = cls.ALPHA if terms else 0.0
        cls._model.change_decoding_strategy(cfg, verbose=False)
        cls._terms = terms
        logging.getLogger("rainb").info("word boosting: %d names %s", len(terms), list(terms)[:12])

    @classmethod
    def _transcribe(cls, audio: np.ndarray):
        import torch

        with torch.no_grad():  # not inference_mode: the boosting tree breaks on inference tensors
            return cls._model.transcribe([audio], batch_size=1, timestamps=True, verbose=False)[0]

    @classmethod
    def words(cls, audio: np.ndarray, terms: list[str] | None = None) -> list[tuple[str, float, float]]:
        if len(audio) < SR * 0.2:
            return []
        cls.load()
        with cls._lock:
            cls._set_terms(tuple(boost_terms(terms)))
            h = cls._transcribe(audio)
        stamps = (h.timestamp or {}).get("word", []) if isinstance(h.timestamp, dict) else []
        return [(w["word"], float(w["start"]), float(w["end"])) for w in stamps if w.get("word")]


US, MY = "us", "my"  # "Which English do you speak?": US/UK or Malaysia/Singapore
ACCENT_LABELS = {US: "US / UK English", MY: "Malaysian English"}


class ASR:
    """Speech engines by accent; each listening session picks one.

    us: NVIDIA Parakeet TDT 0.6B v2 + NeMo word boosting on a CUDA GPU (else Parakeet ONNX on CPU).
    my: Mesolitica's Malaysian Whisper small on a CUDA GPU. On the owner's phone recordings:
        0% WER vs Parakeet's 12%. Needs a GPU; without one, Malaysian falls back to US.
    Both engines stay loaded (~0.5 GB + ~2.5 GB fit a 6 GB GPU), so switching is free.
    """
    us_backend: str | None = None  # "nemo" | "parakeet"
    my_ok: bool | None = None
    my_error = ""

    @classmethod
    def load(cls) -> None:
        log = logging.getLogger("rainb")
        if cls.us_backend is None:
            if config.ASR_BACKEND in ("auto", "nemo"):
                try:
                    ParakeetBoosted.load()
                    cls.us_backend = "nemo"
                except Exception as e:
                    if config.ASR_BACKEND == "nemo":
                        raise
                    log.warning("Parakeet + NeMo boosting unavailable (%s); using Parakeet ONNX on CPU", e)
            if cls.us_backend is None:
                Parakeet.load()
                cls.us_backend = "parakeet"
        if cls.my_ok is None:
            try:
                if not Path(config.WHISPER_MODEL).exists() and "/" not in config.WHISPER_MODEL:
                    raise RuntimeError(f"model not found: {config.WHISPER_MODEL}")
                Whisper.load()
                Whisper.words(np.zeros(SR, dtype=np.float32))  # fails fast if CUDA can't run it
                cls.my_ok = True
            except Exception as e:
                cls.my_ok, cls.my_error = False, str(e)[:200]
                log.warning("Malaysian English engine unavailable (%s); Malaysian falls back to US", e)

    @classmethod
    def resolve(cls, accent: str) -> tuple[str, str | None]:
        """(accent actually used, note for the wearer if it differs from what they chose)."""
        cls.load()
        if accent == MY and not cls.my_ok:
            return US, "Malaysian English needs a GPU on the RainB server, so this session uses US / UK English."
        return (accent if accent in (US, MY) else US), None

    @classmethod
    def name(cls, accent: str = US) -> str:
        accent, _ = cls.resolve(accent)
        if accent == MY:
            return "Malaysian Whisper small by Mesolitica (GPU, primed with your names)"
        return {"nemo": "NVIDIA Parakeet TDT 0.6B v2 + NeMo word boosting of your names (GPU)",
                "parakeet": "NVIDIA Parakeet TDT 0.6B v2 (CPU)"}[cls.us_backend]

    @classmethod
    def words(cls, audio: np.ndarray, terms: list[str] | None = None,
              accent: str = US) -> list[tuple[str, float, float]]:
        accent, _ = cls.resolve(accent)
        if accent == MY:
            return Whisper.words(audio, terms)
        if cls.us_backend == "nemo":
            return ParakeetBoosted.words(audio, terms)
        return Parakeet.words(audio)


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
        self.tail_pending = False

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
            # phone audio arrives quiet (speech ~0.01-0.03 rms, vs ~0.09 from a desk mic), so
            # the absolute floor is low and the adaptive noise estimate does the real work
            speaking = rms > max(0.004, self.noise * 3.0)
            if not speaking:
                # follow the floor down fast but up only slowly: with quiet speech, soft syllables
                # fall under the threshold, and a symmetric average would creep up until all
                # speech counted as "noise" and the rest of the sentence never got transcribed
                rate = 0.1 if rms < self.noise else 0.002
                self.noise += rate * (rms - self.noise)
                self.silence_run += FRAME / SR
            else:
                self.silence_run = 0.0
                self.speech_pending = True
                self.tail_pending = True
            self.speaking = speaking
            self.t += FRAME / SR
        due = self.speech_pending and (
            self.silence_run >= SILENCE_TRIGGER_S or self.t - self.last_tx_at >= FORCE_EVERY_S)
        # a soft last word ("…in Bangsar") can stay under the speech threshold, so after any
        # speech do one more pass once it's been quiet a while, to pick up what the VAD missed
        if not due and self.tail_pending and self.silence_run >= TAIL_CHECK_S:
            due = True
            self.tail_pending = False
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
                if _plain(t) == self.emitted_last_text and abs(s - self.emitted_last_start) < 1.0]
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
    return [WordEvent(t, round(s, 3), round(e, 3)) for t, s, e in ASR.words(audio, accent=US)]
