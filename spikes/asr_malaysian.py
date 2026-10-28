"""Parakeet + boosting vs Whisper turbo vs Malaysian Whisper (mesolitica) on the same clips.

  .venv-nemo\\Scripts\\python spikes\\asr_malaysian.py clip.wav ...
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rainb import brains, config, speech  # noqa: E402

TERMS = brains.get("me").vocabulary()


def load(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


clips = {Path(p).name: load(p) for p in sys.argv[1:]}


def run(label, fn):
    fn(np.zeros(16000, dtype=np.float32))  # warm
    print(f"== {label}")
    for name, a in clips.items():
        t0 = time.perf_counter()
        text = " ".join(w for w, _, _ in fn(a))
        print(f"  {(time.perf_counter() - t0) * 1000:5.0f}ms  {name:24} {text}", flush=True)


run("Parakeet TDT 0.6B v2 + NeMo boosting", lambda a: speech.ParakeetBoosted.words(a, TERMS))
for label, model in [("Whisper large-v3-turbo + names", "large-v3-turbo"),
                     ("Malaysian Whisper turbo (mesolitica) + names", str(config.ROOT / "models" / "malaysian-whisper-turbo-ct2"))]:
    speech.Whisper._model = None
    config.WHISPER_MODEL = model
    run(label, lambda a: speech.Whisper.words(a, TERMS))
