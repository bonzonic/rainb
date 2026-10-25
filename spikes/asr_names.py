"""Which speech model gets the wearer's own names (Trontal, Uokatsu, Camp5) right?

Compares Parakeet (no biasing possible) with Whisper on GPU, plain and primed with the
brain's vocabulary via initial_prompt / hotwords. Prints text + latency per clip.

  python spikes/asr_names.py clip1.wav ...
"""
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

# make the pip-installed CUDA DLLs (cublas, cudnn) visible to CTranslate2 on Windows
import site
for sp in site.getsitepackages():
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"):
        d = Path(sp) / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
TERMS = "Trontal, Uokatsu, Jalan Telawi, Camp5, Khai Wei, Genki, Volume Bubbles, hackathon, bouldering"
PROMPT = f"Um, uh, so I was saying. Names: {TERMS}."


def load(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


clips = {Path(p).name: load(p) for p in sys.argv[1:]}


def run(name, fn):
    fn(next(iter(clips.values())))  # warm
    print(f"== {name}")
    for c, a in clips.items():
        t0 = time.perf_counter()
        text = fn(a)
        print(f"  {(time.perf_counter() - t0) * 1000:5.0f}ms  {c:24} {text}")


import onnx_asr  # noqa: E402

pk = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2", "models/parakeet-tdt-0.6b-v2", quantization="int8")
run("parakeet v2 int8 (cpu)", lambda a: pk.recognize(a, sample_rate=16000))

from faster_whisper import WhisperModel  # noqa: E402

for size in ("large-v3-turbo", "small.en"):
    wm = WhisperModel(size, device="cuda", compute_type="float16", download_root="models/whisper")
    tx = lambda a, **kw: " ".join(s.text.strip() for s in wm.transcribe(a, language="en", beam_size=1, vad_filter=False, **kw)[0])
    run(f"whisper {size} (gpu) plain", lambda a: tx(a))
    run(f"whisper {size} (gpu) + prompt", lambda a: tx(a, initial_prompt=PROMPT))
    run(f"whisper {size} (gpu) + hotwords", lambda a: tx(a, hotwords=TERMS))
