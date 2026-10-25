"""Compare Parakeet variants (int8 vs fp32, v2 vs v3) on the same clips: text + latency.

  python spikes/asr_compare.py clip1.wav clip2.wav ...
"""
import sys
import time
import wave

import numpy as np
import onnx_asr

VARIANTS = [
    ("v2 int8", "nemo-parakeet-tdt-0.6b-v2", "models/parakeet-tdt-0.6b-v2", "int8"),
    ("v2 fp32", "nemo-parakeet-tdt-0.6b-v2", "models/parakeet-tdt-0.6b-v2", None),
    ("v3 int8", "nemo-parakeet-tdt-0.6b-v3", "models/parakeet-tdt-0.6b-v3", "int8"),
]


def load(path):
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


clips = {p: load(p) for p in sys.argv[1:]}
for name, model_id, path, quant in VARIANTS:
    try:
        m = onnx_asr.load_model(model_id, path, quantization=quant)
    except Exception as e:
        print(f"== {name}: unavailable ({str(e)[:80]})")
        continue
    m.recognize(next(iter(clips.values())), sample_rate=16000)  # warm
    print(f"== {name}")
    for p, audio in clips.items():
        t0 = time.perf_counter()
        text = m.recognize(audio, sample_rate=16000)
        print(f"  {(time.perf_counter() - t0) * 1000:5.0f}ms  {p.split(chr(92))[-1]:22} {text}")
