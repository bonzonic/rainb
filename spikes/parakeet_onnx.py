"""Spike 1: does Parakeet (ONNX, CPU) keep "uh/um", and are word timestamps usable?

Usage: .venv\\Scripts\\python spikes\\parakeet_onnx.py soundtrack\\test1\\october.wav ...
"""
import sys
import time

import onnx_asr

MODEL = "nemo-parakeet-tdt-0.6b-v2"

t0 = time.perf_counter()
model = onnx_asr.load_model(MODEL, "models/parakeet-tdt-0.6b-v2", quantization="int8").with_timestamps()
print(f"loaded {MODEL} in {time.perf_counter() - t0:.1f}s")

for path in sys.argv[1:]:
    t0 = time.perf_counter()
    res = model.recognize(path)
    dt = time.perf_counter() - t0
    print(f"\n== {path}  ({dt * 1000:.0f} ms)")
    print(res.text)
    # tokens are sub-words; a leading space marks a word start
    line = []
    for tok, ts in zip(res.tokens, res.timestamps):
        line.append(f"{ts:.2f}:{tok!r}")
    print(" ".join(line))
