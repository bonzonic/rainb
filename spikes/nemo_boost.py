"""Spike: Parakeet TDT 0.6B v2 in NeMo (GPU) with and without word boosting of the brain's names.

Measures, on the same clips: which names come out right, whether "uh/um" survive, and
latency per clip. Run with the NeMo venv:

  .venv-nemo\\Scripts\\python spikes\\nemo_boost.py <clips...>
"""
import re
import sys
import time
import wave

import numpy as np
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel

TERMS = ["Trontal", "Uokatsu", "Jalan Telawi", "Camp5", "Khai Wei", "Genki", "Volume Bubbles",
         "Rezolve AI", "Champion Court", "Redis", "hackathon", "bouldering"]
EXPECT = {  # clip-name prefix -> words that must appear
    "trontal": ["trontal"], "trontal2": ["trontal", "redis"], "uokatsu": ["uokatsu", "jalan", "telawi"],
    "hackathon": ["hackathon"], "bouldering": ["bouldering"],
}


def load(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


clips = {p.replace("\\", "/").split("/")[-1]: load(p) for p in sys.argv[1:]}
model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").cuda().eval()


NEGATIVE = {  # must NOT be rewritten into a boosted name
    "neg_toronto": "toronto", "neg_control": "control", "neg_camp": "camping",
}


def set_boost(on: bool, score: float = 1.0, alpha: float = 0.5):
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        # the checkpoint's config predates boosting; add the section explicitly
        cfg.greedy.boosting_tree = {
            "key_phrases_list": TERMS if on else None,
            "context_score": score,
            "depth_scaling": 2.0,
            "use_triton": False,  # no Triton on Windows
        }
        cfg.greedy.boosting_tree_alpha = alpha if on else 0.0
    model.change_decoding_strategy(cfg, verbose=False)


def transcribe(audio):
    with torch.inference_mode():
        out = model.transcribe([audio], batch_size=1, verbose=False)
    h = out[0]
    return h.text if hasattr(h, "text") else str(h)


def score(name, text):
    key = re.sub(r"-.*", "", name.removesuffix(".wav"))
    if key in NEGATIVE:
        return "[1/1]" if NEGATIVE[key] in text.lower() else "[0/1]!"
    want = EXPECT.get(key)
    if not want:
        return ""
    t = text.lower()
    hit = sum(w in t for w in want)
    return f"[{hit}/{len(want)}]"


CONFIGS = [("plain", False, 0, 0), ("score 2 a0.5", True, 2.0, 0.5), ("score 3 a0.5", True, 3.0, 0.5),
           ("score 2 a1.0", True, 2.0, 1.0), ("score 4 a1.0", True, 4.0, 1.0)]
summary = []
for label, on, sc, al in CONFIGS:
    set_boost(on, sc, al)
    transcribe(next(iter(clips.values())))  # warm
    lat, hits, total, negok, negn = [], 0, 0, 0, 0
    print(f"== parakeet tdt v2 (NeMo, GPU) {label}")
    for name, audio in clips.items():
        t0 = time.perf_counter()
        text = transcribe(audio)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        s = score(name, text)
        if s:
            a, b = map(int, s.strip("[]!").split("/"))
            if name.startswith("neg_"):
                negok, negn = negok + a, negn + b
            else:
                hits, total = hits + a, total + b
        print(f"  {ms:5.0f}ms {s:7} {name:24} {text}")
    line = f"{label:14} names {hits}/{total}  untouched negatives {negok}/{negn}  median {np.median(lat):.0f} ms  max {max(lat):.0f} ms"
    summary.append(line)
    print("  -> " + line)
print("\nSUMMARY")
print("\n".join(summary))
