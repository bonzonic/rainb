"""Compare speech engines on the owner's recorded test sentences (recordings/testset).

  .venv-nemo\\Scripts\\python eval\\asr_testset.py [--engines parakeet,mal-medium,mal-small,whisper-turbo]

Per engine: word error rate (fillers excluded), names/key words right, whether "uh/um" were
kept, and seconds per clip (a clip is about one streaming chunk). Stop the live server first
so the engines aren't fighting it for the 6 GB GPU.
"""
import argparse
import json
import re
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rainb import brains, config, speech  # noqa: E402

CLIPS = ROOT / "recordings" / "testset"
FILLERS = {"uh", "um", "er", "erm", "ah", "eh", "hmm", "mm"}
KEY = ["trontal", "khai", "wei", "uokatsu", "bouldering", "camp5", "badminton", "champion", "court", "ss2",
       "tan", "bangsar", "redis", "vincent", "rezolve", "singapore", "genki", "mid", "valley", "only", "human",
       "caleb", "thirtieth", "october"]
NUMS = {"30th": "thirtieth", "camp": "camp", "5": "five"}
ENGINES = {
    "parakeet": ("Parakeet TDT 0.6B v2 + NeMo boosting", None),
    "mal-medium": ("Malaysian Whisper medium-v2 + names", ROOT / "models" / "malaysian-whisper-medium-ct2"),
    "mal-small": ("Malaysian Whisper small-v3 + names", ROOT / "models" / "malaysian-whisper-small-ct2"),
    "whisper-turbo": ("Whisper large-v3-turbo + names", "large-v3-turbo"),
}


def words(text: str) -> list[str]:
    t = text.lower().replace("camp 5", "camp5").replace("camp five", "camp5").replace("ss 2", "ss2")
    out = []
    for w in re.findall(r"[a-z0-9']+", t):
        w = NUMS.get(w, w)
        out.append("uh" if re.fullmatch(r"u+h+|u+m+|e+r+m*|a+h+|e+h+|h+m+|m+", w) else w)
    return out


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def load(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", default="parakeet,mal-medium,mal-small")
    a = p.parse_args()
    refs = json.loads((CLIPS / "refs.json").read_text(encoding="utf-8"))
    clips = [(name, load(CLIPS / name), r["text"]) for name, r in sorted(refs.items())]
    terms = brains.get("me").vocabulary()
    summary = []
    for key in a.engines.split(","):
        label, model = ENGINES[key]
        if key == "parakeet":
            fn = lambda x: speech.ParakeetBoosted.words(x, terms)  # noqa: E731
        else:
            speech.Whisper._model = None
            config.WHISPER_MODEL = str(model)
            fn = lambda x: speech.Whisper.words(x, terms)  # noqa: E731
        fn(np.zeros(16000, dtype=np.float32))  # warm up
        print(f"\n== {label}")
        wers, lats, key_hit, key_n, fill_kept, fill_n = [], [], 0, 0, 0, 0
        for name, audio, ref in clips:
            t0 = time.perf_counter()
            hyp_text = " ".join(w for w, _, _ in fn(audio))
            lat = time.perf_counter() - t0
            r, h = words(ref), words(hyp_text)
            wers.append(wer([w for w in r if w != "uh"], [w for w in h if w != "uh"]))
            lats.append(lat)
            for k in KEY:
                if k in r:
                    key_n += 1
                    key_hit += k in h
            if "uh" in r:
                fill_n += 1
                fill_kept += "uh" in h
            print(f"  {lat * 1000:5.0f}ms wer {wers[-1]:.2f}  {name}  {hyp_text}", flush=True)
        line = (f"{label:40} WER {statistics.mean(wers):.2f}  key words {key_hit}/{key_n}  "
                f"'uh' kept {fill_kept}/{fill_n}  median {statistics.median(lats) * 1000:.0f} ms  "
                f"max {max(lats) * 1000:.0f} ms")
        summary.append(line)
        print("  -> " + line, flush=True)
        if key == "parakeet":  # free the GPU for the Whisper models
            import torch

            speech.ParakeetBoosted._model = None
            torch.cuda.empty_cache()
    print("\nSUMMARY (references:", ", ".join(f"{n}: {r!r}" for n, _, r in clips[:2]), "...)")
    print("\n".join(summary))


main()
