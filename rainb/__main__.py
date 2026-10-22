"""rainb CLI.

  python -m rainb sync <folder-or-file> [--brain me]   index notes (read-only on the source)
  python -m rainb demo                                 (re)build the demo persona brain
  python -m rainb ask "<half sentence>" [--brain me]   one recall, as if you stalled there
  python -m rainb replay <file.wav> [--brain me]       stream a wav through the real pipeline
  python -m rainb transcribe <file.wav>                print WordEvents as JSONL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

from . import brains
from .detector import HesitationDetector
from .embedder import Embedder
from .events import Cancel, StuckCandidate
from .recall import recall
from .tokenfactory import TokenFactory


async def cmd_sync(a):
    from .ingest import sync

    print(await sync(brains.get(a.brain), Embedder(), Path(a.path)))


async def cmd_demo(a):
    print(await brains.build_demo(Embedder()))


async def cmd_ask(a):
    tf, emb = TokenFactory(), Embedder()
    c = StuckCandidate(1, "manual", 0.0, 0.0, a.text)
    t0 = time.perf_counter()
    r = await recall(brains.get(a.brain), emb, tf, c)
    ms = (time.perf_counter() - t0) * 1000
    for i, h in enumerate(r.hits, 1):
        print(f"  [{i}] cos={h.cosine:.3f} rrf={h.score:.4f} {h.text[:90]}")
    print(f"answer={r.answer!r} raw={r.raw} {r.timings_ms} total={ms:.0f}ms")
    await tf.close()


def cmd_transcribe(a):
    from .speech import transcribe_file

    for w in transcribe_file(a.wav):
        print(json.dumps(w.to_dict()))


async def cmd_replay(a):
    """Feed a wav in 100 ms frames on a simulated clock, exactly like the phone does."""
    from .speech import Parakeet, SpeechStream

    Parakeet.load()
    tf, emb = TokenFactory(), Embedder()
    emb.warm()
    brain = brains.get(a.brain)
    with wave.open(a.wav, "rb") as w:
        pcm = w.readframes(w.getnframes())
    pcm += b"\x00\x00" * 16000 * 2  # 2 s of trailing silence so a final stall can register
    stream, det = SpeechStream(), HesitationDetector()
    step = 3200  # 100 ms of int16
    tasks = []
    for i in range(0, len(pcm), step):
        due, _ = stream.push(pcm[i:i + step])
        if due:
            audio, start = stream.snapshot()
            t0 = time.perf_counter()
            words = stream.accept(Parakeet.words(audio), start, start + len(audio) / 16000)
            asr_ms = (time.perf_counter() - t0) * 1000
            if words:
                print(f"{stream.t:6.2f}s  asr {asr_ms:4.0f}ms  " + " ".join(w.text for w in words))
            for w in words:
                for ev in det.feed(w):
                    tasks.append(_handle(ev, brain, emb, tf, stream.t))
        for ev in det.tick(stream.t):
            tasks.append(_handle(ev, brain, emb, tf, stream.t))
    await asyncio.gather(*tasks)
    await tf.close()


async def _handle(ev, brain, emb, tf, now):
    if isinstance(ev, Cancel):
        print(f"{now:6.2f}s  CANCEL #{ev.id} ({ev.reason})")
        return
    print(f"{now:6.2f}s  CANDIDATE #{ev.id} {ev.reason} {ev.signals} pause={now - ev.pause_start:.2f}s "
          f"…{ev.context[-60:]!r}")
    t0 = time.perf_counter()
    r = await recall(brain, emb, tf, ev)
    ms = (time.perf_counter() - t0) * 1000
    print(f"         -> #{ev.id} answer={r.answer!r} recall={ms:.0f}ms {r.timings_ms} "
          f"since-pause≈{(now - ev.pause_start) * 1000 + ms:.0f}ms")


def main():
    p = argparse.ArgumentParser(prog="rainb")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync"); s.add_argument("path"); s.add_argument("--brain", default="me")
    sub.add_parser("demo")
    s = sub.add_parser("ask"); s.add_argument("text"); s.add_argument("--brain", default="me")
    s = sub.add_parser("replay"); s.add_argument("wav"); s.add_argument("--brain", default="me")
    s = sub.add_parser("transcribe"); s.add_argument("wav")
    a = p.parse_args()
    if a.cmd == "transcribe":
        cmd_transcribe(a)
    else:
        asyncio.run({"sync": cmd_sync, "demo": cmd_demo, "ask": cmd_ask, "replay": cmd_replay}[a.cmd](a))


if __name__ == "__main__":
    main()
