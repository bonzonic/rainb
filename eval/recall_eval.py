"""Answer-accuracy eval for the recall agent against a golden set.

  python eval/recall_eval.py [eval/recall_golden.jsonl] [--model <chat model>]

A case passes if the expected text appears in the answer (expect=null: must stay silent).
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rainb import brains, config  # noqa: E402
from rainb.embedder import Embedder  # noqa: E402
from rainb.events import StuckCandidate  # noqa: E402
from rainb.recall import recall  # noqa: E402
from rainb.tokenfactory import TokenFactory  # noqa: E402


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("golden", nargs="?", default=str(Path(__file__).parent / "recall_golden.jsonl"))
    p.add_argument("--model")
    a = p.parse_args()
    if a.model:
        config.CHAT_MODEL = a.model
    cases = [json.loads(l) for l in Path(a.golden).read_text(encoding="utf-8").splitlines() if l.strip()]
    tf, emb = TokenFactory(), Embedder()
    emb.warm()
    await tf.chat([{"role": "user", "content": "hi"}], max_tokens=1)
    passed, lat = 0, []
    for i, case in enumerate(cases, 1):
        c = StuckCandidate(i, "structural", 0, 0, case["context"])
        t0 = time.perf_counter()
        r = await recall(brains.get(case["brain"]), emb, tf, c)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        got, exp = r.answer, case["expect"]
        ok = (got is None) if exp is None else (got is not None and exp.lower() in got.lower())
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {ms:5.0f}ms {r.timings_ms}  {case['context'][-45:]!r:50} "
              f"-> {got!r}  raw={r.raw}")
    lat.sort()
    print(f"\n{passed}/{len(cases)} passed  model={config.CHAT_MODEL}  "
          f"median={statistics.median(lat):.0f}ms  p95={lat[int(0.95 * (len(lat) - 1))]:.0f}ms")
    await tf.close()


asyncio.run(main())
