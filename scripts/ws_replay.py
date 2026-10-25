"""Stream a wav into a running RainB server over the WebSocket, exactly like the phone.

  python scripts/ws_replay.py clip.wav [--owner] [--url ws://127.0.0.1:8000/ws]

--owner pairs as the owner (reads RAINB_OWNER_KEY from .env) so the session uses your brain.
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rainb import config  # noqa: E402


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--owner", action="store_true")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    a = p.parse_args()
    headers = {}
    if a.owner:
        token = hashlib.sha256(("rainb-owner:" + config.OWNER_KEY).encode()).hexdigest()
        headers["Cookie"] = f"rainb_owner={token}"
    with wave.open(a.wav, "rb") as w:
        pcm = w.readframes(w.getnframes()) + b"\x00\x00" * 16000 * 2
    async with websockets.connect(a.url, additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "start", "source": "sample"}))

        async def reader():
            async for m in ws:
                msg = json.loads(m)
                if msg["type"] == "ready":
                    print("brain:", msg["brain"])
                elif msg["type"] == "words":
                    print("words:", " ".join(w["text"] + ("*" if w.get("fixed") else "") for w in msg["words"]))
                elif msg["type"] in ("answer", "silent", "candidate"):
                    print(msg["type"], {k: msg.get(k) for k in ("reason", "answer", "latency_ms", "why")})

        task = asyncio.create_task(reader())
        t0 = time.perf_counter()
        for i in range(0, len(pcm), 3200):  # 100 ms frames in real time
            await ws.send(pcm[i:i + 3200])
            await asyncio.sleep(max(0, (i / 32000) - (time.perf_counter() - t0)))
        await asyncio.sleep(3)
        await ws.send(json.dumps({"type": "stop"}))
        task.cancel()


asyncio.run(main())
