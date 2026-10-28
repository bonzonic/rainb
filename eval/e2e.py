"""End-to-end voice tests: synthesise each scenario with Windows TTS, stream it into a running
RainB server over the WebSocket exactly like the phone, and check the answer and latency.

  python eval/e2e.py [--scenarios eval/e2e_scenarios.json] [--url ws://127.0.0.1:8000/ws] [--only id,...]

Runs as the paired owner (your brain, RAINB_OWNER_KEY from .env) with memory writes off.
A scenario passes if an answer containing `expect` arrives within MAX_LATENCY_MS of the
pause (or, for expect=null, no answer arrives at all).
"""
import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rainb import config  # noqa: E402

AUDIO = ROOT / "eval" / "e2e_audio"  # *.wav is git-ignored; regenerated on demand
MAX_LATENCY_MS = 2500

TTS = r"""
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' -and $_.VoiceInfo.Name -like '*{voice}*' } | Select-Object -First 1
if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }
$s.SetOutputToWaveFile('{path}', $fmt)
$s.SpeakSsml('{ssml}')
$s.SetOutputToNull(); $s.Dispose()
"""


def synth(sid: str, say: str, voice: str) -> Path:
    path = AUDIO / f"{sid}-{voice}.wav"
    if path.exists():
        return path
    AUDIO.mkdir(parents=True, exist_ok=True)
    ssml = ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f"{say} <break time='2500ms'/></speak>")
    script = TTS.replace("{path}", str(path)).replace("{ssml}", ssml.replace("'", "''")).replace("{voice}", voice)
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True)
    return path


async def run(url: str, wav: Path, accent: str | None = None) -> dict:
    token = hashlib.sha256(("rainb-owner:" + config.OWNER_KEY).encode()).hexdigest()
    with wave.open(str(wav), "rb") as w:
        pcm = w.readframes(w.getnframes()) + b"\x00\x00" * 16000 * 2
    words, answers, brain = [], [], None
    async with websockets.connect(url, additional_headers={"Cookie": f"rainb_owner={token}"}) as ws:
        start = {"type": "start", "source": "e2e", "remember": False}
        if accent:
            start["accent"] = accent
        await ws.send(json.dumps(start))

        async def reader():
            nonlocal brain
            async for m in ws:
                msg = json.loads(m)
                if msg["type"] == "ready":
                    brain = msg["brain"]
                elif msg["type"] == "engine":
                    print(f"   engine: {msg['name']}" + (f" ({msg['note']})" if msg.get("note") else ""),
                          flush=True) if run.verbose else None
                    run.verbose = False
                elif msg["type"] == "words":
                    words.extend(w["text"] for w in msg["words"])
                elif msg["type"] == "answer":
                    answers.append(msg)

        task = asyncio.create_task(reader())
        t0 = time.perf_counter()
        for i in range(0, len(pcm), 3200):  # 100 ms frames in real time, like the phone
            await ws.send(pcm[i:i + 3200])
            await asyncio.sleep(max(0, i / 32000 - (time.perf_counter() - t0)))
        await asyncio.sleep(3)
        task.cancel()
    return {"brain": brain, "transcript": " ".join(words), "answers": answers}


def judge(sc: dict, res: dict) -> tuple[bool, str]:
    exp = sc["expect"]
    shown = [a for a in res["answers"] if not a.get("recovered")]
    if sc.get("transcript_has") and sc["transcript_has"].lower() not in res["transcript"].lower():
        return False, f"transcript lacks {sc['transcript_has']!r}"
    if exp is None:
        return (not shown), ("silent" if not shown else f"answered {shown[0]['answer']!r}")
    hit = next((a for a in shown if exp.lower() in a["answer"].lower()), None)
    if not hit:
        got = ", ".join(repr(a["answer"]) for a in shown) or "nothing"
        return False, f"got {got}"
    if hit["latency_ms"] > MAX_LATENCY_MS:
        return False, f"{hit['answer']!r} but {hit['latency_ms']} ms"
    return True, f"{hit['answer']!r} {hit['latency_ms']} ms"


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", default=str(ROOT / "eval" / "e2e_scenarios.json"))
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--voices", default="David,Zira")
    p.add_argument("--only")
    p.add_argument("--accent", choices=["us", "my"], help="default: the brain's default (yours: my)")
    a = p.parse_args()
    run.verbose = True
    scenarios = json.loads(Path(a.scenarios).read_text(encoding="utf-8"))
    if a.only:
        keep = set(a.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    rows, passed = [], 0
    for sc in scenarios:
        for voice in a.voices.split(","):
            res = await run(a.url, synth(sc["id"], sc["say"], voice), a.accent)
            if res["brain"] != "me":
                sys.exit(f"server gave brain {res['brain']!r}, not 'me': is RAINB_OWNER_KEY set?")
            ok, detail = judge(sc, res)
            passed += ok
            rows.append((ok, sc["id"], voice, sc["expect"], detail, res["transcript"]))
            print(f"{'PASS' if ok else 'FAIL'}  {sc['id']:22} {voice:6} expect={sc['expect']!r:18} {detail:32} | {res['transcript']}",
                  flush=True)
    lat = sorted(int(r[4].split()[-2]) for r in rows if r[0] and r[3] and r[4].endswith(" ms"))
    print(f"\n{passed}/{len(rows)} passed" + (f"   answer latency median {lat[len(lat) // 2]} ms, max {lat[-1]} ms" if lat else ""))
    (ROOT / "eval" / "e2e_last.json").write_text(json.dumps(
        [dict(zip(("ok", "id", "voice", "expect", "detail", "transcript"), r)) for r in rows], indent=1), encoding="utf-8")
    sys.exit(0 if passed == len(rows) else 1)


asyncio.run(main())
