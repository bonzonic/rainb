"""One listening session = one WebSocket: audio in, transcript + pause meter + answers out."""
from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from fastapi import WebSocket

from . import config
from .brain import Brain
from .correct import correct
from .detector import HesitationDetector, norm
from .embedder import Embedder
from .events import ME, Cancel, StuckCandidate, WordEvent
from .recall import recall
from .speech import ACCENT_LABELS, ASR, MY, US, SpeechStream
from .tokenfactory import TokenFactory

log = logging.getLogger("rainb.session")
UTTERANCE_GAP_S = 1.5


class Session:
    def __init__(self, ws: WebSocket, brain: Brain, emb: Embedder, tf: TokenFactory, remember: bool,
                 accent: str = US):
        self.ws, self.brain, self.emb, self.tf = ws, brain, emb, tf
        self.remember = remember  # write conversation memory into this brain?
        self.accent, self.accent_note = ASR.resolve(accent)
        self.stream = SpeechStream()
        self.det = HesitationDetector()
        self.asr_busy = False
        self.asr_again = False
        self.cancelled: set[int] = set()
        self.tasks: set[asyncio.Task] = set()
        self.utterance: list[WordEvent] = []
        self.transcript: list[WordEvent] = []
        self.last_meter = 0.0
        self.source = "mic"
        self.closed = False
        self._stat_buf: list = []

    def engine_info(self) -> dict:
        return {"type": "engine", "accent": self.accent, "label": ACCENT_LABELS[self.accent],
                "name": ASR.name(self.accent), "note": self.accent_note}

    async def send(self, msg: dict):
        if not self.closed:
            try:
                await self.ws.send_text(json.dumps(msg))
            except Exception:
                self.closed = True

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)

    # -- inputs ---------------------------------------------------------------

    def _audio_stats(self, pcm: bytes):
        """Every ~5 s of sound, log level, clipping and how much energy is above 3.4 kHz.
        Speech normally has plenty up there (s, t, b bursts); a narrowband or mis-rated phone
        stream doesn't. Numbers only: no audio is kept."""
        self._stat_buf.append(np.frombuffer(pcm, dtype="<i2"))
        if sum(len(b) for b in self._stat_buf) < 16000 * 5:
            return
        x = np.concatenate(self._stat_buf).astype(np.float32) / 32768.0
        self._stat_buf = []
        rms = float(np.sqrt(np.mean(x * x)))
        if rms < 0.005:
            return
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / 16000)
        hf = float(spec[freqs > 3400].sum() / max(spec[freqs > 80].sum(), 1e-12))
        log.info("audio: rms %.3f peak %.2f clipped %.2f%% energy>3.4kHz %.1f%%", rms, float(np.abs(x).max()),
                 100 * float(np.mean(np.abs(x) > 0.99)), 100 * hf)

    async def on_audio(self, pcm: bytes):
        if self.source == "mic":
            self._audio_stats(pcm)
        due, level = self.stream.push(pcm)
        if due:
            self._spawn(self._transcribe())
        for c in self.det.tick(self.stream.t):
            await self._candidate(c)
        if self.stream.t - self.last_meter >= 0.1:
            self.last_meter = self.stream.t
            st = self.det.status(self.stream.t)
            await self.send({"type": "meter", "t": round(self.stream.t, 2), "rms": round(level.rms, 4),
                             "speaking": level.speaking, "pause": round(st.pause_s, 2) if not level.speaking else 0,
                             "threshold": round(st.threshold_s, 2), "armed": st.armed})

    async def on_control(self, msg: dict):
        kind = msg.get("type")
        if kind == "start":
            self.source = msg.get("source", "mic")
            if msg.get("remember") is False:  # test runs: don't write anything into memory
                self.remember = False
            if msg.get("accent") in (US, MY):  # the phone's "Which English do you speak?" choice
                self.accent, self.accent_note = ASR.resolve(msg["accent"])
            await self.send(self.engine_info())
        elif kind == "mic":
            log.info("mic: context %s Hz, settings %s, device %r, %s", msg.get("context_rate"),
                     msg.get("settings"), msg.get("label"), (msg.get("ua") or "")[:80])
        elif kind == "stuck":
            await self._candidate(self.det.manual(self.stream.t))
        elif kind == "not_it" and msg.get("moment_id"):
            self.brain.set_outcome(int(msg["moment_id"]), "not_it")
        elif kind == "stop":
            await self.finish()

    # -- speech ---------------------------------------------------------------

    async def _transcribe(self):
        if self.asr_busy:
            self.asr_again = True
            return
        self.asr_busy = True
        try:
            while True:
                self.asr_again = False
                audio, start = self.stream.snapshot()
                end = start + len(audio) / 16000
                raw = await asyncio.to_thread(ASR.words, audio, self.brain.vocabulary(), self.accent)
                words = self.stream.accept(raw, start, end)
                fixed: set[int] = set()
                if words:
                    aliases = self.brain.aliases()
                    for i, w in enumerate(words):
                        before = [x.text for x in self.transcript[-8:]] + [x.text for x in words[:i]]
                        name = aliases.fix(w.text, before[-8:])
                        if name:  # "Toronto" -> "Trontal": a name from your notes that sounds the same
                            log.info("sound-alike: %r -> %r", w.text, name)
                            words[i] = WordEvent(name, w.start, w.end, w.speaker)
                            fixed.add(i)
                    log.info("heard: %s", " ".join(w.text for w in words))
                    await self.send({"type": "words", "words": [
                        {**w.to_dict(), "fixed": i in fixed} for i, w in enumerate(words)]})
                for w in words:
                    self._remember_word(w)
                    for ev in self.det.feed(w):
                        if isinstance(ev, Cancel):
                            self.cancelled.add(ev.id)
                            await self.send({"type": "cancel", "id": ev.id, "reason": ev.reason})
                        else:
                            await self._candidate(ev)
                if not self.asr_again:
                    break
        except Exception:
            log.exception("transcription failed")
        finally:
            self.asr_busy = False

    # -- recall ---------------------------------------------------------------

    async def _candidate(self, c: StuckCandidate):
        await self.send({"type": "candidate", "id": c.id, "reason": c.reason, "signals": list(c.signals),
                         "pause_start": c.pause_start, "context": c.context})
        self._spawn(self._recall(c))

    async def _recall(self, c: StuckCandidate):
        t0 = time.perf_counter()
        try:
            r = await recall(self.brain, self.emb, self.tf, c)
        except Exception as e:
            log.exception("recall failed")
            await self.send({"type": "silent", "id": c.id, "why": f"error: {e}"[:200]})
            return
        recall_ms = (time.perf_counter() - t0) * 1000
        # the audio clock runs at real time, so "now - pause start" is what the wearer felt
        since_pause_ms = round((self.stream.t - c.pause_start) * 1000)
        late = since_pause_ms > config.LATE_AFTER_S * 1000
        recovered = c.id in self.cancelled
        outcome = "answered" if r.ok else "silent"
        if r.ok and recovered:
            outcome = "self_recovered"
        moment_id = self.brain.log_moment(
            created_at=time.time(), reason=c.reason, context=c.context, reaching_for="",
            answer=r.answer, source_chunk_id=r.source.chunk_id if r.source else None, confidence=None,
            latency_ms=since_pause_ms, late=int(late), outcome=outcome)
        if not r.ok:
            self.det.found_nothing(c.id)  # a silent lookup shouldn't block the next real question
            await self.send({"type": "silent", "id": c.id, "moment_id": moment_id,
                             "why": "no matching note", "recall_ms": round(recall_ms)})
            return
        src = r.source
        await self.send({
            "type": "answer", "id": c.id, "moment_id": moment_id, "answer": r.answer,
            "source": {"title": src.title or ("conversation" if src.kind == "utterance" else src.kind),
                       "heading": src.heading, "text": src.text, "kind": src.kind,
                       "created_at": src.created_at},
            "buzz": not late and not recovered, "late": late, "recovered": recovered,
            "latency_ms": since_pause_ms, "recall_ms": round(recall_ms), "timings": r.timings_ms,
        })

    # -- conversation memory --------------------------------------------------

    def _remember_word(self, w: WordEvent):
        self.transcript.append(w)
        if self.utterance and w.start - self.utterance[-1].end > UTTERANCE_GAP_S:
            self._flush_utterance()
        self.utterance.append(w)
        if w.text.rstrip().endswith((".", "?", "!")) and len(self.utterance) >= 4:
            self._flush_utterance()

    def _flush_utterance(self):
        words, self.utterance = self.utterance, []
        if words:
            self._spawn(self._finish_utterance(words))

    async def _finish_utterance(self, words: list[WordEvent]):
        text = " ".join(w.text for w in words).strip()
        fixed = text
        if config.LLM_NAME_FIX:
            try:
                fixed = await correct(self.tf, text, self.brain.vocabulary())
            except Exception:
                log.exception("correction failed")
        if fixed != text:
            await self.send({"type": "correct", "start": words[0].start, "end": words[-1].end, "text": fixed})
        if not self.remember or len([w for w in words if norm(w.text)]) < 4:
            return
        if fixed.rstrip().endswith("?"):
            return  # questions you asked aren't facts to remember (they'd be "found" as answers later)
        speaker = words[0].speaker
        expires = None if speaker == ME else time.time() + config.OTHERS_RETENTION_H * 3600
        vec = self.emb.passage(fixed)
        self.brain.add_memory("utterance", fixed, vec, speaker, expires, heading="conversation")

    async def finish(self):
        self._flush_utterance()
        if self.remember and len(self.transcript) >= 20:
            self._spawn(extract_facts(self.brain, self.emb, self.tf,
                                      " ".join(w.text for w in self.transcript)))

    async def close(self):
        await self.finish()
        self.closed = True
        for t in list(self.tasks):
            if not t.done():
                try:
                    await asyncio.wait_for(asyncio.shield(t), timeout=10)
                except Exception:
                    pass


FACTS_PROMPT = """Extract durable personal facts from this transcript of the wearer talking.
Keep only names, places, dates, numbers, decisions and relationships worth remembering
("Met Daniel; he works at Grab and likes Qdrant"). One fact per line, max 8 lines, no preamble.
If there is nothing worth keeping, reply NONE."""


async def extract_facts(brain: Brain, emb: Embedder, tf: TokenFactory, transcript: str) -> int:
    try:
        text = await tf.chat([{"role": "system", "content": FACTS_PROMPT},
                              {"role": "user", "content": transcript[-6000:]}],
                             max_tokens=300, model=config.FACTS_MODEL)
    except Exception:
        log.exception("fact extraction failed")
        return 0
    facts = [l.strip("-• ").strip() for l in text.splitlines() if l.strip() and l.strip().upper() != "NONE"]
    if facts:
        vecs = await emb.passages(facts)
        for f, v in zip(facts, vecs):
            brain.add_memory("fact", f, v, ME, None, heading="from conversation")
    return len(facts)
