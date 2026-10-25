"""One listening session = one WebSocket: audio in, transcript + pause meter + answers out."""
from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket

from . import config
from .brain import Brain
from .correct import correct
from .detector import HesitationDetector, norm
from .embedder import Embedder
from .events import ME, Cancel, StuckCandidate, WordEvent
from .recall import recall
from .speech import Parakeet, SpeechStream
from .tokenfactory import TokenFactory

log = logging.getLogger("rainb.session")
UTTERANCE_GAP_S = 1.5


class Session:
    def __init__(self, ws: WebSocket, brain: Brain, emb: Embedder, tf: TokenFactory, remember: bool):
        self.ws, self.brain, self.emb, self.tf = ws, brain, emb, tf
        self.remember = remember  # write conversation memory into this brain?
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

    async def on_audio(self, pcm: bytes):
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
                raw = await asyncio.to_thread(Parakeet.words, audio)
                words = self.stream.accept(raw, start, end)
                if words:
                    await self.send({"type": "words", "words": [w.to_dict() for w in words]})
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
        try:
            fixed = await correct(self.tf, text, self.brain.vocabulary())
        except Exception:
            log.exception("correction failed")
            fixed = text
        if fixed != text:
            await self.send({"type": "correct", "start": words[0].start, "end": words[-1].end, "text": fixed})
        if not self.remember or len([w for w in words if norm(w.text)]) < 4:
            return
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
