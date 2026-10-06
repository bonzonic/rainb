# RainB — Session Handoff (6 Oct 2026)

Paste this into a new Claude Code session to continue.

## The hackathon

- **Nebius x NVIDIA Global AI Hackathon** — https://nebiusglobalaihackathon.devpost.com/
- **Track: Personal AI** ("private assistants with persistent memory and tool integration")
- **Deadline: 30 Oct 2026, 10:00 PDT** — internal target: submit by **28 Oct**
- Prizes: $20k / $10k / $6k; track winners get a Jetson Orin Nano; $3k Best Use of Tavily
- Judged on: Technological Implementation, Design, Potential Impact, Quality of Idea (unweighted)
- **Rules that shape the design:**
  - Must run on Nebius **Token Factory or AI Cloud** using **≥1 NVIDIA open model** (Nemotron preferred)
  - **Hosted, working demo URL**
  - Demo video ≤3 min whose **audio explicitly explains Nebius + NVIDIA model usage**
  - Public repo, OSS license (MIT chosen), README, tools feedback, pre-existing-work disclosure
  - Non-sponsor models (e.g. Claude) are **not** banned

## The idea — RainB

"An extension of your brain": when you blank mid-sentence ("we should use that paper, the one with the, uh…"), RainB detects **your own hesitation**, works out what you're reaching for, searches your notes + past conversations, and pushes the answer to your **Android phone** in **< 2 s**.

- **Positioning:** memory products (Omi, Bee→Amazon, Limitless→Meta, MS Recall, Screenpipe) recall *after the fact*. RainB recalls *in the moment*. Novelty is the **trigger** (your hesitation), not transcription.
- **Score (hackathon-idea-evaluator): ~47/60** — Novelty 7, Feasibility 8, Scalability 7, Impact 7, Demo 9, Domain Fit 9.
- History: abandoned **MarketGuard** (crypto manipulation surveillance, 51/60) for lack of motivation → "ambient listener + Claude Code searches PC" (33/60: Claude as the brain sidelined sponsors + too slow) → reshaped onto Nebius/NVIDIA (42) → Personal AI track + diarization (44) → hesitation trigger (47).

## Decisions made

1. **Phone first, Android.** Web app in Android Chrome (Web Push works without install). PC-mic daily-use setup later.
2. **Lapel mic** (wired USB-C lav for demo). Near-field mic makes "is it *me* hesitating?" easy via loudness.
3. **Sources:** files/notes (Obsidian vault + PDFs) + conversation memory in v1. Gmail/Calendar/Drive = phase 2.
4. **Demo:** fictional persona brain by default; judges can upload their own notes; "Play a sample conversation" button runs a pre-recorded clip through the real pipeline (works without mic permission); manual "I'm stuck" button as last-resort fallback.
5. **Retention policy C:** raw audio never stored; other speakers' words deleted after 24 h; your words + extracted facts kept per user setting.
6. **Approach 1 — cheap-to-expensive pipeline** + manual fallback. Rejected: single Nemotron Omni model (chunked, ≥3 s) and manual-only (kills the novelty).
7. **"You" = diarization speaker #1** via a 5 s "say hi" enrolment (Nemotron 3 Diarization orders speakers by arrival) + lapel loudness. No voice-ID model.
8. **No brain picker in the UI** (latest change): server infers the brain — paired phone (owner-key cookie) → your brain; anyone else → demo persona; after upload → their temporary brain. Listen screen shows a status line, e.g. "Using: demo brain · [Use your own notes]".

## Architecture (design sections 1–4)

```
lapel mic → phone (Chrome) ─audio WS─▶ Speech service  [Nebius AI Cloud GPU VM]
                                          Parakeet → words + timestamps
                                          Nemotron 3 Diarization → speaker label
                                        ▼ WordEvent{text,start,end,speaker}
                                       Hesitation detector  (pure Python, no model)
                                        ▼ StuckCandidate / Cancel
                                       Recall agent  [Token Factory]
                                          Nemotron Nano confirm ∥ speculative search
                                          hybrid search: Nemotron 3 Embed + SQLite FTS5
                                          Nemotron Nano grounded 1-line answer
                                        ▼
phone notification ◀──Web Push (VAPID)── Notifier
```

- Token Factory serves LLMs + embeddings but **no ASR** → speech models run on **AI Cloud**.
- **Latency budget ≈ 1.6 s:** pause 0.6 + Nano confirm 0.3 + search 0.1 + Nano answer 0.4 + push 0.2.

**Section 2 — detector + recall (approved)**
- `HesitationDetector.feed(WordEvent)` + `tick(now)` (a pause is the *absence* of events; clock is injected → deterministic tests).
- States: FLUENT → PAUSED → SENT → COOLDOWN (20 s).
- Rules: **explicit** retrieval phrase ("what's it called") fires immediately; **structural** = pause > max(600 ms, your median + 3·MAD) **and** last word unfinished (the/with/by/of…); **boosters** ("uh", repetition) lower the threshold 150 ms each, floor 400 ms, never fire alone.
- Other speaker talking during your pause → suppress (turn handover). You resume talking → `Cancel(id)` (self-recovery).
- Recall: Nano confirm `{stuck, reaching_for, query, confidence}` (drop if < 0.6) **in parallel with** speculative search on raw words → hybrid search (vectors + keyword; keyword matters for proper nouns) → grounded answer `{answer ≤8 words, source_id, confidence}` or null = silence.
- **> 2.5 s since pause → no buzz**, goes to the Moments list instead.
- Notification: answer as title, source + date as body, buttons **[Not it]** (labelled training data) and **[More]**.
- Every candidate logged with outcome → that log is the eval dataset.

**Section 3 — brain store + privacy (approved)**
- **One SQLite file per brain** (`data/brains/<id>.db`: persona / me / judge-<random>) → physical isolation.
- Tables: documents, chunks (+ heading path), chunks_fts, chunks_vec, utterances, facts, moments.
- Utterances + facts are also indexed as chunks → one search path.
- `rainb sync "<vault path>"` on the PC: read-only, hash-based incremental upload, `.rainbignore` to exclude folders. Never writes to the vault/OneDrive.
- Facts extracted by Nano every 5 min + session end; facts survive the 24 h deletion of others' words (deletable individually).
- `expires_at` set at write; hourly job deletes from chunks **+ FTS + vectors** (test checks *search results*, not row counts).
- Persona brain read-only + nightly reset; judge brains deleted after 24 h (≤20 files / 10 MB, md/pdf/txt); owner-key cookie pairing, never in URL.

**Section 4 — frontend, deploy, testing (presented; awaiting final OK after the brain-picker change)**
- Screens: **Listen** (transcript by speaker, **pause meter** that fills/turns amber/flashes, "I'm stuck", brain status line), **Moments** (history + latency + Not it), **Brain** (sources, upload, sync status), **Privacy**.
- Judge landing: "▶ Play a sample conversation" (no permissions) + "Try it live".
- First run: mic permission → pick external mic → notification permission → "say hi" → listening.
- Android: Screen Wake Lock (browser recording stops when screen locks); probably disable browser autoGainControl/noiseSuppression (verify in spike); AudioWorklet → 16 kHz Int16, 100 ms frames over WS.
- Deploy: one Nebius GPU VM, docker compose: **caddy** (TLS) + **app** (FastAPI: WS, detector, recall, brain, push, serves frontend) + **speech** (NeMo, GPU). Secrets in `.env`.
- Local dev without GPU: convert spike recordings to **WordEvent JSONL fixtures** once; develop detector/recall/brain against them.
- **Risk:** URL must stay up through judging (maybe weeks) → GPU VM cost. Check judging length + credits; fallback = smaller GPU or precomputed sample-conversation output.
- Eval: `rainb eval` replays labelled recordings → **precision ≥ 0.8, recall ≥ 0.5, p95 latency ≤ 2 s**; persona golden set for answer accuracy.
- Stack: Python 3.11+, FastAPI, asyncio, NeMo, OpenAI-compatible SDK for Token Factory, SQLite + FTS5 + vector extension, TypeScript + React (Vite), Web Push (VAPID), Caddy.

## Schedule

| Week | Dates | Goal |
|---|---|---|
| 1 | 30 Sep – 6 Oct | Nebius accounts, mic, **3 spikes**, fixtures, **dumb end-to-end path** (mic → push) with latency measured |
| 2 | 7 – 13 Oct | Detector + recall + brain store, eval harness |
| 3 | 14 – 20 Oct | Frontend screens, memory + facts, privacy, persona brain |
| 4 | 21 – 30 Oct | Deploy, precision tuning, video, **submit by 28 Oct** |

**The 3 spikes:** (1) does Parakeet keep "uh/um" + word-timestamp accuracy on lapel audio; (2) Nemotron 3 Diarization streaming (0.32 s config) + does enrolment make you speaker #1; (3) end-to-end latency ≤ 2 s.

## Where things are (on the first PC: `C:\Users\wongn\Documents\…`, *not* the OneDrive Documents tree)

- **Repo:** `Projects\rainb` — origin `https://github.com/bonzonic/rainb.git`, branch `main`. First commit (6 Oct): LICENSE (MIT, bonzonic), README.md stub, .gitignore (ignores recordings/audio, data/, *.db, .env, *.pem), this HANDOFF.md, `docs/notes/`.
- **Obsidian notes:** `Obsidian Vault\rainb\` (copies as of 6 Oct in this repo's `docs/notes/` — the vault is the source of truth; re-copy if you edit there) — `kanban.md` (~50 cards, Todo / In Progress / Done, tagged), `RainB - Architecture.md`, `RainB - Glossary.md` (hesitation signals → computable features, starting thresholds), empty `architecture/` and `plans/`.
- Reference project for style: `Projects\marketguard` + `Obsidian Vault\marketguard\`.

## Next steps

1. Final OK on design section 4 (with brain picker removed).
2. Write the spec → `docs/superpowers/specs/2026-09-30-rainb-design.md` in the repo, self-review it, user reviews it. Ask before committing/pushing.
3. Then `superpowers:writing-plans` → implementation plan.
4. Start week-1 work (now behind schedule): Nebius account + Token Factory keys (confirm Nemotron Nano + Nemotron 3 Embed via list-models), GPU VM pricing/credits, buy lapel mic, run the 3 spikes, record + label 30 min of own speech.

## How to work with me

- Using **superpowers:brainstorming** (architectural path) → spec → **writing-plans**. Questions one at a time, design approved section by section.
- **Learning output style:** include `★ Insight` blocks; leave meaningful 5–10 line decisions for me to write. Earmarked for me: the **structural firing rule** in the hesitation detector (pause threshold, unfinished-word list, booster amounts).
- Use **hackathon-idea-evaluator** for any re-scoring.
- Commit/push only when I ask.
