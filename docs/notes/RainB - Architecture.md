# RainB — Architecture

> An extension of your brain: the moment you blank mid-sentence, the answer is already on your phone.
> Nebius x NVIDIA Global AI Hackathon — **Personal AI** track.
> Deadline: **30 Oct 2026, 10:00 PDT**

See also: [[RainB - Glossary]]

---

## The pitch

Everyone knows the feeling: *"we should use that paper, the one with the, uh…"* — and the word is gone.

Memory wearables (Omi, Bee → Amazon, Limitless → Meta) and screen recall (Microsoft Recall, Screenpipe) all record your life — but recall is **after the fact**. You open an app later and ask. By then the conversation has moved on.

RainB works **in the moment**. It notices *you* hesitating, works out what you're reaching for, searches your own notes and past conversations, and buzzes your phone with the answer in under two seconds.

**Not novel as a category. Novel in the trigger.** Everyone else competes on transcription (solved). RainB competes on *knowing when you're stuck* — and your own hesitation is the signal, so it doesn't have to guess.

---

## Scope

| Piece | Status | Why |
|---|---|---|
| **Files + notes** (vault, PDFs) | v1 | Real content from day one |
| **Conversation memory** | v1 | The track's "persistent memory" requirement; makes "extension of your brain" literally true |
| **Hesitation trigger** | v1 | The entire novelty |
| **Manual "I'm stuck" button** | v1 | Live demo can never fully fail |
| Connected apps (Gmail, Calendar, Drive) | phase 2 | OAuth per app eats time |
| Claude Code "dig deeper" | phase 2 | Local companion on the PC; slow path only |
| Native Android app | phase 2 | Only needed for pocket / screen-locked recording |

---

## System diagram

```
 Android phone (Chrome)                     Nebius AI Cloud — 1 GPU VM (HTTPS)
┌───────────────────────┐  audio (WS)  ┌──────────────────────────────────────────┐
│ 1. Capture client     │─────────────▶│ 2. Speech service                        │
│  lapel mic → 16 kHz   │              │  Parakeet → words + timestamps           │
│  live transcript      │              │  Nemotron 3 Diarization → speaker label  │
│  fallback "stuck" btn │              └──────────────┬───────────────────────────┘
│  privacy screen       │                  WordEvent{text,start,end,speaker}
│                       │              ┌──────────────▼───────────────────────────┐
│                       │              │ 3. Hesitation detector  ← NO MODEL. Free │
│                       │              │  your speech only → StuckCandidate       │
│                       │              └──────────────┬───────────────────────────┘
│                       │              ┌──────────────▼───────────────────────────┐
│                       │              │ 4. Recall agent        ── Token Factory ─┤
│                       │              │  Nano: confirm + write query (JSON)      │
│                       │              │  Embed: search  ◀──▶ 5. Brain store      │
│                       │  web push    │  Nano: 1-line answer + source            │
│ 6. Notification ◀─────│◀─────────────│ 6. Notifier (Web Push, VAPID)            │
└───────────────────────┘              └──────────────────────────────────────────┘
```

| # | Unit | One job | Knows nothing about |
|---|---|---|---|
| 1 | Capture client | Stream mic audio; show transcript; fallback button; privacy settings | Models |
| 2 | Speech service | Audio → `WordEvent`s | Hesitation |
| 3 | Hesitation detector | `WordEvent` stream → `StuckCandidate`s. **Pure function** | Network, models |
| 4 | Recall agent | Confirm, search, compose answer. Only caller of Token Factory | Audio |
| 5 | Brain store | Doc chunks + conversation memory, per-brain namespaces, retention | Speech |
| 6 | Notifier | Web Push to Android; in-page banner fallback | Content |

### Governing principle

**Cheap, dumb checks first; expensive, smart ones last.** Pause timing is free, runs on every word, and produces *candidates*. Nemotron Nano only sees candidates. Search and answer composition only run on confirmed moments.

This is the answer to *"why Token Factory?"* — an open, fast, cheap model is the right tool for a check that runs dozens of times an hour, and it can be fine-tuned on your own "interrupt / don't interrupt" examples. Frontier-API pricing on every sentence would be absurd.

### Division of labour

| | Hesitation detector | Nemotron Nano |
|---|---|---|
| Cost | Free | Fractions of a cent per call |
| Speed | Per word | ~300 ms |
| Produces | **Candidates** | **Confirmations + queries** |
| Understands meaning? | No | Yes |
| Can tell "uh" (thinking) from "uh" (stuck)? | Partly (timing, syntax) | Yes (context) |

### Latency budget

| Stage | Target |
|---|---|
| Pause detected | ~600 ms (the threshold itself) |
| Nano confirm + query | ~300 ms |
| Embed search | ~100 ms |
| Nano answer | ~400 ms |
| Web push delivery | ~200 ms |
| **Total** | **≈ 1.6 s** — must stay under 2 s |

### Knowing who "you" are

Nemotron 3 Diarization orders speakers **by first arrival**. Every session starts with a 5-second "say hi" enrolment, so the user is always speaker #1. Lapel-mic loudness is a second check — your voice is centimetres away, everyone else's is metres. No separate voice-ID model needed.

---

## NVIDIA + Nebius usage

| Stage | Model | Runs on |
|---|---|---|
| Speech → text + word timestamps | **Parakeet** (TDT) | Nebius AI Cloud GPU |
| Who is speaking | **Nemotron 3 Diarization** (released 23 Sep 2026, streaming, ≤8 speakers) | Nebius AI Cloud GPU |
| Confirm + query rewrite | **Nemotron Nano** | Token Factory |
| Search | **Nemotron 3 Embed** | Token Factory |
| Answer | **Nemotron Nano** | Token Factory |

> Token Factory docs list LLM + embeddings, **no ASR** — so speech models live on AI Cloud. Both platforms satisfy the rules. Confirm model availability with the list-models API once the account exists.

---

## Stack (proposed — pending design review)

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.11+, FastAPI, asyncio | Same muscle memory as MarketGuard |
| Speech | NVIDIA NeMo (Parakeet + Diarization) | Streaming inference on the VM |
| LLM / embeddings | Token Factory via OpenAI-compatible SDK | Structured JSON output for the confirm step |
| Storage | SQLite + vector extension | One file per brain; dozens of writes/hour |
| Frontend | TypeScript + React (Vite), installable web app | Served by the backend, same origin |
| Push | Web Push (VAPID) | Android Chrome supports it without install |
| Transport | WebSocket, phone ↔ backend | Audio up, transcript + events down |
| TLS | Caddy | Mic **and** push both require HTTPS |

---

## Data

### Brains

Each brain is an isolated namespace: **demo persona** (default), **you**, **judge upload**.

- **Documents** — Obsidian markdown + PDFs, chunked and embedded
- **Conversation memory** — transcripts + extracted facts ("met Daniel, works at Grab, into RAG")

### Retention (decision: policy C)

| What | Kept |
|---|---|
| Raw audio | **Never stored** |
| Other speakers' words | 24 h, then deleted |
| Your own words | Longer (user-configurable) |
| Extracted facts | Longer (user-configurable) |

Visible privacy controls screen — defensible pitch *and* points for Design.

---

## Demo strategy

**The hosted demo defaults to a demo persona, with "use your own notes" as an option.**

A judge opening the URL at 3am must get the wow moment with zero setup: preloaded persona brain + on-screen *"Try saying: 'we should use that paper, the one about, uh…'"*.

Second layer of reliability: a **"play a sample conversation"** button that streams a pre-recorded clip through the *real* pipeline — works even if the judge won't grant mic permission. Same lesson as MarketGuard's replay-by-default.

Third layer: the manual **"I'm stuck"** button.

### The money shot

You stall mid-sentence on camera — and the phone on the table **lights up with the answer before you've finished saying "uh"**. Show the lock-screen notification, not a dashboard.

---

## Deployment (proposed)

⚠️ Speech models need a GPU and a **long-running** streaming process — cannot be serverless.

| Piece | Where |
|---|---|
| Speech + backend + frontend | 1 Nebius AI Cloud GPU VM, behind Caddy |
| Nano + Embed | Token Factory (serverless) |

Stop the VM when not in use — GPU hours are the main cost.

---

## Submission requirements

- [ ] Working project on Token Factory / AI Cloud using ≥1 NVIDIA open model
- [ ] Track selection — **Personal AI**
- [ ] Written description: what it does, why
- [ ] **Working demo URL** — hosted, live
- [ ] **Demo video ≤3 min with audio explicitly explaining Nebius + NVIDIA model usage**
- [ ] **Public repo with open-source license** (Apache 2.0 / MIT / MPL 2.0) — MIT, in root
- [ ] README with setup + execution instructions
- [ ] Feedback on Nebius + NVIDIA tools (separate prize)
- [ ] Pre-existing work disclosure — none; new repo started 29 Sep 2026

---

## Open risks

| Risk | Mitigation |
|---|---|
| **False triggers** — people say "uh" constantly | Pause alone never fires; needs pause + unfinished phrase, or an explicit retrieval phrase. **Silent is better than wrong.** |
| **ASR drops fillers** — many ASR models normalise "uh/um" away | Spike first. If dropped, the filler still shows up as a *timestamp gap*, which is what we detect anyway. |
| Latency > 2 s kills the magic | Measure the budget end-to-end in week 1 before building features |
| Recording other people | Own-mic only, visible indicator, 24 h retention for others, no raw audio |
| Browser stops recording when screen locks | Demo with screen on; native app is phase 2 |
| GPU VM cost | Stop when idle; check credits |
| Judge denies mic permission | "Play sample conversation" through the real pipeline |

---

## Scoring (hackathon-idea-evaluator)

**~47/60 (estimate)** — Novelty 7, Feasibility 8, Scalability 7, Impact 7, Demo-ability 9, Domain Fit 9.

Evolved from "ambient listener → Claude Code searches PC" (33/60 as stated: Claude as the brain sidelined the sponsors and put the slowest step where the wow moment should be) → reshaped Nebius/NVIDIA fast path (42) → Personal AI track + diarization (44) → hesitation trigger (47). Chosen over MarketGuard (51/60) on motivation.
