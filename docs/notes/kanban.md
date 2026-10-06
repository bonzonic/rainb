---

kanban-plugin: board

---

## Todo

- [ ] **Nebius account + Token Factory keys** — confirm Nemotron Nano + Nemotron 3 Embed availability via the list-models API. #llm #blocker
- [ ] **Nebius AI Cloud GPU VM** — which GPU, cost/hour, credits. Speech models can't run on Token Factory (no ASR endpoints). #deploy #blocker
- [ ] **Buy a lapel mic** — wired USB-C lavalier is enough for the demo; small wireless clip-on for daily use. Must show up as a normal mic in Android Chrome. #hardware
- [ ] **Employment-agreement check covers RainB** — same outside-activities clause as MarketGuard; gates the public repo. #legal
- [ ] **Spike: does Parakeet keep "uh/um"?** — and how accurate are word timestamps on lapel audio? Decides whether filled pauses are a signal or just gaps. #spike #speech #critical-path
- [ ] **Spike: Nemotron 3 Diarization streaming** — 0.32 s latency config on the VM; does "say hi" enrolment reliably make you speaker #1? #spike #speech #critical-path
- [ ] **Spike: end-to-end latency ≤ 2 s** — mic → push, measured, before any feature work. Above 2 s the magic is gone. #spike #critical-path
- [ ] **Record + label 30 min of own speech** — real conversations with real tip-of-the-tongue moments, timestamps marked. This is the detector's test set. Never commit it. #detector #critical-path
- [ ] **Repo scaffold** — Python 3.11+, FastAPI, pinned deps; Vite + React frontend. #backend
- [ ] **Speech service** — WS audio in → `WordEvent{text,start,end,speaker}` out. Knows nothing about hesitation. #speech
- [ ] **"Say hi" enrolment step** — 5 s at session start; diarization orders by arrival, so the user is always speaker #1. #speech
- [ ] **Hesitation detector — pure function** — `WordEvent` stream → `StuckCandidate`. No network, no models. Rules from [[RainB - Glossary]]. #detector #critical-path
- [ ] **Personal pause baseline** — median + MAD of *your* inter-word gaps, so slow speakers aren't flagged constantly. #detector
- [ ] **Self-recovery cancel + 20 s cool-down** — found the word yourself → cancel the push. #detector
- [ ] **Nano confirm + query rewrite** — structured JSON: `{stuck: bool, reaching_for: str, query: str}`. #recall #critical-path
- [ ] **Nemotron 3 Embed indexing + search** — chunk + embed; top-k over docs and memory. #recall #brain
- [ ] **Nano answer composer** — one line + source ("from your notes, 14 Sept"). No confident match → stay silent. #recall
- [ ] **Precision tuning pass** — silent is better than wrong. A wrong buzz mid-conversation is worse than none. #risk #critical-path
- [ ] **Brain store — SQLite + vector extension** — one namespace per brain: demo persona / you / judge upload. #brain
- [ ] **Vault ingest** — Obsidian markdown + PDFs. Read-only; never write into OneDrive. #brain
- [ ] **`rainb sync` command + `.rainbignore`** — PC-side, read-only, hash-based incremental upload of the vault. #brain
- [ ] **Owner-key pairing** — pair phone once, key in a secure cookie, never in a URL. #brain #privacy
- [ ] **Persona nightly reset + judge brains deleted after 24 h** — upload cap ~20 files / 10 MB, md/pdf/txt. #brain #demo
- [ ] **Retention test: expired text gone from FTS + vector search** — not just row counts. #brain #privacy #critical-path
- [ ] **Conversation memory writer** — transcripts by speaker into the brain store. #brain
- [ ] **Retention job — policy C** — others' words deleted after 24 h; your words + facts kept longer; raw audio never stored. #brain #privacy
- [ ] **Fact extraction from conversations** — Nano: "met Daniel, works at Grab, into RAG". Makes memory searchable by meaning. #brain
- [ ] **Demo persona brain** — fictional person: small vault, papers, a few "past conversations" with planted names/terms to blank on. #brain #demo #critical-path
- [ ] **Web Push (VAPID) + service worker** — Android Chrome, lock-screen notification. #notify #critical-path
- [ ] **In-page banner fallback** — when push permission is denied. #notify
- [ ] **Capture client** — pick external mic, AudioWorklet → 16 kHz mono PCM → WS. #frontend
- [ ] **Live transcript with hesitation highlight** — show the pause being measured; makes the detector legible to judges. #frontend
- [ ] **"I'm stuck" fallback button** — the live demo can never fully fail. #frontend
- [ ] **"Play a sample conversation"** — pre-recorded clip streamed through the *real* pipeline; works without mic permission. Reliability strategy, not a gimmick. #frontend #demo #critical-path
- [ ] **"Use your own notes" upload** — judge-mode brain. #frontend
- [ ] **Privacy controls screen** — retention settings, delete memory, recording indicator. Points for Design. #frontend #privacy
- [ ] **HTTPS via Caddy** — mic and push both require a secure context. #deploy
- [ ] **Deploy on the Nebius VM** — long-running streaming process; cannot be serverless. #deploy
- [ ] **End-to-end check on the hosted URL** — cold open from another Android phone, another network. #deploy
- [ ] **README with setup + execution instructions** — submission requirement. #submission
- [ ] **Demo video ≤ 3 min on YouTube** — audio must *explicitly* name Parakeet, Nemotron 3 Diarization, Nemotron Embed, Nemotron Nano, Token Factory, AI Cloud. Build it around the money shot: stall mid-sentence → phone lights up. #submission #critical-path
- [ ] **State the consent limitation on camera** — own mic only, 24 h retention for others. Naming it is credibility. #submission #privacy
- [ ] **Platform/tools feedback** — separately eligible for a prize. #submission
- [ ] **Submit — track: Personal AI** — hard deadline **30 Oct 2026, 10:00 PDT**. #submission #blocker
- [ ] **Connected apps — Gmail, Calendar, Drive** — phase 2 sources into the same brain store. #stretch
- [ ] **Claude Code "dig deeper"** — local companion runs `claude -p` over the whole PC. Slow path only. #stretch
- [ ] **Tavily web fallback** — nothing local matches → search the web. Separate $3k prize. #stretch
- [ ] **Nemotron 3 Nano Omni for borderline cases** — hears tone, not just words. #stretch
- [ ] **Fine-tune Nano on interrupt / don't-interrupt** — via Token Factory post-training, from the labelled recordings. #stretch
- [ ] **Native Android app** — background recording with the screen locked. #stretch


## In Progress

- [ ] **Design review — section 4: frontend, deploy, testing** #design


## Done

**Complete**
- [x] **Idea selected and scored — ~47/60** — Novelty 7, Feasibility 8, Scalability 7, Impact 7, Demo-ability 9, Domain Fit 9. Chosen over MarketGuard (51/60) on motivation.
- [x] **Track chosen — Personal AI** (Nebius × NVIDIA Global AI Hackathon) — "private assistants with persistent memory and tool integration" is the idea's own description.
- [x] **Positioning settled** — memory products recall *after the fact*; RainB recalls *in the moment*, triggered by your own hesitation.
- [x] **[[RainB - Architecture]] written** — pipeline, units, latency budget, stack, demo, risks.
- [x] **[[RainB - Glossary]] written** — hesitation signals → computable features, with starting thresholds.
- [x] **Decision: phone first, Android** — web app on the phone for judges; PC-mic daily-use setup later.
- [x] **Decision: sources = files/notes + conversation memory** — connected apps are phase 2.
- [x] **Decision: demo persona by default, "use your own notes" optional** — zero-setup wow for a judge at 3am.
- [x] **Decision: retention policy C** — others' words 24 h, yours + facts longer, raw audio never.
- [x] **Decision: cheap-to-expensive pipeline (Approach 1) + manual fallback** — rejected single Omni model (chunked, ≥3 s) and manual-only (throws away the novelty).
- [x] **Decision: "you" = diarization speaker #1 via enrolment** — arrival-order speaker cache + lapel loudness; no voice-ID model.
- [x] **Design section 2 approved — detector + recall** — `feed()` + `tick(now)` pure detector; explicit / structural / booster rules; other-speaker handover suppresses; self-recovery `Cancel`. Recall: Nano confirm ∥ speculative search → hybrid (Embed + FTS5) → grounded answer; > 2.5 s = no buzz, goes to Moments; "Not it" is the only feedback.
- [x] **Design section 3 approved — brain store + privacy** — one SQLite file per brain (persona / me / judge-*); utterances + facts indexed as chunks, one search path; `rainb sync` read-only with `.rainbignore`; facts outlive the 24 h deletion of others' words; expiry deletes from chunks + FTS + vectors; owner-key cookie pairing.
- [x] **Repo created with MIT LICENSE** — `Projects/rainb`, day one.




%% kanban:settings
```
{"kanban-plugin":"board","list-collapse":[false,false,false],"show-checkboxes":true,"tag-colors":[]}
```
%%
