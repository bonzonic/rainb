---
title: RainB
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Fills in the word you blank on, mid-sentence
---

# RainB

An extension of your brain: the moment you blank mid-sentence, the answer is
already on your phone.

Memory wearables and screen-recall tools record your life, but recall happens
*after the fact*: you open an app later and ask. RainB works *in the moment*.
It notices you hesitating ("we should cite that paper, the one about retrieval
augmented generation. It's by…"), works out what you're reaching for, searches
your own notes and past conversations, and puts the answer on your phone in
under two seconds.

> Built for the Nebius x NVIDIA Global AI Hackathon, Personal AI track.

---

## How it works

```
mic (phone, Chrome) ──100 ms PCM frames over WebSocket──▶ Speech        NVIDIA Parakeet TDT 0.6B (ONNX, CPU)
                                                            │            words + timestamps, keeps "uh/um"
                                                            ▼
                                                     Hesitation detector  no model: pause timing + syntax
                                                            │ candidates only
                                                            ▼
                                                     Recall agent         hybrid search (local embeddings + SQLite FTS5)
                                                            │             NVIDIA Nemotron on Nebius Token Factory:
                                                            ▼             "is this a stall, and which note completes it?"
phone: the blank fills in + lock-screen notification ◀──────┘
```

**Cheap checks first, expensive ones last.** Pause timing runs on every word
for free. The model only sees moments that already look like you're stuck.

### The hesitation detector

A pure state machine (`rainb/detector.py`, fully unit-tested with an injected clock):

| Signal | Strength | Rule |
|---|---|---|
| Retrieval phrase ("what's it called", "what's his name") | fires alone | regex on your last 8 words |
| Pause after an unfinished phrase ("…the one with the ▁▁▁") | fires | gap > max(600 ms, your median + 3·MAD) and the last word hangs (the/of/by/called…) or trails into "uh" |
| Filler, repetition | weak | each lowers the threshold by 150 ms, floor 400 ms; never fires alone |
| You start talking again | cancel | self-recovery: no buzz for an answer you already found |
| Someone else talks during your pause | suppress | turn handover, not a stall |

After a buzz there is a 20 s cool-down. Answers that would land more than 2.5 s
after the pause go to the Moments list instead of interrupting you.

### The recall agent

One model round-trip instead of two: on a candidate, RainB searches immediately
(about 10–60 ms, local), then a single Nemotron call both confirms the stall and
picks the answer from the top 5 notes. The answer must literally appear in the
cited note, or RainB stays silent. **Silent is better than wrong.**

## NVIDIA + Nebius usage

| Stage | Model | Runs on |
|---|---|---|
| Speech → words + timestamps | **NVIDIA Parakeet TDT 0.6B v2** (int8 ONNX) | Nebius AI Cloud VM (CPU is enough) |
| Stall confirmation + answer | **NVIDIA Nemotron 3.5 Lightning** | **Nebius Token Factory** |
| Fact extraction from conversations | **NVIDIA Nemotron 3 Nano 30B-A3B** | **Nebius Token Factory** |
| Search | BGE-base embeddings (local ONNX) + SQLite FTS5 | same VM |

Why these choices, from measurements on 22–23 Oct 2026:

- **Lightning vs Nano for recall:** Lightning answered **29/29** golden cases
  correctly (13 on the owner's notes, 16 on the demo persona) at **~380 ms
  median**. Nano 30B-A3B got 11/13 on the owner's set at ~920 ms. A recall
  that runs mid-sentence needs the fast one.
- **Local embeddings:** Token Factory's embedding endpoint (Qwen3-Embedding-8B;
  there is no Nemotron embedding model on Token Factory yet) measured 0.3–9 s
  per query, too slow for a 2 s budget. A local BGE model takes about 5–15 ms.
- **Parakeet on CPU:** a 10 s clip transcribes in 0.5–1 s with int8 weights,
  and Parakeet keeps "uh" in the transcript, which the detector uses. No GPU
  needed, so the demo can stay up cheaply through judging.

## Try it

The hosted demo opens on a fictional persona's brain (Maya, `personas/demo/`).

1. Tap **Play a sample conversation**. A recorded clip streams through the real
   pipeline, with no mic permission needed.
2. Or tap **Start listening** and stall mid-sentence: *"our design system is
   called, uh…"*
3. Or open **Brain → Use your own notes** and upload Markdown, text, JSON or PDF
   files. They go into a private brain that is deleted after 24 hours.

## Run it locally

Requires Python 3.11 and a Nebius Token Factory API key.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # macOS/Linux: .venv/bin/pip
.venv/Scripts/python scripts/download_models.py     # Parakeet int8 + embeddings, ~1 GB
cp .env.example .env                                 # then set NEBIUS_API_KEY
.venv/Scripts/python -m uvicorn rainb.server:app --port 8000
```

Open http://localhost:8000. The browser allows the mic on `localhost`. For a
phone you need HTTPS: see Deploy.

### CLI

```bash
python -m rainb sync ~/ObsidianVault --brain me      # index your notes (read-only; honours .rainbignore)
python -m rainb ask "our design system is called uh" --brain demo
python -m rainb replay web/samples/paper.wav --brain demo   # stream a wav through the full pipeline
python -m rainb transcribe some.wav                   # Parakeet words + timestamps as JSONL
python eval/recall_eval.py eval/recall_golden_demo.jsonl
python -m pytest tests
```

## Deploy (Hugging Face Space, free CPU)

The YAML at the top of this README makes the repo a Docker Space (`sdk: docker`,
`app_port: 7860`). Hugging Face serves it over HTTPS, so no Caddy is needed.

1. Create a Space: SDK **Docker**, hardware **CPU basic** (free).
2. Space **Settings → Variables and secrets**: add the secret `NEBIUS_API_KEY`
   (and optionally `RAINB_OWNER_KEY`).
3. Push a snapshot of `mvp` (the wav samples go through Git LFS, which Hugging Face requires for binaries):

```bash
git checkout --orphan hf-space mvp
git lfs track "*.wav"
git add .gitattributes
git add --renormalize web/samples
git commit -m "RainB Space snapshot"
git push --force https://huggingface.co/spaces/<user>/<space> hf-space:main
git checkout mvp
```

Speech on the Space is NVIDIA Parakeet TDT 0.6B v2 (ONNX, CPU) for US / UK English;
Malaysian English needs a GPU and falls back to US / UK. The demo persona (Maya) is
the default brain. Open the app at `https://<user>-<space>.hf.space` rather than
inside the huggingface.co page, so the mic and cookies work. The free tier has no
persistent disk: uploaded notes vanish when the Space restarts, the demo brain is rebuilt.

## Deploy (Nebius AI Cloud)

One small CPU VM (4 vCPU / 8–16 GB RAM is plenty) running Docker:

```bash
git clone https://github.com/bonzonic/rainb && cd rainb
cp .env.example .env    # NEBIUS_API_KEY, DOMAIN (e.g. <public-ip>.sslip.io), optional RAINB_OWNER_KEY
mkdir -p data && sudo chown 1000:1000 data   # the app runs as uid 1000
docker compose up -d --build
```

Caddy gets a TLS certificate for `DOMAIN` automatically. The mic and
notifications both need HTTPS. Open ports 80 and 443 in the VM's security group.

## Privacy

- Raw audio is never stored: it is transcribed in memory and dropped.
- Only the wearer's hesitation can trigger a lookup.
- Other speakers' words are deleted after 24 hours. Your own words and the
  extracted facts stay until you delete them (Privacy → Delete my conversation memory).
- The shared demo brain never stores what visitors say. Uploaded notes live in
  a separate per-browser brain and are deleted after 24 hours.
- Each brain is its own SQLite file. Deleting memory removes it from the
  keyword index and the vector index, not just the table.

Known limitations: without diarization, every voice the mic hears is treated
as the wearer, so use a lapel mic. Android Chrome stops recording when the
screen locks, so the app holds a screen wake lock while listening.

## Repo map

| Path | What |
|---|---|
| `rainb/detector.py` | hesitation detector (pure, tested) |
| `rainb/speech.py` | Parakeet wrapper + VAD-driven streaming |
| `rainb/recall.py` | search + Nemotron answer + grounding check |
| `rainb/brain.py`, `ingest.py`, `brains.py` | per-brain SQLite store, chunking, sync, retention |
| `rainb/session.py`, `server.py` | WebSocket sessions, REST API |
| `web/` | phone client (no build step) |
| `personas/demo/` | the fictional demo brain |
| `eval/` | golden sets + recall eval |
| `spikes/` | the Parakeet-on-CPU spike |

## License

[MIT](LICENSE)
