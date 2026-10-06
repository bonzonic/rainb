# RainB — Glossary & Hesitation Signals

> Every "I'm stuck" moment is a **pattern in word timing and syntax.**
> This note is the translation layer from how people blank → computable signal.

See also: [[RainB - Architecture]]

⚠️ **All thresholds are starting guesses.** They get tuned against a labelled recording of *your own* speech — see the kanban item "Record + label 30 min of own speech".

---

## The core idea

A pause alone means nothing — people pause to breathe, to think, for emphasis. A pause **after an unfinished phrase** means something. An explicit **"what's it called"** means almost everything.

**Signals come in three strengths:**

| Strength | Fires alone? | Example |
|---|---|---|
| **Explicit** | Yes | "what's it called", "what's his name" |
| **Structural** | Only in combination | pause after "the one with the…" |
| **Weak** | Never — only raises confidence | "uh", repetition |

---

## Terms → detection

### Tip-of-the-tongue (TOT)

The state RainB exists for: you know the thing, you can describe it, you can't produce the word. Research term; see Tip-pong (CHI 2023).

### Retrieval phrase — *explicit*

| | |
|---|---|
| **Examples** | "what's it called", "what's his/her name", "what was it", "I forgot the name", "you know the…", "the thingy" |
| **Computed** | Phrase match on your last ~8 words |
| **Fires at** | Any match from the user speaker |

### Silent pause — *structural*

| | |
|---|---|
| **History needed** | Your own normal inter-word gaps (median + MAD, same robust idea as MarketGuard) |
| **Computed** | `gap = next_word.start − prev_word.end` |
| **Fires at** | gap > **600 ms** *and* mid-utterance (not after a sentence end) |

Fluent inter-word gaps are ~100–250 ms. Using *your* median/MAD rather than a fixed number means a slow, thoughtful speaker isn't flagged constantly.

### Unfinished phrase — *structural*

| | |
|---|---|
| **Computed** | Last word before the pause is a determiner, preposition or dangling adjective: *the, a, that, this, with, by, from, of, called* |
| **Fires at** | Unfinished phrase **+** silent pause |

"The paper by ___" is stuck. "The paper was great. ___" is just a breath.

### Filled pause — *weak*

| | |
|---|---|
| **Examples** | uh, um, er, erm, ah |
| **Caveat** | ASR often **drops** fillers from transcripts. If Parakeet does, the filler shows up as a timestamp gap instead — detection still works via silent pause. Spike this first. |

### Repetition / restart — *weak*

| | |
|---|---|
| **Examples** | "the, the, the…", "it's by — it's by…" |
| **Computed** | Same token or bigram repeated within 2 s |

### Self-recovery

If you find the word yourself **before** the answer lands, cancel the push. Detected as: fluent speech resumes within ~300 ms of the candidate and the Nano query is answered by what you said.

### Cool-down

At most one push per **20 s**. A second buzz while you're still recovering is worse than none.

### Speaker gate

Only `speaker == you` (diarization channel #1 from enrolment, confirmed by lapel loudness) can trigger. Someone else hesitating is not your problem.

---

## Confidence → action

| Confidence | Action |
|---|---|
| Explicit phrase | Send to Nano immediately |
| Structural combo | Send to Nano |
| Weak signals only | Log, don't send |
| Nano says "not stuck" or no good match | **Stay silent** — silent is better than wrong |
