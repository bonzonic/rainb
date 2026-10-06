# RainB

An extension of your brain: the moment you blank mid-sentence, the answer is
already on your phone.

Memory wearables and screen-recall tools record your life, but recall happens
*after the fact* — you open an app later and ask. RainB works *in the moment*.
It notices you hesitating ("we should use that paper, the one with the, uh…"),
works out what you're reaching for, searches your own notes and past
conversations, and pushes the answer to your phone in under two seconds.

> Built for the Nebius x NVIDIA Global AI Hackathon — Personal AI track.

---

## Status

| Component | State |
|---|---|
| Design | 🔨 in review |
| Speech service (Parakeet + Nemotron 3 Diarization) | 🔜 |
| Hesitation detector | 🔜 |
| Recall agent (Nemotron Nano + Embed on Token Factory) | 🔜 |
| Brain store + conversation memory | 🔜 |
| Phone client + Web Push | 🔜 |

---

## Architecture

```
lapel mic → phone (Chrome) ──audio──▶ Speech service      Parakeet: words + timestamps
                                         │                 Nemotron 3 Diarization: who's speaking
                                         ▼
                                  Hesitation detector      no model — pause + unfinished phrase
                                         │  candidates only
                                         ▼
                                  Recall agent             Nemotron Nano: stuck? reaching for what?
                                         │                 Nemotron Embed: search your brain
                                         ▼                 Nemotron Nano: one-line answer + source
phone notification ◀──web push──  Notifier
```

Cheap checks first, expensive ones last: pause timing runs on every word for
free; the model only sees moments that already look like you're stuck.

---

## Privacy

- Raw audio is never stored.
- Other speakers' words are deleted after 24 hours.
- Only the wearer's hesitation can trigger a lookup.

---

## Setup

Coming soon.

## License

[MIT](LICENSE)
