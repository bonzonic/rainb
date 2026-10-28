# RainB demo video script (≤ 3 min)

Read slowly; ~2:40 total. Record each scene separately, keep the best take.

## 1. Hook (face, ~20 s)
"You're mid-sentence, and the name is just gone. The restaurant, the paper, the person.
You'll remember it an hour later. RainB gives it to you now, while you're still talking.
It's an extension of your brain, for the moment you blank."

## 2. Money shot (you + phone in frame, ~35 s)
Say naturally, and actually pause: "Let's go to the Japanese place near work. The place, uh..."
→ phone lights up: **Uokatsu**.
Then: "I didn't ask it anything. It noticed I got stuck, and finished my sentence from my own notes."

## 3. How it works (screen recording, ~65 s)
Show the web app, play a demo sample, show the answer appear.
"My phone streams audio to a server on **Nebius AI Cloud**.
Speech is transcribed by **NVIDIA Parakeet TDT 0.6B**, which keeps the 'uh's and 'um's and gives word timestamps.
A small hesitation detector, with no model at all, watches pause timing and unfinished sentences.
Only when I look stuck does it search my notes, using local embeddings plus keyword search,
and ask **NVIDIA Nemotron 3.5 Lightning** on **Nebius Token Factory**: is this a real stall, and which note completes it?
If Lightning is busy, it falls back to **Nemotron Nano**.
From the pause to the answer on my phone takes about one second."

## 4. Privacy and limits (face, ~25 s)
"Raw audio is never stored. Other people's words are deleted after 24 hours.
The public demo brain, Maya, never remembers what visitors say.
One honest limit: it can't tell voices apart yet, so it should be used with a lapel mic close to you."

## 5. Close (~10 s)
"RainB. Try it at 213-239-173-96.sslip.io. The code is on GitHub."
On screen: https://213-239-173-96.sslip.io and https://github.com/bonzonic/rainb
