// RainB web client: mic or sample clip -> 100 ms PCM frames over a WebSocket; renders the
// transcript, the growing gap while you pause, and the answer when RainB finds it.
const $ = (id) => document.getElementById(id);
const SR = 16000;
const KEEP_WORDS = 40;
const FILLERS = new Set(["uh", "um", "er", "erm", "ah", "eh", "hmm", "mm"]);

let ws = null;
let mode = null; // "mic" | "sample"
let mic = null; // { ctx, stream, node }
let sample = null; // { audio, pcm, sent, timer, btn }
let wakeLock = null;
let swReg = null;
let currentMoment = null;
let lastAnswer = null;
let info = null;

// ---------- tabs ----------
function show(tab) {
  for (const s of document.querySelectorAll(".screen")) s.hidden = s.id !== tab;
  for (const b of document.querySelectorAll(".tabs button")) {
    if (b.dataset.tab === tab) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  }
  if (tab === "moments") loadMoments();
  if (tab === "brain") loadBrain();
  history.replaceState(null, "", tab === "listen" ? "/" : `#${tab}`);
}
document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-tab]");
  if (t) show(t.dataset.tab);
});

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 3500);
}

async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: "include", ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${r.status}`);
  return body;
}

// ---------- who am I ----------
async function loadInfo() {
  info = await api("/api/whoami");
  $("brain-status").textContent = `Using: ${info.label}`;
  $("models").textContent =
    `Speech: ${info.models.asr}, on the RainB server. ` +
    `Recall: ${info.models.llm} on Nebius Token Factory. Search: local embeddings + keyword index.`;
  $("pair-card").hidden = !info.owner_pairing;
  const ul = $("samples");
  ul.innerHTML = "";
  for (const s of info.samples) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.className = "sample";
    b.innerHTML = `<strong></strong><span></span>`;
    b.querySelector("strong").textContent = `▶ ${s.title}`;
    b.querySelector("span").textContent = s.hint;
    b.addEventListener("click", () => toggleSample(s, b));
    li.append(b);
    ul.append(li);
  }
}

// ---------- socket ----------
function openSocket(source) {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "start", source }));
      resolve();
    };
    ws.onerror = () => reject(new Error("Can't reach the RainB server."));
    ws.onclose = () => stopAll(false);
    ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  });
}

function onMessage(m) {
  switch (m.type) {
    case "words": addWords(m.words); break;
    case "meter": meter(m); break;
    case "candidate": candidate(m); break;
    case "answer": answer(m); break;
    case "silent": silent(m); break;
    case "correct": correctWords(m); break;
    case "cancel": break;
  }
}

// a finished sentence came back with misheard names fixed: swap its words in place
function correctWords(m) {
  const spans = [...$("transcript").querySelectorAll(".w")].filter((el) => {
    const t = parseFloat(el.dataset.start);
    return t >= m.start - 0.01 && t <= m.end + 0.01;
  });
  if (!spans.length) return;
  const old = spans.map((s) => s.textContent.trim().toLowerCase());
  const anchor = spans[0];
  for (const word of m.text.split(/\s+/)) {
    const s = document.createElement("span");
    s.className = "w" + (old.includes(word.toLowerCase()) ? "" : " fixed") + (anchor.classList.contains("old") ? " old" : "");
    s.dataset.start = m.start;
    s.textContent = word + " ";
    anchor.before(s);
  }
  spans.forEach((s) => s.remove());
}

// ---------- transcript + gap ----------
function resetStage() {
  const tr = $("transcript");
  tr.innerHTML = "";
  tr.append(newGap());
}
function newGap() {
  const g = document.createElement("span");
  g.className = "gap";
  g.id = "gap";
  g.setAttribute("aria-hidden", "true");
  g.innerHTML = `<span class="gap-fill"></span><span class="gap-answer"></span>`;
  return g;
}
function addWords(words) {
  const tr = $("transcript");
  let gap = $("gap");
  if (gap.classList.contains("found")) {
    gap.removeAttribute("id"); // keep the filled blank in the sentence
    gap.classList.remove("visible", "armed");
    gap = newGap();
    tr.append(gap);
  }
  for (const w of words) {
    const s = document.createElement("span");
    const bare = w.text.toLowerCase().replace(/[^\w']/g, "");
    s.className = "w" + (FILLERS.has(bare) || /^(u+h+|u+m+|e+r+m*|a+h+|h+m+|m+)$/.test(bare) ? " filler" : "");
    s.dataset.start = w.start;
    s.textContent = w.text + " ";
    tr.insertBefore(s, gap);
  }
  const all = tr.querySelectorAll(".w");
  all.forEach((el, i) => el.classList.toggle("old", i < all.length - 14));
  for (let i = 0; i < all.length - KEEP_WORDS; i++) all[i].remove();
  gap.classList.remove("visible", "armed");
}
function meter(m) {
  const gap = $("gap");
  const ratio = m.threshold ? m.pause / m.threshold : 0;
  if (!gap.classList.contains("found")) {
    gap.classList.toggle("visible", m.pause > 0.15);
    gap.classList.toggle("armed", m.armed && m.pause > 0.15);
    gap.style.setProperty("--p", Math.min(ratio, 1.6) / 1.6);
    gap.style.setProperty("--f", Math.min(ratio, 1));
  }
  $("meter").classList.toggle("armed", m.armed);
  $("meter-bar").style.width = `${Math.min(m.pause / (m.threshold * 1.5), 1) * 100}%`;
  $("meter-mark").style.left = `${100 / 1.5}%`;
  $("meter-pause").textContent = `${m.pause.toFixed(2)} s`;
  $("meter-thr").textContent = `${m.threshold.toFixed(2)} s`;
}
function candidate(m) {
  $("hint").textContent = m.reason === "explicit" ? "Heard “what’s it called”: looking…" :
    m.reason === "manual" ? "Looking…" : "That pause looked like a stall: looking…";
}

// ---------- answers ----------
function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
function sourceLine(src) {
  if (src.kind === "utterance") return `From something you said · ${fmtDate(src.created_at)}`;
  if (src.kind === "fact") return `From a past conversation · ${fmtDate(src.created_at)}`;
  const section = src.heading ? src.heading.split(" > ").pop() : "";
  return `From your notes: ${src.title}${section ? ` › ${section}` : ""}`;
}
function answer(m) {
  lastAnswer = m;
  currentMoment = m.moment_id;
  const gap = $("gap");
  if (!m.recovered) {
    gap.classList.add("found");
    gap.classList.remove("visible", "armed");
    gap.querySelector(".gap-answer").textContent = m.answer;
  }
  $("answer").hidden = false;
  $("answer-text").textContent = m.answer;
  $("answer-source").textContent = sourceLine(m.source);
  $("answer-kicker").textContent = `${(m.latency_ms / 1000).toFixed(1)} s after you paused` +
    (m.recovered ? " · you got there yourself" : m.late ? " · too late to buzz, saved to Moments" : "");
  $("answer-more").hidden = true;
  $("answer-more").textContent = `“${m.source.text}”`;
  $("not-it").disabled = false;
  $("hint").textContent = `Recall ${m.recall_ms} ms (search ${m.timings.search} ms, model ${m.timings.model} ms).`;
  if (m.buzz) notify(m);
}
function silent(m) {
  $("hint").textContent = "Nothing in your brain matched, so RainB stayed quiet.";
}
async function notify(m) {
  if (navigator.vibrate) navigator.vibrate([60, 40, 60]);
  if (!swReg || Notification.permission !== "granted") return;
  swReg.showNotification(m.answer, {
    body: sourceLine(m.source),
    tag: "rainb-answer",
    renotify: true,
    silent: false,
    icon: "/icon.svg",
    data: { moment_id: m.moment_id },
    actions: [{ action: "not-it", title: "Not it" }, { action: "more", title: "More" }],
  }).catch(() => {});
}
$("not-it").addEventListener("click", async () => {
  if (!currentMoment) return;
  await api(`/api/moments/${currentMoment}/not-it`, { method: "POST" });
  $("not-it").disabled = true;
  toast("Marked as not it. Thanks, that helps tune RainB.");
});
$("more").addEventListener("click", () => ($("answer-more").hidden = !$("answer-more").hidden));

// ---------- mic ----------
async function startMic() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser can't record audio. Try Chrome.");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new AudioContext({ sampleRate: SR });
  await ctx.audioWorklet.addModule("/pcm-worklet.js");
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-framer");
  node.port.onmessage = (e) => ws?.readyState === 1 && ws.send(e.data);
  src.connect(node);
  mic = { ctx, stream, node };
}
async function toggleListen() {
  if (mode) return stopAll(true);
  try {
    await prepare();
    await openSocket("mic");
    await startMic();
    mode = "mic";
    $("listen-btn").textContent = "Stop listening";
    $("listen-btn").classList.add("on");
    $("stuck-btn").disabled = false;
    $("hint").textContent = "Listening. Talk normally; stall on a name and wait.";
  } catch (e) {
    stopAll(false);
    $("hint").textContent = e.name === "NotAllowedError"
      ? "Microphone access is blocked. Allow it in the browser’s site settings, or play a sample below."
      : e.message;
  }
}
$("listen-btn").addEventListener("click", toggleListen);
$("stuck-btn").addEventListener("click", () => ws?.readyState === 1 && ws.send(JSON.stringify({ type: "stuck" })));

// ---------- samples: the same pipeline, fed from a file ----------
async function toggleSample(s, btn) {
  if (mode) {
    const same = sample?.btn === btn;
    stopAll(true);
    if (same) return;
  }
  try {
    await prepare();
    $("stage").scrollIntoView({ behavior: "smooth", block: "center" });
    const buf = await (await fetch(`/samples/${s.id}.wav`)).arrayBuffer();
    const pcm = wavPcm(buf);
    await openSocket("sample");
    const audio = new Audio(`/samples/${s.id}.wav`);
    sample = { audio, pcm, sent: 0, btn, timer: null };
    mode = "sample";
    btn.classList.add("playing");
    btn.querySelector("strong").textContent = `■ ${s.title}`;
    $("stuck-btn").disabled = false;
    $("hint").textContent = "Playing a sample through the live pipeline…";
    await audio.play();
    sample.timer = setInterval(pumpSample, 50);
  } catch (e) {
    stopAll(false);
    $("hint").textContent = `Couldn't play the sample: ${e.message}`;
  }
}
function pumpSample() {
  const sm = sample;
  if (!sm || ws?.readyState !== 1) return;
  const target = sm.audio.ended ? sm.pcm.length + SR * 2 : Math.floor(sm.audio.currentTime * SR);
  while (sm.sent + 1600 <= target) {
    const frame = new Int16Array(1600);
    if (sm.sent < sm.pcm.length) frame.set(sm.pcm.subarray(sm.sent, Math.min(sm.sent + 1600, sm.pcm.length)));
    ws.send(frame.buffer);
    sm.sent += 1600;
  }
  if (sm.audio.ended && sm.sent >= sm.pcm.length + SR * 2) setTimeout(() => sample === sm && stopAll(true), 2500);
}
function wavPcm(buf) {
  const v = new DataView(buf);
  let off = 12;
  while (off < v.byteLength) {
    const id = String.fromCharCode(v.getUint8(off), v.getUint8(off + 1), v.getUint8(off + 2), v.getUint8(off + 3));
    const size = v.getUint32(off + 4, true);
    if (id === "data") return new Int16Array(buf.slice(off + 8, off + 8 + size));
    off += 8 + size + (size % 2);
  }
  throw new Error("not a WAV file");
}

// ---------- lifecycle ----------
async function prepare() {
  resetStage();
  $("answer").hidden = true;
  if ("serviceWorker" in navigator && !swReg) swReg = await navigator.serviceWorker.register("/sw.js").catch(() => null);
  if ("Notification" in window && Notification.permission === "default") await Notification.requestPermission().catch(() => {});
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
}
function stopAll(sendStop) {
  if (sendStop && ws?.readyState === 1) ws.send(JSON.stringify({ type: "stop" }));
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
  if (mic) { mic.stream.getTracks().forEach((t) => t.stop()); mic.ctx.close(); mic = null; }
  if (sample) {
    clearInterval(sample.timer);
    sample.audio.pause();
    sample.btn.classList.remove("playing");
    sample.btn.querySelector("strong").textContent = sample.btn.querySelector("strong").textContent.replace("■", "▶");
    sample = null;
  }
  wakeLock?.release?.().catch?.(() => {});
  wakeLock = null;
  mode = null;
  $("listen-btn").textContent = "Start listening";
  $("listen-btn").classList.remove("on");
  $("stuck-btn").disabled = true;
  meter({ pause: 0, threshold: 0.6, armed: false });
}
document.addEventListener("visibilitychange", async () => {
  if (mode && document.visibilityState === "visible" && !wakeLock) {
    try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
  }
});

// ---------- moments ----------
async function loadMoments() {
  const { moments } = await api("/api/moments");
  const ol = $("moment-list");
  ol.innerHTML = "";
  if (!moments.length) {
    ol.innerHTML = `<li class="empty">No moments yet. Start listening or play a sample, then come back.</li>`;
    return;
  }
  for (const m of moments) {
    const li = document.createElement("li");
    li.className = "moment";
    const ctx = m.context.split(" ").slice(-12).join(" ");
    li.innerHTML = `<q></q><p class="ans"></p><div class="meta"></div>`;
    li.querySelector("q").textContent = ctx;
    const a = li.querySelector(".ans");
    a.textContent = m.answer || "Stayed quiet: nothing matched";
    a.classList.toggle("none", !m.answer);
    const meta = [
      new Date(m.created_at * 1000).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }),
      `${(m.latency_ms / 1000).toFixed(1)} s`,
      m.reason,
      m.outcome.replace("_", " "),
    ];
    if (m.source_title) meta.push(m.source_title);
    li.querySelector(".meta").textContent = meta.join("  ·  ");
    ol.append(li);
  }
}

// ---------- brain ----------
async function loadBrain() {
  const b = await api("/api/brain");
  $("brain-label").textContent = `Using: ${b.label}`;
  $("use-demo").hidden = !b.brain.startsWith("judge-");
  const st = $("stats");
  st.innerHTML = "";
  for (const [k, v] of [["files", b.stats.documents], ["notes", b.stats.chunks], ["memories", b.stats.utterances + b.stats.facts]]) {
    const d = document.createElement("div");
    d.innerHTML = `<dt></dt><dd></dd>`;
    d.querySelector("dt").textContent = k;
    d.querySelector("dd").textContent = v;
    st.append(d);
  }
  const docs = $("docs");
  docs.innerHTML = "";
  for (const d of b.documents) {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span></span>`;
    li.children[0].textContent = d.title;
    li.children[1].textContent = `${d.chunks} notes`;
    docs.append(li);
  }
  const mems = $("mems");
  mems.innerHTML = "";
  $("mem-head").hidden = !b.memories.length;
  for (const m of b.memories.slice(0, 30)) {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span></span>`;
    li.children[0].textContent = m.text;
    li.children[1].textContent = m.kind === "fact" ? "fact" : m.expires_at ? "24 h" : "you";
    mems.append(li);
  }
}
$("upload").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  if (!files.length) return;
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  $("upload-status").textContent = `Indexing ${files.length} file(s)…`;
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    $("upload-status").textContent = `Added ${r.added.map((a) => `${a.file} (${a.chunks})`).join(", ")}.`;
    await loadInfo();
    loadBrain();
  } catch (err) {
    $("upload-status").textContent = `Upload failed: ${err.message}`;
  }
  e.target.value = "";
});
$("use-demo").addEventListener("click", async () => {
  await api("/api/use-demo", { method: "POST" });
  await loadInfo();
  loadBrain();
  toast("Your uploaded notes were deleted. Using the demo brain.");
});
$("pair-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("/api/pair", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key: $("pair-key").value }) });
    $("pair-key").value = "";
    $("pair-status").textContent = "Paired. This phone now uses your brain.";
    await loadInfo();
    loadBrain();
  } catch {
    $("pair-status").textContent = "That key didn't match.";
  }
});
$("forget").addEventListener("click", async () => {
  if (!confirm("Delete everything RainB remembers from your conversations? Your notes stay.")) return;
  const r = await api("/api/memory", { method: "DELETE" });
  $("forget-status").textContent = `Deleted ${r.deleted} memories.`;
});

// ---------- boot ----------
loadInfo().catch(() => ($("hint").textContent = "Can't reach the RainB server."));
const start = location.hash.slice(1);
if (["moments", "brain", "privacy"].includes(start)) show(start);
