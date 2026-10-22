#!/usr/bin/env node
// Semantic search over notes.json using Nebius Token Factory (OpenAI-compatible).
// Usage:
//   node search.mjs --models
//   node search.mjs "the Japanese place near work" [--answer]
// Env: NEBIUS_API_KEY (or a local .env), optional EMBED_MODEL.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const BASE_URL = "https://api.tokenfactory.nebius.com/v1";
const CHAT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B";
const NOTES_PATH = join(HERE, "notes.json");
const CACHE_PATH = join(HERE, "embeddings.json");

// Minimal .env loader so the key never has to be pasted anywhere.
function loadDotEnv() {
  const p = join(HERE, ".env");
  if (!existsSync(p)) return;
  for (const line of readFileSync(p, "utf8").split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (!m || process.env[m[1]]) continue;
    process.env[m[1]] = m[2].replace(/^(['"])(.*)\1$/, "$2");
  }
}
loadDotEnv();

const API_KEY = process.env.NEBIUS_API_KEY;
if (!API_KEY) {
  console.error('NEBIUS_API_KEY is not set. In PowerShell: $env:NEBIUS_API_KEY="..."');
  process.exit(1);
}

async function api(path, body) {
  const res = await fetch(BASE_URL + path, {
    method: body ? "POST" : "GET",
    headers: {
      Authorization: `Bearer ${API_KEY}`,
      ...(body && { "Content-Type": "application/json" }),
    },
    body: body && JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    const err = new Error(`${res.status} ${res.statusText} on ${path}: ${text.slice(0, 500)}`);
    err.status = res.status;
    throw err;
  }
  return JSON.parse(text);
}

async function listModels() {
  const { data } = await api("/models");
  return data.map((m) => m.id).sort();
}

async function pickEmbedModel() {
  if (process.env.EMBED_MODEL) return process.env.EMBED_MODEL;
  const embeds = (await listModels()).filter((id) => /embed/i.test(id));
  const pick =
    embeds.find((id) => /nvidia|nemotron/i.test(id)) ?? embeds[0];
  if (!pick) throw new Error("No embedding model found; set EMBED_MODEL.");
  return pick;
}

let inputTypeSupported = true;
async function embed(model, input, inputType) {
  if (inputTypeSupported) {
    try {
      return await api("/embeddings", { model, input, input_type: inputType });
    } catch (e) {
      if (e.status !== 400 && e.status !== 422) throw e;
      console.warn(`(input_type rejected, retrying without it: ${e.message.slice(0, 120)})`);
      inputTypeSupported = false;
    }
  }
  return api("/embeddings", { model, input });
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

async function noteVectors(model, notes) {
  const key = createHash("sha256")
    .update(model + "\n" + notes.map((n) => n.text).join("\n"))
    .digest("hex");
  if (existsSync(CACHE_PATH)) {
    const cache = JSON.parse(readFileSync(CACHE_PATH, "utf8"));
    if (cache.key === key) return { vectors: cache.vectors, cached: true };
  }
  const res = await embed(model, notes.map((n) => n.text), "passage");
  const vectors = res.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
  writeFileSync(CACHE_PATH, JSON.stringify({ key, model, vectors }));
  return { vectors, cached: false };
}

async function answer(question, top) {
  const notes = top.map((t) => `- ${t.text}`).join("\n");
  const t0 = performance.now();
  const res = await api("/chat/completions", {
    model: CHAT_MODEL,
    temperature: 0,
    max_tokens: 40,
    chat_template_kwargs: { enable_thinking: false },
    messages: [
      {
        role: "system",
        content:
          "The user has started a sentence and trailed off. Complete the half-finished " +
          "sentence using ONLY the notes below. Reply with just the missing words " +
          "(e.g. a name or place), nothing else. If the notes don't contain it, reply \"?\".\n\n" +
          "Notes:\n" + notes,
      },
      { role: "user", content: question },
    ],
  });
  return { text: res.choices[0].message.content?.trim(), ms: performance.now() - t0 };
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes("--models")) {
    for (const id of await listModels()) console.log(`${/embed/i.test(id) ? "[embed] " : "        "}${id}`);
    return;
  }
  const wantAnswer = args.includes("--answer");
  const question = args.find((a) => !a.startsWith("--")) ?? "the Japanese place near work";
  const notes = JSON.parse(readFileSync(NOTES_PATH, "utf8"));

  let t0 = performance.now();
  const model = await pickEmbedModel();
  const { vectors, cached } = await noteVectors(model, notes);
  const tNotes = performance.now() - t0;

  t0 = performance.now();
  const q = (await embed(model, [question], "query")).data[0].embedding;
  const tQuery = performance.now() - t0;

  const top = notes
    .map((n, i) => ({ ...n, score: cosine(q, vectors[i]) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);

  console.log(`model: ${model}`);
  console.log(`question: "${question}"`);
  top.forEach((t, i) => console.log(`  ${i + 1}. ${t.score.toFixed(4)}  [${t.id}] ${t.text}`));
  console.log(
    `timings: notes ${tNotes.toFixed(0)} ms${cached ? " (cached)" : ""}, query embed ${tQuery.toFixed(0)} ms`
  );

  if (wantAnswer) {
    const a = await answer(question, top);
    console.log(`answer (${CHAT_MODEL}): ${a.text}`);
    console.log(`answer latency: ${a.ms.toFixed(0)} ms`);
  }
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
