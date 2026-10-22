"""Nebius Token Factory client (OpenAI-compatible). The only module that talks to models."""
from __future__ import annotations

import json
import re

import httpx
import numpy as np

from . import config

# Qwen3-Embedding is instruction-tuned on the query side; documents are embedded raw.
QUERY_INSTRUCT = (
    "Instruct: Given a half-finished sentence someone is trying to complete, "
    "retrieve the personal note that contains the missing name, place or fact\nQuery: "
)


class TokenFactory:
    def __init__(self, api_key: str = config.NEBIUS_API_KEY):
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY is not set (env or .env)")
        # one pooled client: keep-alive matters inside a 2 s budget
        self.http = httpx.AsyncClient(
            base_url=config.TF_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(20.0, connect=5.0),
            http2=False,
        )

    async def close(self):
        await self.http.aclose()

    async def _post(self, path: str, body: dict) -> dict:
        r = await self.http.post(path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} on {path}: {r.text[:300]}")
        return r.json()

    async def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            if query:
                batch = [QUERY_INSTRUCT + t for t in batch]
            res = await self._post("/embeddings", {"model": config.EMBED_MODEL, "input": batch})
            out += [d["embedding"] for d in sorted(res["data"], key=lambda d: d["index"])]
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    async def chat(self, messages: list[dict], max_tokens: int = 120, temperature: float = 0.0,
                   model: str | None = None) -> str:
        res = await self._post("/chat/completions", {
            "model": model or config.CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        return (res["choices"][0]["message"].get("content") or "").strip()

    async def chat_json(self, messages: list[dict], max_tokens: int = 120) -> dict:
        text = await self.chat(messages, max_tokens=max_tokens)
        return parse_json(text)


def parse_json(text: str) -> dict:
    """Small models wrap JSON in prose or code fences; take the first object."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
