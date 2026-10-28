"""Nebius Token Factory client (OpenAI-compatible). The only module that talks to models."""
from __future__ import annotations

import json
import logging
import re

import httpx
import numpy as np

from . import config

log = logging.getLogger("rainb.tokenfactory")

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

    async def _post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        kw = {"timeout": httpx.Timeout(timeout, connect=5.0)} if timeout else {}
        r = await self.http.post(path, json=body, **kw)
        if r.status_code >= 400:
            raise TokenFactoryError(r.status_code, f"{r.status_code} on {path}: {r.text[:300]}")
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
        """Chat with the primary model; if Token Factory says it's overloaded (or it's too slow
        to matter mid-sentence), ask the fallback model once instead."""
        body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        primary = model or config.CHAT_MODEL
        fallback = config.CHAT_FALLBACK_MODEL if model is None else None
        try:
            res = await self._post("/chat/completions", {**body, "model": primary},
                                   timeout=config.CHAT_TIMEOUT_S if fallback else None)
        except (TokenFactoryError, httpx.TimeoutException, httpx.TransportError) as e:
            if not fallback or fallback == primary or not _retryable(e):
                raise
            log.warning("%s unavailable (%s); falling back to %s", primary, str(e)[:120], fallback)
            res = await self._post("/chat/completions", {**body, "model": fallback})
        return (res["choices"][0]["message"].get("content") or "").strip()

    async def chat_json(self, messages: list[dict], max_tokens: int = 120) -> dict:
        text = await self.chat(messages, max_tokens=max_tokens)
        return parse_json(text)


class TokenFactoryError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _retryable(e: Exception) -> bool:
    """Overloaded / rate-limited / server-side trouble, or a timeout: worth another model.
    A 400 (bad request) or 401 (bad key) would fail the same way on any model."""
    if isinstance(e, TokenFactoryError):
        return e.status == 429 or e.status >= 500
    return True


def parse_json(text: str) -> dict:
    """Small models wrap JSON in prose or code fences; take the first object."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
