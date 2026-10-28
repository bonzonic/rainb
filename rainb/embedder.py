"""Text embeddings for the brain store.

Default: a small local ONNX model (~5-15 ms per query on CPU). Token Factory's embedding
endpoint measured 0.3-9 s per query, which doesn't fit a 2 s budget, so it is opt-in
(EMBED_BACKEND=tokenfactory) for offline indexing experiments.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from . import config

BACKEND = os.environ.get("EMBED_BACKEND", "local")
LOCAL_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# BGE retrieval instruction for queries; passages are embedded raw
BGE_QUERY = "Represent this sentence for searching relevant passages: "


class Embedder:
    _lock = threading.Lock()
    _model = None

    def __init__(self, tf=None):
        self.tf = tf

    @property
    def id(self) -> str:
        return f"local:{LOCAL_MODEL}" if BACKEND == "local" else f"tf:{config.EMBED_MODEL}"

    @classmethod
    def _local(cls):
        with cls._lock:
            if cls._model is None:
                from fastembed import TextEmbedding

                cls._model = TextEmbedding(LOCAL_MODEL, cache_dir=str(config.ROOT / "models" / "fastembed"))
        return cls._model

    def warm(self):
        if BACKEND == "local":
            self.query("warm up")

    def query(self, text: str) -> np.ndarray:
        """Synchronous, local only: used on the hot path."""
        m = self._local()
        v = next(iter(m.embed([BGE_QUERY + text])))
        return (v / np.linalg.norm(v)).astype(np.float32)

    async def aquery(self, text: str) -> np.ndarray:
        if BACKEND == "local":
            return self.query(text)
        return (await self.tf.embed([text], query=True))[0]

    def passage(self, text: str) -> np.ndarray:
        """Synchronous single passage (conversation memory)."""
        v = next(iter(self._local().embed([text])))
        return (v / np.linalg.norm(v)).astype(np.float32)

    async def passages(self, texts: list[str]) -> np.ndarray:
        if BACKEND == "local":
            m = self._local()
            v = np.asarray(list(m.embed(texts, batch_size=32)), dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        return await self.tf.embed(texts)
