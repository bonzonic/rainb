"""Settings from the environment (and a local .env, never committed)."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m and m[1] not in os.environ:
            os.environ[m[1]] = re.sub(r"^(['\"])(.*)\1$", r"\2", m[2])


_load_dotenv()

TF_BASE_URL = os.environ.get("TF_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
# Lightning: 13/13 on eval/recall_golden.jsonl at ~380 ms median; Nano 30B-A3B: 11/13 at ~920 ms
CHAT_MODEL = os.environ.get("CHAT_MODEL", "nvidia/Nemotron-3_5-Lightning")
FACTS_MODEL = os.environ.get("FACTS_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")  # off the hot path
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

DATA_DIR = Path(os.environ.get("RAINB_DATA", ROOT / "data"))
PERSONA_DIR = ROOT / "personas"
ASR_MODEL_DIR = Path(os.environ.get("ASR_MODEL_DIR", ROOT / "models" / "parakeet-tdt-0.6b-v2"))

# Pairing: whoever presents this key gets the owner's brain. Unset = no owner brain.
OWNER_KEY = os.environ.get("RAINB_OWNER_KEY", "")

LATE_AFTER_S = 2.5  # answer later than this after the pause began -> Moments list, no buzz
OTHERS_RETENTION_H = 24
JUDGE_BRAIN_TTL_H = 24
