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
# When Token Factory reports Lightning overloaded (529/5xx/429) or it takes longer than
# CHAT_TIMEOUT_S (useless mid-sentence anyway), retry once on Nano: 11/13 vs 13/13, ~0.9 s.
CHAT_FALLBACK_MODEL = os.environ.get("CHAT_FALLBACK_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
CHAT_TIMEOUT_S = float(os.environ.get("CHAT_TIMEOUT_S", "2.5"))
FACTS_MODEL =os.environ.get("FACTS_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")  # off the hot path
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

DATA_DIR = Path(os.environ.get("RAINB_DATA", ROOT / "data"))
PERSONA_DIR = ROOT / "personas"
# US / UK engine. auto: Parakeet + NeMo boosting on a CUDA GPU, else Parakeet ONNX on CPU.
# Force one with nemo | parakeet. (Malaysian English always uses WHISPER_MODEL, GPU only.)
ASR_BACKEND = os.environ.get("ASR_BACKEND", "auto")
# After-the-fact LLM fixing of misheard names. Off: it dropped words and rewrote real
# "Toronto" into "Trontal" in tests; NeMo word boosting fixes names at the source instead.
LLM_NAME_FIX = os.environ.get("LLM_NAME_FIX", "0") == "1"
# Mesolitica's Malaysian Whisper small, converted to CTranslate2. On the owner's 10 phone-recorded
# sentences: 0% WER, 24/24 names, keeps "uh", ~190 ms/clip on an RTX 2060, vs Parakeet + boosting
# 12% WER, 16/24 names (eval/asr_testset.py).
_MALAYSIAN = ROOT / "models" / "malaysian-whisper-small-ct2"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", str(_MALAYSIAN) if _MALAYSIAN.exists() else "large-v3-turbo")
ASR_MODEL_DIR = Path(os.environ.get("ASR_MODEL_DIR", ROOT / "models" / "parakeet-tdt-0.6b-v2"))

# Pairing: whoever presents this key gets the owner's brain. Unset = no owner brain.
OWNER_KEY = os.environ.get("RAINB_OWNER_KEY", "")

LATE_AFTER_S = 2.5  # answer later than this after the pause began -> Moments list, no buzz
OTHERS_RETENTION_H = 24
JUDGE_BRAIN_TTL_H = 24
