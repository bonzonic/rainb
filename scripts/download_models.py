"""Fetch model weights into ./models (plain files, no symlinks: ONNX Runtime refuses
external-data paths that resolve outside the model folder, which the HF cache does)."""
from huggingface_hub import snapshot_download

snapshot_download(
    "istupakov/parakeet-tdt-0.6b-v2-onnx",
    local_dir="models/parakeet-tdt-0.6b-v2",
    allow_patterns=["*.int8.onnx", "nemo128.onnx", "vocab.txt", "config.json"],
)

from rainb.embedder import Embedder  # noqa: E402  (downloads the local embedding model)

Embedder().warm()
print("models ready")
