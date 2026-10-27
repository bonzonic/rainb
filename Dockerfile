# Runs as-is on a free Hugging Face Docker Space (CPU basic) and behind Caddy on a VM.
# CPU only: US / UK speech is NVIDIA Parakeet ONNX; Malaysian English needs a GPU and falls back.
FROM python:3.11-slim

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 ASR_BACKEND=parakeet
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user rainb rainb
COPY --chown=user scripts/download_models.py scripts/
# Bake the weights into the image so a cold start never waits on Hugging Face
RUN python scripts/download_models.py

COPY --chown=user personas personas
COPY --chown=user web web
# Brains live here; on a free Space this is wiped on restart, so the demo brain is rebuilt at startup
RUN mkdir -p data

EXPOSE 7860
CMD ["uvicorn", "rainb.server:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
