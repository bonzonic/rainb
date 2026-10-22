FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rainb rainb
COPY scripts/download_models.py scripts/
# Bake the weights into the image so a cold start never waits on Hugging Face
RUN python scripts/download_models.py

COPY personas personas
COPY web web

EXPOSE 8000
CMD ["uvicorn", "rainb.server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
