"""FastAPI app: WebSocket listening sessions, brain + moments API, and the web client.

  python -m uvicorn rainb.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import brains, config
from .embedder import Embedder
from .ingest import SUPPORTED, ingest_file
from .session import Session
from .speech import ASR
from .tokenfactory import TokenFactory

log = logging.getLogger("rainb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

WEB = config.ROOT / "web"
OWNER_COOKIE, JUDGE_COOKIE = "rainb_owner", "rainb_brain"
MAX_UPLOAD_FILES, MAX_UPLOAD_BYTES = 20, 10 * 1024 * 1024

SAMPLES = [
    {"id": "vector-db", "title": "Coffee with Daniel", "hint": "…the vector database they used at Grab. It was called, uh…"},
    {"id": "paper", "title": "Writing the memo", "hint": "…that paper about retrieval augmented generation. It's by…"},
    {"id": "ramen", "title": "Lunch plans", "hint": "…the ramen place near the office, uh, what's it called…"},
]

state: dict = {}


def owner_token() -> str:
    return hashlib.sha256(("rainb-owner:" + config.OWNER_KEY).encode()).hexdigest()


def brain_for(cookies) -> tuple[str, bool]:
    """(brain id, remember conversations?). No picker: the server infers it."""
    if config.OWNER_KEY and hmac.compare_digest(cookies.get(OWNER_COOKIE, ""), owner_token()):
        return "me", True
    judge = cookies.get(JUDGE_COOKIE)
    if brains.valid_id(judge) and judge.startswith("judge-") and brains.exists(judge):
        return judge, True
    return "demo", False  # the shared persona never stores what visitors say


async def housekeeping():
    last_day = time.strftime("%Y-%m-%d")
    while True:
        await asyncio.sleep(3600)
        try:
            n = brains.purge_expired_all()
            j = brains.purge_judge_brains()
            if time.strftime("%Y-%m-%d") != last_day:
                last_day = time.strftime("%Y-%m-%d")
                brains.reset_demo_memory()
            log.info("housekeeping: %d expired chunks, %d judge brains removed", n, j)
        except Exception:
            log.exception("housekeeping failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    emb = Embedder()
    tf = TokenFactory()
    state.update(emb=emb, tf=tf)
    await asyncio.to_thread(emb.warm)
    log.info("speech: %s", await asyncio.to_thread(ASR.name))
    if config.OWNER_KEY and brains.exists("me"):  # load the owner's names now, not on their first sentence
        import numpy as np

        await asyncio.to_thread(ASR.words, np.zeros(16000, dtype=np.float32), brains.get("me").vocabulary())
    log.info("demo brain: %s", await brains.build_demo(emb, log=log.info))
    try:  # open the HTTPS pool before the first stall
        await tf.chat([{"role": "user", "content": "hi"}], max_tokens=1)
    except Exception as e:
        log.warning("Token Factory warm-up failed: %s", e)
    task = asyncio.create_task(housekeeping())
    yield
    task.cancel()
    await tf.close()


app = FastAPI(title="RainB", lifespan=lifespan)


# -- pages -------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/sw.js")
async def service_worker():
    return FileResponse(WEB / "sw.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# -- brain -------------------------------------------------------------------

@app.get("/api/whoami")
async def whoami(request: Request):
    bid, remember = brain_for(request.cookies)
    b = brains.get(bid)
    return {"brain": bid, "label": b.label, "remember": remember, "stats": b.stats(),
            "owner_pairing": bool(config.OWNER_KEY), "samples": SAMPLES,
            "models": {"asr": ASR.name(), "llm": config.CHAT_MODEL}}


@app.get("/api/brain")
async def brain_info(request: Request):
    bid, _ = brain_for(request.cookies)
    b = brains.get(bid)
    return {"brain": bid, "label": b.label, "stats": b.stats(), "documents": b.documents(),
            "memories": b.memories() if bid != "demo" else []}


@app.post("/api/upload")
async def upload(request: Request, response: Response, files: list[UploadFile] = File(...)):
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(400, f"at most {MAX_UPLOAD_FILES} files")
    bid, _ = brain_for(request.cookies)
    brain = brains.get(bid) if bid != "demo" else brains.new_judge_brain()
    total, added = 0, []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED:
            continue
        data = await f.read()
        total += len(data)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(400, "upload limit is 10 MB")
        name = Path(f.filename).name
        tmp = config.DATA_DIR / "uploads" / brain.id / name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)  # pypdf wants a path; deleted with the brain
        n = await ingest_file(brain, state["emb"], tmp, source=f"upload:{name}", data=data)
        added.append({"file": name, "chunks": n})
    if not added:
        raise HTTPException(400, "no .md, .txt, .json or .pdf files")
    if brain.id.startswith("judge-"):
        response.set_cookie(JUDGE_COOKIE, brain.id, max_age=config.JUDGE_BRAIN_TTL_H * 3600,
                            httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return {"brain": brain.id, "added": added, "stats": brain.stats()}


@app.post("/api/use-demo")
async def use_demo(request: Request, response: Response):
    judge = request.cookies.get(JUDGE_COOKIE)
    if brains.valid_id(judge) and judge.startswith("judge-"):
        brains.wipe(judge)
    response.delete_cookie(JUDGE_COOKIE)
    return {"brain": "demo"}


@app.post("/api/pair")
async def pair(request: Request, response: Response):
    body = await request.json()
    if not config.OWNER_KEY or not hmac.compare_digest(str(body.get("key", "")), config.OWNER_KEY):
        raise HTTPException(403, "wrong key")
    response.set_cookie(OWNER_COOKIE, owner_token(), max_age=365 * 86400, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https")
    return {"brain": "me"}


@app.post("/api/unpair")
async def unpair(response: Response):
    response.delete_cookie(OWNER_COOKIE)
    return {"brain": "demo"}


@app.delete("/api/memory")
async def forget_memory(request: Request):
    bid, _ = brain_for(request.cookies)
    if bid == "demo":
        return {"deleted": 0}
    return {"deleted": brains.get(bid).forget()}


@app.delete("/api/memory/{chunk_id}")
async def forget_one(chunk_id: int, request: Request):
    bid, _ = brain_for(request.cookies)
    if bid == "demo":
        raise HTTPException(403, "the demo brain is read-only")
    brains.get(bid).delete_chunk(chunk_id)
    return {"ok": True}


@app.get("/api/moments")
async def moments(request: Request):
    bid, _ = brain_for(request.cookies)
    return {"moments": brains.get(bid).moments()}


@app.post("/api/moments/{moment_id}/not-it")
async def not_it(moment_id: int, request: Request):
    bid, _ = brain_for(request.cookies)
    brains.get(bid).set_outcome(moment_id, "not_it")
    return {"ok": True}


# -- live session ------------------------------------------------------------

@app.websocket("/ws")
async def ws_listen(ws: WebSocket):
    await ws.accept()
    bid, remember = brain_for(ws.cookies)
    brain = brains.get(bid)
    s = Session(ws, brain, state["emb"], state["tf"], remember=remember)
    await s.send({"type": "ready", "brain": bid, "label": brain.label, "remember": remember})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await s.on_audio(msg["bytes"])
            elif msg.get("text"):
                import json

                await s.on_control(json.loads(msg["text"]))
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("session crashed")
    finally:
        await s.close()


app.mount("/", StaticFiles(directory=WEB), name="web")


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse({"detail": "not found"}, status_code=404)
