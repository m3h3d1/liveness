"""HTTP API: session with a random challenge, frame upload, decision, profile photo choice.

No images are written to disk. After a PASS, up to 3 photo candidates stay in memory
for PHOTO_TTL so the user can pick one; the rest are dropped.
"""
import asyncio
import json
import logging
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from app import challenge, photos
from app.decision import decide
from app.pipeline import Landmarks, Passive

SESSION_TTL = 60.0
PHOTO_TTL = 300.0
SESSIONS_PER_MIN = 10            # per client IP
MIN_FRAMES, MAX_FRAMES = 15, 250
MAX_FRAME_BYTES = 300 * 1024
MAX_BODY_BYTES = 20 * 1024 * 1024
MIN_FPS, MAX_FPS = 5, 20
MAX_SIDE = 640
PASSIVE_FRAMES = 5               # frames sampled for the passive check
MIN_GOOD_FRAMES = 3
MIN_FACE = 0.2                   # face size, fraction of the frame
BRIGHTNESS = (40, 220)
MIN_SHARPNESS = 15.0
STATIC = Path(__file__).resolve().parent.parent / "static"

log = logging.getLogger("uvicorn.error")
sessions = {}
candidates = {}                  # sid -> {created, images}, only after PASS
recent = defaultdict(deque)
lock = asyncio.Lock()
analyzers = {}


@asynccontextmanager
async def lifespan(app):
    analyzers["passive"] = Passive()
    analyzers["landmarks"] = Landmarks()
    yield
    analyzers["landmarks"].close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def pick_actions():
    return secrets.SystemRandom().sample(challenge.ACTIONS, 2)


def client_ip(request):
    # Envoy appends the client address as the last X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[-1].strip() if forwarded else request.client.host


def drop_old(now):
    for sid in [s for s, v in sessions.items() if now - v["created"] > 2 * SESSION_TTL]:
        del sessions[sid]
    for sid in [s for s, v in candidates.items() if now - v["created"] > PHOTO_TTL]:
        del candidates[sid]
    for ip in [i for i, q in recent.items() if not q or now - q[-1] > 60]:
        del recent[ip]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/models/face_landmarker.task")
def landmarker_model():
    # Same model file the server uses, for the page's live guidance.
    return FileResponse(STATIC.parent / "models" / "face_landmarker.task", media_type="application/octet-stream",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/session")
def new_session(request: Request):
    now = time.monotonic()
    drop_old(now)
    q = recent[client_ip(request)]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= SESSIONS_PER_MIN:
        raise HTTPException(429, "too many sessions, try again in a minute")
    q.append(now)
    sid = secrets.token_urlsafe(16)
    actions = pick_actions()
    sessions[sid] = {"actions": actions, "created": now}
    return {"id": sid, "actions": actions, "expires_in": int(SESSION_TTL)}


@app.post("/api/session/{sid}/frames")
async def submit(sid: str, request: Request):
    size = request.headers.get("content-length")
    if size is None:
        raise HTTPException(411, "Content-Length required")
    if int(size) > MAX_BODY_BYTES:
        raise HTTPException(413, "upload larger than 20 MB")
    session = sessions.pop(sid, None)   # single use, even if the upload is rejected
    if session is None:
        raise HTTPException(404, "unknown or already used session")
    if time.monotonic() - session["created"] > SESSION_TTL:
        return {"decision": "FAIL", "reasons": ["EXPIRED"], "actions": session["actions"],
                "scores": {"passive": None, "v2": None, "v1se": None}, "frames": 0, "ms": 0, "details": {},
                "photos": []}

    form = await request.form(max_files=MAX_FRAMES, max_fields=2)
    files = form.getlist("frames")
    try:
        stamps = json.loads(form.get("times") or "")
    except ValueError:
        raise HTTPException(400, "times must be a JSON list of capture times in ms")
    if not MIN_FRAMES <= len(files) <= MAX_FRAMES or len(stamps) != len(files):
        raise HTTPException(400, f"send {MIN_FRAMES}-{MAX_FRAMES} frames with one capture time each")

    images = []
    for i, f in enumerate(files):
        data = await f.read(MAX_FRAME_BYTES + 1)
        if len(data) > MAX_FRAME_BYTES:
            raise HTTPException(413, f"frame {i} larger than 300 KB")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"frame {i} is not an image")
        h, w = img.shape[:2]
        if max(h, w) > MAX_SIDE:
            k = MAX_SIDE / max(h, w)
            img = cv2.resize(img, (round(w * k), round(h * k)))
        images.append(img)
    times = [(float(t) - float(stamps[0])) / 1000 for t in stamps]

    async with lock:
        result, picked = await run_in_threadpool(analyze, session["actions"], images, times)
    if picked:
        candidates[sid] = {"created": time.monotonic(), "images": picked}
    result["photos"] = [photos.preview(img) for img in picked]
    log.info("liveness %s… asked=%s %s %s passive=%s %d ms photos=%d %s", sid[:6], ",".join(session["actions"]),
             result["decision"], ",".join(result["reasons"]) or "-", result["scores"]["passive"], result["ms"],
             len(picked), result["details"])
    return result


@app.post("/api/session/{sid}/photo")
async def choose_photo(sid: str, request: Request):
    entry = candidates.get(sid)
    if entry is None or time.monotonic() - entry["created"] > PHOTO_TTL:
        raise HTTPException(404, "no photos for this session, or they expired")
    try:
        choice = (await request.json())["choice"]
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, 'send {"choice": <photo number>}')
    if not isinstance(choice, int) or not 0 <= choice < len(entry["images"]):
        raise HTTPException(400, f"choice must be 0-{len(entry['images']) - 1}")
    del candidates[sid]   # single use; the other photos are dropped
    return Response(photos.jpeg(entry["images"][choice]), media_type="image/jpeg",
                    headers={"Content-Disposition": 'attachment; filename="profile.jpg"'})


def stream_reasons(times):
    gaps = [b - a for a, b in zip(times, times[1:])]
    if any(g <= 0 for g in gaps):
        return ["BAD_FRAMES"]
    fps = (len(times) - 1) / times[-1]
    return [] if MIN_FPS <= fps <= MAX_FPS else ["BAD_FRAMES"]


def passive_check(images, frames):
    """Average passive score on up to 5 frontal, open-eye, good-quality frames."""
    seen = [f for f in frames if f.faces]
    if not seen:
        return None, [], {}
    base_yaw, _, base_blink = challenge.baseline(frames)
    picks = [i for i, f in enumerate(frames)
             if f.faces and abs(f.yaw - base_yaw) < challenge.STILL_DEG and f.blink < base_blink + 0.2]
    step = max(1, len(picks) // PASSIVE_FRAMES)
    passive, scores, checked = analyzers["passive"], [], []
    for i in picks[::step][:PASSIVE_FRAMES]:
        box = passive.face_box(images[i])
        if box is None:
            continue
        brightness, sharpness = passive.quality(images[i], box)
        checked.append((brightness, sharpness))
        if not BRIGHTNESS[0] <= brightness <= BRIGHTNESS[1] or sharpness < MIN_SHARPNESS:
            continue
        scores.append(passive.score(images[i], box))
    reasons = []
    sizes = sorted(f.size for f in seen)
    if sizes[len(sizes) // 2] < MIN_FACE or len(scores) < MIN_GOOD_FRAMES:
        reasons.append("LOW_QUALITY")
    stats = {"face_size": round(sizes[len(sizes) // 2], 2), "passive_frames": len(scores)}
    if checked:
        stats["brightness"] = round(sorted(b for b, _ in checked)[len(checked) // 2])
        stats["sharpness"] = round(sorted(s for _, s in checked)[len(checked) // 2])
    if not scores:
        return None, reasons, stats
    return {k: round(sum(s[k] for s in scores) / len(scores), 3) for k in scores[0]}, reasons, stats


def analyze(actions, images, times):
    start = time.perf_counter()
    frames = [analyzers["landmarks"].analyze(img) for img in images]
    reasons = stream_reasons(times) + challenge.check(actions, frames, times)
    passive, quality, stats = passive_check(images, frames)
    reasons = list(dict.fromkeys(reasons + quality))
    decision, reasons = decide(reasons, passive and passive["real"])
    picked = photos.candidates(images, frames, analyzers["passive"]) if decision == "PASS" else []
    return {
        "decision": decision,
        "reasons": reasons,
        "actions": actions,
        "scores": {"passive": passive and passive["real"], "v2": passive and passive["v2"],
                   "v1se": passive and passive["v1se"]},
        "frames": len(images),
        "ms": round((time.perf_counter() - start) * 1000),
        "details": {**challenge.summary(frames, times), **stats},
    }, picked
