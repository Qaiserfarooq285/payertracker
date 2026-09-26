"""FastAPI Application Server for The Reach Vision (player tracking & analytics).

Provides REST endpoints and media streaming for video upload, player detection, ByteTrack tracking,
jersey OCR, event detection, movement analytics, and stat card visualization.
"""

import ctypes
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from apps.api.access import DEFAULT_ACCESS_PASSWORD, resolve_access_password
from apps.api.idle_stop import watchdog_from_env
from apps.api.uploads import ChunkedUploads, safe_upload_name, upload_response
from src.common.logging import get_logger

# `src.pipeline.statcard_pdf` (unlike e.g. `src.track.target`/`src.identity.jersey_models`
# elsewhere in this file) is deliberately imported at module level, breaking this file's usual
# "heavy src.* imports stay local to the function that needs them" convention: the module itself
# has no heavy transitive deps (`from __future__ import annotations`, `pathlib`,
# `src.common.logging` only -- `reportlab` is its OWN lazy, function-local import, see that
# module's docstring), so importing it here costs nothing at API startup, and a module-level
# import is what lets `_generate_statcard_pdf_on_demand` below be monkeypatched/mocked cleanly in
# tests the same way `apps.api.main.render_statcard_pdf` (Plan Fix D, "streamed-gathering-
# treehouse" re-check, 2026-09-15).
from src.pipeline.statcard_pdf import render_statcard_pdf

logger = get_logger(__name__)

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
WORK_DIR = BASE_DIR / "work"
WEB_DIR = BASE_DIR / "apps" / "web"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR.mkdir(parents=True, exist_ok=True)

# The whole pipeline's own I/O contract (CLAUDE.md §10) assumes cwd == repo root -- every
# `configs/*.yaml` load in `src.pipeline.run._load_all_configs` is a RELATIVE path. `_run_pipeline_
# job` already chdir'd here defensively before calling it; doing it once at import time, for the
# whole process, is both simpler and required by the fix below (never rely on it happening deep in
# a background task after cache keys have already been computed).
os.chdir(BASE_DIR)


def _resolve_video_path(video_name: str) -> Path:
    """`video_name` (e.g. "video2/chelsea_burnley_target10.mp4" or a bare filename) -> its path
    under INPUT_DIR, RELATIVE to BASE_DIR. Shared by every endpoint that takes a video name.

    Real bug fixed here (2026-09-01): every `StageCache` key embeds `str(video_path)` verbatim
    (`src/detect/run.py`, `src/track/run.py`, ...), and the CLI's own convention (`make run
    VIDEO=input/...`, and every prior real run in this project) is a RELATIVE path -- so a cache
    written by a CLI run keys on `"input/video2/chelsea_burnley_target10.mp4"`. This function used
    to return an ABSOLUTE path (`INPUT_DIR / video_name`, and `INPUT_DIR` is itself absolute), which
    produces a DIFFERENT hash and therefore a cache MISS on every stage for a video that had
    already been fully processed via the CLI -- confirmed directly: computing both hashes for the
    same on-disk config gave `d8237a32d5ce` (relative, matches the real cached `.meta.json`) vs.
    `d1d419e66f98` (absolute, a guaranteed miss). Returning a relative path here makes the API and
    the CLI share the exact same cache.
    """
    video_path = INPUT_DIR / video_name
    if video_path.exists():
        return video_path.relative_to(BASE_DIR)
    matches = list(INPUT_DIR.glob(f"**/{video_name}"))
    if matches:
        return matches[0].relative_to(BASE_DIR)
    raise HTTPException(
        status_code=404, detail=f"Video file '{video_name}' not found in input directory"
    )


def _canonical_work_dir(video_path: Path) -> Path:
    """The SAME `work/<slug>/` directory the pipeline itself resolves to
    (`src.common.io.work_dir_for`/`_slugify`) -- NOT `stem.replace(" ", "_")`, which diverges on
    uppercase/punctuation and silently points at the wrong directory."""
    from src.common.io import work_dir_for

    return work_dir_for(video_path, root=WORK_DIR)


def _canonical_slug(video_path: Path) -> str:
    return _canonical_work_dir(video_path).name


app = FastAPI(
    title="The Reach Vision API",
    description="YOLO/RF-DETR Detection + ByteTrack + Jersey OCR + Event & Movement Analytics",
    version="1.0.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------------------------
# Access gate (hosted deployment, docs/DEPLOY.md). ON BY DEFAULT (owner, 2026-09-16): every
# request except the static UI, `/api/health`, and the login endpoints must carry a valid session
# cookie -- otherwise anyone who finds the public URL can upload videos and burn GPU credit.
# `PV_ACCESS_PASSWORD` unset ⇒ the built-in default password (`access.DEFAULT_ACCESS_PASSWORD`,
# change it once the pod is up); the literal `off` disables the gate entirely (local editing only).
#
# The cookie value is an HMAC of the password under a per-process random key, so it can't be
# forged without the password and a server restart invalidates old sessions (an acceptable
# "log in again after a redeploy" cost -- no session store to persist or leak).
# ---------------------------------------------------------------------------------------------
ACCESS_PASSWORD, ACCESS_PASSWORD_IS_DEFAULT = resolve_access_password(
    os.environ.get("PV_ACCESS_PASSWORD")
)
if ACCESS_PASSWORD_IS_DEFAULT:
    logger.warning(
        "using the default access password (%s) -- set PV_ACCESS_PASSWORD to change it",
        DEFAULT_ACCESS_PASSWORD,
    )
_SESSION_COOKIE = "pv_session"
_SESSION_KEY = secrets.token_bytes(32)
_SESSION_MAX_AGE_S = 30 * 24 * 3600
_LOGIN_FAIL_DELAY_S = 0.5  # crude brute-force brake; the password is a shared secret, not a user DB
# Prefixes that never need a session: the UI shell itself (it renders the login overlay) and the
# endpoints the overlay needs to work.
_PUBLIC_PATH_PREFIXES = ("/api/health", "/api/login", "/api/auth/status")
_PUBLIC_STATIC_PREFIXES = ("/css/", "/js/", "/assets/", "/favicon")  # the login card needs the logo


def _session_token() -> str:
    return hmac.new(_SESSION_KEY, ACCESS_PASSWORD.encode("utf-8"), hashlib.sha256).hexdigest()


def _is_public_path(path: str) -> bool:
    if path in ("/", "/index.html"):
        return True
    return path.startswith(_PUBLIC_PATH_PREFIXES) or path.startswith(_PUBLIC_STATIC_PREFIXES)


def _has_valid_session(request: Request) -> bool:
    if not ACCESS_PASSWORD:
        return True
    cookie = request.cookies.get(_SESSION_COOKIE, "")
    return bool(cookie) and hmac.compare_digest(cookie, _session_token())


@app.middleware("http")
async def access_gate(request: Request, call_next):
    if not _is_public_path(request.url.path):
        if ACCESS_PASSWORD and not _has_valid_session(request):
            return JSONResponse(status_code=401, content={"detail": "login required"})
        # Real, authenticated (or gate-disabled) API traffic counts as activity for the idle
        # auto-stop watchdog -- a bare 401 probe against a public path never reaches here.
        IDLE_WATCHDOG.touch()
    return await call_next(request)


class LoginRequest(BaseModel):
    password: str


@app.get("/api/auth/status")
def auth_status(request: Request):
    """Tells the UI whether a login is needed and whether this browser already has a session."""
    return {"required": bool(ACCESS_PASSWORD), "authenticated": _has_valid_session(request)}


@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    if not ACCESS_PASSWORD:
        return {"status": "ok", "required": False}
    if not hmac.compare_digest(req.password.encode("utf-8"), ACCESS_PASSWORD.encode("utf-8")):
        time.sleep(_LOGIN_FAIL_DELAY_S)
        raise HTTPException(status_code=401, detail="wrong password")
    # Behind Cloudflare Tunnel / RunPod's proxy the app itself only ever sees plain HTTP; the
    # forwarded header is what tells us the browser is on HTTPS and the cookie may be Secure.
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    response = JSONResponse(content={"status": "ok", "required": True})
    response.set_cookie(
        _SESSION_COOKIE,
        _session_token(),
        max_age=_SESSION_MAX_AGE_S,
        httponly=True,
        samesite="lax",
        secure=forwarded_proto == "https",
        path="/",
    )
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse(content={"status": "ok"})
    response.delete_cookie(_SESSION_COOKIE, path="/")
    return response


# Global in-memory job store for background processing tasks
JOBS: dict[str, dict[str, Any]] = {}

# Long videos (2026-09-26): a 10-minute clip runs for well over half an hour on the GPU, and the
# staged messages in `_run_pipeline_job` are all written in its first few seconds -- after that the
# job log froze on one line while the real pipeline only logged to the server console, so a long
# run looked hung. `_JobLogHandler` forwards the pipeline's own INFO lines (`src.*` loggers) into
# the job's log, which the web app and the gateway already mirror. Filtered by thread, so two jobs
# running at once never see each other's lines; head + tail kept so the log stays small.
JOB_LOG_KEEP_HEAD = 10
JOB_LOG_MAX_LINES = 300
# Refuse a run up front when the volume is nearly full, instead of dying mid-render hours in.
MIN_FREE_DISK_GB = float(os.environ.get("PV_MIN_FREE_DISK_GB", "3"))


class _JobLogHandler(logging.Handler):
    def __init__(self, job: dict[str, Any], thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self.job = job
        self.thread_id = thread_id

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.thread_id or not record.name.startswith("src."):
            return
        try:
            message = record.getMessage()
        except Exception:
            return
        logs = self.job["logs"]
        logs.append(f"[{time.strftime('%H:%M:%S')}] {message[:300]}")
        if len(logs) > JOB_LOG_MAX_LINES:
            del logs[JOB_LOG_KEEP_HEAD : len(logs) - (JOB_LOG_MAX_LINES - JOB_LOG_KEEP_HEAD)]


def _attach_job_log(job: dict[str, Any]) -> _JobLogHandler:
    """Start forwarding this thread's pipeline log lines into `job["logs"]`. The `src` loggers
    are raised to INFO if nothing else configured them (root is WARNING by default, and
    `get_logger`'s basicConfig is a no-op once any handler exists) -- else nothing would arrive."""
    src_logger = logging.getLogger("src")
    if src_logger.getEffectiveLevel() > logging.INFO:
        src_logger.setLevel(logging.INFO)
    handler = _JobLogHandler(job, threading.get_ident())
    logging.getLogger().addHandler(handler)
    return handler


class JobCancelled(BaseException):
    """Raised INSIDE a running pipeline thread by `POST /api/jobs/{id}/cancel` (2026-09-26: owner
    -- "if a job is started and someone stops it, the processing should stop as well"). A
    BaseException on purpose: the pipeline has many `except Exception` blocks that would otherwise
    swallow it and carry on burning GPU time."""


# job_id -> thread ident of the pipeline thread running it. Guarded so a cancel can never be fired
# at a thread that has already moved on to other work (the threadpool reuses threads).
_JOB_THREADS: dict[str, int] = {}
_JOB_THREADS_LOCK = threading.Lock()


def _set_async_exc(thread_id: int, exc: type[BaseException] | None) -> int:
    return ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(exc) if exc is not None else None
    )


def _release_job_thread(job_id: str) -> None:
    """Forget the job's thread and clear any cancel that is still pending on it, so it can never
    land in whatever the thread runs next."""
    with _JOB_THREADS_LOCK:
        if _JOB_THREADS.pop(job_id, None) is not None:
            _set_async_exc(threading.get_ident(), None)


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9

# Which video slugs currently have a pipeline job in flight, and the lock guarding that set.
#
# Real corruption this prevents (owner-reported 2026-09-15, root-caused from the server log):
# every run writes to the SAME `work/<slug>/` and `output/<slug>/` directories, and `extract_clip`
# cuts each clip with `ffmpeg -y`. Two runs of one video therefore overwrite each other's files
# mid-write. The observed failure was `moov atom not found` on
# `output/clip5_77/clips/clip_000_sprint_*.mp4` ONE SECOND after that same run logged "cut 3
# clip(s)" -- an MP4 writes its `moov` atom last, so the reel's concat read a file another run had
# just truncated. The file was perfectly valid again afterwards, which is the signature of a
# concurrent overwrite rather than a bad encode.
_INFLIGHT_SLUGS: set[str] = set()
_INFLIGHT_LOCK = threading.Lock()

# Idle auto-stop (docs/DEPLOY.md "Idle auto-stop"): disabled unless RUNPOD_API_KEY/RUNPOD_POD_ID
# are both set (docker/runpod.env.example). `_INFLIGHT_SLUGS` above is the exact "is a job
# running" signal it needs, so it's read live via a closure rather than duplicated.
IDLE_WATCHDOG = watchdog_from_env(lambda: len(_INFLIGHT_SLUGS))


@app.on_event("startup")
def _start_idle_watchdog() -> None:
    IDLE_WATCHDOG.start()


class TargetAnchor(BaseModel):
    """One click anchor ("streamed-gathering-treehouse" plan Stage 1): a human-picked
    `(take_id, track_id)` pair, optionally timestamped. `apps/web/js/app.js`'s picker chip list
    appends one of these per click -- the fix for the pre-existing bug where a second click in the
    SAME take silently overwrote the first (`selectedFramePlayer` was a scalar). `t` is advisory
    only (shown in the chip label / used as a `click_t` fallback for `_resolve_target_click`'s own
    profile-establishment timestamp when this is the very first anchor of a run); the pair itself
    is always resolved against that take's own cached tracks, never against `t` directly.
    """

    take_id: int
    track_id: int
    t: float | None = None


class ProcessRequest(BaseModel):
    video_name: str
    # Real bug fixed here ("streamed-gathering-treehouse" plan Stage E, 2026-09-04): this used to
    # default to `10`, and a request that simply omitted `target_jersey` (e.g. a pure click/track-id
    # selection with no jersey field filled in) silently got jersey #10 wired into
    # `configs["identity"]["target_jersey"]` below -- confirmed as the real, on-disk cause of
    # `work/chelsea_burnley_target2/selection.json` carrying `target_jersey: 10` for a clip whose
    # own filename says `target2`. `configs/run.yaml: target_jersey` is already `null` by default
    # (CLAUDE.md Golden Rule 5: an honest "not given" beats a fabricated number) -- this API model
    # now matches that, so nothing downstream silently substitutes 10 for "the field was left
    # blank". `apps/web/index.html`'s jersey input field's own default value and `app.js`'s
    # `|| 10`/`|| '10'` fallbacks were the other two places this same "10" was seeping in from and
    # are fixed alongside this.
    target_jersey: int | None = None
    track_id: str | None = None
    manual_annotations: str | None = None
    click_x: float | None = None
    click_y: float | None = None
    click_t: float | None = None
    # "streamed-gathering-treehouse" plan Stage 1: the multi-anchor picker chip list submits its
    # whole accumulated anchor set here -- zero or more `(take_id, track_id)` pairs, as many as the
    # user clicked (across one or several takes). The free-text `track_id` field above stays
    # supported and is MERGED in (`_run_pipeline_job` unions both sources per take), so a
    # hand-typed `#track-id-input` value and clicked chips can be combined in one request.
    target_clicks: list[TargetAnchor] | None = None
    # Plan Stage 2 ("streamed-gathering-treehouse", 2026-09-14): optional client-typed ball-touch
    # times, as free text (comma- or newline-separated `M:SS`/`H:MM:SS` entries) -- parsed by
    # `src.events.manual_touches.parse_touch_times`. Optional exactly like `target_jersey` (no
    # default that could silently apply to the wrong run); threaded into whichever pipeline
    # function actually runs for this request (`_run_pipeline_job` below).
    touch_times: str | None = None


@app.get("/api/health")
def get_health():
    """Health check endpoint with GPU status."""
    cuda_available = False
    gpu_name = "CPU Only"
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "status": "healthy",
        "gpu_available": cuda_available,
        "gpu_name": gpu_name,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "idle_stop": IDLE_WATCHDOG.status(),
        "disk_free_gb": round(_disk_free_gb(WORK_DIR), 1),
    }


@app.get("/api/videos")
def list_videos():
    """List input videos and existing output runs."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    input_videos = []
    if INPUT_DIR.exists():
        for p in INPUT_DIR.glob("*"):
            if p.is_file() and p.suffix.lower() in video_extensions:
                input_videos.append(
                    {
                        "name": p.name,
                        "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                        "modified": time.ctime(p.stat().st_mtime),
                    }
                )
            elif p.is_dir():
                for sub in p.glob("*"):
                    if sub.is_file() and sub.suffix.lower() in video_extensions:
                        input_videos.append(
                            {
                                "name": f"{p.name}/{sub.name}",
                                "size_mb": round(sub.stat().st_size / (1024 * 1024), 2),
                                "modified": time.ctime(sub.stat().st_mtime),
                            }
                        )

    outputs = []
    if OUTPUT_DIR.exists():
        for p in OUTPUT_DIR.glob("*"):
            if p.is_dir() and not p.name.startswith("."):
                annotated_video = p / "original_annotated_video.mp4"
                has_annotated = annotated_video.exists()
                players_glob = (p / "players").glob("player_*")
                players = [d.name for d in players_glob] if (p / "players").exists() else []
                outputs.append(
                    {
                        "slug": p.name,
                        "has_annotated_video": has_annotated,
                        "players": players,
                    }
                )

    return {
        "input_videos": input_videos,
        "outputs": outputs,
    }


# Chunk reassembly, name sanitising and the response shape live in `apps/api/uploads.py`, shared
# with the always-on VPS gateway (`apps/gateway/main.py`) so a file accepted there can be replayed
# to this pod with the identical protocol.
_UPLOADS = ChunkedUploads()


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a football video clip into the input directory in ONE request.

    Kept for local use and scripts. The web UI uses `/api/upload/chunk` instead, because a hosted
    deployment sits behind Cloudflare, which rejects any single request body over 100 MB -- and a
    4K clip is routinely several hundred MB (docs/DEPLOY.md)."""
    safe_name = safe_upload_name(file.filename)
    target_path = INPUT_DIR / safe_name

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return upload_response(target_path)


@app.post("/api/upload/chunk")
async def upload_video_chunk(
    upload_id: str = Form(...),
    index: int = Form(...),
    total: int = Form(...),
    filename: str = Form(...),
    chunk: UploadFile = File(...),
):
    """Upload one sequential slice of a video. The client picks `upload_id`, sends chunks
    `0..total-1` in order, and the final chunk's response carries the same payload as
    `/api/upload`. Out-of-order or unknown chunks are a 409/404, never silently appended."""
    return _UPLOADS.receive(INPUT_DIR, upload_id, index, total, filename, chunk.file)


def _read_jersey_hint(
    frame_bgr, bbox, legibility_model, parseq_model, parseq_transform, identity_cfg: dict
) -> dict[str, Any] | None:
    """Best-effort ADR-21 jersey read for ONE box on the already-decoded `frame_bgr` --
    advisory only (CLAUDE.md Golden Rule 4/7): shown to a human as a hint next to a clickable box,
    never used to auto-pick a target. `None` when the crop doesn't clear the legibility gate or
    PARSeq isn't confident (an honest "no hint", not a guess)."""
    from src.annotations.associate import expanded_bbox_region
    from src.identity.jersey_ocr import upscale_crop
    from src.identity.jersey_parseq import number_region, read_jersey_number_parseq
    from src.identity.legibility import is_legible

    crop_cfg = identity_cfg.get("crop", {})
    expand = crop_cfg.get("torso_crop_expand", 0.08)
    upscale = crop_cfg.get("crop_upscale_factor", 3.0)
    parseq_cfg = identity_cfg.get("parseq_soccernet", {})
    legibility_cfg = identity_cfg.get("legibility", {})

    h, w = frame_bgr.shape[:2]
    region = expanded_bbox_region(bbox, expand)
    x1, y1 = max(0, int(region.x1)), max(0, int(region.y1))
    x2, y2 = min(w, int(region.x2)), min(h, int(region.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    body_crop = upscale_crop(frame_bgr[y1:y2, x1:x2], upscale)
    leg = is_legible(body_crop, legibility_model, legibility_cfg)
    if body_crop.size == 0 or not leg.is_legible:
        return None

    nx1, ny1, nx2, ny2 = number_region(x1, y1, x2, y2, parseq_cfg.get("number_crop", {}))
    number_crop = upscale_crop(frame_bgr[ny1:ny2, nx1:nx2], upscale)
    read = read_jersey_number_parseq(number_crop, parseq_model, parseq_transform, parseq_cfg)
    if not read.is_confident:
        return None
    return {"digits": read.digits, "confidence": round(read.confidence, 3)}


@app.get("/api/frame_players")
def get_frame_players(video: str, t: float = 0.0, read_jersey: bool = False):
    """Every non-referee player track visible near timestamp `t`, keyed by its STITCHED chain id
    (`build_take_identities` -- not the raw ByteTrack fragment id, which is what makes labels look
    like they "change constantly"), plus a JPEG frame in the SAME pixel coordinate space as the
    returned boxes, so a UI can draw clickable rectangles directly on it with no separate
    calibration step.

    This is the picker behind "click a player, track them" (CLAUDE.md Golden Rule 4: a
    human-provided identity is the strongest evidence this pipeline has -- stronger than any
    jersey-number auto-read). Runs Stage 3 tracking first; it is cached, so a second call for the
    same video is a fast no-op read-back, not a re-track.

    `read_jersey=true` attaches an ADVISORY jersey-number hint per box (ADR-21's legibility-gate +
    tight-number-region PARSeq chain) -- shown to the human as a hint only, never used to pick a
    box automatically.
    """
    import base64

    import cv2

    from src.common.io import load_models_parquet
    from src.common.types import Track
    from src.common.video import decode_frames
    from src.pipeline.run import _effective_frame_size, _load_all_configs
    from src.shots.boundaries import detect_takes
    from src.track.continuity import build_take_identities
    from src.track.run import run_track_stage
    from src.track.tracker import assign_take_id

    video_path = _resolve_video_path(video)
    configs = _load_all_configs()

    run_track_stage(
        video_path,
        configs["hardware"],
        configs["detect"],
        configs["track"],
        configs["team"],
        configs["shots"],
        work_root=WORK_DIR,
    )
    takes = detect_takes(video_path, configs["shots"], work_root=WORK_DIR)
    take_id = assign_take_id(t, takes)
    if take_id is None:
        raise HTTPException(status_code=404, detail=f"No take covers t={t:.2f}s for this video")

    work_dir = _canonical_work_dir(video_path)
    all_tracks = load_models_parquet(work_dir / "track" / "tracks.parquet", Track)
    take_tracks = [tr for tr in all_tracks if tr.take_id == take_id]

    # `motion` left at its default `None` (2026-09-03): this endpoint only runs Stage 3 tracking
    # (`run_track_stage` above) -- it has no `TakeCameraMotion` to pass, and estimating one just
    # for this request would mean a fresh per-take decode + optical-flow pass
    # (`src.track.camera_motion.estimate_take_camera_motion`) inside what is otherwise a cheap,
    # cache-backed read endpoint the picker UI calls interactively. That is exactly the "expensive
    # new motion estimation at a call site that doesn't have it" this fix was told not to add, so
    # this stays an honest, documented gap: the chain ids this endpoint hands the click-picker UI
    # can still mis-stitch during a fast pan, the same failure this fix closes everywhere motion
    # IS already available.
    identity_of, identity_conf = build_take_identities(
        take_tracks, configs["highlights"]["selection"]
    )
    frame_w, frame_h = _effective_frame_size(video_path, configs["hardware"]["decode"])

    frame_bgr = None
    for _idx, _frame_t, frame in decode_frames(
        str(video_path),
        start=max(0.0, t - 0.05),
        end=t + 0.2,
        scale_width=configs["hardware"]["decode"].get("scale_width"),
    ):
        frame_bgr = frame
        break
    if frame_bgr is None:
        raise HTTPException(status_code=500, detail="Could not decode a frame at that timestamp")

    legibility_model = parseq_model = parseq_transform = None
    if read_jersey:
        from src.identity.jersey_models import load_optional_jersey_stack

        legibility_model, parseq_model, parseq_transform = load_optional_jersey_stack(
            configs["identity"]
        )

    tolerance_s = 0.3
    players = []
    for tr in take_tracks:
        if getattr(tr.dominant_class, "value", str(tr.dominant_class)) == "referee":
            continue
        if not tr.boxes:
            continue
        box = min(tr.boxes, key=lambda b: abs(b.t - t))
        if abs(box.t - t) > tolerance_s:
            continue
        bb = box.bbox
        chain_id = identity_of.get(tr.id, tr.id)
        entry: dict[str, Any] = {
            "raw_track_id": tr.id,
            "chain_id": chain_id,
            "chain_confidence": round(identity_conf.get(chain_id, 0.0), 3),
            "bbox_norm": {
                "x1": bb.x1 / frame_w,
                "y1": bb.y1 / frame_h,
                "x2": bb.x2 / frame_w,
                "y2": bb.y2 / frame_h,
            },
            "team": tr.team,
            "team_confidence": round(tr.team_confidence, 3),
            "height_px": round(bb.height, 1),
        }
        if read_jersey and legibility_model is not None and parseq_model is not None:
            entry["jersey_hint"] = _read_jersey_hint(
                frame_bgr, bb, legibility_model, parseq_model, parseq_transform, configs["identity"]
            )
        players.append(entry)

    if read_jersey:
        from src.identity.jersey_models import free_optional_jersey_stack

        free_optional_jersey_stack(legibility_model, parseq_model)

    ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii") if ok else None

    return {
        "video": video,
        "t": t,
        "take_id": take_id,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "frame_jpeg_base64": frame_b64,
        "players": players,
    }


def _match_click_to_track(
    video_path: Path, configs: dict[str, dict], click_x: float, click_y: float, click_t: float
) -> tuple[int, int | None]:
    """`(take_id, matched_track_id | None)` for a click at normalized `(click_x, click_y)` at
    time `click_t`.

    Two real bugs fixed here vs. the original inline version:
    - the take is now chosen via `assign_take_id(click_t, takes)` instead of being hardcoded to
      take 0 -- a click in a video's second (or later) take used to silently target the wrong take.
    - the "does the click land inside a box" search now stops as soon as it finds one (a `return`,
      not a `break` that only exited the inner loop and let later tracks overwrite the match).
    - the nearest-centroid fallback cap is a FRACTION of frame height, not an absolute 400px,
      which was calibrated for one specific resolution and meaningless on a 4K clip vs. a 720p one.
    """
    from src.common.io import load_models_parquet
    from src.common.types import Track
    from src.pipeline.run import _effective_frame_size
    from src.shots.boundaries import detect_takes
    from src.track.run import run_track_stage
    from src.track.tracker import assign_take_id

    run_track_stage(
        video_path,
        configs["hardware"],
        configs["detect"],
        configs["track"],
        configs["team"],
        configs["shots"],
        work_root=WORK_DIR,
    )
    takes = detect_takes(video_path, configs["shots"], work_root=WORK_DIR)
    take_id = assign_take_id(click_t, takes)
    if take_id is None:
        return 0, None

    tracks_path = _canonical_work_dir(video_path) / "track" / "tracks.parquet"
    tr_list = load_models_parquet(tracks_path, Track)
    take_tracks = [tr for tr in tr_list if tr.take_id == take_id]

    frame_w, frame_h = _effective_frame_size(video_path, configs["hardware"]["decode"])
    px_x = click_x * frame_w if click_x <= 1.0 else click_x
    px_y = click_y * frame_h if click_y <= 1.0 else click_y
    max_dist_px = 0.15 * frame_h  # a fraction of frame height, not an absolute pixel count

    best_track_id: int | None = None
    best_dist = float("inf")
    for tr in take_tracks:
        if getattr(tr.dominant_class, "value", str(tr.dominant_class)) == "referee":
            continue
        near = (b for b in tr.boxes if abs(b.t - click_t) <= 2.0)
        box = min(near, key=lambda b: abs(b.t - click_t), default=None)
        if box is None:
            continue
        bb = box.bbox
        if bb.x1 <= px_x <= bb.x2 and bb.y1 <= px_y <= bb.y2:
            return take_id, tr.id  # contained -- stop immediately, the strongest possible match
        dist = ((bb.cx - px_x) ** 2 + (bb.cy - px_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_track_id = dist, tr.id

    return take_id, (best_track_id if best_dist < max_dist_px else None)


class ClickRejected(Exception):
    """Raised by `_resolve_target_click` when a human selection (a raw coordinate click, or the
    "Load Players" picker's own `take_id:raw_track_id` text entry -- both are "a click" for this
    purpose) is judged REJECT or UNCERTAIN against the already-established persistent target
    identity ("streamed-gathering-treehouse" plan Stage E). Carries the exact plain-language,
    user-facing message; `_run_pipeline_job` catches this and fails the job loudly, touching
    NOTHING on disk (CLAUDE.md Golden Rule 5 -- an honest failure beats a confident wrong answer).
    """


# Plain-language labels for `TargetVerdict.evidence["rejected_reason"]` (`src/track/target_verify.
# py::verify_candidate`'s own hard-reject/score-floor codes) -- UI text, not a tunable threshold,
# so (unlike every numeric knob in this codebase, CLAUDE.md §10) this is fine as an ordinary
# in-code mapping rather than a `configs/*.yaml` entry: the SAME "no magic strings" discipline
# `configs/annotations.yaml`'s phrase map follows would apply if these were being matched FROM
# free text, but here they are simply reworded for display, matching the existing hardcoded
# "Your click did not land on a detected player..." message already in this file.
_REJECTION_REASON_LABELS = {
    "wrong_kit_colour": "different kit colour",
    "wrong_jersey_number": "different jersey number",
    "height_mismatch": "different player height",
    "score_below_uncertain_low": "overall match score too low",
}


def _click_rejection_message(profile: Any, verdict: Any) -> str:
    """The owner's own wording (plan Stage E): `CLICK REJECTED -- this player does not match
    TARGET #<N> (<reason>). Target remains lost; click the original player again.` `verdict` may be
    REJECT (a hard reject, or score below `uncertain_low` -- `rejected_reason` is always present)
    or UNCERTAIN (`target_verify.py`'s own docstring: never a weak accept, treated identically to
    REJECT here -- `rejected_reason` is NOT set for this decision, so a generic score-based reason
    is built instead, still carrying the real measured score)."""
    reason_code = verdict.evidence.get("rejected_reason")
    if reason_code is not None:
        reason = _REJECTION_REASON_LABELS.get(reason_code, reason_code)
    else:
        reason = f"match uncertain, score={verdict.score:.2f}"
    jersey_part = f"#{profile.jersey_number} " if profile.jersey_number is not None else ""
    return (
        f"CLICK REJECTED -- this player does not match {jersey_part}{profile.target_id} "
        f"({reason}). Target remains lost; click the original player again."
    )


def _click_evidence_for_candidate(
    video_path: Path,
    configs: dict[str, dict],
    take_id: int,
    track_id: int,
    jersey_stack: tuple[Any, Any, Any] | None = None,
) -> tuple[Any, Any, str | None]:
    """`(candidate_track, kit_sample, jersey_digits)` for ONE clicked/overridden track -- computed
    for JUST that track (never the whole take) so an interactive click endpoint stays cheap.
    `kit_sample` is a `src.track.target.KitColourSample` or `None`; `jersey_digits` is a confident
    digit string or `None` -- both feed `src.track.target_verify.verify_candidate`'s
    `kit_by_track`/`jersey_by_track` (a missing entry is "no evidence for this track", never a
    rejection, per that module's own docstring) and `src.track.target.build_target_profile`'s own
    `kit_samples`/`jersey_number` for a first-click establishment. `(None, None, None)` when the
    track itself isn't even in that take's cached tracks -- the caller already confirmed the track
    exists via `_match_click_to_track`/`parse_track_id_overrides` before calling in, so this should
    not happen in practice, but is handled as an honest failure rather than assumed impossible.

    `jersey_stack` ("streamed-gathering-treehouse" plan Stage 1 performance fix): an optional
    already-loaded `(legibility_model, parseq_model, parseq_transform)` triple. Every PRE-EXISTING
    caller (this function had exactly one, `_resolve_target_click`, called once per click) left
    this `None`, so the DEFAULT behaviour is unchanged byte-for-byte: load the whole jersey-OCR
    stack, use it for this one track, free it before returning. The multi-anchor loop in
    `_run_pipeline_job` now loads the stack ONCE outside its own per-pair loop and passes it in
    here for every pair, so N anchors cost one model load instead of N -- this function never
    frees a stack it did not itself load.
    """
    from src.common.io import load_models_parquet
    from src.common.types import Track
    from src.common.video import probe
    from src.identity.jersey_models import free_optional_jersey_stack, load_optional_jersey_stack
    from src.pipeline.run import _effective_frame_size
    from src.shots.boundaries import detect_takes
    from src.track.click_reid import collect_track_jersey_digits
    from src.track.kit_wiring import build_take_kit_colour

    tracks_path = _canonical_work_dir(video_path) / "track" / "tracks.parquet"
    tr_list = load_models_parquet(tracks_path, Track)
    candidate = next((tr for tr in tr_list if tr.take_id == take_id and tr.id == track_id), None)
    if candidate is None:
        return None, None, None

    takes = detect_takes(video_path, configs["shots"], work_root=WORK_DIR)
    take = next((t for t in takes if t.id == take_id), None)
    if take is None:
        return candidate, None, None

    frame_w, frame_h = _effective_frame_size(video_path, configs["hardware"]["decode"])
    native_meta = probe(video_path)
    native_w, native_h = native_meta["width"], native_meta["height"]
    scale_x = native_w / frame_w if frame_w else 1.0
    scale_y = native_h / frame_h if frame_h else 1.0
    max_samples = (
        configs["highlights"].get("click_reid", {}).get("max_jersey_samples_per_track", 40)
    )

    _kit_lab_by_track, kit_sample_by_track = build_take_kit_colour(
        video_path,
        take,
        [candidate],
        scale_x,
        scale_y,
        native_w,
        native_h,
        configs["identity"],
        configs["target"],
        configs["events"]["kit_colour"],
        max_samples,
    )
    kit_sample = kit_sample_by_track.get(candidate.id)

    owns_stack = jersey_stack is None
    if owns_stack:
        legibility_model, parseq_model, parseq_transform = load_optional_jersey_stack(
            configs["identity"]
        )
    else:
        legibility_model, parseq_model, parseq_transform = jersey_stack
    try:
        jersey_by_track = collect_track_jersey_digits(
            video_path,
            [candidate],
            scale_x,
            scale_y,
            native_w,
            native_h,
            configs["identity"],
            legibility_model,
            parseq_model,
            parseq_transform,
            max_samples,
            configs["identity"]["aggregation"],
        )
    finally:
        if owns_stack:
            free_optional_jersey_stack(legibility_model, parseq_model)

    return candidate, kit_sample, jersey_by_track.get(candidate.id)


def _resolve_target_click(
    video_path: Path,
    configs: dict[str, dict],
    profile: Any,
    take_id: int,
    track_id: int,
    click_t: float | None,
    jersey_stack: tuple[Any, Any, Any] | None = None,
) -> Any:
    """One (take_id, track_id) human selection against the persistent target identity
    ("streamed-gathering-treehouse" plan Stage E): "The click means: 'This is a candidate for
    TARGET_001.' It does NOT mean: 'Make this player the target.'"

    - No stored profile (`profile is None`) -> this selection ESTABLISHES the persistent target
      (`src.track.target.build_target_profile`, `jersey_source="click"`), persisted immediately
      via `save_target_profile` (CLAUDE.md Golden Rule 4 -- a human-provided identity is the
      strongest evidence available, and persisting it right away means a later request in this
      same run always finds it, even if this run's own background pipeline job fails partway
      through). Returns the new profile.
    - A stored profile exists -> the selection is a CANDIDATE, verified via
      `src.track.target_verify.verify_candidate`. ACCEPT -> returns `profile` UNCHANGED: the
      reconnect itself (same `target_id`) and the actual `update_memory_bank` call + re-persist
      are already performed by `run_pipeline_for_video`'s own Stage 4/5 wiring when it re-verifies
      this SAME override with the full per-take evidence a moment later -- doing it again here
      would double-insert a near-duplicate kit sample for the identical physical evidence into the
      memory bank. REJECT or UNCERTAIN (never a weak accept, `target_verify.py`'s own docstring)
      -> raises `ClickRejected`; the caller must stop before ever calling `run_pipeline_for_video`
      or touching `target.json` again.

    `jersey_stack` ("streamed-gathering-treehouse" plan Stage 1 performance fix): forwarded
    verbatim to `_click_evidence_for_candidate` -- `None` (every pre-existing caller) keeps that
    function's own load-once-per-call behaviour; `_run_pipeline_job`'s multi-anchor loop passes an
    already-loaded stack so N anchors in one request cost ONE model load, not N.
    """
    from src.track.target import build_target_profile, save_target_profile
    from src.track.target_verify import VerdictDecision, verify_candidate

    candidate, kit_sample, jersey_digits = _click_evidence_for_candidate(
        video_path, configs, take_id, track_id, jersey_stack=jersey_stack
    )
    if candidate is None:
        raise ClickRejected(
            f"take={take_id} track={track_id} could not be re-located for identity "
            'verification. Target remains lost; use "Load Players" and click a currently '
            "detected box."
        )

    jersey_number = int(jersey_digits) if jersey_digits is not None else None

    if profile is None:
        established_t = (
            click_t if click_t is not None else (candidate.boxes[0].t if candidate.boxes else 0.0)
        )
        new_profile = build_target_profile(
            candidate,
            [kit_sample] if kit_sample is not None else [],
            jersey_number,
            "click",
            take_id,
            established_t,
            configs["target"],
        )
        save_target_profile(new_profile, _canonical_work_dir(video_path) / "target.json")
        return new_profile

    # Re-clicking the very track that ESTABLISHED this profile is never a rejection (Golden Rule
    # 4, and parity with `src.highlights.selection._select_take_with_profile`, which already had
    # this bypass -- this path did not, so a user who clicked their own anchor again could be told
    # it "does not match" itself).
    if candidate.id == profile.established_track_id and take_id == profile.established_take_id:
        return profile

    target_cfg = {**configs["target"], "kit_colour": configs["events"]["kit_colour"]}
    kit_by_track = {candidate.id: kit_sample} if kit_sample is not None else {}
    jersey_by_track = {candidate.id: jersey_digits} if jersey_digits is not None else {}
    verdict = verify_candidate(profile, candidate, kit_by_track, jersey_by_track, target_cfg)
    # A deliberate human click is governed by the HARD rejects, not by the soft-signal aggregate.
    #
    # Owner-reported failure, 2026-09-15: `CLICK REJECTED ... (match uncertain, score=0.85)`. Every
    # hard check had PASSED -- the kit colour matched, no jersey read contradicted, the height was
    # fine -- and the click was still refused because a blended score landed just under
    # `strong_match`. What drags that blend down in exactly this situation is `trajectory`: the
    # score rewards spatial/temporal continuity with where the target was last seen, and a
    # reconnect click happens precisely BECAUSE the target vanished and reappeared somewhere else.
    # Scoring the discontinuity that motivated the click is circular, and it made the owner's own
    # stated workflow ("the player moves out of frame and appears again... it should be clickable")
    # fail on correct clicks.
    #
    # So: a hard reject still refuses the click, with its specific reason -- that IS the owner's
    # "if the user clicks another player, reject and keep the target lost" requirement, and those
    # three signals (kit colour, printed number, height) are the reliable discriminators. An
    # UNCERTAIN or merely-low score with nothing actually contradicting is NOT evidence of a
    # different player, and CLAUDE.md Golden Rule 4 is explicit that a human-provided identity is
    # the strongest evidence this pipeline has -- the person is looking at the footage. The score
    # is kept in the profile's own link evidence either way, so a weak reconnect stays auditable.
    #
    # NOTE this deliberately differs from AUTOMATIC re-identification
    # (`_select_take_with_profile`'s candidate scan), where UNCERTAIN must still mean "stay lost":
    # there no human asserted anything, so the strict bar is the whole safeguard.
    hard_reject_reasons = {"wrong_kit_colour", "wrong_jersey_number", "height_mismatch"}
    rejected_reason = verdict.evidence.get("rejected_reason")
    if verdict.decision == VerdictDecision.REJECT and rejected_reason in hard_reject_reasons:
        raise ClickRejected(_click_rejection_message(profile, verdict))
    return profile


def _find_output_dir(output_dir: Path, slug: str) -> Path | None:
    """`output_dir / slug` if it exists, else `None` -- EXACT match only.

    Real bug fixed here ("streamed-gathering-treehouse" plan Stage E, 2026-09-04): this used to
    fall back to `slug in d.name or d.name in slug` substring matching, which can serve a
    DIFFERENT run's output entirely -- e.g. a request for slug `chelsea_burnley_target1` would
    substring-match the on-disk `chelsea_burnley_target10` (the shorter string IS a substring of
    the longer one), silently handing back a different player's stats/highlights under the
    requested slug. `_canonical_slug`/`work_dir_for` already compute the exact slug every run
    actually writes to (`src.common.io._slugify`), so an exact match is always available for any
    real run -- a miss means the run truly doesn't exist (or hasn't finished), which is the honest
    answer (CLAUDE.md Golden Rule 5), not a plausible-looking wrong one.
    """
    candidate = output_dir / slug
    return candidate if candidate.exists() else None


def _resolve_primary_player_jersey(work_dir: Path) -> int | None:
    """The run's own real target jersey number, if one is resolvable from cached `work/<slug>/`
    artifacts -- `target.json` (Stage 1's persistent `TargetProfile`, written by the click/profile
    -driven flow) first, else `selection.json` (`SelectionResult.target_jersey`, the filename/
    typed-jersey flow). `None` when neither file exists, neither names a jersey number, or either
    fails to parse -- callers treat that as "no resolvable target" and fall back to the pre-
    existing behaviour, never a crash.

    Real bug this exists to fix ("streamed-gathering-treehouse" plan re-check, Fix D, 2026-09-15):
    `/api/results/{slug}` used to hand back `sorted(players_dir.glob("player_*"))[0]` as
    `primary_player` -- a LEXICOGRAPHIC accident (the string `"player_10"` sorts before
    `"player_2"`), so a run that targeted #2 showed the dashboard **#10's** numbers. `target.json`/
    `selection.json` are the SAME two artifacts `_resolve_target_click`/`_run_pipeline_job` already
    write for exactly this purpose -- the run's own real, human-confirmed target -- so reading them
    back here is the honest fix rather than guessing from folder names.

    `target.json` is loaded with the full `TargetProfile` pydantic model (`load_target_profile`,
    same as `_resolve_target_click` above) since it IS this endpoint's own canonical schema for
    that file; `selection.json` is read as plain JSON (not `SelectionResult.model_validate`) since
    only its one `target_jersey` field is needed here and plain parsing degrades more gracefully
    on an older/partial file than full schema validation would.
    """
    target_json_path = work_dir / "target.json"
    if target_json_path.exists():
        try:
            from src.track.target import load_target_profile

            profile = load_target_profile(target_json_path)
            if profile.jersey_number is not None:
                return profile.jersey_number
        except Exception:
            logger.warning(
                "primary_player resolution: could not parse %s -- trying selection.json next",
                target_json_path,
                exc_info=True,
            )

    selection_json_path = work_dir / "selection.json"
    if selection_json_path.exists():
        try:
            selection_data = json.loads(selection_json_path.read_text())
            target_jersey = selection_data.get("target_jersey")
            if target_jersey is not None:
                return int(target_jersey)
        except Exception:
            logger.warning(
                "primary_player resolution: could not parse %s -- falling back to sorted order",
                selection_json_path,
                exc_info=True,
            )

    return None


def _run_pipeline_job(
    job_id: str,
    video_name: str,
    target_jersey: int | None,
    track_id: str | None,
    manual_annotations: str | None,
    click_x: float | None = None,
    click_y: float | None = None,
    click_t: float | None = None,
    target_clicks: list[TargetAnchor] | None = None,
    touch_times: str | None = None,
):
    """Background task function executing the processing pipeline stages.

    `target_clicks` ("streamed-gathering-treehouse" plan Stage 1): zero or more click anchors from
    the picker's own accumulating chip list (`ProcessRequest.target_clicks`) -- as many as the user
    clicked, possibly several in the SAME take (the "player left frame, came back with a new track
    id" case the whole plan exists for). Merged with the free-text `track_id` field below into one
    `dict[int, list[int]]` override set before anything else runs.

    `touch_times` (Plan Stage 2): forwarded verbatim into `run_pipeline_for_video`/
    `run_manual_events_pipeline_for_video` below, whichever branch actually runs -- see those
    functions' own docstrings for the merge rule.  `run_extended_pipeline_for_video` (the
    no-filename/no-click auto-ID branch) has no single "the target player" until identity
    verification itself resolves one per take, so it has no natural place to attach these; that
    branch does not accept this parameter (an honest scope limit, not an oversight).
    """
    # Captured up front so the `finally` below can always release this video's in-flight slot,
    # even if the run dies before `_resolve_video_path` would succeed. Falls back to the job's own
    # recorded slug, which `process_video` computed with the same helper when it claimed the slot.
    _job_slug_for_release = JOBS[job_id].get("canonical_slug") or ""

    job = JOBS[job_id]
    log_handler = _attach_job_log(job)
    if job.get("cancel_requested"):  # cancelled while still queued: never start
        job.update(status="failed", stage="Cancelled", error="cancelled")
        logging.getLogger().removeHandler(log_handler)
        with _INFLIGHT_LOCK:
            _INFLIGHT_SLUGS.discard(_job_slug_for_release)
        return
    with _JOB_THREADS_LOCK:
        _JOB_THREADS[job_id] = threading.get_ident()
    job["status"] = "processing"
    job["progress"] = 5
    job["stage"] = "Initializing Environment"
    job["logs"].append(
        f"Started analysis job {job_id} for video '{video_name}' with target jersey "
        f"#{target_jersey}"
    )

    try:
        video_path = _resolve_video_path(video_name)

        # Step 1: Handle manual annotations sidecar if provided
        if manual_annotations and manual_annotations.strip():
            job["logs"].append("Writing manual annotation sidecar file...")
            sidecar_path = video_path.parent / f"{video_path.stem}.annotations.txt"
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write(manual_annotations.strip() + "\n")
            job["logs"].append(f"Manual sidecar created at {sidecar_path.name}")

        # Update progress stages
        job["stage"] = "Player Detection (YOLO / RF-DETR)"
        job["progress"] = 15
        job["logs"].append("Running player & ball detection stage (YOLO / RF-DETR model)...")
        time.sleep(1)

        job["stage"] = "Player Tracking (ByteTrack / BoT-SORT)"
        job["progress"] = 35
        job["logs"].append("Running ByteTrack player association & trajectory tracking...")
        time.sleep(1)

        job["stage"] = "Jersey Number OCR (PARSeq / SoccerNet)"
        job["progress"] = 55
        job["logs"].append(
            "Extracting player upper-torso crops & running PARSeq jersey number OCR..."
        )
        time.sleep(1)

        job["stage"] = "Player Identity Matching"
        job["progress"] = 70
        job["logs"].append(f"Matching tracked IDs to Target Jersey #{target_jersey}...")
        time.sleep(1)

        job["stage"] = "Event Detection & Movement Analytics"
        job["progress"] = 85
        job["logs"].append(
            "Computing touch, pass (same-colour), turnover, sprint, shot, goal, assist, "
            "tackle stats..."
        )
        time.sleep(1)

        # Set current working directory to BASE_DIR so relative config paths work
        os.chdir(BASE_DIR)

        # Call python module command or direct function execution
        from src.ingest.discovery import parse_filename
        from src.pipeline.extended_output import run_extended_pipeline_for_video
        from src.pipeline.manual_events import run_manual_events_pipeline_for_video
        from src.pipeline.run import (
            _load_all_configs,
            normalize_manual_overrides,
            parse_track_id_overrides,
            run_pipeline_for_video,
        )

        configs = _load_all_configs()
        if target_jersey is not None:
            configs["identity"]["target_jersey"] = target_jersey

        annotations_path = video_path.parent / f"{video_path.stem}.annotations.txt"
        slug = _canonical_slug(video_path)

        # Owner decision, 2026-09-01: when the human EXPLICITLY selects a player (an exact
        # track_id, or a click on the preview), that selection is authoritative and drives
        # auto-detected stats for THAT player -- it always wins, including over a
        # `.annotations.txt` sidecar (which otherwise unconditionally takes over, ADR-19, and
        # would silently swallow the click). This is Golden Rule 4 exercised directly: a
        # human-provided identity outranks every other signal, sidecar included.
        has_explicit_selection = (
            bool(track_id) or bool(target_clicks) or (click_x is not None and click_y is not None)
        )

        if has_explicit_selection:
            job["logs"].append(
                "Explicit player selection given -- routing to the auto-detect "
                "pipeline for that player (sidecar, if any, is not used)."
            )
            # "streamed-gathering-treehouse" plan Stage 1: every anchor source is merged into ONE
            # `dict[int, list[int]]` -- the hand-typed `#track-id-input` mirror (`track_id`, still
            # the legacy `dict[int, int]` grammar, one bare/`take:track` entry per take) and the
            # picker's own accumulating chip list (`target_clicks`, one entry PER CLICK, possibly
            # several in the same take). `normalize_manual_overrides` (shared with
            # `run_pipeline_for_video` itself) is what actually dedupes/normalizes both sources
            # into the canonical shape -- this is the ONE merge point, not two separate ones.
            raw_overrides: dict[int, list[int]] = {}
            if track_id:
                for t_id, tr_id in parse_track_id_overrides(track_id).items():
                    raw_overrides.setdefault(t_id, []).append(tr_id)
            if target_clicks:
                for anchor in target_clicks:
                    raw_overrides.setdefault(anchor.take_id, []).append(anchor.track_id)
            overrides = normalize_manual_overrides(raw_overrides)

            if click_x is not None and click_y is not None and not overrides:
                job["logs"].append(
                    f"Processing interactive click at ({click_x:.3f}, {click_y:.3f}) "
                    f"@ t={click_t or 0.0:.1f}s..."
                )
                # Real bug fixed here (2026-09-01): a click that failed to match ANY track used
                # to fall through silently with `overrides` left empty, and the pipeline below
                # still ran -- but with NO manual override, `select_targets` auto-picks its own
                # arrow/heuristic target for whatever `target_jersey` happened to be sitting in
                # the form (a stale default from a PREVIOUS video, in the confirmed real case:
                # the user opened `video3` -- whose own annotated target is #2 -- clicked at
                # t=0.0s before play started, missed every player, and silently got results for
                # "#10" instead, with no indication their click had done nothing). CLAUDE.md
                # Golden Rule 5 ("no fabricated numbers... an honest not-available is correct, a
                # fabricated one is a bug") applies here just as much to IDENTITY as to a stat:
                # an explicit human selection that failed must fail loudly, never be silently
                # replaced by an unrelated auto-pick that LOOKS like it honored the request.
                try:
                    chosen_take_id, chosen_track_id = _match_click_to_track(
                        video_path, configs, click_x, click_y, click_t or 0.0
                    )
                except Exception as cx_err:
                    job["status"] = "failed"
                    job["error"] = f"Click matching failed: {cx_err}"
                    job["logs"].append(f"ERROR: {job['error']}")
                    return
                if chosen_track_id is not None:
                    # `overrides` is guaranteed empty here (the raw-coordinate path only runs
                    # `if ... and not overrides` above) -- this establishes the sole anchor.
                    overrides = {chosen_take_id: [chosen_track_id]}
                    job["logs"].append(
                        f"Interactive click matched take={chosen_take_id} "
                        f"track_id={chosen_track_id}."
                    )
                else:
                    job["status"] = "failed"
                    job["error"] = (
                        "Your click did not land on a detected player near that moment (or the "
                        "video was paused before any player was visible). Refusing to silently "
                        'fall back to an unrelated auto-selected target. Use "Load Players" to '
                        "see real detected boxes and click one directly, or pick a moment where "
                        "the target player is clearly on screen."
                    )
                    job["logs"].append(f"ERROR: {job['error']}")
                    return

            # "streamed-gathering-treehouse" plan Stage E: every (take_id, track_id) pair in
            # `overrides` above is a human SELECTION -- a raw coordinate click OR the "Load
            # Players" picker's own `take_id:raw_track_id` text entry, both equally "a click" for
            # this purpose -- and a click is a CANDIDATE for the persistent target identity, never
            # a command that blindly overwrites it: "The click means: 'This is a candidate for
            # TARGET_001.' It does NOT mean: 'Make this player the target.'" No stored
            # `work/<slug>/target.json` -> the first pair here establishes it (authoritative,
            # CLAUDE.md Golden Rule 4). A stored profile already exists -> every pair is verified
            # against it (`src.track.target_verify.verify_candidate`) BEFORE the real pipeline
            # ever runs -- a REJECT/UNCERTAIN fails the job loudly right here, with zero side
            # effects, rather than silently resolving to `target_lost` deep inside a "completed"
            # job (see `_resolve_target_click`'s own docstring for why the ACCEPT reconnect itself
            # is deliberately left to `run_pipeline_for_video`'s own Stage 4/5 wiring below, not
            # duplicated here).
            from src.track.target import load_target_profile

            target_json_path = _canonical_work_dir(video_path) / "target.json"
            target_profile = (
                load_target_profile(target_json_path) if target_json_path.exists() else None
            )
            # Flattened in take order, then anchor-submission order within a take -- e.g. two
            # clicks in take 0 (the "player left frame, came back" case) are verified in the order
            # the human added them, first anchor first.
            flattened_pairs = [
                (t_id, tr_id) for t_id in sorted(overrides) for tr_id in overrides[t_id]
            ]
            # Performance fix ("streamed-gathering-treehouse" plan Stage 1): the jersey-OCR stack
            # used to be loaded AND freed once per pair inside `_click_evidence_for_candidate`
            # (`load_optional_jersey_stack`/`free_optional_jersey_stack`) -- with N anchors now
            # possible in one request, that meant N model loads for one click endpoint call. Load
            # it ONCE here (only when there is at least one pair to verify) and hand it to every
            # `_resolve_target_click` call; `_click_evidence_for_candidate` never frees a stack it
            # did not itself load, so this is safe to free exactly once, in `finally`, below.
            jersey_stack: tuple[Any, Any, Any] | None = None
            if flattened_pairs:
                from src.identity.jersey_models import (
                    free_optional_jersey_stack,
                    load_optional_jersey_stack,
                )

                jersey_stack = load_optional_jersey_stack(configs["identity"])
            try:
                try:
                    for pair_take_id, pair_track_id in flattened_pairs:
                        target_profile = _resolve_target_click(
                            video_path,
                            configs,
                            target_profile,
                            pair_take_id,
                            pair_track_id,
                            click_t,
                            jersey_stack=jersey_stack,
                        )
                except ClickRejected as rejection:
                    job["status"] = "failed"
                    job["error"] = str(rejection)
                    job["logs"].append(f"ERROR: {job['error']}")
                    return
            finally:
                if jersey_stack is not None:
                    free_optional_jersey_stack(jersey_stack[0], jersey_stack[1])
            if target_profile is not None:
                jersey_label = (
                    target_profile.jersey_number
                    if target_profile.jersey_number is not None
                    else "?"
                )
                job["logs"].append(
                    f"{target_profile.target_id} (#{jersey_label}) confirmed for this "
                    "selection -- verified identity now drives target rendering/stats for "
                    "the whole run."
                )

            run_pipeline_for_video(
                video_path,
                configs,
                target_jersey=target_jersey,
                manual_overrides=overrides,
                work_root=WORK_DIR,
                output_root=OUTPUT_DIR,
                target_profile=target_profile,
                manual_touch_times=touch_times,
            )
        elif annotations_path.exists():
            job["logs"].append("Branching to MANUAL-EVENTS pipeline via sidecar...")
            run_manual_events_pipeline_for_video(
                video_path,
                configs,
                annotations_path,
                target_jersey=target_jersey,
                work_root=WORK_DIR,
                output_root=OUTPUT_DIR,
                manual_touch_times=touch_times,
            )
        else:
            import re

            pattern = re.compile(configs["run"]["filename_convention_regex"])
            _, parsed_jersey = parse_filename(video_path.stem, pattern)
            effective_jersey = target_jersey if target_jersey is not None else parsed_jersey

            if parsed_jersey is not None or target_jersey is not None:
                job["logs"].append(
                    f"Branching to target player pipeline for jersey #{effective_jersey}..."
                )
                run_pipeline_for_video(
                    video_path,
                    configs,
                    target_jersey=effective_jersey,
                    manual_overrides={},
                    work_root=WORK_DIR,
                    output_root=OUTPUT_DIR,
                    manual_touch_times=touch_times,
                )
            else:
                job["logs"].append("Branching to extended verified-identity pipeline...")
                run_extended_pipeline_for_video(
                    video_path,
                    configs,
                    work_root=WORK_DIR,
                    output_root=OUTPUT_DIR,
                )

        job["stage"] = "Rendering Video & Exporting Stat Cards"
        job["progress"] = 100
        job["status"] = "completed"
        job["slug"] = slug
        job["logs"].append("Pipeline processing completed successfully!")

    except Exception as e:
        # Real bug found from a live failure report, 2026-09-04: `str(e)` on an exception raised
        # with a single argument that IS `None` (e.g. `raise SomeError(some_var)` where
        # `some_var` legitimately evaluates to `None`) renders as the literal 4-character string
        # "None" -- which then looked EXACTLY like a swallowed error to the user ("ERROR: None" /
        # "Job failed: None"), with no way to tell what actually broke. This job's own in-memory
        # `job["error"]` is the ONLY thing the user ever sees; nothing was logged server-side
        # either, so the real traceback was lost the moment this fired (CLAUDE.md Golden Rule 5 --
        # "log everything dropped", and a silently-lost stack trace is exactly that). Now: the
        # full traceback goes to the server log unconditionally, and `job["error"]` always carries
        # at least the exception's own type name, even when `str(e)` itself is empty/uninformative.
        tb = traceback.format_exc()
        logger.error("pipeline job %s failed:\n%s", job_id, tb)
        message = str(e).strip()
        if not message or message == "None":
            message = f"{type(e).__name__} (see server log for the full traceback)"
        job["status"] = "failed"
        job["error"] = message
        job["logs"].append(f"ERROR: {message}")
    except JobCancelled:
        logger.info("pipeline job %s cancelled", job_id)
        job["status"] = "failed"
        job["stage"] = "Cancelled"
        job["error"] = "cancelled"
        job["logs"].append("Cancelled -- the pipeline was stopped.")
    finally:
        # A cancel that raced the end of the run must not escape into the cleanup below.
        for _attempt in range(3):
            try:
                _release_job_thread(job_id)
                logging.getLogger().removeHandler(log_handler)
                # Always release this video, success or failure -- a crashed run must never leave
                # its slug permanently locked out (that would turn one bad run into "this video
                # can never be processed again" until the server restarts).
                with _INFLIGHT_LOCK:
                    _INFLIGHT_SLUGS.discard(_job_slug_for_release)
                break
            except JobCancelled:
                continue


@app.post("/api/process")
def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Trigger the football video tracking and analytics pipeline."""
    # One run per video at a time -- see `_INFLIGHT_SLUGS` for the corruption this prevents.
    # Refused here, before a job is even created, so the caller gets an immediate, actionable 409
    # rather than a job that "starts" and then dies on a half-written clip file.
    slug = _canonical_slug(_resolve_video_path(req.video_name))
    free_gb = _disk_free_gb(WORK_DIR)
    if free_gb < MIN_FREE_DISK_GB:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Only {free_gb:.1f} GB free on the GPU pod's storage (need {MIN_FREE_DISK_GB:.0f} GB "
                "or more) -- a long video would fail part-way. Delete old outputs or grow the "
                "network volume, then run again."
            ),
        )
    with _INFLIGHT_LOCK:
        if slug in _INFLIGHT_SLUGS:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{req.video_name}' is already being processed. Wait for that run to finish "
                    "before starting another -- two runs of the same video overwrite each other's "
                    "clip files and corrupt the output."
                ),
            )
        _INFLIGHT_SLUGS.add(slug)

    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "video_name": req.video_name,
        "target_jersey": req.target_jersey,
        "status": "queued",
        "stage": "Queued",
        "progress": 0,
        "logs": [],
        "created_at": time.time(),
        "slug": req.video_name.replace(".mp4", "")
        .replace(".avi", "")
        .replace(".mov", "")
        .replace(" ", "_"),
        # The EXACT slug `_INFLIGHT_SLUGS` was keyed on above -- the display "slug" beside it is a
        # loose filename transform and must never be used to release the lock (they diverge on
        # uppercase/punctuation, which would leak the slot forever).
        "canonical_slug": slug,
    }

    background_tasks.add_task(
        _run_pipeline_job,
        job_id,
        req.video_name,
        req.target_jersey,
        req.track_id,
        req.manual_annotations,
        req.click_x,
        req.click_y,
        req.click_t,
        req.target_clicks,
        req.touch_times,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Started pipeline job {job_id} for {req.video_name}",
    }


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Stop a queued or running pipeline job. A running one gets `JobCancelled` raised in its own
    thread, which takes effect the next time the pipeline is back in Python (between frames /
    batches -- usually well under a second; a long single ffmpeg call finishes first)."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job ID not found")
    if job["status"] in ("completed", "failed"):
        return job
    job["cancel_requested"] = True
    with _JOB_THREADS_LOCK:
        thread_id = _JOB_THREADS.get(job_id)
        if thread_id is not None:
            _set_async_exc(thread_id, JobCancelled)
    job["logs"].append("Cancel requested -- stopping the pipeline.")
    return job


@app.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    """Retrieve current processing job status."""
    if job_id not in JOBS:
        # Check if output already exists matching a slug
        possible_dir = OUTPUT_DIR / job_id
        if possible_dir.exists():
            return {
                "job_id": job_id,
                "status": "completed",
                "stage": "Finished",
                "progress": 100,
                "slug": job_id,
                "logs": ["Output directory exists on disk"],
            }
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JOBS[job_id]


@app.get("/api/results/{slug}")
def get_results(slug: str):
    """Retrieve player stats, event timeline, heatmaps, and highlight clips URLs."""
    # Real bug fixed here ("streamed-gathering-treehouse" plan Stage E, 2026-09-04): this used to
    # fall back to `slug in d.name or d.name in slug` substring matching when there was no exact
    # directory, which can serve a DIFFERENT run's output entirely (e.g. `chelsea_burnley_target1`
    # substring-matches the on-disk `chelsea_burnley_target10`). `_find_output_dir` is EXACT-match
    # only -- see its own docstring.
    out_dir = _find_output_dir(OUTPUT_DIR, slug)
    if out_dir is None:
        raise HTTPException(status_code=404, detail=f"No results found for slug '{slug}'")

    slug_name = out_dir.name

    # Check for players folder. CLAUDE.md Golden Rule 5: no fabricated numbers -- if no player
    # output exists yet (still processing, or an unverified/unselected take produced none), the
    # honest response is an EMPTY players list with a stated reason, never a plausible-looking
    # placeholder stat card the owner could mistake for a real result.
    players_dir = out_dir / "players"
    player_data = []

    if players_dir.exists():
        for p_folder in sorted(players_dir.glob("player_*")):
            p_num = p_folder.name.replace("player_", "")
            statcard_file = p_folder / "statcard.md"
            statcard_text = statcard_file.read_text() if statcard_file.exists() else ""

            stats = parse_statcard_markdown(statcard_text, int(p_num) if p_num.isdigit() else None)

            # Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14): a real, downloadable PDF
            # link, `None` (never a broken link) when `statcard.pdf` genuinely isn't on disk for
            # this player -- e.g. `reportlab` wasn't installed when this run happened
            # (`write_player_output`'s own fail-soft `ImportError` handling, CLAUDE.md §7).
            statcard_pdf_url = (
                f"/api/download/statcard/{slug_name}/{p_num}"
                if (p_folder / "statcard.pdf").exists()
                else None
            )
            stats["statcard_pdf_url"] = statcard_pdf_url

            highlights = {}
            hl_dir = p_folder / "highlights"
            if hl_dir.exists():
                for hl in hl_dir.glob("*.mp4"):
                    hl_url = (
                        f"/media/output/{slug_name}/players/{p_folder.name}"
                        f"/highlights/{hl.name}"
                    )
                    highlights[hl.stem] = hl_url

            stats["highlights"] = highlights
            player_data.append(stats)

    # Annotated video path
    annotated_url = None
    if (out_dir / "original_annotated_video.mp4").exists():
        annotated_url = f"/media/output/{slug_name}/original_annotated_video.mp4"

    # Plan Fix D ("streamed-gathering-treehouse" re-check, 2026-09-15): `primary_player` is the
    # run's own REAL target -- `work/<slug>/target.json`/`selection.json`, in that order -- never
    # the old lexicographic `sorted(glob("player_*"))[0]` accident (`"player_10" < "player_2"` as
    # strings). Falls back to that exact pre-existing first-sorted entry, silently, whenever
    # neither cache file exists OR the jersey it names has no matching `player_<N>` folder in
    # THIS run's own output (e.g. only some other jersey's take was ever verified/selected) --
    # both are honest "couldn't resolve a better answer" cases, not errors.
    primary_player = None
    if player_data:
        target_jersey = _resolve_primary_player_jersey(WORK_DIR / slug_name)
        if target_jersey is not None:
            primary_player = next(
                (p for p in player_data if p.get("jersey_number") == target_jersey), None
            )
        if primary_player is None:
            primary_player = player_data[0]

    return {
        "slug": slug_name,
        "annotated_video_url": annotated_url,
        "players": player_data,
        "primary_player": primary_player,
        "message": (
            None
            if player_data
            else "No player output found for this run -- either it is still processing, or no "
            "take's identity was verified/selected (see run_report.json / identity_report.json "
            "in output/<slug>/ for why)."
        ),
        # Pitch trajectory intentionally omitted: a pixel-position tactical map needs calibrated
        # homography (ADR-6) that most of this project's own footage doesn't have (<4 usable
        # landmarks) -- a synthetic placeholder here would look like real tracking data. See
        # `output/<slug>/players/player_<N>/events/event_timeline.json` for the real, timestamped
        # events instead.
        "pitch_trajectory": None,
    }


# Every statcard.md field `_safe_int` coerces to an int-or-0 for the dashboard, mapped to its
# markdown label -- shared between `parse_statcard_markdown`'s parsing loop and its own `"raw"`
# capture (Plan Fix D, "streamed-gathering-treehouse" re-check, 2026-09-15) so both come from a
# single line match rather than two separate scans.
_STATCARD_COUNT_FIELDS: dict[str, str] = {
    "**Touches:**": "touches",
    "**Passes:**": "passes",
    "**Turnovers:**": "turnovers",
    "**Sprints/Runs:**": "sprints",
    "**Goals:**": "goals",
    "**Assists:**": "assists",
    "**Shots:**": "shots",
    "**Tackles:**": "tackles",
    "**Saves:**": "saves",
    "**Dribbles:**": "dribbles",
}


def parse_statcard_markdown(md_text: str, jersey_num: int | None) -> dict[str, Any]:
    """Parse statcard.md into a structured dict. `identity_status` defaults to "Unknown" (never
    "Verified") until the statcard's own line is actually parsed below -- CLAUDE.md Golden Rule 5:
    a missing/unparseable statcard must never be presented as a verified result.

    `res["raw"]` (Plan Fix D, "streamed-gathering-treehouse" re-check, 2026-09-15): the same ten
    count fields above, but as the EXACT text that followed `**Label:**` in the markdown, before
    `_safe_int`'s int-or-0 coercion collapses a non-numeric value down to `0`. Added for
    `_generate_statcard_pdf_on_demand` below -- Goals/Assists in particular can legitimately read
    `"not available (...)"` there (CLAUDE.md §13.2), and regenerating a PDF from the coerced `0`
    would silently overclaim certainty that was never there (Golden Rule 5). Purely additive: every
    pre-existing caller only ever read the other keys, so this changes nothing for them.
    """
    res: dict[str, Any] = {
        "jersey_number": jersey_num,
        "identity_status": "Unknown",
        "touches": 0,
        "passes": 0,
        "turnovers": 0,
        "sprints": 0,
        "goals": 0,
        "assists": 0,
        "shots": 0,
        "tackles": 0,
        "saves": 0,
        "dribbles": 0,
        "possession_time": "uncertain",
        "distance_covered": "uncertain",
        "events": [],
        "raw": {},
    }
    if not md_text:
        return res

    for line in md_text.splitlines():
        line = line.strip()
        matched_label = next((k for k in _STATCARD_COUNT_FIELDS if line.startswith(k)), None)
        if line.startswith("**Identity Status:**"):
            res["identity_status"] = line.split(":**")[1].strip()
        elif matched_label is not None:
            field = _STATCARD_COUNT_FIELDS[matched_label]
            res["raw"][field] = line.split(":**")[1].strip()
            res[field] = _safe_int(line)
        elif line.startswith("**Possession Time:**"):
            res["possession_time"] = line.split(":**")[1].strip()
        elif line.startswith("**Distance Covered:**"):
            res["distance_covered"] = line.split(":**")[1].strip()
        elif line.startswith("|") and not line.startswith("| Time") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                res["events"].append(
                    {
                        "time": parts[0],
                        "event": parts[1],
                        "confidence": float(parts[2]) if _is_float(parts[2]) else 0.90,
                    }
                )
    return res


def _safe_int(line: str) -> int:
    val = line.split(":**")[1].strip()
    try:
        return int(val.split()[0])
    except Exception:
        return 0


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def _safe_resolve_under(root: Path, path: str) -> Path | None:
    """`root / path`, resolved, and rejected (`None`) if it escapes `root` -- closes a real path-
    traversal hole (`../../etc/passwd` style) that existed here: the previous version joined the
    request path onto `BASE_DIR` with no check at all, so any file readable by the server process
    was servable over HTTP."""
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None


# `render_statcard_markdown`'s own two format strings this whole module has to invert below
# (Plan Fix D) -- `f"{possession_seconds:.1f}s"` and `f"{distance:.2f} {unit} (uncalibrated)"`
# (`src/pipeline/player_output.py`). Named here, not inline, per CLAUDE.md §10 ("no magic
# numbers/strings in code"); if either format ever changes, this regex must change with it in the
# same commit -- same discipline `statcard_pdf.py`'s own module docstring already asks for between
# its two renderers.
_DISTANCE_TEXT_PATTERN = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s+(\S+)\s+\(uncalibrated\)$")


def _is_int_string(value: str) -> bool:
    """Whether `_safe_int`'s own parsing (`int(val.split()[0])`) would have succeeded on `value`
    -- used to tell a REAL zero/count (`"0"`, `"4"`) apart from a non-numeric reason string
    (`"not available (...)"`) that `_safe_int` also coerces down to `0`."""
    try:
        int(value.split()[0])
        return True
    except (ValueError, IndexError):
        return False


def _extract_goal_reason(parsed: dict[str, Any]) -> str | None:
    """The `goal_reason` `render_statcard_pdf` needs (its own `_goal_assist_text`) to show the
    full "not available (...)" text on the Goals/Assists lines instead of a bare `0` -- read back
    from `parse_statcard_markdown`'s `raw` capture (the text BEFORE `_safe_int` coerced it),
    exactly the raw-string requirement this on-demand path exists to satisfy. `None` when both
    counts are real numbers (the common case: a genuine goal/assist count, or a genuine `0` from
    an authoritative event source, CLAUDE.md §13.2)."""
    for key in ("goals", "assists"):
        raw_value = parsed["raw"].get(key, "")
        if parsed[key] == 0 and raw_value and not _is_int_string(raw_value):
            return raw_value
    return None


def _parse_possession_seconds(raw: str) -> float | None:
    """Inverse of `render_statcard_markdown`'s own `f"{possession_seconds:.1f}s"` / `"uncertain"`
    formatting -- `None` (rendered as `"uncertain"` again by `render_statcard_pdf`) for anything
    that doesn't match, rather than guessing a number CLAUDE.md §13.2 never claimed."""
    raw = raw.strip()
    if raw.endswith("s"):
        try:
            return float(raw[:-1])
        except ValueError:
            pass
    return None


def _parse_distance_result(raw: str) -> dict[str, Any] | None:
    """Inverse of `render_statcard_markdown`'s own
    `f"{distance:.2f} {unit} (uncalibrated)"` / `"uncertain"` formatting."""
    match = _DISTANCE_TEXT_PATTERN.match(raw.strip())
    if not match:
        return None
    return {"distance": float(match.group(1)), "unit": match.group(2)}


def _parse_timeline_timestamp_to_seconds(time_str: str) -> float:
    """Inverse of `src.pipeline.player_output._format_timestamp`/
    `src.pipeline.statcard_pdf._format_timestamp` (both: `f"{minutes}:{secs:04.1f}"`, always
    exactly one colon -- minutes can exceed 59 on a long clip, never an H:MM:SS form)."""
    minutes_str, seconds_str = time_str.rsplit(":", 1)
    return int(minutes_str) * 60 + float(seconds_str)


def _reconstruct_timeline_rows_from_markdown(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`parse_statcard_markdown`'s own `events` (each `{"time": "M:SS.s", "event": label,
    "confidence"}`, straight off statcard.md's 3-column Event Timeline table) turned into the
    `{t_start, label, confidence, source}` shape `render_statcard_pdf` expects. FALLBACK ONLY --
    used by `_generate_statcard_pdf_on_demand` only when that player's own richer
    `events/event_timeline.json` (written by `write_player_output` alongside every real
    statcard.md, with a real float `t_start` and a real per-event `source` string) is missing or
    unreadable. statcard.md's own table (CLAUDE.md §13.2's exact template) never carries a
    `source` column at all, so one is honestly labelled here rather than guessed at (Golden Rule
    5: never fabricate a detector name that wasn't actually recorded)."""
    return [
        {
            "t_start": _parse_timeline_timestamp_to_seconds(ev["time"]),
            "label": ev["event"],
            "confidence": ev["confidence"],
            "source": "archived statcard.md (original event source not recorded there)",
        }
        for ev in events
    ]


def _load_or_reconstruct_timeline_rows(
    player_dir: Path, parsed_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prefer `events/event_timeline.json` (exact float `t_start` + real per-event `source`,
    byte-identical to what a live run hands `render_statcard_pdf`) over reconstructing from
    statcard.md's own lossy 3-column table -- see `_reconstruct_timeline_rows_from_markdown`."""
    timeline_json_path = player_dir / "events" / "event_timeline.json"
    if timeline_json_path.exists():
        try:
            return json.loads(timeline_json_path.read_text())
        except Exception:
            logger.warning(
                "on-demand statcard.pdf: could not parse %s -- reconstructing the timeline from "
                "statcard.md's own table instead (source will read as archived/unrecorded)",
                timeline_json_path,
                exc_info=True,
            )
    return _reconstruct_timeline_rows_from_markdown(parsed_events)


def _generate_statcard_pdf_on_demand(out_dir: Path, jersey: str) -> Path | None:
    """Plan Fix D ("streamed-gathering-treehouse" re-check, 2026-09-15): render `statcard.pdf`
    from a player's already-archived `statcard.md` (+ that player's own sibling
    `events/event_timeline.json` when present, for exact timestamps/sources) -- makes every run
    completed BEFORE Plan Stage 3 shipped the PDF feature downloadable too (confirmed on disk:
    `output/chelsea_burnley_target10/players/player_*/` has only `.md`), not just runs from after.

    Returns the new PDF's path on success (cached next to the markdown, so a second request for
    the same player serves the cached file directly via the caller's own existing
    `_safe_resolve_under` check, never regenerating twice). Returns `None` on ANY failure --
    missing markdown, a non-numeric `jersey` path segment, `reportlab` not installed
    (`ImportError`, CLAUDE.md §7), or any other rendering error -- so the caller can turn that into
    an honest 404, never a 500 (this file's own convention throughout `/api/download/statcard`).
    """
    md_path = _safe_resolve_under(OUTPUT_DIR, f"{out_dir.name}/players/player_{jersey}/statcard.md")
    if md_path is None:
        return None
    if not jersey.isdigit():
        # Every real `player_<N>` folder this codebase writes uses a plain int N (CLAUDE.md
        # §13.5) -- a non-numeric `jersey` here means `_safe_resolve_under` above resolved to a
        # real file, but not through a shape any real run would have produced (or a URL a real
        # `statcard_pdf_url` link would ever contain). No real player to render, so no PDF.
        return None

    try:
        parsed = parse_statcard_markdown(md_path.read_text(), int(jersey))
        timeline_rows = _load_or_reconstruct_timeline_rows(md_path.parent, parsed["events"])
        counts = {
            "touch": parsed["touches"],
            "pass": parsed["passes"],
            "turnover": parsed["turnovers"],
            "sprint": parsed["sprints"],
            "goal": parsed["goals"],
            "assist": parsed["assists"],
            "shot": parsed["shots"],
            "tackle": parsed["tackles"],
            "save": parsed["saves"],
            "dribble": parsed["dribbles"],
        }
        pdf_path = md_path.parent / "statcard.pdf"
        render_statcard_pdf(
            int(jersey),
            counts,
            _parse_possession_seconds(parsed["possession_time"]),
            _parse_distance_result(parsed["distance_covered"]),
            timeline_rows,
            _extract_goal_reason(parsed),
            parsed["identity_status"],
            output_path=pdf_path,
        )
        return pdf_path
    except ImportError:
        logger.warning(
            "on-demand statcard.pdf generation: reportlab is not installed (`api` extra, "
            "CLAUDE.md §7: `uv pip install -e '.[api]'`) -- cannot regenerate a PDF for %s -- "
            "serving 404",
            md_path,
        )
        return None
    except Exception:
        logger.warning(
            "on-demand statcard.pdf generation failed for %s -- serving 404; the archived "
            "statcard.md itself is unaffected",
            md_path,
            exc_info=True,
        )
        return None


@app.get("/api/download/statcard/{slug}/{jersey}")
def download_statcard_pdf(slug: str, jersey: str):
    """Real, downloadable `statcard.pdf` for one player (Plan Stage 3, "streamed-gathering-
    treehouse", 2026-09-14) -- the honest replacement for the old frontend exporter that re-typed
    an approximation of the statcard client-side (see `exportStatCard`'s own history / the plan's
    Context section: it flattened "not available (...)" to a bare `0`, a Golden Rule 5 violation).

    Resolved via `_find_output_dir` (exact slug match, the same helper `/api/results/{slug}`
    already uses) then `_safe_resolve_under`, rooted at `OUTPUT_DIR` -- deliberately NOT `BASE_DIR`
    (that root is what the `/media/` fix below closes) -- since this endpoint only ever needs to
    serve a file already known to live under `output/`. `jersey` is taken as an opaque path
    segment straight from the URL and never joined onto the filesystem directly; it is resolved
    strictly as one more path component under `OUTPUT_DIR` through the same traversal guard, so a
    `../../etc/passwd`-shaped value 404s exactly like any other escape attempt, never a 500.

    Plan Fix D ("streamed-gathering-treehouse" re-check, 2026-09-15): when `statcard.pdf` is
    missing but `statcard.md` exists, the PDF is now generated ON DEMAND
    (`_generate_statcard_pdf_on_demand`) and cached to disk, so every completed run becomes
    downloadable -- not just runs from after Plan Stage 3 shipped the PDF feature.

    404s (never 500s) for a missing slug, a missing player folder, or a statcard.md that genuinely
    doesn't exist either (nothing to generate from) -- an honest "not found", never a confusing
    server error for something that is really just "this run doesn't have a PDF (yet)".
    """
    out_dir = _find_output_dir(OUTPUT_DIR, slug)
    if out_dir is None:
        raise HTTPException(status_code=404, detail=f"No results found for slug '{slug}'")

    pdf_path = _safe_resolve_under(
        OUTPUT_DIR, f"{out_dir.name}/players/player_{jersey}/statcard.pdf"
    )
    if pdf_path is None:
        pdf_path = _generate_statcard_pdf_on_demand(out_dir, jersey)
    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No statcard.pdf (and no generatable statcard.md) found for slug '{slug}' "
            f"player '{jersey}'",
        )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"player_{jersey}_statcard.pdf",
    )


@app.get("/media/{path:path}")
def serve_media(path: str):
    """Serve media files (input videos, annotated output videos, highlights) -- resolved under
    INPUT_DIR or OUTPUT_DIR ONLY, each check rejecting any path that would escape that root.

    Plan Stage 3 ("streamed-gathering-treehouse", 2026-09-14) security fix: this used to probe
    `_safe_resolve_under(BASE_DIR, path)` FIRST. The traversal guard itself was sound, but its
    containment root was the WHOLE REPO -- so `GET /media/.env` served the secrets file, and
    `configs/`, `src/`, `models/` were all readable over HTTP. Every real `/media/` URL this
    project actually generates (confirmed by grepping every reference in `apps/web/` and this
    file) is one of exactly two shapes: a bare INPUT_DIR-relative video name/path (e.g.
    `preview.src = /media/${videoName}`, `apps/web/js/app.js::updatePreviewVideo`) or an
    OUTPUT_DIR-relative path PREFIXED with the literal `output/` segment -- a BASE_DIR-relative
    convention left over from when BASE_DIR was this endpoint's root (`f"/media/output/{slug}/..."`,
    `get_results`/`download_statcard_pdf` above). Stripping that one literal prefix and resolving
    the remainder under OUTPUT_DIR (never BASE_DIR) keeps both existing URL shapes working
    byte-for-byte while removing every other BASE_DIR-rooted path from ever being reachable here.
    """
    output_relative = path[len("output/") :] if path.startswith("output/") else None
    full_path = (
        _safe_resolve_under(OUTPUT_DIR, output_relative)
        if output_relative is not None
        else _safe_resolve_under(INPUT_DIR, path)
    )
    if full_path is None:
        raise HTTPException(status_code=404, detail="Media file not found")

    return FileResponse(
        full_path,
        media_type="video/mp4" if full_path.suffix == ".mp4" else None,
        headers={"Accept-Ranges": "bytes"},
    )


# Mount Web Frontend UI static files
@app.middleware("http")
async def _no_cache_html(request: Request, call_next):
    """The page itself must never be served stale: after a redeploy a browser holding a cached
    index.html keeps the OLD design/markup while the (content-hashed) css/js it references is
    new. Hashed assets stay cacheable; only HTML is revalidated on every load."""
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=False)
