"""PitchVision always-on gateway -- runs on the Hostinger VPS in front of the RunPod GPU pod.

Why it exists (owner, 2026-09-17): the pod stops itself when idle and, once stopped, often cannot
be started again (a stopped pod doesn't reserve its GPU), so with plain nginx in front visitors
saw nothing but a "pod is asleep" page. The gateway keeps the FRONTEND up no matter what:

    browser --https--> nginx --> this app (:8100) --https--> <pod>-8000.proxy.runpod.net
                                  |  serves apps/web, the login, uploads (to VPS disk),
                                  |  the job queue and the pod's lifecycle
                                  +-- apps/gateway/pod_manager.py  starts / recreates the pod,
                                      waits for a GPU when RunPod has none
                                  +-- apps/gateway/jobs.py         one worker: pod -> upload ->
                                      /api/process on the pod -> mirror status

Same URL contract as the pod's own `apps/api/main.py`, so `apps/web` is served unchanged: what
needs the GPU is proxied when the pod is online and answered with a 503 + `pod` state when it
isn't; what doesn't (uploads, the video list, the queue) is answered here.

The GPU is the USER's to switch on and off (2026-09-26, after the old auto-wake/auto-recreate
policy burned the credit): `POST /api/pod/start` and `POST /api/pod/stop` are the only things
that start or stop it, plus an idle auto-stop (`PV_GPU_IDLE_MINUTES`, default 10) that only ever
STOPS. Pods fetch their bootstrap + code from this gateway (`/pod/*`), not from GitHub, so an
expired GitHub token can no longer leave pods that never boot.

Configuration is environment only (docker/vps/gateway.env.example):
    PV_ACCESS_PASSWORD   the shared login (same value on the pod)
    RUNPOD_API_KEY       pods read/write -- the gateway starts, creates and terminates the pod
    GITHUB_TOKEN         read-only PAT the pod uses to fetch the private repo at boot
    PV_GATEWAY_DATA      state dir (uploads + jobs.json), default /var/lib/pitchvision
    PV_POD_NAME / PV_VOLUME_NAME / PV_DATACENTER / PV_GPU_TYPES / PV_POD_PORT
    PV_POD_SSH_PUBLIC_KEY, GEMINI_API_KEY, PV_IDLE_STOP_MINUTES   passed through to the pod
    PV_PUBLIC_URL        this gateway's public https URL; pods fetch their code from it
    PV_GPU_IDLE_MINUTES  stop a billing pod after this long with no job and no use (0 = never)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from apps.api.access import DEFAULT_ACCESS_PASSWORD, resolve_access_password
from apps.api.uploads import ALLOWED_UPLOAD_SUFFIXES, ChunkedUploads, safe_upload_name, upload_response
from apps.gateway import runpod_pods as rp
from apps.gateway.jobs import ACTIVE_STATUSES, JobStore, JobWorker
from apps.gateway.pod_client import PodClient, iter_response
from apps.gateway.pod_manager import ERROR, ONLINE, WAITING_FOR_GPU, PodConfig, PodManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gateway")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEB_DIR = BASE_DIR / "apps" / "web"
DATA_DIR = Path(os.environ.get("PV_GATEWAY_DATA", "/var/lib/pitchvision"))
INPUT_DIR = DATA_DIR / "input"
JOBS_FILE = DATA_DIR / "jobs.json"
STATE_REFRESH_S = float(os.environ.get("PV_POD_REFRESH_SECONDS", "30"))
GPU_IDLE_S = float(os.environ.get("PV_GPU_IDLE_MINUTES", "10")) * 60.0
BOOTSTRAP_SCRIPT = BASE_DIR / "docker" / "runpod_bootstrap.sh"

app = FastAPI(title="The Reach Vision gateway")


# ---------------------------------------------------------------- configuration


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def pod_bootstrap_key() -> str:
    """What a pod presents to fetch `/pod/bootstrap.sh` + `/pod/code.tar.gz`. Derived from secrets
    the gateway already has, so it needs no storage and changes when they are rotated."""
    password, _ = resolve_access_password(os.environ.get("PV_ACCESS_PASSWORD"))
    secret = (password or _env("RUNPOD_API_KEY") or "pitchvision").encode("utf-8")
    return hmac.new(secret, b"pv-pod-bootstrap", hashlib.sha256).hexdigest()[:40]


def pod_config_from_env() -> PodConfig:
    api_key = _env("RUNPOD_API_KEY")
    password, _ = resolve_access_password(os.environ.get("PV_ACCESS_PASSWORD"))
    pod_env = {
        "PV_ACCESS_PASSWORD": password or "off",
        "PV_IDLE_STOP_MINUTES": _env("PV_IDLE_STOP_MINUTES", "30"),
        "PORT": _env("PV_POD_PORT", str(rp.DEFAULT_PORT)),
    }
    # The pod stops itself when idle with the same key (apps/api/idle_stop.py).
    if api_key:
        pod_env["RUNPOD_API_KEY"] = api_key
    for src, dst in (("GITHUB_TOKEN", "GITHUB_TOKEN"), ("GEMINI_API_KEY", "GEMINI_API_KEY"), ("PV_POD_SSH_PUBLIC_KEY", "PUBLIC_KEY")):
        if _env(src):
            pod_env[dst] = _env(src)
    # Code comes from this gateway when it knows its own public URL (runpod_pods.pod_create_body).
    if _env("PV_PUBLIC_URL"):
        pod_env["PV_GATEWAY_URL"] = _env("PV_PUBLIC_URL").rstrip("/")
        pod_env["PV_POD_KEY"] = pod_bootstrap_key()
    gpu_types = tuple(g.strip() for g in _env("PV_GPU_TYPES").split(",") if g.strip()) or rp.DEFAULT_GPU_TYPES
    return PodConfig(
        name=_env("PV_POD_NAME", rp.DEFAULT_POD_NAME),
        volume_name=_env("PV_VOLUME_NAME", rp.DEFAULT_VOLUME_NAME),
        datacenter=_env("PV_DATACENTER", rp.DEFAULT_DATACENTER),
        gpu_types=gpu_types,
        port=int(pod_env["PORT"]),
        pod_env=pod_env,
    )


ACCESS_PASSWORD, ACCESS_PASSWORD_IS_DEFAULT = resolve_access_password(os.environ.get("PV_ACCESS_PASSWORD"))
if ACCESS_PASSWORD_IS_DEFAULT:
    logger.warning("using the default access password (%s) -- set PV_ACCESS_PASSWORD", DEFAULT_ACCESS_PASSWORD)
if not _env("RUNPOD_API_KEY"):
    logger.warning("RUNPOD_API_KEY is not set -- the gateway cannot start or create the pod")
if not _env("PV_PUBLIC_URL") and not _env("GITHUB_TOKEN"):
    logger.warning("neither PV_PUBLIC_URL nor GITHUB_TOKEN is set -- a new pod could not fetch the code")

INPUT_DIR.mkdir(parents=True, exist_ok=True)

PODS = PodManager(rp.RunPodClient(_env("RUNPOD_API_KEY")), pod_config_from_env())
POD_CLIENT = PodClient(lambda: PODS.proxy_url if PODS.online else "", ACCESS_PASSWORD)
JOBS = JobStore(JOBS_FILE)
WORKER = JobWorker(JOBS, PODS, POD_CLIENT, INPUT_DIR)
_UPLOADS = ChunkedUploads()

def _refresh_loop() -> None:
    while True:
        try:
            PODS.refresh()
            # The only automatic action the gateway takes on the GPU: STOP one nobody is using.
            PODS.idle_check(GPU_IDLE_S, busy=bool(JOBS.active()))
        except Exception as exc:
            logger.warning("pod refresh: %s", exc)
        time.sleep(STATE_REFRESH_S)


@app.on_event("startup")
def _startup() -> None:
    if os.environ.get("PV_GATEWAY_NO_THREADS") == "1":  # tests drive the worker by hand
        return
    threading.Thread(target=_refresh_loop, name="gateway-pod-refresh", daemon=True).start()
    WORKER.start()


# ---------------------------------------------------------------- access gate (same as the pod's)

_SESSION_COOKIE = "pv_session"
_SESSION_KEY = secrets.token_bytes(32)
_SESSION_MAX_AGE_S = 30 * 24 * 3600
_LOGIN_FAIL_DELAY_S = 0.5
# `/pod/` is fetched by the pod at boot and checks its own key (`_check_pod_key`).
_PUBLIC_PATH_PREFIXES = ("/api/health", "/api/login", "/api/auth/status", "/pod/")
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
    if not _is_public_path(request.url.path) and ACCESS_PASSWORD and not _has_valid_session(request):
        return JSONResponse(status_code=401, content={"detail": "login required"})
    return await call_next(request)


class LoginRequest(BaseModel):
    password: str


@app.get("/api/auth/status")
def auth_status(request: Request):
    return {"required": bool(ACCESS_PASSWORD), "authenticated": _has_valid_session(request)}


@app.post("/api/login")
def login(req: LoginRequest, request: Request):
    if not ACCESS_PASSWORD:
        return {"status": "ok", "required": False}
    if not hmac.compare_digest(req.password.encode("utf-8"), ACCESS_PASSWORD.encode("utf-8")):
        time.sleep(_LOGIN_FAIL_DELAY_S)
        raise HTTPException(status_code=401, detail="wrong password")
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    response = JSONResponse(content={"status": "ok", "required": True})
    response.set_cookie(
        _SESSION_COOKIE, _session_token(), max_age=_SESSION_MAX_AGE_S, httponly=True,
        samesite="lax", secure=forwarded_proto == "https", path="/",
    )
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse(content={"status": "ok"})
    response.delete_cookie(_SESSION_COOKIE, path="/")
    return response


# ---------------------------------------------------------------- health + pod state


@app.get("/api/health")
def get_health():
    st = PODS.state
    return {
        "status": "healthy",
        "gateway": True,
        "gpu_available": st.phase == ONLINE,
        "gpu_name": st.gpu_name if st.phase == ONLINE else "",
        "pod": st.public(),
        "queue": len(JOBS.active()),
    }


# RunPod account balance + current spend, for the GPU control card. Cached: it is polled.
_ACCOUNT: dict[str, Any] = {"at": 0.0, "data": None, "error": ""}
_ACCOUNT_TTL_S = 60.0
_ACCOUNT_LOCK = threading.Lock()


def _account() -> tuple[dict[str, float] | None, str]:
    with _ACCOUNT_LOCK:
        if time.time() - _ACCOUNT["at"] >= _ACCOUNT_TTL_S:
            try:
                _ACCOUNT.update(at=time.time(), data=rp.fetch_account(_env("RUNPOD_API_KEY")), error="")
            except rp.RunPodError as exc:
                logger.warning("balance lookup failed: %s", exc)
                _ACCOUNT.update(at=time.time(), error=str(exc))
        return _ACCOUNT["data"], _ACCOUNT["error"]


@app.get("/api/pod")
def get_pod():
    account, account_error = _account()
    return {
        "pod": PODS.state.full(),
        "starting": PODS.starting,
        "active_jobs": JOBS.active(),
        "idle_stop_minutes": GPU_IDLE_S / 60.0,
        "account": account,
        "account_error": account_error or None,
    }


class PodStartRequest(BaseModel):
    gpu_tier: str | None = None


@app.post("/api/pod/start")
def start_pod(req: PodStartRequest | None = None):
    """The Start GPU button -- the ONLY way a GPU pod is created or started."""
    tier_id = req.gpu_tier if req else None
    tier = rp.gpu_tier(tier_id) if tier_id else None
    if tier_id and tier is None:
        raise HTTPException(status_code=400, detail=f"unknown gpu_tier {tier_id!r}")
    if PODS.online and not PODS.starting:
        return {"status": "already_on", "pod": PODS.state.public()}
    started = PODS.start(tier["gpu_type_ids"] if tier else None)
    _ACCOUNT["at"] = 0.0
    return {"status": "starting" if started else "already_starting", "pod": PODS.state.public()}


@app.post("/api/pod/stop")
def stop_pod():
    """The Stop GPU button: cancel what is queued/running, then stop + terminate the pod (on a
    thread -- RunPod can take a minute; the UI follows along via /api/pod)."""
    cancelled = JOBS.cancel_active("the GPU was stopped")
    PODS.state.wanted = False

    def run() -> None:
        PODS.stop()
        _ACCOUNT["at"] = 0.0  # show the new spend rate right away, not after the cache expires

    threading.Thread(target=run, name="gateway-pod-stop", daemon=True).start()
    return {"status": "stopping", "cancelled_jobs": cancelled}


# ---------------------------------------------------------------- what a booting pod fetches


def _check_pod_key(request: Request) -> None:
    if not hmac.compare_digest(request.headers.get("x-pv-pod-key", ""), pod_bootstrap_key()):
        raise HTTPException(status_code=403, detail="bad pod key")


@app.get("/pod/bootstrap.sh")
def pod_bootstrap(request: Request):
    _check_pod_key(request)
    return FileResponse(BOOTSTRAP_SCRIPT, media_type="text/x-shellscript")


_CODE_LOCK = threading.Lock()


@app.get("/pod/code.tar.gz")
def pod_code(request: Request):
    """The gateway's own checkout (exactly what is deployed) as a tarball, cached per commit."""
    _check_pod_key(request)
    with _CODE_LOCK:
        try:
            sha = subprocess.run(["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=500, detail=f"no git checkout to serve: {exc}") from exc
        target = DATA_DIR / f"code-{sha[:12]}.tar.gz"
        if not target.exists():
            for old in DATA_DIR.glob("code-*.tar.gz"):
                old.unlink(missing_ok=True)
            tmp = target.with_suffix(".tmp")
            subprocess.run(["git", "-C", str(BASE_DIR), "archive", "--format=tar.gz", "-o", str(tmp), sha],
                           check=True, timeout=120)
            os.replace(tmp, target)
    return FileResponse(target, media_type="application/gzip", headers={"X-PV-Commit": sha[:12]})


# GPU tiers with RunPod's live per-hour price and per-datacenter stock (2026-09-19). One GraphQL
# call, cached: prices move rarely, and the app polls this on every page load.
GPU_OFFERS_TTL_S = float(os.environ.get("PV_GPU_PRICE_TTL_SECONDS", "300"))
_GPU_OFFERS: dict[str, Any] = {"at": 0.0, "offers": {}, "error": ""}
_GPU_OFFERS_LOCK = threading.Lock()


def _gpu_offers() -> tuple[dict[str, dict[str, Any]], str]:
    with _GPU_OFFERS_LOCK:
        if time.time() - _GPU_OFFERS["at"] < GPU_OFFERS_TTL_S:
            return _GPU_OFFERS["offers"], _GPU_OFFERS["error"]
        try:
            offers = rp.fetch_gpu_offers(_env("RUNPOD_API_KEY"), PODS.cfg.datacenter)
            _GPU_OFFERS.update(at=time.time(), offers=offers, error="")
        except rp.RunPodError as exc:
            # Keep whatever we had; say why it may be stale.
            logger.warning("GPU price lookup failed: %s", exc)
            _GPU_OFFERS.update(at=time.time(), error=str(exc))
        return _GPU_OFFERS["offers"], _GPU_OFFERS["error"]


@app.get("/api/gpus")
def list_gpus():
    """The selectable GPU tiers, each with the live price/stock of its cards in the volume's
    datacenter, plus which card the pod is on right now (so the app can say "switching GPU
    means a restart")."""
    offers, error = _gpu_offers()
    # RunPod's pod record does not always carry the card id (seen live 2026-09-19: online pod,
    # empty gpuTypeId); the name torch reports on the pod is the same string RunPod uses.
    current = PODS.state.gpu_type or PODS.state.gpu_name
    tiers = []
    for tier in rp.GPU_TIERS:
        cards = []
        for gid in tier["gpu_type_ids"]:
            offer = offers.get(gid, {})
            cards.append({
                "gpu_type_id": gid,
                "display_name": offer.get("display_name") or gid.replace("NVIDIA ", ""),
                "memory_gb": offer.get("memory_gb"),
                "price_per_hr": offer.get("price_per_hr"),
                "stock": offer.get("stock"),
            })
        # what the user will most likely get: the first card in stock, else the first card
        primary = next((c for c in cards if c["stock"]), cards[0])
        tiers.append({
            "id": tier["id"],
            "label": tier["label"],
            "blurb": tier["blurb"],
            "primary": primary,
            "cards": cards,
            "in_stock": any(c["stock"] for c in cards),
            "is_current": bool(current) and current in tier["gpu_type_ids"],
        })
    return {
        "tiers": tiers,
        "default": rp.DEFAULT_GPU_TIER,
        "datacenter": PODS.cfg.datacenter,
        "current_gpu_type": current,
        "prices_error": error or None,
        "prices_at": _GPU_OFFERS["at"] or None,
    }


def _pod_unavailable(action: str) -> HTTPException:
    st = PODS.state
    if st.phase == WAITING_FOR_GPU:
        detail = f"No GPU free on RunPod yet -- {action} will work once the GPU is on. {st.message}"
    elif st.phase == ERROR:
        detail = f"GPU error: {st.message}"
    elif PODS.starting or st.wanted:
        detail = f"The GPU is starting -- {action} will work in a few minutes. ({st.message})"
    else:
        detail = f"The GPU is off -- press Start GPU to use {action} (it bills only while on)."
    return HTTPException(status_code=503, detail=detail, headers={"Retry-After": "30"})


# ---------------------------------------------------------------- videos + uploads (local)


def _local_videos() -> list[dict[str, Any]]:
    out = []
    for p in sorted(INPUT_DIR.glob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_UPLOAD_SUFFIXES:
            out.append({"name": p.name, "size_mb": round(p.stat().st_size / (1024 * 1024), 2), "modified": time.ctime(p.stat().st_mtime)})
    return out


@app.get("/api/videos")
def list_videos():
    """Videos uploaded here, merged with what the pod's volume has when it is online; outputs
    from the pod when online, else the slugs of jobs this gateway finished."""
    local = _local_videos()
    outputs: list[dict[str, Any]] = []
    if PODS.online:
        try:
            remote = POD_CLIENT.list_videos()
            seen = {v["name"] for v in local}
            local += [v for v in remote.get("input_videos", []) if v.get("name") not in seen]
            outputs = remote.get("outputs", [])
        except requests.RequestException as exc:
            logger.warning("pod video list: %s", exc)
    if not outputs:
        slugs = {j["slug"] for j in JOBS.list() if j["status"] == "completed" and j.get("slug")}
        outputs = [{"slug": s, "has_annotated_video": True, "players": []} for s in sorted(slugs)]
    return {"input_videos": local, "outputs": outputs}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    target_path = INPUT_DIR / safe_upload_name(file.filename)
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
    return _UPLOADS.receive(INPUT_DIR, upload_id, index, total, filename, chunk.file)


# ---------------------------------------------------------------- jobs


@app.post("/api/process")
async def process_video(request: Request):
    """Queue a run. The payload is the pod's `ProcessRequest`, forwarded verbatim once the pod
    is up -- the gateway only needs `video_name` to find the file and refuse duplicates."""
    payload = await request.json()
    if not isinstance(payload, dict) or not payload.get("video_name"):
        raise HTTPException(status_code=400, detail="video_name is required")
    video_name = str(payload["video_name"])
    if payload.get("gpu_tier") and rp.gpu_tier(payload["gpu_tier"]) is None:
        raise HTTPException(status_code=400, detail=f"unknown gpu_tier {payload['gpu_tier']!r}")
    if not (PODS.online or PODS.starting or PODS.state.wanted):
        # Run never switches the GPU on by itself (2026-09-26) -- that is the Start GPU button.
        raise HTTPException(status_code=409, detail="The GPU is off. Press Start GPU first -- it bills "
                            "only while on, and stops itself when idle.")
    if JOBS.video_in_flight(video_name):
        raise HTTPException(status_code=409, detail=f"'{video_name}' is already queued or being processed.")
    if not (INPUT_DIR / video_name).exists() and not PODS.online:
        # Nothing to push and no pod to ask: refuse now instead of failing minutes later.
        raise HTTPException(status_code=404, detail=f"'{video_name}' is not uploaded here; upload it first.")
    job = JOBS.create(video_name, payload)
    return {"job_id": job["job_id"], "status": job["status"], "message": f"Queued job {job['job_id']} for {video_name}"}


@app.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JOBS.public(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = JOBS.cancel(job_id, "cancelled by you")
    if job is None:
        raise HTTPException(status_code=409, detail="That job is not queued or running.")
    return JOBS.public(job)


@app.get("/api/jobs")
def list_jobs():
    jobs = JOBS.list()
    return {"jobs": jobs, "active": [j for j in jobs if j["status"] in ACTIVE_STATUSES]}


# ---------------------------------------------------------------- everything the GPU answers

_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host", "cookie"}


async def _proxy(request: Request, action: str) -> StreamingResponse:
    if not PODS.online:
        raise _pod_unavailable(action)  # never wakes the GPU -- that is the user's Start button
    PODS.touch()  # browsing results is GPU use: keep the idle auto-stop from firing under it
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    path_qs = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    try:
        # `requests` blocks until the pod's response headers arrive -- keep that off the event loop.
        resp = await run_in_threadpool(POD_CLIENT.proxy, request.method, path_qs, headers, body or None)
    except requests.RequestException as exc:
        PODS.refresh()
        raise HTTPException(status_code=502, detail=f"GPU pod did not answer: {exc}") from exc
    # `requests` hands us the DECODED body, so a compressed answer must not keep its encoding /
    # length headers; an uncompressed one (every video) keeps Content-Length so seeking works.
    drop = _HOP_BY_HOP | {"set-cookie", "content-encoding"}
    if "content-encoding" in resp.headers:
        drop = drop | {"content-length"}
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
    return StreamingResponse(iter_response(resp), status_code=resp.status_code, headers=out_headers,
                             media_type=resp.headers.get("content-type"))


@app.get("/api/frame_players")
async def frame_players(request: Request):
    return await _proxy(request, "player detection")


@app.get("/api/results/{slug}")
async def get_results(request: Request, slug: str):
    return await _proxy(request, "loading results")


@app.get("/api/download/{path:path}")
async def download(request: Request, path: str):
    return await _proxy(request, "the download")


@app.get("/media/{path:path}")
async def serve_media(request: Request, path: str):
    """Uploaded inputs are served from the VPS (so previews work with the pod asleep); anything
    else -- annotated outputs, clips, reels -- lives on the pod's volume and is proxied."""
    if not path.startswith("output/"):
        local = (INPUT_DIR / path).resolve()
        if local.is_file() and INPUT_DIR.resolve() in local.parents:
            return FileResponse(local, media_type="video/mp4" if local.suffix == ".mp4" else None, headers={"Accept-Ranges": "bytes"})
    return await _proxy(request, "playing this video")


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
