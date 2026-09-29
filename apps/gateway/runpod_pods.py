"""Thin RunPod REST client for the ONE pod this project runs (docs/DEPLOY.md "Always-on gateway").

Everything the gateway (`apps/gateway/pod_manager.py`) and the command-line provisioner
(`docker/runpod_provision.py`) need to say to RunPod lives here, once: find the pod by name,
start / stop / terminate it, create a fresh one on the network volume, and probe its HTTP proxy.
The pod *spec* (`pod_create_body`) is the single source of truth for what a PitchVision pod looks
like -- CLAUDE.md §10, no duplicated magic values between the two callers.

Stdlib + `requests` only (a core dependency already), so the VPS gateway's venv stays tiny.
Nothing here retries or sleeps: callers own the waiting policy.

RunPod REST quirks learned the hard way (2026-09-17):
  * `POST /pods/{id}/start|stop` answer HTTP 500 to a body-less POST -- always send `{}`.
  * Cloudflare (error 1010) rejects urllib's / requests' default User-Agent -- set our own.
  * The `<id>-<port>.proxy.runpod.net` host answers a stopped pod, and a booting one whose port
    isn't registered yet, with an EMPTY 404 -- not a 502.
And one about the network (2026-09-30): a pod in EUR-IS-1 cannot open a TCP connection to the
Hostinger VPS at all (nor the VPS to the pod's public IP) -- only RunPod's proxy connects them.
So the gateway PUSHES the code to a booting pod through that proxy (`push_code`); a pod never
calls the gateway.
"""

from __future__ import annotations

import base64
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

REST_BASE = "https://rest.runpod.io/v1"
USER_AGENT = "pitchvision-gateway/1.0"
REQUEST_TIMEOUT_S = 60.0
HEALTH_TIMEOUT_S = 15.0
CODE_PUSH_TIMEOUT_S = 120.0

# What a PitchVision pod is made of. On every boot the pod first runs `docker/pod_receiver.py`
# (shipped in its env as `PV_POD_RECEIVER`), which waits on the app port for the gateway to push
# the code; the code's `docker/runpod_bootstrap.sh` then runs the app. The network volume at
# /workspace carries the checkout, venv, models, uploads and outputs across pod recreations.
# (`docker/runpod_provision.py` without a gateway still fetches from GitHub with a GITHUB_TOKEN.)
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
BOOTSTRAP_URL = (
    "https://raw.githubusercontent.com/Qaiserfarooq285/payertracker/master/docker/runpod_bootstrap.sh"
)
POD_RECEIVER_SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "pod_receiver.py"
CODE_PUSH_PATH = "/pv/code"  # docker/pod_receiver.py CODE_PATH
POD_CODE_TGZ = "/tmp/pv-code.tar.gz"
POD_BOOTSTRAP = "/tmp/pv-bootstrap.sh"
CONTAINER_DISK_GB = 20
VOLUME_MOUNT_PATH = "/workspace"
DEFAULT_POD_NAME = "pitchvision"
DEFAULT_VOLUME_NAME = "pitchvision-data"
DEFAULT_VOLUME_GB = 40  # models 2 GB + venv ~7 GB + room for clips and outputs; grows, never shrinks
DEFAULT_PORT = 8000
DEFAULT_DATACENTER = "EUR-IS-1"
DEFAULT_CLOUD = "SECURE"  # network volumes only attach to Secure Cloud pods
# The GPU tiers a user can pick in the web app (2026-09-19). Each tier lists its GPU type ids in
# order of preference -- RunPod creates the pod on the first one with stock, so a tier is a
# PRICE/SPEED CLASS, not a single card. Hard constraints behind the picks:
#   * the pod image is CUDA 12.4 / PyTorch 2.4 (`IMAGE`), so Blackwell cards (RTX 5090, RTX PRO
#     6000, B200/B300) are out -- they need CUDA 12.8+;
#   * the network volume lives in `DEFAULT_DATACENTER`, and a pod can only attach it from there,
#     so only cards RunPod stocks in that datacenter matter (`fetch_gpu_offers` reports per-DC
#     stock, and the app shows it);
#   * the RTX 2000 Ada is excluded on purpose -- every boot on that host died on the model
#     download (2026-09-17).
# Prices are NOT hard-coded: `fetch_gpu_offers` reads RunPod's live per-hour Secure Cloud price.
GPU_TIERS: tuple[dict[str, Any], ...] = (
    {
        "id": "budget",
        "label": "Budget",
        "blurb": "Cheapest. Fine for short clips; roughly 2-3x slower than Standard.",
        "gpu_type_ids": (
            "NVIDIA RTX 4000 Ada Generation",
            "NVIDIA RTX A5000",
            "NVIDIA RTX A4500",
        ),
    },
    {
        "id": "standard",
        "label": "Standard",
        "blurb": "RTX 4090 -- the proven default for this pipeline.",
        "gpu_type_ids": ("NVIDIA GeForce RTX 4090",),
    },
    {
        "id": "pro",
        "label": "Pro",
        "blurb": "A100 80 GB -- for long or 4K matches; the most memory.",
        "gpu_type_ids": ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"),
    },
)
DEFAULT_GPU_TIER = "standard"
# What the gateway uses when a job names no tier (and `PV_GPU_TYPES` is unset).
DEFAULT_GPU_TYPES = GPU_TIERS[1]["gpu_type_ids"]

GRAPHQL_URL = "https://api.runpod.io/graphql"


def gpu_tier(tier_id: str | None) -> dict[str, Any] | None:
    """The tier record for `tier_id` (`None`/unknown -> `None`)."""
    for tier in GPU_TIERS:
        if tier["id"] == tier_id:
            return tier
    return None


def fetch_gpu_offers(
    api_key: str,
    datacenter: str = DEFAULT_DATACENTER,
    session: requests.Session | None = None,
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> dict[str, dict[str, Any]]:
    """RunPod's LIVE Secure Cloud offer for every GPU type in `GPU_TIERS`, keyed by GPU type id:
    `{"display_name", "memory_gb", "price_per_hr", "stock"}`. `price_per_hr` is the on-demand
    (uninterruptible, 1 GPU) price in `datacenter` when RunPod quotes one there, else the
    catalogue Secure Cloud price; `stock` is RunPod's own "High"/"Medium"/"Low" for that
    datacenter, or `None` when the card is not offered there at all.

    The REST API has no GPU catalogue endpoint (checked 2026-09-19: `/v1/gputypes` is not in its
    spec), so this is the one GraphQL call the gateway makes. Raises `RunPodError` on any
    failure -- callers decide whether stale/absent prices are acceptable.
    """
    wanted = sorted({gid for tier in GPU_TIERS for gid in tier["gpu_type_ids"]})
    query = (
        "query($ids: [String!], $dc: String) { gpuTypes(input: {ids: $ids}) { id displayName "
        "memoryInGb securePrice lowestPrice(input: {gpuCount: 1, secureCloud: true, "
        "dataCenterId: $dc}) { uninterruptablePrice stockStatus } } }"
    )
    http = session or requests
    try:
        resp = http.post(
            GRAPHQL_URL,
            params={"api_key": api_key},
            json={"query": query, "variables": {"ids": wanted, "dc": datacenter}},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        raise RunPodError(f"GPU price lookup failed: {exc}") from exc
    if resp.status_code != 200:
        raise RunPodError(f"GPU price lookup: HTTP {resp.status_code}", resp.status_code)
    body = resp.json()
    if body.get("errors"):
        raise RunPodError(f"GPU price lookup: {body['errors'][0].get('message', body['errors'])}")
    offers: dict[str, dict[str, Any]] = {}
    for g in (body.get("data") or {}).get("gpuTypes") or []:
        lowest = g.get("lowestPrice") or {}
        price = lowest.get("uninterruptablePrice")
        offers[str(g["id"])] = {
            "display_name": g.get("displayName") or g["id"],
            "memory_gb": int(g.get("memoryInGb") or 0),
            "price_per_hr": float(price if price is not None else (g.get("securePrice") or 0.0)),
            "stock": lowest.get("stockStatus"),
        }
    return offers


def fetch_account(
    api_key: str,
    session: requests.Session | None = None,
    timeout_s: float = REQUEST_TIMEOUT_S,
) -> dict[str, float]:
    """The account's credit balance and what it is being billed per hour right now (every pod +
    the network volume), from RunPod's GraphQL `myself`. Shown next to the Start/Stop GPU buttons
    (2026-09-26) so the owner can see at a glance whether anything is burning credit. Raises
    `RunPodError` on any failure."""
    http = session or requests
    try:
        resp = http.post(
            GRAPHQL_URL,
            params={"api_key": api_key},
            json={"query": "{ myself { clientBalance currentSpendPerHr } }"},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
        )
    except requests.RequestException as exc:
        raise RunPodError(f"balance lookup failed: {exc}") from exc
    if resp.status_code != 200:
        raise RunPodError(f"balance lookup: HTTP {resp.status_code}", resp.status_code)
    body = resp.json()
    if body.get("errors"):
        raise RunPodError(f"balance lookup: {body['errors'][0].get('message', body['errors'])}")
    me = (body.get("data") or {}).get("myself") or {}
    return {
        "balance": float(me.get("clientBalance") or 0.0),
        "spend_per_hr": float(me.get("currentSpendPerHr") or 0.0),
    }


def is_low_balance_error(message: str) -> bool:
    """RunPod refuses to create/start a pod on an empty account with HTTP 500 "Your account
    balance is too low to rent a pod" -- not a stock problem, and retrying cannot fix it."""
    text = message.lower()
    return "balance is too low" in text or "add funds" in text


class RunPodError(Exception):
    """A RunPod REST call that did not succeed. `status` is the HTTP status (0 = no response)."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status

    @property
    def is_auth_error(self) -> bool:
        return self.status in (401, 403)


def proxy_url(pod_id: str, port: int = DEFAULT_PORT) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def pod_receiver_env() -> str:
    """`docker/pod_receiver.py`, zlib-compressed and base64-encoded, for the pod's
    `PV_POD_RECEIVER` env var -- the start command decompresses and execs it (`pod_create_body`).
    It travels in the pod spec because a booting pod has no other way to get code."""
    return base64.b64encode(zlib.compress(POD_RECEIVER_SCRIPT.read_bytes(), 9)).decode("ascii")


def pod_create_body(
    *,
    name: str,
    gpu_type_ids: list[str],
    volume_id: str,
    env: dict[str, str],
    port: int = DEFAULT_PORT,
    cloud: str = DEFAULT_CLOUD,
    debug: bool = False,
) -> dict[str, Any]:
    """The `POST /pods` payload for a PitchVision pod. `env` must already carry the secrets the
    bootstrap needs (`PV_ACCESS_PASSWORD`, optional `RUNPOD_API_KEY` for idle auto-stop,
    `PUBLIC_KEY` for SSH) and says where the code comes from:

    * `PV_POD_RECEIVER` + `PV_POD_KEY` (what the VPS gateway sets, 2026-09-30): the pod waits for
      the gateway to PUSH the code through RunPod's proxy (`docker/pod_receiver.py`,
      `push_code`). A pod cannot reach the VPS itself -- the 2026-09-26 design had it fetch from
      the gateway's public URL, and on its first real run every fetch timed out.
    * otherwise (`docker/runpod_provision.py` by hand): fetch the bootstrap from the private repo
      with `GITHUB_TOKEN`, left for the container's shell to expand so it never appears in the
      command.
    """
    if env.get("PV_POD_RECEIVER") and env.get("PV_POD_KEY"):
        get_code = (
            f"export PV_CODE_TGZ={POD_CODE_TGZ} PV_BOOTSTRAP={POD_BOOTSTRAP}; "
            "python3 -c 'import base64,os,zlib; "
            'exec(zlib.decompress(base64.b64decode(os.environ["PV_POD_RECEIVER"])))\''
        )
    else:
        get_code = f'curl -fsSL -H "Authorization: token $GITHUB_TOKEN" {BOOTSTRAP_URL} -o {POD_BOOTSTRAP}'
    if debug:
        # Keep the container alive after a failed bootstrap and leave its output on the volume,
        # so a crash-looping pod can be inspected over SSH instead of guessed at from a blank
        # log viewer.
        start_cmd = (
            "/start.sh >/workspace/runpod-start.log 2>&1 & export PV_SKIP_RUNPOD_SERVICES=1; "
            f"{get_code} && bash {POD_BOOTSTRAP} >/workspace/bootstrap.log 2>&1; "
            "echo EXIT=$? >>/workspace/bootstrap.log; sleep infinity"
        )
    else:
        # If the code never arrives, the bootstrap fails, or the app exits with an error, the pod
        # STOPS ITSELF (RunPod injects RUNPOD_POD_ID; the gateway passes RUNPOD_API_KEY) instead
        # of letting RunPod restart the container in a billing loop -- a guard that does not
        # depend on the gateway being up. (`curl | bash` hid a failed fetch: bash exits 0 on no
        # input.) The gateway, for its part, never starts a pod that stopped itself mid-boot.
        stop_self = (
            'echo "[pitchvision] bootstrap/app failed -- stopping this pod so it does not bill"; '
            '[ -n "$RUNPOD_API_KEY" ] && [ -n "$RUNPOD_POD_ID" ] && curl -fsS -X POST '
            '-H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" '
            f"-A {USER_AGENT} -d '{{}}' \"{REST_BASE}/pods/$RUNPOD_POD_ID/stop\"; sleep 60"
        )
        start_cmd = f"{get_code} && bash {POD_BOOTSTRAP} || {{ {stop_self}; }}"
    return {
        "name": name,
        "imageName": IMAGE,
        "cloudType": cloud,
        "gpuTypeIds": list(gpu_type_ids),
        "gpuCount": 1,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "volumeMountPath": VOLUME_MOUNT_PATH,
        "networkVolumeId": volume_id,
        "ports": [f"{port}/http", "22/tcp"],
        "env": dict(env),
        "dockerStartCmd": ["bash", "-c", start_cmd],
        "supportPublicIp": True,
    }


@dataclass(frozen=True)
class PodInfo:
    """The few fields of a RunPod pod record the gateway reasons about."""

    id: str
    name: str
    desired_status: str  # "RUNNING" | "EXITED" | ... as RunPod reports it
    gpu_type: str
    gpu_count: int
    uptime_s: int
    cost_per_hr: float

    @property
    def is_running(self) -> bool:
        return self.desired_status == "RUNNING"

    @property
    def has_gpu(self) -> bool:
        """False for a pod RunPod resumed on CPU only -- its "Start Pod using CPUs" fallback when
        the stopped pod's card is gone (observed 2026-09-17: `gpuCount` null, `machine` empty,
        still billed $0.37/h). Useless to us: the app reports "CPU Only"."""
        return self.gpu_count > 0 or bool(self.gpu_type)

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> PodInfo:
        machine = raw.get("machine") or {}
        runtime = raw.get("runtime") or {}
        return cls(
            id=str(raw.get("id", "")),
            name=str(raw.get("name", "")),
            desired_status=str(raw.get("desiredStatus", "")),
            gpu_type=str(machine.get("gpuTypeId") or raw.get("gpuTypeId") or ""),
            gpu_count=int(raw.get("gpuCount") or 0),
            uptime_s=int(runtime.get("uptimeInSeconds") or 0),
            cost_per_hr=float(raw.get("costPerHr") or 0.0),
        )


class RunPodClient:
    """One authenticated REST session. `session` is injectable so tests never touch the network."""

    def __init__(
        self,
        api_key: str,
        base_url: str = REST_BASE,
        session: requests.Session | None = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_s = timeout_s

    # ---------------------------------------------------------------- raw call

    def _call(self, method: str, path: str, body: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        try:
            resp = self.session.request(method, url, json=body, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise RunPodError(f"{method} {path}: {type(exc).__name__}: {exc}") from exc
        if not 200 <= resp.status_code < 300:
            raise RunPodError(f"{method} {path} -> HTTP {resp.status_code}: {resp.text[:400]}", resp.status_code)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            return {}

    # ---------------------------------------------------------------- pods

    def list_pods(self) -> list[PodInfo]:
        raw = self._call("GET", "/pods") or []
        return [PodInfo.from_api(p) for p in raw]

    def find_pod(self, name: str) -> PodInfo | None:
        for pod in self.list_pods():
            if pod.name == name:
                return pod
        return None

    def get_pod(self, pod_id: str) -> PodInfo | None:
        try:
            return PodInfo.from_api(self._call("GET", f"/pods/{pod_id}"))
        except RunPodError as exc:
            if exc.status == 404:
                return None
            raise

    def start_pod(self, pod_id: str) -> PodInfo:
        # `{}` is load-bearing: a body-less POST is answered with HTTP 500.
        return PodInfo.from_api(self._call("POST", f"/pods/{pod_id}/start", {}))

    def stop_pod(self, pod_id: str) -> None:
        self._call("POST", f"/pods/{pod_id}/stop", {})

    def terminate_pod(self, pod_id: str) -> None:
        self._call("DELETE", f"/pods/{pod_id}")

    def create_pod(self, body: dict[str, Any]) -> PodInfo:
        return PodInfo.from_api(self._call("POST", "/pods", body))

    # ---------------------------------------------------------------- network volume

    def find_volume(self, name: str) -> dict[str, Any] | None:
        for vol in self._call("GET", "/networkvolumes") or []:
            if vol.get("name") == name:
                return vol
        return None

    def create_volume(self, name: str, size_gb: int, datacenter: str) -> dict[str, Any]:
        return self._call("POST", "/networkvolumes", {"name": name, "size": size_gb, "dataCenterId": datacenter})


def pod_awaits_code(url: str, session: requests.Session | None = None, timeout_s: float = HEALTH_TIMEOUT_S) -> bool:
    """Is the pod's receiver (`docker/pod_receiver.py`) up and still waiting for the code? False
    while RunPod's proxy has not registered the port, while the bootstrap installs (nothing
    listens), once the app runs (its own 401/404), and on any network error."""
    sess = session or requests.Session()
    try:
        resp = sess.get(f"{url.rstrip('/')}{CODE_PUSH_PATH}", timeout=timeout_s, headers={"User-Agent": USER_AGENT})
        return resp.status_code == 200 and resp.json().get("awaiting") is True
    except (requests.RequestException, ValueError, AttributeError):
        return False


def push_code(
    url: str,
    key: str,
    bundle: bytes,
    commit: str,
    session: requests.Session | None = None,
    timeout_s: float = CODE_PUSH_TIMEOUT_S,
) -> bool:
    """Hand a waiting pod its code (the gateway's `git archive` tar.gz) through RunPod's proxy.
    True once the pod accepted it; False on a network error or a proxy hiccup (the caller asks
    again on its next poll). Raises `RunPodError` when the pod REFUSES the bundle -- wrong key,
    not a bundle -- which asking again cannot fix."""
    sess = session or requests.Session()
    headers = {"User-Agent": USER_AGENT, "X-PV-Pod-Key": key, "X-PV-Commit": commit,
               "Content-Type": "application/gzip"}
    try:
        resp = sess.post(f"{url.rstrip('/')}{CODE_PUSH_PATH}", data=bundle, timeout=timeout_s, headers=headers)
    except requests.RequestException as exc:
        logger.warning("code push to %s failed: %s", url, exc)
        return False
    if resp.status_code == 200:
        return True
    if resp.status_code in (400, 403, 413):
        raise RunPodError(f"the pod refused the code: HTTP {resp.status_code} {resp.text[:200]}", resp.status_code)
    logger.warning("code push to %s: HTTP %s", url, resp.status_code)
    return False


def probe_health(url: str, session: requests.Session | None = None, timeout_s: float = HEALTH_TIMEOUT_S) -> dict | None:
    """GET `<proxy>/api/health`. The parsed JSON when the app answers 200, else `None` (stopped,
    booting, or unreachable -- the caller doesn't need to tell those apart)."""
    sess = session or requests.Session()
    try:
        resp = sess.get(f"{url.rstrip('/')}/api/health", timeout=timeout_s, headers={"User-Agent": USER_AGENT})
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
