"""Runs the GPU pod exactly when the user says so (docs/DEPLOY.md "Always-on gateway").

The gateway on the VPS is always up; the RunPod pod is not. Since 2026-09-26 the pod is under the
USER's control, not the gateway's: it is switched on only by the Start GPU button and off by the
Stop GPU button, by the idle auto-stop, or by a failed start. Nothing else -- not a page load,
not a results view, not a queued job, not a gateway restart -- may create or start a pod.

Why (owner, 2026-09-26): the old "wake whenever anyone needs it, recreate whatever fails" policy
burned the whole credit. The pods' GitHub token had expired, so every new pod died at its first
command; the gateway kept "replacing" pods that could never boot (~13 GPU-hours, zero output)
until the balance hit zero, then reported the refusals as "No GPU available". So:

    `wanted`          set only by `start()`, cleared by `stop()` and by any failed start
    no pod            -> create one on the chosen GPU tier
    create refused    -> WAITING_FOR_GPU, retried every `retry_s` for at most `max_gpu_wait_s`,
                         then give up (a GPU appearing hours later must not start billing then)
    balance too low   -> ERROR at once, with the reason -- retrying cannot fix it
    pod stopped       -> Start; if RunPod refuses (GPU taken) -> terminate it, then create
    pod running       -> wait for /api/health (BOOTING); a boot that is not up within
                         `boot_cap_s` is STOPPED + terminated and reported -- never recreated
                         in a loop

Every transition is reported through `on_progress(phase, message)`. `stop()` interrupts a start
in progress (the waits are on a cancel event), and `idle_check()` stops a billing pod nobody has
used for a while. `clock` and `sleep` are injectable, and the RunPod client is a parameter, so the
whole state machine is unit-tested with a fake API and no real time.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

from apps.gateway import runpod_pods as rp
from apps.gateway.runpod_pods import PodInfo, RunPodClient, RunPodError

logger = logging.getLogger(__name__)

# Phases, as shown to the user. Strings on purpose: they go straight into JSON.
UNKNOWN = "unknown"  # never looked yet, or RunPod unreachable
OFFLINE = "offline"  # pod stopped (or none exists) and nobody is asking for it
STARTING = "starting"  # Start accepted / pod created; waiting for RunPod to run the container
BOOTING = "booting"  # container running, app not answering yet
WAITING_FOR_GPU = "waiting_for_gpu"  # RunPod has no GPU of the wanted types right now
ONLINE = "online"  # /api/health answers
ERROR = "error"  # something a retry won't fix (auth); a human must look

DEFAULT_POLL_S = 10.0  # while starting/booting
DEFAULT_RETRY_S = 60.0  # while waiting for a GPU
# A boot is apt + code fetch + (only when pyproject.toml changed) deps; ~3-5 min normally, ~10 on a
# fresh volume. Past this the pod is broken, and every further minute is billed for nothing.
DEFAULT_BOOT_CAP_S = 20 * 60.0
# How long a Start keeps asking RunPod for a card before giving up. Short on purpose: the user is
# watching, and a GPU that frees up hours later must not start billing while nobody is there.
DEFAULT_MAX_GPU_WAIT_S = 15 * 60.0
DEFAULT_TERMINATE_WAIT_S = 90.0
BILLING_URL = "https://www.runpod.io/console/user/billing"
# A replacement pod is created WITH a gpuTypeId, so it coming up GPU-less means something is
# wrong at RunPod's end; don't burn credit recreating forever.
MAX_NO_GPU_REPLACEMENTS = 3


@dataclass
class PodConfig:
    name: str = rp.DEFAULT_POD_NAME
    volume_name: str = rp.DEFAULT_VOLUME_NAME
    volume_gb: int = rp.DEFAULT_VOLUME_GB
    datacenter: str = rp.DEFAULT_DATACENTER
    cloud: str = rp.DEFAULT_CLOUD
    gpu_types: tuple[str, ...] = rp.DEFAULT_GPU_TYPES
    port: int = rp.DEFAULT_PORT
    # Everything the bootstrap needs on the pod (docker/runpod.env.example).
    pod_env: dict[str, str] = field(default_factory=dict)
    poll_s: float = DEFAULT_POLL_S
    retry_s: float = DEFAULT_RETRY_S
    boot_cap_s: float = DEFAULT_BOOT_CAP_S
    max_gpu_wait_s: float = DEFAULT_MAX_GPU_WAIT_S
    terminate_wait_s: float = DEFAULT_TERMINATE_WAIT_S


@dataclass
class PodState:
    phase: str = UNKNOWN
    message: str = ""
    pod_id: str = ""
    proxy_url: str = ""
    gpu_type: str = ""
    gpu_name: str = ""  # what torch on the pod reports, once online
    cost_per_hr: float = 0.0
    since: float = 0.0  # wall-clock when `phase` was entered
    checked_at: float = 0.0
    last_error: str = ""
    waiting_since: float = 0.0  # wall-clock when WAITING_FOR_GPU began (0 when not waiting)
    wanted: bool = False  # the user pressed Start GPU and has not stopped it (nor has it failed)
    billing: bool = False  # RunPod has a pod RUNNING under our name -- i.e. credit is being spent
    idle_stop_in_s: float | None = None  # seconds until the idle auto-stop, when it is counting

    def public(self) -> dict[str, Any]:
        """What `/api/health` exposes without a login: no pod id, no cost."""
        return {
            "phase": self.phase,
            "message": self.message,
            "gpu_name": self.gpu_name,
            "wanted": self.wanted,
            "billing": self.billing,
        }

    def full(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "message": self.message,
            "pod_id": self.pod_id,
            "proxy_url": self.proxy_url,
            "gpu_type": self.gpu_type,
            "gpu_name": self.gpu_name,
            "cost_per_hr": self.cost_per_hr,
            "since": self.since,
            "checked_at": self.checked_at,
            "last_error": self.last_error,
            "waiting_since": self.waiting_since,
            "wanted": self.wanted,
            "billing": self.billing,
            "idle_stop_in_s": self.idle_stop_in_s,
        }


class PodUnavailable(Exception):
    """`ensure_online` gave up: the GPU is off / was stopped, the start failed in a way the user
    has to see (no GPU in time, empty balance, a pod that will not boot, bad key), or a
    caller-imposed deadline passed."""


ProgressFn = Callable[[str, str], None]


def _no_progress(_phase: str, _message: str) -> None:
    pass


def _fmt_minutes(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    return f"{minutes} min"


class PodManager:
    def __init__(
        self,
        client: RunPodClient,
        config: PodConfig,
        *,
        health_session: requests.Session | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] | None = None,
    ):
        self.client = client
        self.cfg = config
        self._health_session = health_session or requests.Session()
        self._clock = clock
        # `stop()` sets this; every wait inside a start returns early on it.
        self._cancel = threading.Event()
        self._sleep = sleep or (lambda seconds: self._cancel.wait(seconds))
        self._lock = threading.Lock()  # serialises ensure_online / stop; refresh() never blocks on it
        self.state = PodState()
        self._volume_id: str = ""
        self._start_thread: threading.Thread | None = None
        self.activity_at = clock()  # last use of the GPU, for the idle auto-stop

    # ---------------------------------------------------------------- state helpers

    def _set(self, phase: str, message: str, pod: PodInfo | None = None, error: str = "") -> None:
        now = self._clock()
        st = self.state
        if phase != st.phase:
            st.since = now
        st.phase = phase
        st.message = message
        st.checked_at = now
        if error:
            st.last_error = error
        if pod is not None:
            st.billing = pod.is_running
            st.pod_id = pod.id
            st.proxy_url = rp.proxy_url(pod.id, self.cfg.port)
            st.gpu_type = pod.gpu_type
            st.cost_per_hr = pod.cost_per_hr
        elif phase in (OFFLINE, WAITING_FOR_GPU):
            st.billing = False
            st.pod_id = ""
            st.proxy_url = ""
            st.gpu_type = ""
            st.cost_per_hr = 0.0
        if phase == WAITING_FOR_GPU:
            if not st.waiting_since:
                st.waiting_since = now
        else:
            st.waiting_since = 0.0
        if phase != ONLINE:
            st.gpu_name = ""
        logger.info("pod %s: %s", phase, message)

    @property
    def online(self) -> bool:
        return self.state.phase == ONLINE

    @property
    def proxy_url(self) -> str:
        return self.state.proxy_url

    # ---------------------------------------------------------------- read-only refresh

    def refresh(self) -> PodState:
        """Look at RunPod + the proxy once and update `state` WITHOUT changing anything. Used by
        the background status poller so the UI is honest even when no job is running. Skipped
        while a start/stop holds the lock -- that one reports its own progress."""
        if self._lock.locked() or self.starting:
            return self.state
        try:
            pod = self.client.find_pod(self.cfg.name)
        except RunPodError as exc:
            self._set(self.state.phase if self.state.phase != UNKNOWN else UNKNOWN,
                      f"Cannot reach RunPod: {exc}", error=str(exc))
            return self.state
        was_billing = self.state.billing
        self.state.billing = pod is not None and pod.is_running
        if self.state.billing and not was_billing:
            self.touch()  # a pod we did not see running before: its idle countdown starts now
        if pod is None or not pod.is_running:
            if self.state.phase == ERROR:
                return self.state  # keep the reason a start failed on screen until the next Start/Stop
            self._set(OFFLINE, "GPU is off -- nothing is billing. Press Start GPU when you want to "
                      "process a video.", pod)
            return self.state
        if not pod.has_gpu:
            self._set(OFFLINE, "A pod is running WITHOUT a GPU (RunPod resumed it on CPU) and is "
                      "billing -- press Stop GPU, then Start GPU.", pod)
            return self.state
        health = rp.probe_health(rp.proxy_url(pod.id, self.cfg.port), self._health_session)
        if health and not health.get("gpu_available", True):
            self._set(OFFLINE, "The pod is up but sees no GPU -- press Stop GPU, then Start GPU.", pod)
        elif health:
            self._mark_online(pod, health)
        else:
            self._set(BOOTING, f"GPU pod is running, app still starting ({_fmt_minutes(pod.uptime_s)} up).", pod)
        return self.state

    # ---------------------------------------------------------------- user control

    @property
    def starting(self) -> bool:
        """A Start GPU press is still being worked on (creating / booting / waiting for stock)."""
        return self._start_thread is not None and self._start_thread.is_alive()

    def touch(self) -> None:
        """Someone used the GPU (a job step, a results view): restart the idle countdown."""
        self.activity_at = self._clock()

    def start(self, gpu_types: tuple[str, ...] | list[str] | None = None) -> bool:
        """The Start GPU button. Brings the pod up on a background thread; returns False when a
        start is already in progress (the button was pressed twice)."""
        if self.starting:
            return False
        self._cancel.clear()
        self.state.wanted = True
        self.state.last_error = ""
        self.touch()
        self._set(STARTING, "Starting the GPU...")

        def run() -> None:
            try:
                self.ensure_online(gpu_types=gpu_types)
            except PodUnavailable as exc:
                logger.info("start ended without a GPU: %s", exc)
            except Exception:  # never let the thread die silently
                logger.exception("start crashed")
                self.state.wanted = False
                self._set(ERROR, "Starting the GPU crashed -- see the gateway log.", error="start crashed")

        self._start_thread = threading.Thread(target=run, name="gateway-pod-start", daemon=True)
        self._start_thread.start()
        return True

    def stop(self, reason: str = "stopped by you") -> PodState:
        """The Stop GPU button (and the idle auto-stop): abort any start in progress, then stop
        AND terminate the pod so nothing keeps billing. The network volume -- uploads, results --
        is untouched; the next Start creates a fresh pod on it."""
        self.state.wanted = False
        self._cancel.set()
        with self._lock:  # waits for a running start to notice the cancel and let go
            self._cancel.clear()
            self.state.wanted = False
            try:
                pod = self.client.find_pod(self.cfg.name)
            except RunPodError as exc:
                self._set(ERROR, f"Could not reach RunPod to stop the GPU ({exc}) -- check the RunPod "
                          "console.", error=str(exc))
                return self.state
            if pod is not None and not self._shut_down(pod.id):
                self._set(ERROR, "RunPod did not stop the GPU pod -- stop it in the RunPod console; "
                          "it is still billing.", pod, error="stop failed")
                self.state.billing = True
                return self.state
            self.state.billing = False
            self._set(OFFLINE, f"GPU is off ({reason}) -- nothing is billing. Press Start GPU when "
                      "you want to process a video.")
            return self.state

    def idle_check(self, idle_s: float, busy: bool) -> float | None:
        """Called by the gateway's poller. While a pod is billing, nobody is starting/stopping
        it, and no job is using it, count down `idle_s` from the last use and stop it at zero.
        Returns (and records in `state`) the seconds left, or None when not counting."""
        st = self.state
        if busy:
            self.touch()
        if idle_s <= 0 or busy or not st.billing or self._lock.locked():
            st.idle_stop_in_s = None
            return None
        left = self.activity_at + idle_s - self._clock()
        if left > 0:
            st.idle_stop_in_s = left
            return left
        st.idle_stop_in_s = None
        logger.info("idle auto-stop: no GPU use for %.0f min", idle_s / 60)
        self.stop(f"auto-stopped after {int(idle_s // 60)} min with nothing to do")
        return 0.0

    def _mark_online(self, pod: PodInfo, health: dict) -> None:
        gpu = str(health.get("gpu_name") or pod.gpu_type or "GPU")
        if self.state.phase != ONLINE:
            self.touch()  # the idle countdown starts when the GPU is actually usable
        self._set(ONLINE, f"GPU online ({gpu}) -- billing until you press Stop GPU.", pod)
        self.state.gpu_name = gpu

    # ---------------------------------------------------------------- the state machine

    def ensure_online(
        self,
        on_progress: ProgressFn = _no_progress,
        deadline_s: float | None = None,
        gpu_types: tuple[str, ...] | list[str] | None = None,
    ) -> PodState:
        """Block until the pod's app answers `/api/health`, doing whatever RunPod needs along the
        way. Reports each phase change via `on_progress`. Raises `PodUnavailable` only for an
        auth error or when `deadline_s` (seconds from now, `None` = wait forever) passes.

        `gpu_types` (2026-09-19, the user's GPU pick in the web app -- `rp.GPU_TIERS`): the GPU
        type ids acceptable for THIS request, in order of preference. A pod that exists on a card
        outside that list is replaced (its volume -- uploads, results -- is untouched) so the run
        really happens on what the user chose and is billed at. `None` = the configured default,
        used only when a pod has to be CREATED -- an existing pod on any card is kept as before
        (a plain wake-up must never throw a healthy pod away)."""
        with self._lock:
            if not self.state.wanted:
                raise PodUnavailable("The GPU is off -- press Start GPU.")
            try:
                return self._ensure_online_locked(
                    on_progress,
                    deadline_s,
                    tuple(gpu_types) if gpu_types else self.cfg.gpu_types,
                    strict_gpu=bool(gpu_types),
                )
            except PodUnavailable:
                # Any failed or cancelled start leaves the GPU OFF: it is never retried behind
                # the user's back. `billing` is re-read by the next refresh.
                self.state.wanted = False
                raise

    def _ensure_online_locked(
        self,
        on_progress: ProgressFn,
        deadline_s: float | None,
        wanted: tuple[str, ...],
        strict_gpu: bool = False,
    ) -> PodState:
        started = self._clock()
        deadline = started + deadline_s if deadline_s is not None else None
        last_reported: tuple[str, str] | None = None
        boot_started: float | None = None
        no_gpu_replacements = 0

        def report() -> None:
            nonlocal last_reported
            key = (self.state.phase, self.state.message)
            if key != last_reported:
                last_reported = key
                on_progress(*key)

        def wait(seconds: float) -> None:
            if self._cancel.is_set() or not self.state.wanted:
                raise PodUnavailable("stopped")
            if deadline is not None and self._clock() + seconds > deadline:
                raise PodUnavailable(self.state.message or "gave up waiting for the GPU pod")
            self._sleep(seconds)
            if self._cancel.is_set() or not self.state.wanted:
                raise PodUnavailable("stopped")

        def low_balance(exc: RunPodError) -> None:
            self._set(ERROR, "RunPod refused: the account balance is too low to rent a GPU. Add "
                      f"funds at {BILLING_URL}, then press Start GPU.", error=str(exc))
            report()
            raise PodUnavailable(self.state.message) from exc

        while True:
            try:
                pod = self.client.find_pod(self.cfg.name)
            except RunPodError as exc:
                if exc.is_auth_error:
                    self._set(ERROR, f"RunPod rejected the API key: {exc}", error=str(exc))
                    report()
                    raise PodUnavailable(self.state.message) from exc
                self._set(self.state.phase, f"Cannot reach RunPod ({exc}); retrying.", error=str(exc))
                report()
                wait(self.cfg.poll_s)
                continue

            # --- pod on the wrong card for this request: replace it ------------------------
            # Only when RunPod tells us the card (a stopped pod sometimes reports none); an
            # unknown card is started as before rather than thrown away on a guess.
            if strict_gpu and pod is not None and pod.gpu_type and pod.gpu_type not in wanted:
                self._set(
                    STARTING,
                    f"Switching GPU: the pod is on {pod.gpu_type}, this run asked for "
                    f"{wanted[0]} -- replacing the pod (your uploads and results are on the "
                    "persistent volume).",
                    pod,
                    error="gpu type mismatch",
                )
                report()
                self._terminate_and_wait(pod.id)
                boot_started = None
                continue

            # --- no pod: create one -------------------------------------------------------
            if pod is None:
                try:
                    pod = self._create_pod(wanted)
                except RunPodError as exc:
                    if exc.is_auth_error:
                        self._set(ERROR, f"RunPod rejected the API key: {exc}", error=str(exc))
                        report()
                        raise PodUnavailable(self.state.message) from exc
                    if rp.is_low_balance_error(str(exc)):
                        low_balance(exc)
                    # Anything else from POST /pods is, in practice, "no stock for these GPU
                    # types right now" -- RunPod phrases it several ways. Keep the raw reason.
                    waited = self._clock() - (self.state.waiting_since or self._clock())
                    if waited >= self.cfg.max_gpu_wait_s:
                        self._set(
                            OFFLINE,
                            f"No {wanted[0].replace('NVIDIA ', '')} came free on RunPod in "
                            f"{_fmt_minutes(waited)} -- gave up so nothing starts billing while you "
                            "are away. Press Start GPU to try again, or pick another GPU tier.",
                            error=str(exc),
                        )
                        report()
                        raise PodUnavailable(self.state.message) from exc
                    self._set(
                        WAITING_FOR_GPU,
                        "No GPU of this tier free on RunPod right now -- asking again every "
                        f"{int(self.cfg.retry_s)} s (waiting {_fmt_minutes(waited)} of at most "
                        f"{_fmt_minutes(self.cfg.max_gpu_wait_s)}). Nothing is billing yet.",
                        error=str(exc),
                    )
                    report()
                    wait(self.cfg.retry_s)
                    continue
                boot_started = self._clock()
                self._set(STARTING, f"GPU pod created on {pod.gpu_type or 'a GPU'}; starting up.", pod)
                report()
                wait(self.cfg.poll_s)
                continue

            # --- pod exists but is stopped: Start, or tear down if its GPU is gone ----------
            if not pod.is_running:
                try:
                    pod = self.client.start_pod(pod.id)
                except RunPodError as exc:
                    if exc.is_auth_error:
                        self._set(ERROR, f"RunPod rejected the API key: {exc}", error=str(exc))
                        report()
                        raise PodUnavailable(self.state.message) from exc
                    if rp.is_low_balance_error(str(exc)):
                        low_balance(exc)
                    if exc.status == 0:
                        # Network blip -- not evidence the GPU is gone. Try again shortly.
                        self._set(OFFLINE, f"Cannot reach RunPod ({exc}); retrying.", pod, error=str(exc))
                        report()
                        wait(self.cfg.poll_s)
                        continue
                    self._set(
                        STARTING,
                        "The stopped pod's GPU was taken by someone else -- replacing the pod "
                        "(your uploads and results are on the persistent volume).",
                        pod,
                        error=str(exc),
                    )
                    report()
                    self._terminate_and_wait(pod.id)
                    continue  # next loop: no pod -> create
                if not pod.is_running or not pod.has_gpu:
                    # 2xx, but either the pod is still not RUNNING or RunPod resumed it on CPU
                    # only (its fallback when the card is gone) -- both are a refusal to us.
                    self._set(
                        STARTING,
                        "RunPod could not give the stopped pod its GPU back -- replacing the pod "
                        "(your uploads and results are on the persistent volume).",
                        pod,
                        error="resumed without a GPU" if pod.is_running else "start did not take",
                    )
                    report()
                    self._terminate_and_wait(pod.id)
                    continue
                boot_started = self._clock()
                self._set(STARTING, f"GPU pod starting on {pod.gpu_type or 'its GPU'}.", pod)
                report()
                wait(self.cfg.poll_s)
                continue

            # --- pod running: does it have a GPU, and is the app up? -------------------------
            if not pod.has_gpu:
                no_gpu_replacements += 1
                if no_gpu_replacements > MAX_NO_GPU_REPLACEMENTS:
                    self._set(ERROR, "Every replacement pod comes up without a GPU -- check the "
                              "RunPod console.", pod, error="repeated CPU-only pods")
                    report()
                    raise PodUnavailable(self.state.message)
                self._set(STARTING, "The running pod has no GPU (RunPod resumed it on CPU) -- replacing it.",
                          pod, error="running without a GPU")
                report()
                self._terminate_and_wait(pod.id)
                boot_started = None
                continue
            health = rp.probe_health(rp.proxy_url(pod.id, self.cfg.port), self._health_session)
            if health and not health.get("gpu_available", True):
                # The app itself is the last word: no CUDA device means no GPU, whatever RunPod says.
                no_gpu_replacements += 1
                if no_gpu_replacements > MAX_NO_GPU_REPLACEMENTS:
                    self._set(ERROR, "Every replacement pod comes up without a GPU -- check the "
                              "RunPod console.", pod, error="repeated CPU-only pods")
                    report()
                    raise PodUnavailable(self.state.message)
                self._set(STARTING, "The pod booted without a usable GPU -- replacing it.", pod,
                          error="app reports no CUDA device")
                report()
                self._terminate_and_wait(pod.id)
                boot_started = None
                continue
            if health:
                self._mark_online(pod, health)
                report()
                return self.state
            if boot_started is None:
                # It was already running when we arrived; count from RunPod's own uptime.
                boot_started = self._clock() - pod.uptime_s
            booting_for = self._clock() - boot_started
            if booting_for > self.cfg.boot_cap_s:
                # A pod that cannot boot will not boot on a second try either (2026-09-24: an
                # expired token made EVERY new pod die the same way, and replacing them in a loop
                # billed ~13 GPU-hours). Shut it down and tell the user; never recreate here.
                stopped = self._shut_down(pod.id)
                self._set(
                    ERROR,
                    f"The GPU pod did not finish starting in {_fmt_minutes(booting_for)} -- "
                    + ("it has been stopped so it does not keep billing. " if stopped else
                       "and RunPod did not stop it: stop it in the RunPod console now. ")
                    + "Check the pod's logs in the RunPod console, then press Start GPU to try again.",
                    error="boot cap exceeded",
                )
                self.state.billing = not stopped
                report()
                raise PodUnavailable(self.state.message)
            self._set(BOOTING, f"GPU pod is running; app starting ({_fmt_minutes(booting_for)} so far).", pod)
            report()
            wait(self.cfg.poll_s)

    # ---------------------------------------------------------------- RunPod actions

    def _volume(self) -> str:
        if self._volume_id:
            return self._volume_id
        vol = self.client.find_volume(self.cfg.volume_name)
        if vol is None:
            logger.info("creating network volume %r (%d GB, %s)", self.cfg.volume_name, self.cfg.volume_gb, self.cfg.datacenter)
            vol = self.client.create_volume(self.cfg.volume_name, self.cfg.volume_gb, self.cfg.datacenter)
        self._volume_id = str(vol["id"])
        return self._volume_id

    def _create_pod(self, gpu_types: tuple[str, ...] | None = None) -> PodInfo:
        body = rp.pod_create_body(
            name=self.cfg.name,
            gpu_type_ids=list(gpu_types or self.cfg.gpu_types),
            volume_id=self._volume(),
            env=self.cfg.pod_env,
            port=self.cfg.port,
            cloud=self.cfg.cloud,
        )
        return self.client.create_pod(body)

    def _shut_down(self, pod_id: str) -> bool:
        """Stop, then terminate, the pod; True once RunPod no longer has it running. Stop first
        because RunPod refuses to DELETE a pod it has "locked" (seen 2026-09-24 on a crash-looping
        pod, which then stayed up and billed) -- a stopped pod bills nothing for its GPU."""
        try:
            self.client.stop_pod(pod_id)
        except RunPodError as exc:
            if exc.status == 404:
                return True
            logger.warning("stop %s: %s", pod_id, exc)
        self._terminate_and_wait(pod_id)
        try:
            pod = self.client.get_pod(pod_id)
        except RunPodError as exc:
            logger.warning("get_pod after shut-down: %s", exc)
            return False
        return pod is None or not pod.is_running

    def _terminate_and_wait(self, pod_id: str) -> None:
        """DELETE the pod and wait until RunPod no longer lists it (the volume is untouched). If
        RunPod will not delete it, at least stop it, so it cannot keep billing."""
        waited = 0.0
        stop_tried = False
        while True:
            try:
                self.client.terminate_pod(pod_id)
            except RunPodError as exc:
                if exc.status == 404:
                    return
                logger.warning("terminate %s: %s", pod_id, exc)
                if not stop_tried:
                    stop_tried = True
                    try:
                        self.client.stop_pod(pod_id)
                    except RunPodError as stop_exc:
                        logger.warning("stop %s: %s", pod_id, stop_exc)
            self._sleep(self.cfg.poll_s)
            waited += self.cfg.poll_s
            try:
                if self.client.get_pod(pod_id) is None:
                    return
            except RunPodError as exc:
                logger.warning("get_pod after terminate: %s", exc)
            if waited >= self.cfg.terminate_wait_s:
                logger.warning("pod %s still listed %.0fs after terminate; carrying on", pod_id, waited)
                return
