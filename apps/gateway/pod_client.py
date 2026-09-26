"""HTTP client for the app running ON the pod, as seen from the VPS gateway.

The pod's own access gate (`apps/api/main.py::access_gate`) is still on -- its proxy hostname is
public on runpod.net -- so the gateway logs in with the shared `PV_ACCESS_PASSWORD` and keeps the
session cookie. A 401 (the pod restarted: its session key is per-process) means "log in again and
retry once", exactly what the browser does in `apps/web/js/app.js`.

`base_url` is re-read on every call because the pod id -- and therefore the proxy host -- changes
whenever the pod is recreated (`apps/gateway/pod_manager.py`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import requests

from apps.api.uploads import UPLOAD_CHUNK_BYTES
from apps.gateway.runpod_pods import USER_AGENT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 60.0
UPLOAD_TIMEOUT_S = 300.0
STREAM_TIMEOUT_S = 300.0


class PodClient:
    def __init__(
        self,
        base_url_fn: Callable[[], str],
        password: str,
        session: requests.Session | None = None,
    ):
        self._base_url_fn = base_url_fn
        self._password = password
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    @property
    def base_url(self) -> str:
        url = self._base_url_fn()
        if not url:
            raise requests.ConnectionError("GPU pod is not online")
        return url.rstrip("/")

    # ---------------------------------------------------------------- auth

    def login(self) -> None:
        if not self._password:
            return
        resp = self.session.post(
            f"{self.base_url}/api/login", json={"password": self._password}, timeout=DEFAULT_TIMEOUT_S
        )
        if resp.status_code != 200:
            raise requests.HTTPError(f"pod login failed: HTTP {resp.status_code} {resp.text[:200]}")

    def request(self, method: str, path: str, *, stream: bool = False, timeout: float = DEFAULT_TIMEOUT_S, **kw) -> requests.Response:
        """One call with a single transparent re-login on 401."""
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, stream=stream, timeout=timeout, **kw)
        if resp.status_code == 401:
            resp.close()
            self.login()
            resp = self.session.request(method, url, stream=stream, timeout=timeout, **kw)
        return resp

    # ---------------------------------------------------------------- app calls

    def list_videos(self) -> dict[str, Any]:
        resp = self.request("GET", "/api/videos")
        resp.raise_for_status()
        return resp.json()

    def has_input_video(self, name: str, size_bytes: int) -> bool:
        """Is this exact file (name + size within rounding) already in the pod's `input/`? The
        network volume keeps uploads across pod recreations, so usually yes after the first run."""
        want_mb = round(size_bytes / (1024 * 1024), 2)
        for v in self.list_videos().get("input_videos", []):
            if v.get("name") == name and abs(float(v.get("size_mb", -1)) - want_mb) <= 0.02:
                return True
        return False

    def upload_video(self, path: Path, upload_id: str, on_progress: Callable[[float], None] | None = None) -> dict[str, Any]:
        """Replay a local file to the pod's `/api/upload/chunk`, the same protocol the browser used."""
        size = path.stat().st_size
        total = max(1, -(-size // UPLOAD_CHUNK_BYTES))
        data: dict[str, Any] = {}
        with open(path, "rb") as fh:
            for index in range(total):
                blob = fh.read(UPLOAD_CHUNK_BYTES)
                resp = self.request(
                    "POST",
                    "/api/upload/chunk",
                    data={"upload_id": upload_id, "index": str(index), "total": str(total), "filename": path.name},
                    files={"chunk": (path.name, blob, "application/octet-stream")},
                    timeout=UPLOAD_TIMEOUT_S,
                )
                if resp.status_code != 200:
                    raise requests.HTTPError(f"chunk {index + 1}/{total}: HTTP {resp.status_code} {resp.text[:200]}")
                data = resp.json()
                if on_progress:
                    on_progress((index + 1) / total)
        return data

    def start_process(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.request("POST", "/api/process", json=payload)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = resp.text[:200]
            raise requests.HTTPError(f"pod refused the job (HTTP {resp.status_code}): {detail}")
        return resp.json()

    def job_status(self, pod_job_id: str) -> dict[str, Any]:
        resp = self.request("GET", f"/api/status/{pod_job_id}")
        resp.raise_for_status()
        return resp.json()

    def cancel_job(self, pod_job_id: str) -> None:
        """Stop the pipeline on the pod (`apps/api/main.py::cancel_job`). Best effort: raises on
        HTTP failure, and the caller decides what that means."""
        resp = self.request("POST", f"/api/jobs/{pod_job_id}/cancel")
        resp.raise_for_status()

    # ---------------------------------------------------------------- generic proxy

    def proxy(self, method: str, path_qs: str, headers: dict[str, str], body: bytes | None) -> requests.Response:
        """Forward one browser request as-is (streamed response). Hop-by-hop headers are the
        caller's problem; `Range` and content-type pass through so video seeking keeps working."""
        return self.request(method, path_qs, headers=headers, data=body, stream=True, timeout=STREAM_TIMEOUT_S)


def iter_response(resp: requests.Response, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
    try:
        yield from resp.iter_content(chunk_size=chunk_size)
    finally:
        resp.close()
