"""The first thing a pod runs: wait on its HTTP port for the gateway to hand over the code.

Why (2026-09-30, the first real GPU run after ADR-24): a pod in EUR-IS-1 cannot open a TCP
connection to the Hostinger VPS at all -- measured both ways, the SYNs time out -- so a pod can
never PULL its code from the gateway (every curl to /pod/bootstrap.sh hung for 13 min, then the
pod stopped itself and the gateway started it again, billing an A100 for nothing). The gateway CAN
reach the pod through RunPod's HTTPS proxy -- the path every upload, status poll and result
download already takes -- so the gateway PUSHES the code instead (`runpod_pods.push_code`).

The gateway passes this file to the pod zlib+base64-compressed in `PV_POD_RECEIVER`, and the pod's
start command execs it (`runpod_pods.pod_create_body`) -- stdlib only, nothing is installed yet.

    GET  /pv/code   200 {"awaiting": true}: "I'm up and still need the code"
    POST /pv/code   the gateway's `git archive` tar.gz; X-PV-Pod-Key must equal PV_POD_KEY.
                    Writes PV_CODE_TGZ (+ ".commit") and the bundle's docker/runpod_bootstrap.sh
                    to PV_BOOTSTRAP, answers 200, and exits 0 -> the start command runs the bootstrap
    anything else   404, so the gateway's /api/health probe keeps reading "not up yet"

Exits 1 when nothing valid arrives within PV_CODE_WAIT_S (default 15 min), so a pod whose gateway
is gone stops itself (the start command's `|| stop-self`) instead of billing.
"""

import hmac
import io
import json
import os
import sys
import tarfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

CODE_PATH = "/pv/code"
BOOTSTRAP_MEMBER = "docker/runpod_bootstrap.sh"
MAX_BYTES = 64 * 1024 * 1024  # the bundle is ~1 MB; anything near this is not ours
CLIENT_TIMEOUT_S = 60  # one thread serves everything: a stalled client must not hold it


def log(msg):
    print(f"[pitchvision] {msg}", flush=True)


def write_atomic(path, data, mode=0o644):
    tmp = f"{path}.part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def make_handler(key, tgz_path, bootstrap_path, received):
    class Handler(BaseHTTPRequestHandler):
        timeout = CLIENT_TIMEOUT_S

        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            # RunPod's proxy may re-send a sized body chunked; accept both.
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                parts, total = [], 0
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        return b"".join(parts)
                    total += size
                    if total > MAX_BYTES:
                        raise ValueError("bundle too large")
                    parts.append(self.rfile.read(size))
                    self.rfile.readline()
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BYTES:
                raise ValueError("bundle too large")
            return self.rfile.read(length)

        def do_GET(self):
            if self.path == CODE_PATH:
                self._reply(200, {"awaiting": not received})
            else:
                self._reply(404, {"detail": "the app is not running yet"})

        def do_POST(self):
            if self.path != CODE_PATH:
                self._reply(404, {"detail": "not found"})
                return
            if not hmac.compare_digest(self.headers.get("X-PV-Pod-Key", "").encode(), key.encode()):
                self._reply(403, {"detail": "bad pod key"})
                return
            try:
                data = self._body()
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                    bootstrap = tf.extractfile(tf.getmember(BOOTSTRAP_MEMBER)).read()
            except (ValueError, KeyError, AttributeError, OSError, EOFError, tarfile.TarError) as exc:
                self._reply(400, {"detail": f"not a code bundle: {exc}"})
                return
            commit = "".join(c for c in self.headers.get("X-PV-Commit", "") if c.isalnum())[:40]
            write_atomic(tgz_path, data)
            write_atomic(f"{tgz_path}.commit", commit.encode())
            write_atomic(bootstrap_path, bootstrap, 0o755)
            received["commit"] = commit or "unknown"
            self._reply(200, {"ok": True, "bytes": len(data), "commit": received["commit"]})

        def log_message(self, fmt, *args):
            log(f"receiver: {self.command} {self.path} -> {fmt % args}")

    return Handler


def main():
    key = os.environ.get("PV_POD_KEY", "")
    if not key:
        log("PV_POD_KEY is not set -- the gateway's code cannot be accepted")
        return 1
    port = int(os.environ.get("PORT") or "8000")
    tgz_path = os.environ.get("PV_CODE_TGZ") or "/tmp/pv-code.tar.gz"
    bootstrap_path = os.environ.get("PV_BOOTSTRAP") or "/tmp/pv-bootstrap.sh"
    wait_s = float(os.environ.get("PV_CODE_WAIT_S") or "900")
    received = {}
    server = HTTPServer(("0.0.0.0", port), make_handler(key, tgz_path, bootstrap_path, received))
    server.timeout = 5  # handle_request() returns this often, so the deadline is honoured
    log(f"waiting for the gateway to send the code (port {port}, at most {int(wait_s)} s)")
    deadline = time.time() + wait_s
    try:
        while not received and time.time() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if not received:
        log(f"no code arrived in {int(wait_s)} s -- giving up")
        return 1
    log(f"code {received['commit']} received")
    return 0


if __name__ == "__main__":
    sys.exit(main())
