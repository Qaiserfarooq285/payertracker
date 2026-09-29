#!/usr/bin/env python3
"""Unified Launcher for The Reach Vision (player tracking & analytics) web server.

Environment knobs (all optional; docs/DEPLOY.md):
  PORT     -- listen port (default 8000; RunPod's HTTP proxy expects the port you expose there)
  PV_DEV=1 -- enable uvicorn's auto-reload. OFF by default on purpose: the reloader restarts the
              server whenever a file under the repo changes, which kills any pipeline job that is
              mid-run (observed 2026-09-15 as a bare "Job failed: None"). Turn it on only while
              editing code locally, never on a hosted pod.
  PV_ACCESS_PASSWORD -- login gate password. ON by default (the built-in `admin1122`) even when
              unset; the literal `off` disables it (local editing only). See `apps/api/access.py`.
"""

import os
import sys
from pathlib import Path

import uvicorn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.access import resolve_access_password  # noqa: E402 -- needs sys.path set up above
from apps.api.cpu_threads import limit_cpu_threads  # noqa: E402

if __name__ == "__main__":
    # Before anything imports numpy / torch / cv2: size their thread pools to the pod's CPU quota
    # (a RunPod pod sees all of its host's CPUs; 544 threads on 27 CPUs stalled the GPU, 2026-09-30).
    cpu_threads = limit_cpu_threads()
    if cpu_threads:
        import cv2  # OpenCV keeps its own pool; numpy under it already sees OPENBLAS_NUM_THREADS

        cv2.setNumThreads(cpu_threads)
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("PV_DEV", "") == "1"
    password, is_default = resolve_access_password(os.environ.get("PV_ACCESS_PASSWORD"))
    if not password:
        gate_label = "off"
    elif is_default:
        gate_label = "ON (default password — set PV_ACCESS_PASSWORD to change)"
    else:
        gate_label = "ON"
    print("=" * 70)
    print("  THE REACH VISION — Player Tracking & Analytics System")
    print(f"  Server starting at: http://localhost:{port}")
    print(f"  Auto-reload: {'ON (dev)' if reload else 'off'}")
    print(f"  Access gate: {gate_label}")
    print(f"  CPU threads: {cpu_threads or 'library defaults'}" + (" (the pod's CPU quota)" if cpu_threads else ""))
    print("=" * 70)
    # ONE worker, always: JOBS, the in-flight-slug guard and chunked-upload state are in-process.
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=port, reload=reload, workers=1)
