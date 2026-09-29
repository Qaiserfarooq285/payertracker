"""Size every CPU thread pool to the pod's CPU QUOTA, not to its host's CPU count.

A RunPod container sees every CPU of its host (`nproc` said 255 on the A100 host, 2026-09-30) but
may only use its cgroup quota (27 CPUs there). PyTorch/OpenMP, NumPy's BLAS and OpenCV each size
their pools from the host count, so the first real hosted run had 544 threads fighting over 27
CPUs: throttled in 95% of scheduler periods, the A100 11% busy, a 10-min 480p clip heading for
2.5 hours. `limit_cpu_threads` must run before numpy / torch / cv2 are imported -- the pools read
these variables once, when the libraries load (`start_app.py` calls it first thing).

Only the pool SIZES change, never what is computed. Off a cgroup quota (a laptop, `make serve`)
nothing is touched, and a variable that is already set is left alone.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")
POOL_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS")


def cgroup_cpu_limit(root: Path = CGROUP_ROOT) -> int | None:
    """Whole CPUs the container's CFS quota allows (at least 1), or None when it has no quota."""
    try:
        if (root / "cpu.max").exists():  # cgroup v2: "<quota> <period>", quota may be "max"
            quota, period = (root / "cpu.max").read_text().split()[:2]
        else:  # cgroup v1 -- what RunPod pods use
            quota = (root / "cpu" / "cpu.cfs_quota_us").read_text().strip()
            period = (root / "cpu" / "cpu.cfs_period_us").read_text().strip()
        if quota == "max" or int(quota) <= 0:
            return None
        return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        return None


def limit_cpu_threads(environ: MutableMapping[str, str] = os.environ, root: Path = CGROUP_ROOT) -> int | None:
    """Point every pool-size variable at the quota (unless already set). Returns the OpenMP
    thread count now in effect, or None when there is no quota to respect."""
    limit = cgroup_cpu_limit(root)
    if limit is None:
        return None
    for name in POOL_ENV_VARS:
        environ.setdefault(name, str(limit))
    return int(environ["OMP_NUM_THREADS"])
