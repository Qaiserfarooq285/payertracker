"""Thread pools sized to the pod's CPU quota (`apps/api/cpu_threads.py`). The case that drove it
(2026-09-30): an A100 pod whose cgroup allowed 27 CPUs while `nproc` said 255 -- 544 threads,
throttled in 95% of scheduler periods, the GPU 11% busy."""

from __future__ import annotations

from pathlib import Path

from apps.api.cpu_threads import POOL_ENV_VARS, cgroup_cpu_limit, limit_cpu_threads


def _v1(root: Path, quota: str, period: str = "100000") -> Path:
    (root / "cpu").mkdir(parents=True)
    (root / "cpu" / "cpu.cfs_quota_us").write_text(f"{quota}\n")
    (root / "cpu" / "cpu.cfs_period_us").write_text(f"{period}\n")
    return root


def test_reads_the_runpod_pods_cgroup_v1_quota(tmp_path):
    assert cgroup_cpu_limit(_v1(tmp_path, "2720000")) == 27  # what the A100 pod reported


def test_reads_a_cgroup_v2_quota_and_never_goes_below_one(tmp_path):
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    assert cgroup_cpu_limit(tmp_path) == 8
    (tmp_path / "cpu.max").write_text("50000 100000\n")
    assert cgroup_cpu_limit(tmp_path) == 1


def test_no_quota_means_hands_off(tmp_path):
    assert cgroup_cpu_limit(tmp_path) is None  # no cgroup files at all (a Mac)
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cgroup_cpu_limit(tmp_path) is None
    assert cgroup_cpu_limit(_v1(tmp_path / "v1", "-1")) is None
    env: dict[str, str] = {}
    assert limit_cpu_threads(env, tmp_path / "nothing") is None and env == {}


def test_sets_every_pool_to_the_quota_but_keeps_an_explicit_setting(tmp_path):
    root = _v1(tmp_path, "2720000")
    env: dict[str, str] = {}
    assert limit_cpu_threads(env, root) == 27
    assert {name: env[name] for name in POOL_ENV_VARS} == dict.fromkeys(POOL_ENV_VARS, "27")
    env = {"OMP_NUM_THREADS": "4"}
    assert limit_cpu_threads(env, root) == 4 and env["OPENBLAS_NUM_THREADS"] == "27"
