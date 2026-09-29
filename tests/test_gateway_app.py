"""The always-on gateway's HTTP surface (`apps/gateway/main.py`) and job worker
(`apps/gateway/jobs.py`) with the pod faked -- nothing here reaches RunPod or a real pod. The
point of the gateway is that the frontend works with NO pod: uploads land, Run queues, the status
says why it is waiting."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

import apps.gateway.jobs as jobs_mod
import apps.gateway.pod_manager as pm
import apps.gateway.runpod_pods as main_rp
from apps.gateway.jobs import JobStore, JobWorker


@pytest.fixture
def gw(tmp_path, monkeypatch):
    """A fresh gateway app bound to `tmp_path`, threads off, login gate off."""
    monkeypatch.setenv("PV_GATEWAY_DATA", str(tmp_path))
    monkeypatch.setenv("PV_GATEWAY_NO_THREADS", "1")
    monkeypatch.setenv("PV_ACCESS_PASSWORD", "off")
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    import importlib

    import apps.gateway.main as main

    main = importlib.reload(main)
    main.PODS.state = pm.PodState(phase=pm.OFFLINE, message="GPU is off")
    with TestClient(main.app) as client:
        yield client, main


def _upload(client, name: str, data: bytes, chunk: int = 4):
    total = max(1, -(-len(data) // chunk))
    out = None
    for i in range(total):
        out = client.post(
            "/api/upload/chunk",
            data={"upload_id": "abcdefgh1234", "index": i, "total": total, "filename": name},
            files={"chunk": (name, io.BytesIO(data[i * chunk : (i + 1) * chunk]))},
        )
        assert out.status_code == 200, out.text
    return out.json()


def test_frontend_and_uploads_work_with_pod_offline(gw, tmp_path):
    client, main = gw
    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert health["gateway"] is True and health["gpu_available"] is False
    assert health["pod"]["phase"] == "offline"

    out = _upload(client, "my clip.mp4", b"0123456789")
    assert out["status"] == "success" and out["filename"] == "my_clip.mp4"
    assert (tmp_path / "input" / "my_clip.mp4").read_bytes() == b"0123456789"
    vids = client.get("/api/videos").json()
    assert [v["name"] for v in vids["input_videos"]] == ["my_clip.mp4"]
    # The preview streams from the VPS copy -- no pod needed.
    assert client.get("/media/my_clip.mp4").content == b"0123456789"


def test_run_queues_and_status_explains_waiting(gw):
    client, main = gw
    _upload(client, "a.mp4", b"xx")
    main.PODS.state.wanted = True  # the user pressed Start GPU; it is still coming up
    res = client.post("/api/process", json={"video_name": "a.mp4", "target_jersey": 7})
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]
    st = client.get(f"/api/status/{job_id}").json()
    assert st["status"] == "queued" and st["logs"]
    assert "payload" not in st
    # Same video twice while queued -> refused, like the pod's own in-flight guard.
    assert client.post("/api/process", json={"video_name": "a.mp4"}).status_code == 409
    # A file nobody uploaded, with no pod to ask -> immediate 404, not a job that fails later.
    assert client.post("/api/process", json={"video_name": "ghost.mp4"}).status_code == 404
    assert client.get("/api/jobs").json()["active"][0]["job_id"] == job_id


def test_gpu_only_endpoints_say_why_when_pod_is_down(gw):
    client, main = gw
    main.PODS.state = pm.PodState(phase=pm.WAITING_FOR_GPU, message="No GPU available on RunPod right now")
    res = client.get("/api/frame_players?video=a.mp4&t=1")
    assert res.status_code == 503
    assert "No GPU free" in res.json()["detail"]
    assert client.get("/api/results/some_slug").status_code == 503
    assert client.get("/media/output/x/y.mp4").status_code == 503
    # /api/health stays public and cheap, and carries the phase for the badge.
    assert client.get("/api/health").json()["pod"]["phase"] == "waiting_for_gpu"


def test_login_gate_applies_to_the_gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("PV_GATEWAY_DATA", str(tmp_path))
    monkeypatch.setenv("PV_GATEWAY_NO_THREADS", "1")
    monkeypatch.setenv("PV_ACCESS_PASSWORD", "s3cret")
    import importlib

    import apps.gateway.main as main

    main = importlib.reload(main)
    with TestClient(main.app) as client:
        assert client.get("/api/videos").status_code == 401
        assert client.get("/api/health").status_code == 200
        assert client.post("/api/login", json={"password": "nope"}).status_code == 401
        assert client.post("/api/login", json={"password": "s3cret"}).status_code == 200
        assert client.get("/api/videos").status_code == 200


# ---------------------------------------------------------------- the worker, with a fake pod


class FakePodClient:
    """Stands in for `apps/gateway/pod_client.PodClient`: remembers what was uploaded and
    walks a scripted list of pod-side status snapshots."""

    def __init__(self, statuses, has_video=False):
        self.statuses = list(statuses)
        self.has_video = has_video
        self.uploaded: list[str] = []
        self.process_payloads: list[dict] = []

    def has_input_video(self, name, size):
        return self.has_video

    def upload_video(self, path, upload_id, on_progress=None):
        self.uploaded.append(path.name)
        if on_progress:
            on_progress(1.0)
        return {"status": "success", "filename": path.name}

    def start_process(self, payload):
        self.process_payloads.append(payload)
        return {"job_id": "podjob1", "status": "queued"}

    def job_status(self, pod_job_id):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def cancel_job(self, pod_job_id):
        self.cancelled = getattr(self, "cancelled", []) + [pod_job_id]


class FakePods:
    """The PodManager surface the worker uses. `script` is the phases the user's Start walks
    through, one per poll, before the GPU is online; `wanted=False` = nobody pressed Start."""

    def __init__(self, script=(), wanted=True):
        self._script = list(script)
        self._state = pm.PodState(phase=pm.OFFLINE, wanted=wanted)
        self.touched = 0
        self.start_calls = 0

    @property
    def state(self):
        if self._state.wanted and self._state.phase != pm.ONLINE:
            if self._script:
                phase, msg = self._script.pop(0)
                self._state = pm.PodState(phase=phase, message=msg, wanted=True)
            else:
                self._state = pm.PodState(phase=pm.ONLINE, proxy_url="https://pod", gpu_name="RTX 4090",
                                          wanted=True, billing=True)
        return self._state

    @property
    def starting(self):
        return False

    @property
    def online(self):
        return self._state.phase == pm.ONLINE

    def touch(self):
        self.touched += 1

    def refresh(self):
        return self._state

    def ensure_online(self, *a, **k):  # pragma: no cover - the worker must never call this
        raise AssertionError("the job worker must never start the GPU itself")

    def start(self, *a, **k):  # pragma: no cover
        raise AssertionError("the job worker must never start the GPU itself")


def _worker(tmp_path, pod, pods):
    store = JobStore(tmp_path / "jobs.json")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.mp4").write_bytes(b"video")
    worker = JobWorker(store, pods, pod, tmp_path / "input", sleep=lambda s: None, clock=lambda: 0.0)
    return store, worker


def test_worker_waits_for_gpu_then_uploads_then_mirrors_pod_progress(tmp_path):
    pod = FakePodClient([
        {"status": "running", "stage": "Detection", "progress": 20, "logs": ["detecting"]},
        {"status": "running", "stage": "Tracking", "progress": 60, "logs": ["detecting", "tracking"]},
        {"status": "completed", "stage": "Finished", "progress": 100, "slug": "a", "logs": ["detecting", "tracking", "done"]},
    ])
    pods = FakePods([(pm.WAITING_FOR_GPU, "No GPU of this tier free on RunPod right now"),
                     (pm.STARTING, "GPU pod created"), (pm.BOOTING, "app starting")])
    store, worker = _worker(tmp_path, pod, pods)
    job = store.create("a.mp4", {"video_name": "a.mp4", "target_jersey": 9})

    worker.run_job(job)

    final = store.get(job["job_id"])
    assert final["status"] == "completed" and final["slug"] == "a" and final["progress"] == 100
    assert pod.uploaded == ["a.mp4"]
    assert pod.process_payloads == [{"video_name": "a.mp4", "target_jersey": 9}]
    logs = final["logs"]
    assert any("No GPU of this tier free" in line for line in logs)
    assert pods.touched  # job activity keeps the idle auto-stop away
    assert logs[-3:] == ["[pod] detecting", "[pod] tracking", "[pod] done"]
    # Persisted: a fresh store from the same file sees the finished job.
    assert JobStore(tmp_path / "jobs.json").get(job["job_id"])["status"] == "completed"


def test_worker_skips_upload_when_pod_volume_already_has_the_file(tmp_path):
    pod = FakePodClient([{"status": "completed", "stage": "Finished", "progress": 100, "slug": "a", "logs": []}], has_video=True)
    store, worker = _worker(tmp_path, pod, FakePods())
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    worker.run_job(job)
    assert pod.uploaded == []
    assert store.get(job["job_id"])["status"] == "completed"


def test_worker_marks_failed_when_pod_pipeline_fails(tmp_path):
    pod = FakePodClient([{"status": "failed", "stage": "Detection", "progress": 10, "error": "boom", "logs": ["x"]}])
    store, worker = _worker(tmp_path, pod, FakePods())
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    worker.run_job(job)
    final = store.get(job["job_id"])
    assert final["status"] == "failed" and final["error"] == "boom"


def test_worker_reattaches_to_a_running_pod_job_after_restart(tmp_path):
    pod = FakePodClient([{"status": "completed", "stage": "Finished", "progress": 100, "slug": "a", "logs": []}])
    store, worker = _worker(tmp_path, pod, FakePods())
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    store.update(job["job_id"], status=jobs_mod.RUNNING, pod_job_id="podjob1")
    worker.run_job(store.get(job["job_id"]))
    assert pod.uploaded == [] and pod.process_payloads == []
    assert store.get(job["job_id"])["status"] == "completed"


def test_worker_fails_job_when_pod_vanishes_mid_run(tmp_path):
    class Vanishing(FakePodClient):
        def job_status(self, pod_job_id):
            raise requests.ConnectionError("gone")

    now = {"t": 0.0}

    def clock():
        now["t"] += 120.0
        return now["t"]

    pod = Vanishing([])
    store = JobStore(tmp_path / "jobs.json")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.mp4").write_bytes(b"v")
    worker = JobWorker(store, FakePods(), pod, tmp_path / "input", sleep=lambda s: None, clock=clock)
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    with pytest.raises(RuntimeError, match="stopped answering"):
        worker.run_job(job)


# ---------------------------------------------------------------------------
# GPU tiers with live prices (2026-09-19)
# ---------------------------------------------------------------------------


def test_gpus_endpoint_lists_tiers_with_live_prices_and_stock(gw, monkeypatch):
    client, main = gw
    def offer(name, gb, price, stock):
        return {"display_name": name, "memory_gb": gb, "price_per_hr": price, "stock": stock}

    offers = {
        "NVIDIA GeForce RTX 4090": offer("RTX 4090", 24, 0.74, "High"),
        "NVIDIA RTX 4000 Ada Generation": offer("RTX 4000 Ada", 20, 0.28, "Low"),
        "NVIDIA A100-SXM4-80GB": offer("A100 SXM", 80, 1.59, None),
    }
    calls = []

    def fake_fetch(api_key, datacenter, **kw):
        calls.append((api_key, datacenter))
        return offers

    monkeypatch.setattr(main.rp, "fetch_gpu_offers", fake_fetch)
    main._GPU_OFFERS.update(at=0.0, offers={}, error="")
    main.PODS.state.gpu_type = "NVIDIA GeForce RTX 4090"

    data = client.get("/api/gpus").json()
    assert data["default"] == "standard"
    assert calls == [("k", main.PODS.cfg.datacenter)]
    by_id = {t["id"]: t for t in data["tiers"]}
    assert list(by_id) == ["budget", "standard", "pro"]
    assert by_id["standard"]["primary"]["price_per_hr"] == 0.74
    assert by_id["standard"]["is_current"] is True and by_id["budget"]["is_current"] is False
    assert by_id["budget"]["primary"]["display_name"] == "RTX 4000 Ada"
    # the A100 is not stocked in this datacenter -> tier says so, still listed with its price
    assert by_id["pro"]["in_stock"] is False and by_id["pro"]["primary"]["price_per_hr"] == 1.59
    assert data["prices_error"] is None

    # cached: a second call does not hit RunPod again
    client.get("/api/gpus")
    assert len(calls) == 1


def test_gpus_endpoint_survives_a_failed_price_lookup(gw, monkeypatch):
    client, main = gw

    def broken(*a, **k):
        raise main.rp.RunPodError("boom")

    monkeypatch.setattr(main.rp, "fetch_gpu_offers", broken)
    main._GPU_OFFERS.update(at=0.0, offers={}, error="")
    data = client.get("/api/gpus").json()
    assert data["prices_error"] == "boom"
    assert [t["id"] for t in data["tiers"]] == ["budget", "standard", "pro"]
    assert data["tiers"][1]["primary"]["price_per_hr"] is None  # unknown, never a made-up number


def test_process_rejects_an_unknown_gpu_tier_and_keeps_a_known_one(gw):
    client, main = gw
    _upload(client, "g.mp4", b"xx")
    main.PODS.state.wanted = True
    bad = client.post("/api/process", json={"video_name": "g.mp4", "gpu_tier": "quantum"})
    assert bad.status_code == 400
    ok = client.post(
        "/api/process", json={"video_name": "g.mp4", "gpu_tier": "budget", "target_jersey": 7}
    )
    assert ok.status_code == 200, ok.text
    job = main.JOBS.get(ok.json()["job_id"])
    assert job["gpu_tier"] == "budget"
    assert "gpu_tier" not in job["payload"] and job["payload"]["target_jersey"] == 7


DONE = {"status": "completed", "stage": "Finished", "progress": 100, "slug": "a", "logs": []}


def test_worker_forwards_a_clean_payload_and_keeps_the_tier_for_display(tmp_path):
    pod = FakePodClient([DONE], has_video=True)
    store, worker = _worker(tmp_path, pod, FakePods())
    job = store.create("a.mp4", {"video_name": "a.mp4", "gpu_tier": "budget", "target_jersey": 9})
    worker.run_job(job)
    assert pod.process_payloads[-1] == {"video_name": "a.mp4", "target_jersey": 9}
    final = store.get(job["job_id"])
    assert final["status"] == "completed" and final["gpu_tier"] == "budget"


# ---------------------------------------------------------------------------
# User-controlled GPU (2026-09-26)
# ---------------------------------------------------------------------------


def test_worker_waits_for_the_users_start_and_never_starts_the_gpu(tmp_path):
    """The Trim_.mp4 case: a queued job must wait -- for free -- rather than create pods."""
    pod = FakePodClient([DONE], has_video=True)
    pods = FakePods(wanted=False)
    store = JobStore(tmp_path / "jobs.json")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.mp4").write_bytes(b"v")
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    polls = {"n": 0}

    def sleep(_s):
        polls["n"] += 1
        if polls["n"] == 3:
            st = store.get(job["job_id"])
            assert st["status"] == jobs_mod.WAITING_GPU and "Start GPU" in st["stage"]
            store.cancel(job["job_id"], "cancelled by you")

    worker = JobWorker(store, pods, pod, tmp_path / "input", sleep=sleep, clock=lambda: 0.0)
    with pytest.raises(jobs_mod.JobCancelled):
        worker.run_job(job)
    final = store.get(job["job_id"])
    assert final["status"] == "failed" and final["stage"] == "Cancelled"
    assert pod.process_payloads == []


def test_cancel_is_final_even_if_the_worker_writes_after_it(tmp_path):
    store = JobStore(tmp_path / "jobs.json")
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    assert store.cancel(job["job_id"], "cancelled by you")["status"] == "failed"
    store.update(job["job_id"], status=jobs_mod.RUNNING, stage="Detection", progress=40, log="late line")
    st = store.get(job["job_id"])
    assert st["status"] == "failed" and st["stage"] == "Cancelled" and st["progress"] == 0
    assert store.cancel(job["job_id"], "again") is None  # not active any more
    assert store.next_queued() is None


def test_run_is_refused_while_the_gpu_is_off(gw):
    client, main = gw
    _upload(client, "b.mp4", b"xx")
    res = client.post("/api/process", json={"video_name": "b.mp4"})
    assert res.status_code == 409 and "Start GPU" in res.json()["detail"]
    assert client.get("/api/jobs").json()["active"] == []


def test_gpu_endpoints_never_wake_the_pod(gw, monkeypatch):
    client, main = gw
    monkeypatch.setattr(main.PODS, "start", lambda *a, **k: pytest.fail("a view started the GPU"))
    monkeypatch.setattr(main.PODS, "ensure_online", lambda *a, **k: pytest.fail("a view started the GPU"))
    res = client.get("/api/results/some_slug")
    assert res.status_code == 503 and "press Start GPU" in res.json()["detail"]
    assert client.get("/media/output/x/y.mp4").status_code == 503
    assert client.get("/api/frame_players?video=a.mp4&t=1").status_code == 503
    assert client.post("/api/pod/wake").status_code in (404, 405)  # the old auto-wake is gone


def test_start_and_stop_endpoints(gw, monkeypatch):
    client, main = gw
    started = []
    monkeypatch.setattr(main.PODS, "start", lambda gpu_types=None: started.append(gpu_types) or True)
    assert client.post("/api/pod/start", json={"gpu_tier": "quantum"}).status_code == 400
    res = client.post("/api/pod/start", json={"gpu_tier": "budget"})
    assert res.status_code == 200 and res.json()["status"] == "starting"
    assert started == [main_rp.gpu_tier("budget")["gpu_type_ids"]]

    _upload(client, "c.mp4", b"xx")
    main.PODS.state.wanted = True
    job_id = client.post("/api/process", json={"video_name": "c.mp4"}).json()["job_id"]
    stops = []
    monkeypatch.setattr(main.PODS, "stop", lambda *a, **k: stops.append(a))
    res = client.post("/api/pod/stop")
    assert res.status_code == 200 and res.json()["cancelled_jobs"] == [job_id]
    assert client.get(f"/api/status/{job_id}").json()["stage"] == "Cancelled"
    assert main.PODS.state.wanted is False
    import time as _t

    for _ in range(50):
        if stops:
            break
        _t.sleep(0.02)
    assert stops, "Stop GPU must stop the pod"


def test_cancel_endpoint(gw):
    client, main = gw
    _upload(client, "d.mp4", b"xx")
    main.PODS.state.wanted = True
    job_id = client.post("/api/process", json={"video_name": "d.mp4"}).json()["job_id"]
    res = client.post(f"/api/jobs/{job_id}/cancel")
    assert res.status_code == 200 and res.json()["status"] == "failed"
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_pod_state_carries_balance_and_idle_settings(gw, monkeypatch):
    client, main = gw
    monkeypatch.setattr(main.rp, "fetch_account", lambda key: {"balance": 12.5, "spend_per_hr": 0.004})
    main._ACCOUNT.update(at=0.0, data=None, error="")
    data = client.get("/api/pod").json()
    assert data["account"] == {"balance": 12.5, "spend_per_hr": 0.004}
    assert data["idle_stop_minutes"] == 10 and data["starting"] is False
    assert data["pod"]["wanted"] is False and data["pod"]["billing"] is False


def test_code_bundle_is_the_deployed_checkout(gw):
    """What a booting pod is handed (2026-09-30: pushed through RunPod's proxy, because a pod
    cannot reach the VPS): a tar.gz of the committed checkout, bootstrap included."""
    import tarfile

    client, main = gw
    bundle, commit = main.code_bundle()
    assert bundle[:2] == b"\x1f\x8b" and len(commit) == 12
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
        names = set(tf.getnames())
    assert {"docker/runpod_bootstrap.sh", "start_app.py", "apps/api/main.py"} <= names
    assert main.code_bundle() == (bundle, commit)  # cached per commit


def test_pods_are_told_to_wait_for_the_code_not_to_fetch_it(gw):
    client, main = gw
    env = main.PODS.cfg.pod_env
    assert env["PV_POD_KEY"] == main.pod_bootstrap_key()
    assert env["PV_POD_RECEIVER"] == main_rp.pod_receiver_env()
    assert "PV_GATEWAY_URL" not in env and "GITHUB_TOKEN" not in env
    # the old pull endpoints are gone: nothing on the gateway serves code any more
    assert client.get("/pod/bootstrap.sh", headers={"X-PV-Pod-Key": env["PV_POD_KEY"]}).status_code == 404
    assert client.get("/pod/code.tar.gz", headers={"X-PV-Pod-Key": env["PV_POD_KEY"]}).status_code == 404


def test_pod_create_body_waits_for_the_pushed_code():
    env = {"PV_POD_RECEIVER": main_rp.pod_receiver_env(), "PV_POD_KEY": "k"}
    body = main_rp.pod_create_body(name="p", gpu_type_ids=["x"], volume_id="v", env=env)
    cmd = body["dockerStartCmd"][-1]
    assert "PV_POD_RECEIVER" in cmd and "curl -fsSL" not in cmd and "GITHUB_TOKEN" not in cmd
    # missing code / a failed bootstrap / a failed app stops the pod itself, never a billing loop
    assert "|| {" in cmd and "/pods/$RUNPOD_POD_ID/stop" in cmd and "| bash" not in cmd
    legacy = main_rp.pod_create_body(name="p", gpu_type_ids=["x"], volume_id="v", env={})
    assert "raw.githubusercontent.com" in legacy["dockerStartCmd"][-1]


def test_cancelling_a_running_job_stops_the_pipeline_on_the_pod(tmp_path):
    """Cancel job / Stop GPU while the pod is processing: the pod is told to stop the pipeline,
    so it does not keep the GPU busy for a result nobody wants."""
    running = {"status": "running", "stage": "Detection", "progress": 20, "logs": ["detecting"]}
    pod = FakePodClient([running], has_video=True)
    store = JobStore(tmp_path / "jobs.json")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "a.mp4").write_bytes(b"v")
    job = store.create("a.mp4", {"video_name": "a.mp4"})
    polls = {"n": 0}

    def sleep(_s):
        polls["n"] += 1
        if polls["n"] == 3:  # a few polls into the run, the user presses Cancel job
            store.cancel(job["job_id"], "cancelled by you")

    worker = JobWorker(store, FakePods(), pod, tmp_path / "input", sleep=sleep, clock=lambda: 0.0)
    with pytest.raises(jobs_mod.JobCancelled):
        worker.run_job(job)
    assert pod.cancelled == ["podjob1"]
    final = store.get(job["job_id"])
    assert final["stage"] == "Cancelled"
    assert final["logs"][-1] == "Stopped the pipeline on the GPU pod."
