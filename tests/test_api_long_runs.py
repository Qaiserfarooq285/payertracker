"""Long-video behaviours of the pod API (`apps/api/main.py`, 2026-09-26): the job log carries the
pipeline's real progress lines (a 10-minute clip runs for well over half an hour, and the log used
to freeze on one line), and a run is refused up front when the volume is nearly full instead of
dying mid-render."""

from __future__ import annotations

import logging
import threading

import pytest
from fastapi import HTTPException

import apps.api.main as api_main


def test_job_log_gets_pipeline_lines_from_its_own_thread_only():
    job = {"logs": []}
    handler = api_main._attach_job_log(job)
    root = logging.getLogger()
    try:
        logging.getLogger("src.detect.run").info("detected %d frames", 1500)
        logging.getLogger("uvicorn.access").info("GET /api/status")  # not the pipeline
        logging.getLogger("src.detect.run").debug("too chatty")  # below INFO

        def other_job():
            logging.getLogger("src.track.run").info("another job's line")

        t = threading.Thread(target=other_job)
        t.start()
        t.join()
    finally:
        root.removeHandler(handler)
    assert len(job["logs"]) == 1 and job["logs"][0].endswith("detected 1500 frames")


def test_job_log_keeps_head_and_tail_when_long():
    job = {"logs": []}
    handler = api_main._attach_job_log(job)
    log = logging.getLogger("src.pipeline.run")
    try:
        for i in range(api_main.JOB_LOG_MAX_LINES + 50):
            log.info("line %d", i)
    finally:
        logging.getLogger().removeHandler(handler)
    assert len(job["logs"]) == api_main.JOB_LOG_MAX_LINES
    assert job["logs"][0].endswith("line 0")
    assert job["logs"][-1].endswith(f"line {api_main.JOB_LOG_MAX_LINES + 49}")


def test_process_refuses_when_the_volume_is_nearly_full(tmp_path, monkeypatch):
    video = tmp_path / "long.mp4"
    video.write_bytes(b"v")
    monkeypatch.setattr(api_main, "_resolve_video_path", lambda name: video)
    monkeypatch.setattr(api_main, "_disk_free_gb", lambda path: 1.2)
    with pytest.raises(HTTPException) as err:
        api_main.process_video(api_main.ProcessRequest(video_name="long.mp4"), background_tasks=None)
    assert err.value.status_code == 507 and "1.2 GB free" in err.value.detail
    assert not api_main._INFLIGHT_SLUGS


def test_cancel_stops_a_running_pipeline_thread(tmp_path, monkeypatch):
    """Owner, 2026-09-26: a started job that is cancelled must actually stop processing on the
    GPU -- including through code that catches `Exception` and carries on."""
    import time

    video = tmp_path / "long.mp4"
    video.write_bytes(b"v")
    monkeypatch.setattr(api_main, "_resolve_video_path", lambda name: video)
    monkeypatch.setattr(api_main.time, "sleep", lambda s: None)  # skip the staged-message pauses
    progress = {"frames": 0, "swallowed": 0}

    def endless_pipeline(*args, **kwargs):
        while True:  # a pipeline stage that would run for hours
            try:
                progress["frames"] += 1
                sum(range(200))
            except Exception:  # the pipeline's own broad handlers must not eat the cancel
                progress["swallowed"] += 1

    # Stand-ins for the heavy pipeline modules (the real ones import torch).
    import sys
    import types

    fakes = {
        "src.pipeline.run": dict(
            run_pipeline_for_video=endless_pipeline,
            _load_all_configs=lambda: {"identity": {}, "run": {"filename_convention_regex": r"(\d+)$"}},
            normalize_manual_overrides=lambda o: o,
            parse_track_id_overrides=lambda raw: {},
        ),
        "src.pipeline.extended_output": dict(run_extended_pipeline_for_video=endless_pipeline),
        "src.pipeline.manual_events": dict(run_manual_events_pipeline_for_video=endless_pipeline),
        "src.ingest.discovery": dict(parse_filename=lambda stem, pattern: (stem, None)),
    }
    for name, attrs in fakes.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    job_id = "cancel01"
    api_main.JOBS[job_id] = {"job_id": job_id, "status": "queued", "logs": [], "canonical_slug": "long"}
    api_main._INFLIGHT_SLUGS.add("long")
    worker = threading.Thread(target=api_main._run_pipeline_job, args=(job_id, "long.mp4", 7, None, None))
    worker.start()
    for _ in range(200):
        if progress["frames"] > 1000:
            break
        time.sleep(0.01)
    assert api_main.JOBS[job_id]["status"] == "processing"

    api_main.cancel_job(job_id)
    worker.join(5)
    try:
        assert not worker.is_alive(), "the pipeline kept running after cancel"
        job = api_main.JOBS[job_id]
        assert job["status"] == "failed" and job["stage"] == "Cancelled"
        assert progress["swallowed"] == 0
        assert "long" not in api_main._INFLIGHT_SLUGS  # the video can be run again
        assert job_id not in api_main._JOB_THREADS
    finally:
        api_main.JOBS.pop(job_id, None)


def test_cancel_before_the_pipeline_starts_means_it_never_starts(tmp_path, monkeypatch):
    job_id = "cancel02"
    api_main.JOBS[job_id] = {"job_id": job_id, "status": "queued", "logs": [], "canonical_slug": "x"}
    api_main._INFLIGHT_SLUGS.add("x")
    try:
        api_main.cancel_job(job_id)
        api_main._run_pipeline_job(job_id, "x.mp4", 7, None, None)
        job = api_main.JOBS[job_id]
        assert job["status"] == "failed" and job["stage"] == "Cancelled"
        assert "x" not in api_main._INFLIGHT_SLUGS
    finally:
        api_main.JOBS.pop(job_id, None)
