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
