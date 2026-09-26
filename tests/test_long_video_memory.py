"""Long videos must not need the whole video in RAM (2026-09-26). Two stages used to keep every
native-resolution frame of a take -- and a single-camera match is often ONE take: the kit-colour
read (`src/track/kit_wiring.build_take_kit_colour`, 2 fps) and the scoreboard OCR
(`src/events/goals.detect_goals_for_take`, 1 fps). Both now stream; these tests pin that the
results are unchanged and that memory stays bounded."""

from __future__ import annotations

import random

import numpy as np
import pytest

import src.events.goals as goals_mod
import src.track.kit_wiring as kw
from src.common.types import BBox, Take, Track, TrackBox

FPS = 2.0


def _frames(n: int, t0: float = 0.0):
    """A decode stream whose frame i carries i in pixel [0, 0, 0] (and 1000+i in [0, 0, 1])."""
    for i in range(n):
        frame = np.zeros((4, 4, 3), dtype=np.int32)
        frame[0, 0, 0] = i
        yield i, t0 + i / FPS, frame


def _reference_nearest(frames, target_t, tol):
    """The pre-streaming algorithm: nearest over ALL decoded frames, first one on a tie."""
    best = min(frames, key=lambda f: abs(f[0] - target_t))
    return best if abs(best[0] - target_t) <= tol else None


@pytest.mark.parametrize("seed", range(5))
def test_kit_colour_streaming_reads_the_same_frames_as_before(monkeypatch, seed):
    rng = random.Random(seed)
    n_frames = 120
    tracks = []
    for tid in range(8):
        # box times deliberately OFF the 2 fps grid (tracker runs at 25 fps), unsorted gaps,
        # and some past the last decoded frame
        ts = sorted({round(rng.uniform(0, n_frames / FPS + 1.5), 2) for _ in range(rng.randint(3, 30))})
        boxes = [TrackBox(frame_index=k, t=t, bbox=BBox(x1=1, y1=1, x2=3, y2=3), conf=0.9) for k, t in enumerate(ts)]
        tracks.append(Track(id=tid, take_id=0, boxes=boxes))
    take = Take(id=0, t_start=0.0, t_end=n_frames / FPS, frame_start=0, frame_end=n_frames)


    def fake_decode(*a, **k):
        return _frames(n_frames)

    reads = []

    def fake_lab(frame, bbox, crop_cfg, kit_cfg):
        reads.append(int(frame[0, 0, 0]))
        return (np.array([float(frame[0, 0, 0]), 0.0, 0.0]), None)

    def fake_bands(frame, bbox, take_id, t, cfg):
        return None

    monkeypatch.setattr(kw, "decode_frames", fake_decode)
    monkeypatch.setattr(kw, "kit_lab_sample", fake_lab)
    monkeypatch.setattr(kw, "sample_kit_bands", fake_bands)
    monkeypatch.setattr(kw, "_scale_bbox_to_native", lambda b, sx, sy, pad, w, h: (1, 1, 3, 3))
    identity_cfg = {"crop": {"identity_fps_sample": FPS}, "parseq_soccernet": {"number_crop": {}}}
    max_samples = 12

    lab_by_track, _ = kw.build_take_kit_colour(
        "v.mp4", take, tracks, 1.0, 1.0, 4, 4, identity_cfg, {}, {}, max_samples
    )

    # reference: every sampled read resolved against the full frame list, the old way
    all_frames = [(t, i) for i, t, _f in _frames(n_frames)]
    tol = 1.0 / FPS
    expected_by_track = {}
    for tr in tracks:
        for t in kw._sample_timestamps(tr, max_samples):
            hit = _reference_nearest(all_frames, t, tol)
            if hit is not None:
                expected_by_track.setdefault(tr.id, []).append(hit[1])
    expected = {tid: np.median(np.array([[float(i), 0.0, 0.0] for i in idx]), axis=0)
                for tid, idx in expected_by_track.items()}
    assert set(lab_by_track) == set(expected)
    for tid in expected:
        assert np.allclose(lab_by_track[tid], expected[tid])
    assert sorted(reads) == sorted(i for idx in expected_by_track.values() for i in idx)


def test_kit_colour_holds_at_most_two_frames(monkeypatch):
    """The whole point: memory no longer grows with the take's length."""
    alive = {"n": 0, "max": 0}

    class Frame(np.ndarray):
        def __del__(self):
            alive["n"] -= 1

    def counting_decode(*a, **k):
        for i in range(2000):  # a long take
            f = np.zeros((2, 2, 3)).view(Frame)
            alive["n"] += 1
            alive["max"] = max(alive["max"], alive["n"])
            yield i, i / FPS, f

    boxes = [TrackBox(frame_index=k, t=k * 5.0, bbox=BBox(x1=0, y1=0, x2=1, y2=1), conf=0.9) for k in range(200)]
    take = Take(id=0, t_start=0.0, t_end=1000.0, frame_start=0, frame_end=2000)
    monkeypatch.setattr(kw, "decode_frames", counting_decode)
    monkeypatch.setattr(kw, "kit_lab_sample", lambda *a: None)
    monkeypatch.setattr(kw, "sample_kit_bands", lambda *a: None)
    monkeypatch.setattr(kw, "_scale_bbox_to_native", lambda *a: (0, 0, 1, 1))
    identity_cfg = {"crop": {"identity_fps_sample": FPS}, "parseq_soccernet": {"number_crop": {}}}
    kw.build_take_kit_colour("v.mp4", take, [Track(id=1, take_id=0, boxes=boxes)], 1, 1, 2, 2,
                             identity_cfg, {}, {}, 50)
    assert alive["max"] <= 3


# ---------------------------------------------------------------- scoreboard OCR


def _goal_cfg(**over):
    cfg = {"activation_frames_count": 4, "increment_debounce_samples": 2, "occurrence_confidence": 0.75}
    cfg.update(over)
    return cfg


def test_no_scoreboard_stops_decoding_after_the_activation_frames(monkeypatch):
    consumed = {"n": 0}

    def stream():
        for i in range(10_000):  # a 3-hour take at 1 fps
            consumed["n"] += 1
            yield float(i), np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(goals_mod, "scan_candidate_regions", lambda r, frames, cfg: (None, {"activated_region": None}))
    events, debug = goals_mod.detect_goals_scoreboard_delta_stream(object(), stream(), 0, _goal_cfg())
    assert events == [] and consumed["n"] == 4


def test_streamed_scoreboard_matches_the_list_version(monkeypatch):
    region = (0.0, 0.0, 0.5, 0.5)
    monkeypatch.setattr(goals_mod, "scan_candidate_regions", lambda r, frames, cfg: (region, {"activated_region": list(region)}))
    script = ["0 - 0", "0 - 0", "0 - 0", "1 - 0", "1 - 0", "1 - 0", "1 - 0", "1 - 1", "1 - 1", "1 - 1"]
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(len(script))]
    times = [float(i * 10) for i in range(len(script))]
    monkeypatch.setattr(goals_mod, "_ocr_region_text", lambda reader, crop: script[int(crop[0, 0, 0])])

    listed, dbg_a = goals_mod.detect_goals_scoreboard_delta(object(), frames, times, 2, _goal_cfg())
    streamed, dbg_b = goals_mod.detect_goals_scoreboard_delta_stream(object(), zip(times, frames), 2, _goal_cfg())
    assert len(listed) == len(streamed) == 2
    assert [(e.t_start, e.t_end, e.evidence) for e in listed] == [(e.t_start, e.t_end, e.evidence) for e in streamed]
    assert dbg_a["score_samples"] == dbg_b["score_samples"]
