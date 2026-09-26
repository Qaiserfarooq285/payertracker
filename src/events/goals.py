"""Stage 4 — REAL goal + assist detection (ADR-17, CLAUDE.md §5.1/§3.2 consequence 3, Golden
Rule 5).

Supersedes the earlier dead-code stub: `detect_goals_scoreboard_delta` used to raise
`NotImplementedError` unconditionally, and `check_goal_availability` used to return a hardcoded
"not available" regardless of input. Both are now real, and — measured, not assumed — still
correctly report "not available" on all 5 original `clip<N> <jersey>.mp4` clips plus
`jordan_thomas_highlight_video`, because none of them has a scoreboard graphic anywhere
(CLAUDE.md §3.2(3)). ADR-17's own design, in four steps:

1. **Occurrence** (`detect_goals_scoreboard_delta`) — scan a SMALL set of candidate scoreboard
   corner regions (`configs/events.yaml: goal.candidate_regions`) across several early sampled
   frames of a take with the project's existing EasyOCR engine. A region only "activates" if it
   yields the SAME stable `digit [separator] digit` reading across multiple frames (a MEASURED
   self-check — `scan_candidate_regions` — never a hardcoded ROI guess, and never a trust in
   `RunProfile.profile`; ADR-11's own lesson in this codebase applies identically here: read the
   measured signal, not the label). Once activated, the region is watched across the whole take
   for a DEBOUNCED score increment (`find_score_increments`) — a reading must climb by exactly one
   and then STAY there for a couple of subsequent samples before it counts as a real goal, so a
   single OCR misread can never fire a phantom one.
2. **Scoring team** (`attribute_goal_team_and_scorer`) — deliberately sidesteps mapping the
   scoreboard's own left/right digit to home/away. Instead, whichever of OUR OWN two
   colour-clustered teams (`src/events/possession.py`, ADR-12) held the ball for the most
   cumulative time in the `possession_lookback_s` window immediately before the digit changed is
   credited.
3. **Scorer** (same function) — the identity with the LAST possession run in that window, gated to
   the credited team.
4. **Assist** (`find_assist_event`) — the owner's own rule, verbatim: "if the target pass the ball
   and the other score goal should be assist". The nearest PRECEDING `PASS` event whose
   `evidence['receiver_identity']` equals the credited scorer, from a teammate, within
   `assist_window_seconds`, credited to the PASSER.

Confidence stacks DOWN at each step (`goal_cfg['occurrence_confidence']` is the ceiling; each
attribution step only ever multiplies it further down) and every emitted `Event.evidence` carries
enough to audit it — which region/frames the OCR locked onto and its raw text, the possession
scores considered, and which `Event.id` supported an assist (Golden Rule 5: no black-box
confidence).

No visual ball-crosses-the-goal-line detection is built here (no goal-post/net detector exists,
and ADR-17 explicitly scoped that out as deferred, not part of this task).

`detect_goals_for_video` is the shared entrypoint both `src.pipeline.run` (the 5 original clips)
and `src.pipeline.extended_output` (ADR-15's filename-less flow) call identically.
"""

from __future__ import annotations

import itertools
import math
import re
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.common.logging import get_logger
from src.common.types import BallDetection, Event, EventType, Take, Track
from src.common.video import decode_frames, probe
from src.events.ball_track import BallState
from src.events.goal_line import ball_crosses_goal_line, goal_line_from_posts
from src.events.possession import (
    detect_passes,
    detect_possession,
    possession_runs_for_take,
    teammates_gate,
)
from src.events.segments import find_threshold_segments
from src.events.shots import ball_speed_series
from src.goal.detect import GoalStructure, goal_bbox_at, goal_posts_at, take_motion_score
from src.identity.jersey_ocr import free_easyocr_reader, load_easyocr_reader

logger = get_logger(__name__)


class GoalDetectionResult(BaseModel):
    """Outcome of Stage 4's goal-detection attempt for one video (Golden Rule 5: explicit, not
    fabricated). `reason` is always human-readable and, when `available` is `False`, always
    literally starts with the string ``"not available"`` — the run report surfaces it verbatim.
    """

    available: bool
    reason: str
    events: list[Event] = []


# ---------------------------------------------------------------------------
# OCR text -> (left, right) parsing (pure)
# ---------------------------------------------------------------------------

_SCORE_PATTERN = re.compile(r"(\d{1,2})\s*[-:]\s*(\d{1,2})")


def parse_scoreboard_text(joined_text: str | None) -> tuple[int, int] | None:
    """Extract a plausible `digit [separator] digit` scoreboard reading (e.g. ``"1 - 0"``,
    ``"2:1"``) from a joined EasyOCR text string. `None` when no such pattern is present — the
    conservative, no-fabrication default (Golden Rule 5); EasyOCR frequently returns several
    fragments per crop (e.g. ``"1"``, ``"-"``, ``"0"``) or none at all.
    """
    if not joined_text:
        return None
    match = _SCORE_PATTERN.search(joined_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _bbox_x1(bbox: Any) -> float:
    """Best-effort left-x of an EasyOCR result box — real EasyOCR boxes are 4 `[x, y]` points, so
    ``bbox[0][0]`` is the top-left corner's x. Defensive against a test double's dummy bbox shape
    (e.g. the `(0, 0, 0, 0)` tuple `tests/test_identity.py` already uses for the sibling
    `jersey_ocr` module) — falls back to `0.0` rather than crashing on an unexpected shape.
    """
    try:
        return float(bbox[0][0])
    except (TypeError, IndexError):
        return 0.0


def _crop_region(
    frame: np.ndarray, region_relative: tuple[float, float, float, float]
) -> np.ndarray:
    """Crop `region_relative` (`[x1, y1, x2, y2]` as a fraction of frame width/height) out of
    `frame`."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region_relative
    return frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def _ocr_region_text(reader: Any, crop: np.ndarray) -> str:
    """Run EasyOCR over `crop` with NO digit allowlist (unlike `src/identity/jersey_ocr.py`'s
    jersey-number OCR — a scoreboard reading needs the separator character too) and join every
    detected text fragment left-to-right into one string for `parse_scoreboard_text`."""
    if crop.size == 0:
        return ""
    raw = reader.readtext(crop)
    ordered = sorted(raw, key=lambda r: _bbox_x1(r[0]))
    return " ".join(str(text) for _bbox, text, _conf in ordered)


# ---------------------------------------------------------------------------
# step 1a: which (if any) candidate region is a real, stable scoreboard
# ---------------------------------------------------------------------------


def scan_candidate_regions(
    reader: Any, frames: list[np.ndarray], goal_cfg: dict
) -> tuple[tuple[float, float, float, float] | None, dict]:
    """Self-check WHICH (if any) of `goal_cfg['candidate_regions']` is a real scoreboard, from
    MEASURED OCR stability across `frames` — never a hardcoded ROI guess, never a trust in
    `RunProfile.profile` (ADR-11's own lesson: read the measured signal, not the label).

    A region "activates" when the SAME parsed `(left, right)` reading appears in at least
    `goal_cfg['min_stable_frames']` of `frames` — a real scoreboard graphic reads the same score
    frame after frame over a short early window (barring an implausibly fast kickoff goal), so
    stability across several samples is exactly the self-check that separates "this is a
    scoreboard" from "this crop of grass/sky/a sponsor board happened to OCR a plausible-looking
    digit pair once or twice".

    Returns `(activated_region_or_None, debug)` — `debug` always carries every candidate region's
    own reading tally (Golden Rule 5: auditable even when nothing activates), checked in
    `goal_cfg['candidate_regions']`'s own order and returning the FIRST region that activates.
    """
    candidates = goal_cfg["candidate_regions"]
    min_stable = goal_cfg["min_stable_frames"]
    debug: dict = {"candidates": [], "activated_region": None}

    for region in candidates:
        region_t = tuple(float(v) for v in region)
        readings: dict[tuple[int, int], int] = {}
        raw_texts: list[str] = []
        for frame in frames:
            text = _ocr_region_text(reader, _crop_region(frame, region_t))
            raw_texts.append(text)
            parsed = parse_scoreboard_text(text)
            if parsed is not None:
                readings[parsed] = readings.get(parsed, 0) + 1

        best_reading = max(readings.items(), key=lambda kv: kv[1]) if readings else None
        debug["candidates"].append(
            {
                "region": list(region_t),
                "raw_texts": raw_texts,
                "readings": {f"{k[0]}-{k[1]}": v for k, v in readings.items()},
                "best_count": best_reading[1] if best_reading else 0,
            }
        )
        if best_reading is not None and best_reading[1] >= min_stable:
            debug["activated_region"] = list(region_t)
            debug["activated_reading"] = list(best_reading[0])
            return region_t, debug

    return None, debug


# ---------------------------------------------------------------------------
# step 1b: debounced score-increment detection over one take's own low-fps series
# ---------------------------------------------------------------------------


def find_score_increments(samples: list[tuple[float, int]], goal_cfg: dict) -> list[dict]:
    """Walk a time-sorted `(t, total_score)` series (already parsed+summed from an activated
    region's own readings — gaps where OCR produced nothing are simply absent, never zero-filled)
    and return every DEBOUNCED `+1` increment as
    `{t_before, t_after, score_before, score_after}`.

    Debounce (`goal_cfg['increment_debounce_samples']`): a candidate `+1` step must hold at the
    NEW value for that many SUBSEQUENT samples before being trusted — a single-sample spike that
    reverts is OCR noise, not a goal. A jump of more than `+1`, or any decrease, is ambiguous (an
    OCR misread, or — if it turns out to persist — a real change this heuristic cannot tell how
    many goals it represents) and is never turned into a goal event (Golden Rule 5: no guessing);
    if such a reading itself proves debounce-stable, the baseline silently resyncs to it (so a
    later, cleaner `+1` from that new baseline can still be detected) without ever emitting an
    event for the ambiguous jump itself.
    """
    debounce = goal_cfg["increment_debounce_samples"]
    increments: list[dict] = []
    if not samples:
        return increments

    baseline_t, baseline_score = samples[0]
    i = 1
    while i < len(samples):
        t, score = samples[i]
        if score == baseline_score:
            i += 1
            continue

        window = samples[i : i + debounce]
        stable = len(window) == debounce and all(s == score for _, s in window)

        if score == baseline_score + 1 and stable:
            increments.append(
                {
                    "t_before": baseline_t,
                    "t_after": t,
                    "score_before": baseline_score,
                    "score_after": score,
                }
            )
            baseline_t, baseline_score = t, score
            i += debounce
            continue

        if stable:
            # An ambiguous (non-+1) but persistent change -- resync without emitting a goal.
            baseline_t, baseline_score = t, score
            i += debounce
            continue

        # Not stable across the debounce window -- a stray misread, ignore this one sample only.
        i += 1

    return increments


# ---------------------------------------------------------------------------
# step 1: occurrence detector (real implementation -- no longer NotImplementedError)
# ---------------------------------------------------------------------------


def detect_goals_scoreboard_delta(
    reader: Any,
    frames: list[np.ndarray],
    times: list[float],
    take_id: int | None,
    goal_cfg: dict,
) -> tuple[list[Event], dict]:
    """ADR-17 step 1 — the real scoreboard-OCR-delta goal-OCCURRENCE detector.

    `frames`/`times` are a LOW-FPS, time-sorted series spanning one take (`detect_goals_for_take`
    decodes them via `configs/events.yaml: goal.scoreboard_ocr_fps_sample`); this function itself
    is decode-agnostic, so it is independently testable against synthetic frames + a mocked
    EasyOCR `reader` (no GPU/video I/O needed).

    Team/scorer/assist attribution is a SEPARATE, later step (`attribute_goal_team_and_scorer` /
    `find_assist_event`) — every `Event` returned here has `player_track_id=None` and
    `confidence=goal_cfg['occurrence_confidence']` (ADR-17: the highest-confidence step in the
    chain; attribution only ever multiplies it further down), so this function stays a pure,
    auditable "did a goal OCCUR" detector.

    Returns `(occurrence_events, debug)`; `debug` is `scan_candidate_regions`'s own audit trail,
    plus (when a region activated) the raw per-sample score series and the increments found.
    """
    if len(times) != len(frames):
        raise ValueError("times and frames must be the same length")
    return detect_goals_scoreboard_delta_stream(reader, zip(times, frames), take_id, goal_cfg)


def detect_goals_scoreboard_delta_stream(
    reader: Any,
    timed_frames: Iterable[tuple[float, np.ndarray]],
    take_id: int | None,
    goal_cfg: dict,
) -> tuple[list[Event], dict]:
    """`detect_goals_scoreboard_delta` over a STREAM of `(t, frame)` -- same result, bounded
    memory (2026-09-26, long videos). Only the first `activation_frames_count` frames are ever held
    whole (the region scan needs them together); after that each frame is reduced to its
    scoreboard crop and dropped. When no region activates -- no scoreboard, the owner's usual
    footage -- the stream is not consumed any further, so the rest of the take is never decoded.
    Holding every native frame of a single-take 90-min match here was tens of GB of RAM."""
    it = iter(timed_frames)
    head: list[tuple[float, np.ndarray]] = []
    for item in it:
        head.append(item)
        if len(head) >= goal_cfg["activation_frames_count"]:
            break
    activated_region, debug = scan_candidate_regions(reader, [f for _t, f in head], goal_cfg)
    if activated_region is None:
        return [], debug

    samples: list[tuple[float, int]] = []
    for t, frame in itertools.chain(head, it):
        crop = _crop_region(frame, activated_region)
        parsed = parse_scoreboard_text(_ocr_region_text(reader, crop))
        if parsed is not None:
            samples.append((t, parsed[0] + parsed[1]))

    increments = find_score_increments(samples, goal_cfg)
    debug["score_samples"] = samples
    debug["increments"] = increments

    events: list[Event] = []
    for inc in increments:
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.GOAL,
                t_start=inc["t_before"],
                t_end=inc["t_after"],
                player_track_id=None,
                take_id=take_id,
                confidence=goal_cfg["occurrence_confidence"],
                source="scoreboard_ocr_delta",
                evidence={
                    "activated_region": list(activated_region),
                    "score_before": inc["score_before"],
                    "score_after": inc["score_after"],
                    "t_before": inc["t_before"],
                    "t_after": inc["t_after"],
                },
            )
        )
    return events, debug


# ---------------------------------------------------------------------------
# steps 2-3: scoring team + scorer attribution
# ---------------------------------------------------------------------------


def attribute_goal_team_and_scorer(
    goal_event: Event,
    possession_events: list[Event],
    tracks_by_id: dict[int, Track],
    goal_cfg: dict,
) -> Event:
    """ADR-17 steps 2-3. Deliberately sidesteps mapping the scoreboard's own left/right digit to
    home/away — instead credits whichever of OUR OWN two colour-clustered teams (`Track.team`,
    ADR-12) held ball possession for the most cumulative time in the
    `goal_cfg['possession_lookback_s']` window immediately before the digit changed
    (`goal_event.evidence['t_after']`), via the already-computed `POSSESSION` events
    (`src/events/possession.py::detect_possession`). The SCORER is then the identity with the
    LAST such possession run in that window, gated to the credited team.

    Returns a NEW `Event` (never mutates `goal_event` in place — Golden Rule 5: every attribution
    step's own evidence must stay independently auditable) with `player_track_id` set to the
    scorer's identity id when determinable; when no team-confident possession exists in the
    window, `player_track_id` stays `None` and `confidence` is penalised, never guessed.
    """
    lookback = goal_cfg["possession_lookback_s"]
    window_end = goal_event.evidence.get("t_after", goal_event.t_end)
    window_start = window_end - lookback

    in_window = [
        ev
        for ev in possession_events
        if ev.type == EventType.POSSESSION and ev.t_end >= window_start and ev.t_start <= window_end
    ]

    def _team_of(ev: Event) -> int | None:
        raw_ids = ev.evidence.get("raw_track_ids") or []
        if not raw_ids:
            return None
        tr = tracks_by_id.get(raw_ids[-1])
        return tr.team if tr is not None else None

    team_duration: dict[int, float] = {}
    for ev in in_window:
        team = _team_of(ev)
        if team is None:
            continue
        team_duration[team] = team_duration.get(team, 0.0) + (ev.t_end - ev.t_start)

    def _clamp(value: float) -> float:
        return min(goal_cfg["max_confidence"], max(goal_cfg["min_confidence"], value))

    if not team_duration:
        new_evidence = {
            **goal_event.evidence,
            "team_attribution": (
                f"undetermined -- no team-confident POSSESSION event found in the {lookback}s "
                "lookback window before the goal"
            ),
            "possession_lookback_s": lookback,
        }
        return goal_event.model_copy(
            update={
                "evidence": new_evidence,
                "confidence": _clamp(
                    goal_event.confidence * goal_cfg["team_unknown_confidence_penalty"]
                ),
            }
        )

    credited_team = max(team_duration, key=lambda k: team_duration[k])
    team_events = [ev for ev in in_window if _team_of(ev) == credited_team]
    scorer_event = max(team_events, key=lambda ev: ev.t_end)
    scorer_identity = scorer_event.player_track_id

    new_evidence = {
        **goal_event.evidence,
        "credited_team": credited_team,
        "team_duration_seconds": team_duration,
        "scorer_identity": scorer_identity,
        "supporting_possession_event_id": scorer_event.id,
        "possession_lookback_s": lookback,
    }
    return goal_event.model_copy(
        update={
            "player_track_id": scorer_identity,
            "evidence": new_evidence,
            "confidence": _clamp(goal_event.confidence * goal_cfg["scorer_confidence_multiplier"]),
        }
    )


# ---------------------------------------------------------------------------
# step 4: assist
# ---------------------------------------------------------------------------


def find_assist_event(
    goal_event: Event,
    pass_events: list[Event],
    assist_cfg: dict,
    tracks_by_id: dict[int, Track] | None = None,
) -> Event | None:
    """ADR-17 step 4 — the owner's own rule, verbatim: "if the target pass the ball and the other
    score goal should be assist". The nearest PRECEDING `PASS` event
    (`src/events/possession.py::detect_passes`) whose `evidence['receiver_identity']` equals
    `goal_event`'s own credited scorer, ending within `assist_cfg['assist_window_seconds']` of the
    goal, from a CONFIRMED TEAMMATE of the scorer, becomes an `ASSIST` `Event` credited to the
    PASSER (`player_track_id` = the pass event's own `player_track_id`). Returns `None` — never
    fabricated — when the scorer itself is unknown (`goal_event.player_track_id is None`) or no
    qualifying teammate pass exists.

    **Bug fix ("streamed-gathering-treehouse" plan, Stage D).** This docstring always claimed "a
    teammate pass", but the code never actually checked it — it only matched
    `evidence['receiver_identity'] == scorer`. Under the DEFAULT config
    (`configs/events.yaml: pass.same_colour_required: true`, ADR-20) every `PASS` event already
    implies same-colour passer/receiver by construction (`src/events/possession.py::detect_passes`
    only emits `PASS`, never `TURNOVER`, when `teammates_gate` is `True`), so this was latent.
    But `pass.same_colour_required: false` is an explicit, documented owner-facing escape hatch
    for footage where colour clustering is unreliable — under that config a `PASS` can go to an
    OPPOSING-coloured receiver, and if that receiver later becomes the credited scorer (e.g. via a
    deflection/own-goal-shaped possession sequence), the old code would have credited an assist to
    a non-teammate. Fix: when `tracks_by_id` is supplied, re-verify passer-vs-scorer team agreement
    directly via the SAME `teammates_gate` tri-state test `pass`/`tackle`/`dribble` already use —
    `True` (confidently same team) is required; `False` (confidently opposing) or `None` (team
    signal not trustworthy enough to judge) both EXCLUDE that candidate pass, same "can't tell,
    don't guess" convention as every other team-gated heuristic in this codebase (Golden Rule 5).
    `tracks_by_id=None` (no track data available to the caller) skips this re-check entirely,
    preserving the exact pre-fix behaviour — every real caller (`detect_goals_for_take`) always has
    `tracks_by_id` on hand and passes it.
    """
    scorer = goal_event.player_track_id
    if scorer is None:
        return None

    window_start = goal_event.t_end - assist_cfg["assist_window_seconds"]
    candidates = [
        ev
        for ev in pass_events
        if ev.type == EventType.PASS
        and ev.evidence.get("receiver_identity") == scorer
        and window_start <= ev.t_end <= goal_event.t_end
    ]
    if tracks_by_id is not None:
        threshold = assist_cfg["team_confidence_threshold"]
        candidates = [
            ev
            for ev in candidates
            if (passer_raw := ev.evidence.get("passer_raw_track_id", ev.player_track_id))
            is not None
            and teammates_gate(passer_raw, scorer, tracks_by_id, threshold) is True
        ]
    if not candidates:
        return None
    supporting_pass = max(candidates, key=lambda ev: ev.t_end)

    confidence = min(
        assist_cfg["max_confidence"],
        max(
            assist_cfg["min_confidence"],
            goal_event.confidence * assist_cfg["assist_confidence_multiplier"],
        ),
    )
    return Event(
        id=str(uuid.uuid4()),
        type=EventType.ASSIST,
        t_start=supporting_pass.t_start,
        t_end=supporting_pass.t_end,
        player_track_id=supporting_pass.player_track_id,
        take_id=goal_event.take_id,
        confidence=confidence,
        source="goal_preceding_pass_heuristic",
        evidence={
            "goal_event_id": goal_event.id,
            "supporting_pass_event_id": supporting_pass.id,
            "passer_identity": supporting_pass.player_track_id,
            "receiver_identity": scorer,
            "assist_window_seconds": assist_cfg["assist_window_seconds"],
        },
    )


# ---------------------------------------------------------------------------
# ADR-20 -- human-marked goal-region detector (a second, independent occurrence source)
# ---------------------------------------------------------------------------


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test, pure/dependency-free (no need for a cv2/shapely
    import for this -- a handful of point tests per take is cheap either way). `polygon` is a list
    of `(x, y)` vertices in the SAME normalised-fraction coordinate space the ball centroid is
    converted into before calling this (see `detect_goals_goal_region`)."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def detect_goals_goal_region(
    balls: list[BallDetection],
    fast_ball_windows: list[tuple[float, float]],
    slug: str,
    region_cfg: dict,
    take_id: int | None,
    source_label: str = "manual",
) -> list[Event]:
    """ADR-20 -- the goal-mouth-region occurrence detector: a SECOND, independent "did a goal
    occur" source alongside `detect_goals_scoreboard_delta`, for the static-camera footage that has
    no scoreboard at all.

    **`source_label` ("streamed-gathering-treehouse" plan, Stage D)**: the geometry test itself
    (`_point_in_polygon`) never changes -- only WHERE `region_cfg['regions']` came from does. The
    caller (`detect_goals_for_take`) passes `"detected"` when `region_cfg['regions'][slug]` came
    from Stage C's auto-tracked goal structure (`src/goal/detect.py`), or the default `"manual"`
    when it came from the owner's own hand-drawn `configs/goal_region.yaml` polygon (ADR-3's
    human-in-the-loop, no-training pattern, kept alive as an explicit fallback for a camera angle
    where auto-detection can't see the goal). Recorded verbatim in each emitted event's own
    `evidence['goal_region_source']` (Golden Rule 5 -- provenance is always auditable).

    `region_cfg['regions']` is `{slug: [[(x, y), ...], ...]}`, `(x, y)` as FRACTIONS of the
    detect-stage frame (same coordinate space as `configs/events.yaml: goal.candidate_regions`) --
    a `slug` absent from that mapping means no polygon was ever drawn for this video, so this
    returns `[]` immediately (Golden Rule 5: no polygon configured is a genuine "not available",
    never a guessed default region).

    A ball sample counts as a goal candidate only when its centroid falls inside ANY of the
    slug's own polygons AND `fast_ball_windows` (this take's own RAW sustained fast-ball-motion
    segments -- `find_threshold_segments` over `ball_speed_series`, computed BEFORE
    `detect_shots`'s own `direction_consistency_ratio` filtering) has a window that ENDED within
    `region_cfg['shot_window_s']` seconds before it. This is deliberately NOT gated on a
    fully-formed `SHOT` `Event`: `detect_shots`'s direction-consistency filter is calibrated for a
    different, stricter purpose (a countable Shots stat/highlight category) and this project's own
    measured data shows it can reject nearly an entire take's ball motion as "not consistent
    enough" (`configs/events.yaml: shot.direction_consistency_ratio`, CLAUDE.md's own note on
    `clip2`) -- which would silently starve this precondition of any qualifying window regardless
    of where the polygon is placed. The raw fast-motion segment is a more permissive plausibility
    gate: "was the ball moving fast right before this crossing" (a goal kick, a keeper's
    distribution, or a corner taken from behind the line rolling slowly through the marked area
    would NOT clear this), without also requiring the ball's own direction to have stayed
    consistent throughout — a struck ball can still curve or deflect on its way into the net.
    Consecutive qualifying samples from the SAME continuous crossing are debounced into one event
    (`shot_window_s` is reused as the debounce gap too -- a real goal is one instant, not one event
    per ~0.03s ball sample crossing the line).
    """
    polygons = (region_cfg.get("regions") or {}).get(slug)
    if not polygons:
        return []

    window = region_cfg["shot_window_s"]
    fast_window_ends = sorted(end for _start, end in fast_ball_windows)

    events: list[Event] = []
    debounce_until = float("-inf")
    for ball in sorted(balls, key=lambda b: b.t):
        if ball.t <= debounce_until:
            continue
        inside = any(_point_in_polygon(ball.bbox.cx, ball.bbox.cy, poly) for poly in polygons)
        if not inside:
            continue
        preceding_fast_window_ends = [
            t for t in fast_window_ends if t <= ball.t and (ball.t - t) <= window
        ]
        if not preceding_fast_window_ends:
            continue

        fast_window_t_end = max(preceding_fast_window_ends)
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.GOAL,
                t_start=fast_window_t_end,
                t_end=ball.t,
                player_track_id=None,
                take_id=take_id,
                confidence=region_cfg["confidence"],
                source="goal_region",
                evidence={
                    "ball_t": ball.t,
                    "ball_cx": ball.bbox.cx,
                    "ball_cy": ball.bbox.cy,
                    "fast_ball_window_t_end": fast_window_t_end,
                    "shot_window_s": window,
                    "goal_region_source": source_label,
                    "slug": slug,
                },
            )
        )
        debounce_until = ball.t + window
    return events


# ---------------------------------------------------------------------------
# Stage D ("streamed-gathering-treehouse" plan) -- feed detect_goals_goal_region from Stage C's
# auto-tracked goal structure instead of only the static configs/goal_region.yaml polygon.
# ---------------------------------------------------------------------------


def _tracked_polygons_for_take(
    goal_structure: GoalStructure,
    video_path: str | Path,
    take: Take,
    hardware_cfg: dict,
    profile_cfg: dict,
    goal_structure_cfg: dict,
    frame_width: float,
    frame_height: float,
    use_nvdec: bool = True,
) -> list[list[tuple[float, float]]]:
    """Query Stage C's tracked goal structure (`src/goal/detect.py`) at a handful of
    evenly-spaced instants across `take` (bounded -- never a per-ball-sample lookup) and convert
    each into a rectangle polygon in the SAME detect-stage-pixel coordinate space
    `detect_goals_goal_region`'s own manual polygons already use. `_point_in_polygon` itself never
    changes (Stage D's own scope, per the plan) -- a ball crossing ANY of these anchor-instant
    polygons counts, which is an honest approximation of "the goal moved with the camera" without
    needing a per-ball-sample affine lookup.

    Number of anchors is driven by THIS TAKE's own measured motion score (`src/goal/detect.
    take_motion_score`, ADR-11: never a fixed interval or the coarse static/panning label) via
    `goal_structure_cfg['tracking']`'s own `min_reestimate_interval_s`/`max_reestimate_interval_s`
    -- capped at 10 total anchors regardless (a reasoned bound on decode/affine-hop cost for a
    long, high-motion take; each anchor is one `goal_bbox_at` query, itself at most one short
    frame-grab + affine fit per side, see that function's own docstring for its cost).
    """
    tracking_cfg = goal_structure_cfg["tracking"]
    motion_cfg = profile_cfg["motion"]
    motion_score = take_motion_score(video_path, take, hardware_cfg, profile_cfg, use_nvdec)

    interval = (
        tracking_cfg["min_reestimate_interval_s"]
        if motion_score > tracking_cfg["reestimate_motion_score_threshold"]
        else tracking_cfg["max_reestimate_interval_s"]
    )
    duration = max(take.t_end - take.t_start, 1e-3)
    n_anchors = max(1, min(10, math.ceil(duration / interval)))
    anchor_ts = [take.t_start + duration * (i + 0.5) / n_anchors for i in range(n_anchors)]

    native_meta = probe(video_path)
    native_w, native_h = native_meta["width"], native_meta["height"]

    polygons: list[list[tuple[float, float]]] = []
    for t in anchor_ts:
        native_bbox = goal_bbox_at(
            goal_structure, video_path, t, take, motion_score, tracking_cfg, motion_cfg, use_nvdec
        )
        x1 = (native_bbox.x1 / native_w) * frame_width
        x2 = (native_bbox.x2 / native_w) * frame_width
        y1 = (native_bbox.y1 / native_h) * frame_height
        y2 = (native_bbox.y2 / native_h) * frame_height
        polygons.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return polygons


def _tracked_goal_lines_for_take(
    goal_structure: GoalStructure,
    video_path: str | Path,
    take: Take,
    hardware_cfg: dict,
    profile_cfg: dict,
    goal_structure_cfg: dict,
    frame_width: float,
    frame_height: float,
    use_nvdec: bool = True,
) -> list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]]:
    """Stage 6 (owner spec, 2026-09-02): the SAME anchor-instant propagation
    `_tracked_polygons_for_take` uses (mirrored, not shared, since that one returns rectangle
    POLYGONS for the unchanged manual/legacy containment path -- this returns goal-line SEGMENTS
    for the new crossing test), but built from `goal_posts_at` instead of `goal_bbox_at` so the
    result is the actual line between the two posts, never a bounding box.

    Returns `[(anchor_t, (left_base, right_base))]` in detect-stage-pixel space (same scaling as
    `_tracked_polygons_for_take`) -- a take whose structure has fewer than 2 tracked posts at every
    anchor contributes no lines at all (honest -- `goal_line_from_posts` already refuses to guess
    a line from one post).
    """
    tracking_cfg = goal_structure_cfg["tracking"]
    motion_cfg = profile_cfg["motion"]
    motion_score = take_motion_score(video_path, take, hardware_cfg, profile_cfg, use_nvdec)

    interval = (
        tracking_cfg["min_reestimate_interval_s"]
        if motion_score > tracking_cfg["reestimate_motion_score_threshold"]
        else tracking_cfg["max_reestimate_interval_s"]
    )
    duration = max(take.t_end - take.t_start, 1e-3)
    n_anchors = max(1, min(10, math.ceil(duration / interval)))
    anchor_ts = [take.t_start + duration * (i + 0.5) / n_anchors for i in range(n_anchors)]

    native_meta = probe(video_path)
    native_w, native_h = native_meta["width"], native_meta["height"]

    lines: list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]] = []
    for t in anchor_ts:
        native_posts = goal_posts_at(
            goal_structure, video_path, t, take, motion_score, tracking_cfg, motion_cfg, use_nvdec
        )
        line = goal_line_from_posts(native_posts)
        if line is None:
            continue
        (lx, ly), (rx, ry) = line
        scaled = (
            (lx / native_w * frame_width, ly / native_h * frame_height),
            (rx / native_w * frame_width, ry / native_h * frame_height),
        )
        lines.append((t, scaled))
    return lines


def detect_goals_line_crossing(
    take_balls: list[BallDetection],
    tracked_lines: list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]],
    take_id: int | None,
    goal_cfg: dict,
) -> list[Event]:
    """Stage 6 (owner spec, 2026-09-02): a GOAL from the auto-tracked `GoalStructure`, decided by
    genuine goal-LINE crossing (`src.events.goal_line.ball_crosses_goal_line`) -- never a
    bounding-box/polygon containment test (the anti-pattern `detect_goals_goal_region` still uses
    for the DIFFERENT, human-marked-polygon path, which stays as-is: a person drawing a box is an
    accepted, explicit design, not an algorithm inventing one).

    `tracked_lines` is `_tracked_goal_lines_for_take`'s own output -- an empty list (no confident
    goal structure for this take, or fewer than 2 posts tracked at every anchor) means this
    function honestly finds nothing, never falls back to a guessed line.

    Walks CONSECUTIVE, sorted-by-time raw ball detections (not the Stage 1 `BallState` series --
    each tracked line is already anchored to its own nearby frame's coordinate space via
    `goal_posts_at`'s own affine propagation, so comparing camera-COMPENSATED ball positions here
    would double-correct for the same camera motion) and tests each consecutive pair against
    whichever tracked line is nearest in time to that pair's own midpoint.
    """
    if not tracked_lines:
        return []
    sorted_balls = sorted(take_balls, key=lambda b: b.t)
    events: list[Event] = []
    for prev_b, curr_b in zip(sorted_balls, sorted_balls[1:], strict=False):
        mid_t = (prev_b.t + curr_b.t) / 2.0
        _anchor_t, line = min(tracked_lines, key=lambda item: abs(item[0] - mid_t))
        prev_state = BallState(
            t=prev_b.t,
            x=prev_b.bbox.cx,
            y=prev_b.bbox.cy,
            vx=0.0,
            vy=0.0,
            speed=0.0,
            direction_deg=None,
            confidence=prev_b.conf,
            is_interpolated=prev_b.interpolated,
            is_camera_compensated=False,
            known=True,
        )
        curr_state = BallState(
            t=curr_b.t,
            x=curr_b.bbox.cx,
            y=curr_b.bbox.cy,
            vx=0.0,
            vy=0.0,
            speed=0.0,
            direction_deg=None,
            confidence=curr_b.conf,
            is_interpolated=curr_b.interpolated,
            is_camera_compensated=False,
            known=True,
        )
        crossing = ball_crosses_goal_line(prev_state, curr_state, line, goal_cfg)
        if crossing is None:
            continue
        events.append(
            Event(
                id=str(uuid.uuid4()),
                type=EventType.GOAL,
                t_start=prev_b.t,
                t_end=curr_b.t,
                player_track_id=None,
                take_id=take_id,
                confidence=goal_cfg["min_ball_confidence_for_goal"],
                source="goal_line_crossing",
                evidence={
                    "goal_region_source": "detected_line_crossing",
                    "crossing_point": crossing,
                    "goal_line": line,
                    "prev_ball_t": prev_b.t,
                    "curr_ball_t": curr_b.t,
                    "prev_ball_conf": prev_b.conf,
                    "curr_ball_conf": curr_b.conf,
                    "calibrated": False,
                    "unit": "pixels_at_detect_stage_resolution",
                },
            )
        )
    return events


# ---------------------------------------------------------------------------
# per-take orchestration (does its own decode; lazily computes possession/pass only if needed)
# ---------------------------------------------------------------------------


def detect_goals_for_take(
    reader: Any,
    video_path: str | Path,
    take: Take,
    take_tracks: list[Track],
    take_balls: list[BallDetection],
    events_cfg: dict,
    identity_of: dict[int, int] | None,
    use_nvdec: bool = True,
    frame_width: float | None = None,
    frame_height: float | None = None,
    goal_region_cfg: dict | None = None,
    slug: str | None = None,
    goal_structure: GoalStructure | None = None,
    goal_structure_cfg: dict | None = None,
    hardware_cfg: dict | None = None,
    profile_cfg: dict | None = None,
    kit_lab_by_track_id: dict[int, np.ndarray] | None = None,
) -> tuple[list[Event], dict]:
    """Full ADR-17 pipeline for ONE take: decode a low-fps frame series spanning it, run the
    scoreboard occurrence scan, and — ONLY when at least one occurrence event (scoreboard- OR
    region-sourced) is actually found — lazily compute this take's own possession runs + passes
    (`src/events/possession.py`) to attribute team/scorer/assist for EVERY goal found, from
    whichever source. A take with no scoreboard activation, no increment, and no region-sourced
    goal either never pays that extra cost, which is what keeps this cheap on goal-less footage
    (measured true for every take of all 5 original clips + `jordan_thomas_highlight_video`, this
    task's own verification run).

    ADR-20 adds a SECOND, independent occurrence source alongside the scoreboard scan: when
    `goal_region_cfg`/`slug`/`frame_width` are given and a polygon exists for `slug`
    (`detect_goals_goal_region`), this take's own RAW sustained fast-ball-motion segments are
    computed (lazily, only then -- `frame_width` being `None` skips this path entirely, same
    "don't pay for what isn't configured" discipline as the scoreboard branch above) via the exact
    same `ball_speed_series` + `find_threshold_segments` primitives `detect_shots` itself uses,
    BEFORE `detect_shots`'s own stricter `direction_consistency_ratio` filtering (see
    `detect_goals_goal_region`'s own docstring for why: that filter is calibrated for a different,
    countable-stat purpose and this project's own measured data shows it can reject nearly a whole
    take's ball motion, which would silently starve the goal-region precondition regardless of
    polygon placement) -- and checked for a ball-crossing goal. Region-sourced `GOAL`s now go
    through the SAME team/scorer/assist attribution as scoreboard-sourced ones (their own lower
    `occurrence_confidence`, i.e. `configs/events.yaml: goal_region.confidence`, still stacks DOWN
    through the identical attribution multipliers, so a region-sourced goal's final confidence
    stays honestly lower than a scoreboard-sourced one's).

    `identity_of` is the SAME `build_take_identities` partition the caller's own event pipeline
    uses for this take (`None` for the original Phase-1 flow, which has no identity-stitching
    step at all and uses raw `Track.id`s directly, exactly like `detect_sprints` already does).

    **Stage D geometry source, revised to LINE CROSSING 2026-09-02 (owner spec: "DO NOT detect
    the penalty box or generic goal box as the primary goal detector... the critical event is BALL
    CROSSES GOAL LINE... between the two goal posts").** `goal_structure` (Stage C's own
    `src/goal/detect.GoalStructure` for THIS take, or `None`) supplies a THIRD, auto-tracked
    geometry source -- decided by `detect_goals_line_crossing` (genuine post-to-post line
    intersection, `src.events.goal_line`), NOT the bounding-box/polygon containment test this used
    before. Priority, per take: a confident `goal_structure` (>= 2 tracked posts at every queried
    anchor) wins over the manual `configs/goal_region.yaml` polygon (`raw_polygons`) when both
    exist; the manual polygon is kept UNCHANGED as an explicit fallback for a camera angle Stage C
    couldn't confidently localize (ADR-20's own human-in-the-loop escape hatch -- a person's own
    drawn box is an accepted, explicit design, not the "box as goal proxy" anti-pattern the owner
    is asking to stop for the AUTO-DETECTED case specifically); neither present -> the region
    source is skipped entirely, exactly as before this plan. `goal_structure_cfg`/`hardware_cfg`/
    `profile_cfg` are only needed alongside `goal_structure` (to query
    `src/goal/detect.goal_posts_at` at a handful of instants across the take) — any of them being
    `None` falls back to the manual-polygon-or-nothing behaviour, never a crash.

    `kit_lab_by_track_id` ("streamed-gathering-treehouse" plan Stage 3, `src.track.kit_wiring.
    build_take_kit_colour`): this take's own per-track grass-suppressed kit Lab, threaded straight
    into the lazy `detect_passes` call below (assist attribution needs the SAME teammate test
    ADR-20 gave `detect_passes` generally) — `None` (the default) reproduces the exact pre-existing
    cluster-only teammate test for any caller that hasn't wired this up yet.
    """
    goal_cfg = events_cfg["goal"]
    assist_cfg = events_cfg["assist"]

    scoreboard_frames = (
        (t, frame.copy())
        for _idx, t, frame in decode_frames(
            video_path,
            fps=goal_cfg["scoreboard_ocr_fps_sample"],
            start=take.t_start,
            end=take.t_end,
            scale_width=None,
            use_nvdec=use_nvdec,
        )
    )
    try:
        occurrence_events, debug = detect_goals_scoreboard_delta_stream(
            reader, scoreboard_frames, take.id, goal_cfg
        )
    finally:
        scoreboard_frames.close()  # stop ffmpeg now when the scan ended the stream early

    region_events: list[Event] = []
    region_attempted = False
    # ADR-20's own config split (CLAUDE.md §5.1): `configs/goal_region.yaml` (`goal_region_cfg`
    # here) carries ONLY human-marked polygons (`{regions: {slug: [...]}}`), never thresholds --
    # `enabled`/`shot_window_s`/`confidence` live in `configs/events.yaml`'s own `goal_region:`
    # block (`events_cfg["goal_region"]`). The two must be merged before calling
    # `detect_goals_goal_region`, which expects all of them on one dict.
    region_behavior_cfg = events_cfg.get("goal_region", {})
    raw_polygons = (goal_region_cfg or {}).get("regions", {}).get(slug) if slug else None

    # Stage 6 (owner spec, 2026-09-02): a confident Stage C goal_structure now decides a goal by
    # actual LINE CROSSING (`detect_goals_line_crossing`), never a bounding-box/polygon
    # containment test -- the anti-pattern this replaces for the AUTO-TRACKED case specifically.
    # `configs/goal_region.yaml`'s own hand-drawn polygon is UNCHANGED and kept as an explicit
    # fallback for a camera angle Stage C couldn't confidently localize (ADR-20's human-in-the-loop
    # escape hatch, still valuable, still a person's own explicit box -- not the same thing as an
    # algorithm inventing one). Detected wins over manual when both are available for this take.
    detected_lines: list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]] = []
    # initialized here (not just in the elif/else below) so the trailing polygon-containment
    # block's own `if polygons_to_use and ...` guard is well-defined even when the `detected_lines`
    # branch below already fully handled this take and never needs a polygon at all.
    polygons_to_use: list[list[tuple[float, float]]] | None = None
    source_label = "manual"
    if (
        goal_structure is not None
        and goal_structure_cfg is not None
        and hardware_cfg is not None
        and profile_cfg is not None
        and frame_width is not None
        and frame_height is not None
    ):
        detected_lines = _tracked_goal_lines_for_take(
            goal_structure,
            video_path,
            take,
            hardware_cfg,
            profile_cfg,
            goal_structure_cfg,
            frame_width,
            frame_height,
            use_nvdec,
        )

    if detected_lines:
        region_attempted = True
        source_label = "detected_line_crossing"
        region_events = detect_goals_line_crossing(take_balls, detected_lines, take.id, goal_cfg)
    elif raw_polygons:
        # configs/goal_region.yaml polygons are authored as [0,1] FRACTIONS of the frame (so one
        # polygon travels correctly across a re-run at a different decode scale_width) --
        # `detect_goals_goal_region` itself is unit-agnostic (it just compares ball coordinates
        # against polygon coordinates directly), so the fraction -> pixel scaling happens exactly
        # once, here, into a per-call copy rather than mutating the shared config.
        polygons_to_use = [
            [(x * frame_width, y * frame_height) for x, y in poly] for poly in raw_polygons
        ]
        source_label = "manual"
    else:
        polygons_to_use = None
        source_label = "manual"

    if (
        polygons_to_use
        and frame_width is not None
        and frame_height is not None
        and region_behavior_cfg.get("enabled", True)
    ):
        region_attempted = True
        scaled_region_cfg = {**region_behavior_cfg, "regions": {slug: polygons_to_use}}
        shot_cfg = events_cfg["shot"]
        times, speeds, _dxs = ball_speed_series(
            take_balls,
            frame_width,
            shot_cfg["smoothing_window_frames"],
            shot_cfg["min_ball_conf"],
        )
        fast_ball_windows = find_threshold_segments(
            times,
            speeds,
            shot_cfg["ball_speed_threshold"],
            shot_cfg["min_duration_s"],
            shot_cfg["merge_gap_s"],
        )
        region_events = detect_goals_goal_region(
            take_balls,
            fast_ball_windows,
            slug,
            scaled_region_cfg,
            take.id,
            source_label=source_label,
        )
    debug["goal_region_attempted"] = region_attempted
    debug["goal_region_events_found"] = len(region_events)
    debug["goal_region_source"] = source_label if region_attempted else None

    if not occurrence_events and not region_events:
        return [], debug

    tracks_by_id = {tr.id: tr for tr in take_tracks}
    runs = possession_runs_for_take(take_balls, take_tracks, events_cfg, identity_of)
    possession_events = detect_possession(runs, take.id, events_cfg)
    pass_events = detect_passes(
        runs, take_tracks, take.id, events_cfg, kit_lab_by_track_id=kit_lab_by_track_id
    )
    debug["n_possession_runs"] = len(runs)

    events: list[Event] = []
    for occ in occurrence_events + region_events:
        attributed = attribute_goal_team_and_scorer(occ, possession_events, tracks_by_id, goal_cfg)
        events.append(attributed)
        assist_event = find_assist_event(attributed, pass_events, assist_cfg, tracks_by_id)
        if assist_event is not None:
            events.append(assist_event)
    return events, debug


# ---------------------------------------------------------------------------
# final reporting step (pure decision table)
# ---------------------------------------------------------------------------


def check_goal_availability(
    scan_attempted: bool,
    activated_any_region: bool,
    events: list[Event],
    goal_cfg: dict,
    goal_region_configured: bool = False,
) -> GoalDetectionResult:
    """The final Golden-Rule-5 reporting step (ADR-17, extended by ADR-20): turns a scan's own
    OUTCOME into an honest `GoalDetectionResult`. Never does I/O itself — `detect_goals_for_video`
    is responsible for actually running the scan(s); this stays a pure, trivially-unit-testable
    decision table, so its result is provably driven by real input rather than a hardcoded "not
    available" regardless of what was actually found (the exact defect ADR-17 fixes).

    ADR-20: the availability decision is now made off `events` (whichever `GOAL`s survived from
    EITHER the scoreboard-OCR-delta source or the human-marked goal-region source) FIRST, not off
    `activated_any_region` (the scoreboard-only activation flag) alone -- a video with a configured
    goal region but no scoreboard anywhere must still correctly report `available=True` off a real
    region-sourced goal, never masked by "no scoreboard activated". `goal_region_configured` names,
    in the "not available" reason, whether a human-marked polygon was even in play for this video
    (`configs/goal_region.yaml`) so the reason always says which of the two sources this module
    itself attempted (a third source, the ADR-19 manual-annotation sidecar, is a separate pipeline
    entirely and is not this function's concern).
    """
    if not scan_attempted:
        return GoalDetectionResult(
            available=False,
            reason="not available (goal detection was not attempted -- this video has no takes)",
            events=[],
        )

    goal_events = [e for e in events if e.type == EventType.GOAL]
    region_note = (
        "a human-marked goal region IS configured for this video (configs/goal_region.yaml, "
        "ADR-20) and was also checked"
        if goal_region_configured
        else "no human-marked goal region is configured for this video either "
        "(configs/goal_region.yaml, ADR-20)"
    )

    if not goal_events:
        if not activated_any_region:
            scoreboard_note = (
                "no legible scoreboard found by the region-activation scan -- CLAUDE.md ADR-17: "
                "EasyOCR was run over each candidate corner region across multiple sampled frames "
                "per take and none yielded a stable digit-separator-digit reading; this is the "
                "measured, per-video result, never an assumption from the source profile label"
            )
        else:
            scoreboard_note = (
                "a scoreboard region activated on this footage, but no debounced score increment "
                "was ever observed across the video"
            )
        return GoalDetectionResult(
            available=False,
            reason=(
                f"not available ({scoreboard_note}; {region_note}, and no ball-crossing-a-marked-"
                "region goal was found)"
            ),
            events=events,
        )

    assist_events = [e for e in events if e.type == EventType.ASSIST]
    scoreboard_n = sum(1 for e in goal_events if e.source == "scoreboard_ocr_delta")
    region_n = sum(1 for e in goal_events if e.source == "goal_region")
    source_bits = []
    if scoreboard_n:
        source_bits.append(f"{scoreboard_n} via scoreboard-OCR delta")
    if region_n:
        source_bits.append(f"{region_n} via a human-marked goal region (ADR-20)")
    source_desc = "; ".join(source_bits) if source_bits else "via an unspecified source"
    return GoalDetectionResult(
        available=True,
        reason=(
            f"{len(goal_events)} goal(s) detected ({source_desc}) "
            f"({len(assist_events)} with an assist credited) -- see each Event.evidence for the "
            "full audit trail (activated region / marked polygon, before/after reading or "
            "crossing point, possession-window scores, supporting pass id), Golden Rule 5"
        ),
        events=events,
    )


# ---------------------------------------------------------------------------
# whole-video orchestration (shared by src.pipeline.run and src.pipeline.extended_output)
# ---------------------------------------------------------------------------


def detect_goals_for_video(
    video_path: str | Path,
    takes: list[Take],
    tracks_by_take: dict[int, list[Track]],
    balls_by_take: dict[int, list[BallDetection]],
    events_cfg: dict,
    identity_of_by_take: dict[int, dict[int, int]] | None = None,
    ocr_cfg: dict | None = None,
    use_nvdec: bool = True,
    frame_width: float | None = None,
    frame_height: float | None = None,
    goal_region_cfg: dict | None = None,
    slug: str | None = None,
    goal_structures_by_take: dict[int, GoalStructure] | None = None,
    goal_structure_cfg: dict | None = None,
    hardware_cfg: dict | None = None,
    profile_cfg: dict | None = None,
    kit_lab_by_track_id_by_take: dict[int, dict[int, np.ndarray]] | None = None,
) -> GoalDetectionResult:
    """ADR-17's real, per-video goal+assist detector (ADR-20 adds the human-marked goal-region
    source; Stage D of the "streamed-gathering-treehouse" plan adds Stage C's auto-tracked one
    alongside it) — the single entrypoint both `src.pipeline.run` (the 5 original filename-based
    clips) and `src.pipeline.extended_output` (ADR-15's filename-less flow) call identically.
    Loads ONE EasyOCR reader for the whole video (CLAUDE.md §11: never reload per-take), matching
    `src/identity/verify.py`'s own load-once/free-in-`finally` convention.

    Never trusts `RunProfile.profile` (ADR-11's own lesson): every take gets the SAME measured
    region-activation scan regardless of what the source profiler labeled the video.

    `frame_width`/`frame_height`/`goal_region_cfg`/`slug` (ADR-20) and `goal_structures_by_take`/
    `goal_structure_cfg`/`hardware_cfg`/`profile_cfg` (Stage D) are all optional and default to
    `None` -- omitting any of them simply skips that source entirely (never a crash, never a
    fabricated region), which is exactly what every caller that doesn't yet pass them gets (safe,
    behaviour-preserving default for any pre-ADR-20/pre-Stage-D call site).

    `kit_lab_by_track_id_by_take` ("streamed-gathering-treehouse" plan Stage 3): per-take kit-Lab
    dicts from `src.track.kit_wiring.build_take_kit_colour`, looked up per take below and passed
    to `detect_goals_for_take` for its own lazy assist attribution. `None` (the default) omits the
    lookup entirely -- every take's own `detect_goals_for_take` call then gets `kit_lab_by_track_id
    =None`, the exact pre-existing cluster-only teammate test.
    """
    goal_cfg = events_cfg["goal"]
    goal_region_configured = bool(slug and (goal_region_cfg or {}).get("regions", {}).get(slug))
    if not takes:
        return check_goal_availability(
            scan_attempted=False,
            activated_any_region=False,
            events=[],
            goal_cfg=goal_cfg,
            goal_region_configured=goal_region_configured,
        )

    reader = load_easyocr_reader(ocr_cfg or {"gpu": True})
    all_events: list[Event] = []
    activated_any = False
    try:
        for take in takes:
            take_tracks = tracks_by_take.get(take.id, [])
            take_balls = balls_by_take.get(take.id, [])
            identity_of = (identity_of_by_take or {}).get(take.id)
            events, debug = detect_goals_for_take(
                reader,
                video_path,
                take,
                take_tracks,
                take_balls,
                events_cfg,
                identity_of,
                use_nvdec=use_nvdec,
                frame_width=frame_width,
                frame_height=frame_height,
                goal_region_cfg=goal_region_cfg,
                slug=slug,
                goal_structure=(goal_structures_by_take or {}).get(take.id),
                goal_structure_cfg=goal_structure_cfg,
                hardware_cfg=hardware_cfg,
                profile_cfg=profile_cfg,
                kit_lab_by_track_id=(kit_lab_by_track_id_by_take or {}).get(take.id),
            )
            if debug.get("activated_region") is not None:
                activated_any = True
            all_events.extend(events)
    finally:
        free_easyocr_reader(reader)

    any_goal = any(e.type == EventType.GOAL for e in all_events)
    logger.info(
        "goal detection: %s (%d take(s) scanned, scoreboard region activated=%s, goal region "
        "configured=%s)",
        "available" if any_goal else "not available",
        len(takes),
        activated_any,
        goal_region_configured,
    )
    return check_goal_availability(
        scan_attempted=True,
        activated_any_region=activated_any,
        events=all_events,
        goal_cfg=goal_cfg,
        goal_region_configured=goal_region_configured,
    )
