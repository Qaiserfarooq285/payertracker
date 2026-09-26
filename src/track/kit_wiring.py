"""Stage 3 of the "streamed-gathering-treehouse" plan (strict persistent target identity, see
`/home/qaiserfarooq/.claude/plans/streamed-gathering-treehouse.md`): wire per-track kit colour into
production for TWO consumers that need two DIFFERENT shapes of the same underlying measurement:

1. `src.events.possession.detect_passes`'s `teammates_test` (via `kit_lab_by_track_id`) needs ONE
   Lab value per track -- a single upper-torso region read (`src.events.kit_colour.kit_lab_sample`,
   the same ADR-21 `number_region` geometry already used for jersey OCR), aggregated across a
   track's own sampled frames via `median_kit_lab`. This is the ADR-20 "grass-suppressed kit colour
   as the primary teammate/opponent signal" measurement -- it was already implemented in
   `src.events.kit_colour` but nothing ever called it: `detect_passes`'s own `kit_lab_by_track_id`
   parameter has been dead code since it was added.
2. `src.track.target_verify.verify_candidate`'s `kit_by_track` needs a per-track `KitColourSample`
   (torso/shorts/socks THREE-band read, `src.track.target.sample_kit_bands`) -- a different crop
   geometry, built for a different purpose (hard-reject a candidate whose kit colour contradicts
   the persistent target's own profile, not "are these two CURRENTLY-tracked players teammates").

**Judgement call (plan Stage 3 explicitly asks for one):** `sample_kit_bands`'s 3-band read is NOT
a substitute for `kit_lab_sample`'s single upper-torso region for consumer (1) -- `detect_passes`
only ever needs one Lab value to compare against another player's, and collapsing 3 bands into 1
would either throw away 2/3 of a band-confidence-weighted read for no reason, or require inventing
a new aggregation rule nobody asked for. Conversely `sample_kit_bands` cannot be replaced by
`kit_lab_sample` for consumer (2) -- `verify_candidate`'s hard-reject step is torso-dominant but
explicitly falls back to shorts/socks when torso itself has no evidence (`_combined_kit_verdict`'s
own docstring), which needs all three bands. So this module computes BOTH aggregates from the SAME
decoded frames rather than picking one: "build it once, use it for both purposes" (the plan's own
words) is satisfied at the DECODE level (one `decode_frames` pass per take, the genuinely expensive
part) even though the two PER-FRAME crop-and-aggregate steps stay separate arithmetic -- reusing an
already-decoded frame for a second, cheap in-memory crop costs nothing next to a second full decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.types import BBox, Take, Track
from src.common.video import decode_frames
from src.events.kit_colour import kit_lab_sample, median_kit_lab
from src.identity.verify import _scale_bbox_to_native

# `_aggregate_kit` is reused verbatim, not copied (same "call, don't rebuild" discipline as
# src.track.click_reid's own imports from src.track.target/src.identity.verify) -- this module
# only ever CALLS Stage 1's own median-aggregation, never reimplements it.
from src.track.target import KitColourSample, sample_kit_bands
from src.track.target import _aggregate_kit as _aggregate_kit_bands


def _sample_timestamps(track: Track, max_samples: int) -> list[float]:
    """Up to `max_samples` timestamps evenly spread across `track`'s own box lifespan. Duplicated
    (not imported) from `src.track.click_reid._sample_track_timestamps` -- same tiny helper,
    independently kept per this project's own "stages independently runnable/debuggable"
    convention already used for e.g. `_effective_frame_size` (CLAUDE.md §10)."""
    if not track.boxes:
        return []
    if len(track.boxes) <= max_samples:
        return [b.t for b in track.boxes]
    step = len(track.boxes) / max_samples
    return [track.boxes[int(i * step)].t for i in range(max_samples)]


def build_take_kit_colour(
    video_path: str | Path,
    take: Take,
    take_tracks: list[Track],
    scale_x: float,
    scale_y: float,
    native_w: int,
    native_h: int,
    identity_cfg: dict,
    target_cfg: dict,
    kit_colour_cfg: dict,
    max_samples_per_track: int,
    use_nvdec: bool = True,
) -> tuple[dict[int, np.ndarray], dict[int, KitColourSample]]:
    """One decode pass over `take`, producing BOTH per-track kit-colour aggregates this plan needs:

    Returns `(kit_lab_by_track_id, kit_sample_by_track_id)`:
      - `kit_lab_by_track_id`: `{raw_track_id: Lab}` -- single upper-torso-region median, for
        `src.events.possession.detect_passes`/`teammates_test` (ADR-20).
      - `kit_sample_by_track_id`: `{raw_track_id: KitColourSample}` -- torso/shorts/socks median,
        for `src.track.target_verify.verify_candidate`'s `kit_by_track` argument (plan Stage 2).

    A track missing from either dict simply had no frame clear that read's own grass-suppression
    floor -- an honest "no colour evidence", never a guess (both callers already treat a missing
    entry as "no evidence either way", never a rejection).

    `identity_cfg` is `configs/identity.yaml` (reuses its `crop.identity_fps_sample` decode cadence
    and `parseq_soccernet.number_crop` geometry for the single-region read, per this module's own
    docstring); `target_cfg` is `configs/target.yaml` (its `bands`/`kit_colour` sections feed
    `sample_kit_bands`); `kit_colour_cfg` is `configs/events.yaml: kit_colour` (feeds
    `kit_lab_sample`'s own grass-suppression floor -- the SAME dict `detect_passes` itself already
    reads as `events_cfg["kit_colour"]`).
    """
    if not take_tracks:
        return {}, {}

    target_ts_by_track = {
        tr.id: _sample_timestamps(tr, max_samples_per_track) for tr in take_tracks
    }
    all_ts = [t for ts in target_ts_by_track.values() for t in ts]
    if not all_ts:
        return {}, {}

    tolerance_s = 1.0 / identity_cfg["crop"]["identity_fps_sample"]
    number_crop_cfg = identity_cfg["parseq_soccernet"]["number_crop"]

    lab_samples_by_track: dict[int, list[np.ndarray]] = {}
    band_samples_by_track: dict[int, list[KitColourSample]] = {}

    # Every (time, track, box) this take wants a colour read at, in time order. The frames are
    # STREAMED past these (2026-09-26, long videos): this used to keep every native-resolution frame
    # of the take in a list, and a single-camera match is often ONE take -- a 90-min 1080p match at
    # 2 fps is ~67 GB, more RAM than the pod has. Only the previous frame is held now; the frame
    # used for each read is still the nearest decoded one (earlier on a tie), exactly as before.
    wanted: list[tuple[float, int, object]] = []
    for tr in take_tracks:
        boxes_by_t = {b.t: b for b in tr.boxes}
        for target_t in target_ts_by_track[tr.id]:
            box = boxes_by_t.get(target_t)
            if box is not None:
                wanted.append((target_t, tr.id, box))
    wanted.sort(key=lambda w: w[0])

    def read_at(frame_entry: tuple[float, np.ndarray], target_t: float, track_id: int, box) -> None:
        frame_t, frame_bgr = frame_entry
        if abs(frame_t - target_t) > tolerance_s:
            return
        x1, y1, x2, y2 = _scale_bbox_to_native(box.bbox, scale_x, scale_y, 0.0, native_w, native_h)
        if x2 <= x1 or y2 <= y1:
            return
        native_bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

        lab_result = kit_lab_sample(frame_bgr, native_bbox, number_crop_cfg, kit_colour_cfg)
        if lab_result is not None:
            lab_samples_by_track.setdefault(track_id, []).append(lab_result[0])

        band_sample = sample_kit_bands(frame_bgr, native_bbox, take.id, target_t, target_cfg)
        if band_sample is not None:
            band_samples_by_track.setdefault(track_id, []).append(band_sample)

    prev: tuple[float, np.ndarray] | None = None
    i = 0
    for _idx, frame_t, frame in decode_frames(
        str(video_path),
        fps=identity_cfg["crop"]["identity_fps_sample"],
        start=max(0.0, min(all_ts) - 0.5),
        end=max(all_ts) + 0.5,
        scale_width=None,
        use_nvdec=use_nvdec,
    ):
        # Reads whose time this frame has reached: nearest is this frame or the previous one.
        while i < len(wanted) and wanted[i][0] <= frame_t:
            target_t, track_id, box = wanted[i]
            nearest = (frame_t, frame)
            if prev is not None and abs(prev[0] - target_t) <= abs(frame_t - target_t):
                nearest = prev
            read_at(nearest, target_t, track_id, box)
            i += 1
        prev = (frame_t, frame)
    if prev is None:
        return {}, {}
    while i < len(wanted):  # reads after the last decoded frame: that frame is the nearest
        target_t, track_id, box = wanted[i]
        read_at(prev, target_t, track_id, box)
        i += 1

    kit_lab_by_track_id: dict[int, np.ndarray] = {}
    for track_id, samples in lab_samples_by_track.items():
        median = median_kit_lab(samples)
        if median is not None:
            kit_lab_by_track_id[track_id] = median

    kit_sample_by_track_id: dict[int, KitColourSample] = {}
    for track_id, samples in band_samples_by_track.items():
        aggregated = _aggregate_kit_bands(samples)
        if aggregated is not None:
            kit_sample_by_track_id[track_id] = aggregated

    return kit_lab_by_track_id, kit_sample_by_track_id
