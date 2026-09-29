# CLAUDE.md — AI Soccer Highlight Analyzer

> Source of truth for how we build this. Keep it current: when a decision changes, update this file in the
> same commit. Read it fully before working.

## 1. What this project is
Ingest a **full broadcast soccer match** (90+ min, TV-style: multiple cameras, cuts, zooms, replays,
on-screen graphics) and produce **per-player highlight reels** + a **basic stat card**.
User flow: `Upload match → choose a player (by jersey #) → process → review actions → auto/select highlights → export reel`.

Input reality we design for: **broadcast** (hardest case) and a **production** quality bar.
Reference blueprint (rationale for every choice): https://claude.ai/code/artifact/e3f76bf9-ac19-4d68-8175-f4bfba418f37

**One-line goal (owner, 2026-08-27):** given *any* clip and a target jersey number, produce (a) an
**annotated copy of the same video** with the target player boxed + labelled with their jersey number and the
**ball tracked**, (b) a **stat card** with a timestamped event timeline, and (c) **per-category highlight
reels** (passes, goals, assists, …) — **fully automatic, correct on the first run**. When the footage is too
poor to read automatically, the client supplies **manual timestamped annotations** and the pipeline runs from
those instead (ADR-19). See §14 for the single auto flow.

## 2. Golden rules (do not break without asking the owner)
1. **Pipeline of specialized models, not one model.** Build stage by stage (§5).
2. **Never train from scratch.** Use pretrained models and **fine-tune** on public soccer data. If from-scratch
   training seems necessary, stop and ask.
3. **Broadcast tax first.** Segment video into continuous **camera takes** and drop replays/close-ups before
   any tracking. Track only *within* a take; reset IDs at every cut.
4. **Human-in-the-loop for player identity.** Cross-cut auto-ID is not production-reliable — design a
   user-confirmation step. Never claim perfect automatic identification. (A **human-provided** jersey number —
   filename metadata, `--target-jersey`, or a manual-annotation sidecar — is the strongest identity evidence
   we have, ADR-18/ADR-19.)
5. **Every stat is traceable + carries a confidence.** No fabricated numbers; each event links to its clip.
   Lead with reliable events (goals, shots, sprints); passes/touches/tackles are best-effort, flagged.
6. **License hygiene.** Prefer Apache-2.0 / MIT / BSD. **RF-DETR, not Ultralytics YOLO. `supervision` tracker,
   not BoxMOT.** Flag any GPL/AGPL/research-only dependency in §7 before adding it.
7. **Ship Phase 1 first** (§6). No jersey auto-ID or fine-grained action stats until Phase 1 is done + evaluated.
8. **Evaluation is part of the build.** Stand up the eval harness early; verify with numbers, not vibes.
9. **One command, first-flow-correct, resumable.** The whole thing runs with a single `make run` that
   auto-branches per input (§14). "Everything runs auto" (owner, 2026-08-27) is a hard requirement, not a
   nicety — but "auto" never means "guessed": an honest `not available`/`unverified`/`uncertain` is a correct
   auto result (Golden Rule 5), a fabricated number is a bug.

## 3. Current phase
**▶ PHASE 1 — within-take core with MANUAL player selection**, plus an **owner-authorized, verified-jersey
auto-ID exception for filename-less inputs** (ADR-15), a **manual-annotation sidecar path for low-quality
footage** (ADR-19), and **team-colour-aware pass/turnover/assist + a human-marked goal region** (ADR-20).
These pull a narrow slice of Phase 2 forward under explicit owner authorization. (See §6 for scope + definition
of done, and §14 for how they compose into one auto flow.)
Update this line as we advance.

### 3.1 Current input set (measured 2026-08-24)
The owner's footage is **not** a broadcast match — it is 5 short **4K single-camera** clips. This is exactly the
"validate on a single-camera clip first" case in §6. The source profiler (Stage 0.5) is still wired from day one
so the broadcast branch stays automatic.

| File | Resolution | FPS | Duration | Frames | Target jersey |
|---|---|---|---|---|---|
| `input/clip1 43.mp4` | 3840×2160 | 30 | 11.1 s | 372 | **43** |
| `input/clip2 77.mp4` | 3840×2160 | 30 | 13.4 s | 415 | **77** |
| `input/clip3 77.mp4` | 3840×2160 | 30 | 13.5 s | 414 | **77** |
| `input/clip4 77.mp4` | 3840×2160 | 30 | 110.9 s | 3448 | **77** |
| `input/clip5 77.mp4` | 3840×2160 | 30 | 14.3 s | 471 | **77** |

**Filename convention:** `clip<N> <jersey>.mp4`. The trailing integer is the **target player's jersey number**.

**How the jersey number is used in Phase 1 (important):** it is **run metadata only**. Per Golden Rule 7 we do
**not** run jersey OCR / auto-ID in Phase 1. The number is parsed from the filename into the run config
(`target_jersey`), used to name outputs (`output/clip1_43/…`) and to label the manual-selection prompt
("pick the track for #43"). The human still selects the track. Auto-ID by number lands in **Phase 2** (§6),
and the seam (`src/identity/`, `Track.jersey_number`, `Track.id_confidence`) exists from day one.

**4K note:** frames are decoded at 3840×2160 and **downscaled per stage** (see `configs/hardware.yaml`).
Never feed 4K straight to a detector — it wastes VRAM for no accuracy gain at these player sizes. Full-res
crops are used only where small-object detail matters (ball via SAHI, and Phase-2 jersey OCR).

### 3.2 Footage characteristics (inspected 2026-08-24 — these drive real design changes)
The clips are **Veo** exports (Veo Technologies AI camera; "veo" watermark bottom-right). Youth/amateur matches,
elevated wide sideline camera, digital pan/zoom cropped out of a stitched panoramic.

**Measured** *(corrected 2026-08-24 after Stage 0.5 shipped — see the two ⚠️ corrections below)*:
- **Cuts** — clip1/2/3/5 → **0 cuts** (single take each). **clip4 → 3 cuts at 60.13 s / 73.63 s / 87.30 s
  ⇒ 4 takes.** clip4 is a **4-venue compilation** of player #77's highlights across different matches
  (grass → turf-with-running-track → cold-weather grass with cones → bright grass).
  > ⚠️ **Correction.** An earlier ffmpeg `scdet @0.35` pass recorded here found only **1** cut and concluded
  > "2 takes". That threshold was too strict and **missed two real cuts**. PySceneDetect found them;
  > frame contact-sheets at 72.5/74.5/86.0/88.5 s confirm four visually distinct venues, and a re-run of
  > `scdet` at 0.15 reproduces all three. **Lesson: a single detector at one threshold is not ground truth.**
  > `configs/shots.yaml` threshold is now **38.0**, set from the *measured* content-value gap on this data
  > (real cuts ≥ 45.6, false positives ≤ 32.0) — not tuned to match a prediction. The rejected spikes at
  > 14.4 / 47.0 / 99.6 s are fast pans and foreground crossings, verified same-venue either side.
- **Camera motion**: sparse KLT + `estimateAffinePartial2D` (RANSAC) on consecutive native frames,
  median-aggregated. Scores (px/frame): clip1 **12.70**, clip2 **3.79**, clip3 **1.95**, clip4 **3.14**,
  clip5 **3.12**.
  > ⚠️ **Correction.** An earlier `vidstabdetect` eyeball recorded "≈ ±1 px/frame, near-static". That was
  > read off raw local-motion vectors and was **wrong**. Dense Farneback was also tried and rejected: its
  > score proved wildly resolution-dependent (0.07 at 4K vs 6.1 at 240 px on the same clip) — textureless
  > grass gives it nothing to lock onto.
- **⚠️ Every clip opens with a genuine ~3–5 s camera-settling pan** before going steady. Undocumented until
  Stage 0.5 measured it. Median aggregation absorbs it on long clips, but on an 11 s clip it is ~36% of the
  runtime, which is why **clip1 reads `single-panning`**.
- **Player scale** varies hugely within a frame: near players ≈ 450 px tall at 4K, far players ≈ 60 px.
- **Jersey numbers are legible at 4K** for mid/near players (#77, #22, #34, #13, #8 all readable even at 1280 px).

**Consequences (each one changes a stage):**
1. **⚠️ A burned-in red arrow graphic** marks the target player in the opening seconds of clip2 and clip4.
   It is an editorial overlay, not an object: it will (a) generate spurious detections and (b) partially
   occlude the target. Stage 2 must mask it. It is *also* a free prior for locating the target player — a
   legitimate Phase-1 aid for **manual** selection (it reads no jersey number, so Golden Rule 7 holds).
   **Measured signature (clip2, 3840×2160):** at t=3.0 s a saturated-red HSV mask
   (`H∈[0,10]∪[170,180], S>120, V>90`) yields **exactly one** connected component >2000 px —
   310×606 px at (2020, 262), area 27,756 (0.34% of frame). At t=4.0 s only 317 red px remain: **the arrow
   is transient**, present for roughly the first few seconds only. So the rule is robust: take the single
   largest saturated-red component above `arrow_min_area_px`, mask it, and use its **tip** as the
   target-player prior; when no such component exists, there is simply no arrow that frame. Thresholds go in
   `configs/detect.yaml`, never inline.
2. **Takes ≠ profile.** clip4 proves a single-camera source can still contain cuts. **Revision to §5:**
   shot-boundary detection now runs **always** (it is cheap and clip4 needs it); only the *replay detector,
   close-up classifier, and scoreboard OCR* are gated on `profile == broadcast`. The original "skip Stage 1
   entirely for single-camera" rule would have silently corrupted clip4's tracking. IDs still reset at cuts.
3. **⚠️ No scoreboard anywhere in this footage** → the Phase-1 "goal = scoreboard delta" detector is **N/A**
   here. It stays in the code for broadcast input, but on these clips it must report *"not available"*, never a
   guessed goal (Golden Rule 5). Usable Phase-1 events on this footage: **sprints** and **shots (heuristic)**.
   *(Goals on this footage now also have the manual-annotation path (ADR-19) and the human-marked goal-region
   path (ADR-20) — both still honest, both still "not available" when neither a sidecar nor a marked region
   nor a scoreboard exists.)*
4. **⚠️ Sparse/ambiguous pitch lines.** clip1 is a shared-use field with **American-football lines painted over
   the soccer pitch** (yard numbers, hashes); clip2/4 show only a touchline and part of a box, with portable
   goals. A pitch-keypoint model trained on pro stadiums will fail on both. This strongly confirms **ADR-3**.
   But note a static camera is *not* enough on its own — where fewer than 4 usable landmarks are visible,
   **do not fabricate metres**: emit speed with `calibrated: false`, a confidence penalty, and pixel-normalised
   units. See ADR-6.
5. **Detection input size**: at 1280 px the far players are ~30 px tall — marginal. Plan to run detection at
   **1536–1920 px** and/or apply **SAHI tiling to players as well as the ball**, then measure. The
   `image_size` knob in `configs/hardware.yaml` is the place to tune this.

### 3.3 Second input: `Jordan Thomas Highlight Video.mp4` (added 2026-08-27 — no filename jersey metadata)

The owner added a sixth clip that **does not** follow the `clip<N> <jersey>.mp4` convention (§3.1) — it is a
**pre-edited highlight compilation** ("HIGHLIGHTS JORDAN THOMAS" title card, an outro "Thank You For Watching"
card with contact info, a burned-in red arrow marking the target player in several segments, no jersey number
in the filename or run config). **Measured 2026-08-27:**

| | |
|---|---|
| Resolution / fps / duration | 3840×2160, 30 fps, 234.16 s, audio AAC 44.1 kHz stereo |
| Structure | title card (~0–3.7 s) → **14 camera takes across what look like several different matches/venues** (night turf under floodlights, a stadium with an American-football-lined field, several day grass fields with different kits/lighting/crowd size) → outro "Thank You For Watching" card (~229–234 s) |
| Cuts (`ContentDetector`, `downscale_width=640`) | **threshold-dependent, unlike clip4.** 21.0/24.0/27.0 all agree on the same **13-cut** set: t≈3.7, 25.9, 43.0, 60.4, 78.0, 90.2, 105.3, 124.9, 169.2, 180.0, 198.0, 215.1, 229.0 ⇒ **14 takes**. The existing global `configs/shots.yaml` threshold (**38.0**, tuned on clip4 — ADR-7) only finds **7** of these ⇒ would wrongly report 8 takes. **All 13 confirmed real** by eyeballing a before/after frame pair at each timestamp (same discipline as ADR-7): every one shows a genuinely distinct venue/lighting/kit change or the intro/outro card boundary — none is a within-take pan/zoom false positive. `configs/shots.yaml` now carries `jordan_thomas_highlight_video: 24.0` as a per-video override — see **ADR-16**. |
| Burned-in red arrow | present intermittently (measured via the existing `find_arrow`, `configs/detect.yaml` thresholds, 2 fps scan): **107 accepted hits** clustered in short bursts inside several takes (~t=30–33, 38–42, 47–49, 64–66, 79–82, 92–95, 107–109, 131–134, 152–155, 170–173, 182–190), not just the "opening seconds of one take" pattern seen on clip2/clip4. Same masking pipeline, no code change needed — it already generalises. |
| Jersey number | **not given anywhere in run metadata.** The intro title card (t≈1 s) *does* show a crisp, unambiguous **"22"** on a club jersey (crest reads "…SOCCER, SINCE 1988") — but that is a **promo photo, not in-game footage**, and the contact-sheet survey (42 frames @ 1-per-6s) shows the player wearing visibly **different kits/colours across different takes** (this is a highlight reel stitched from multiple matches, the same pattern as clip4's 4-venue compilation but more extreme — 8+ segments). The "22" from the title card is logged as **contextual evidence only**; it is **not** auto-applied to any in-game take. See ADR-15. |

This clip is the trigger for ADR-15 (verified-jersey identification, since there is no filename number to use
as even run metadata) and ADR-16 (cut-detection threshold no longer generalizes across a heterogeneous
compilation the way it did across the single-source §3.1 clips).

**Full pipeline run 2026-08-27 (ADR-15/16 built + executed end-to-end):** ADR-16's per-video threshold
correctly reproduced the measured 14-take/13-cut structure. ADR-15's identity verification checked all 13
non-empty takes (take 0, the 3.8s intro card, had zero crops clearing the height threshold) and **verified
zero of them** — every escalated crop got a real, working-model answer (0 VLM failures; see ADR-15's own row
in §5.1 for the fallback-model story and the evidence quotes), never a quota-silenced default. `output/
jordan_thomas_highlight_video/` therefore correctly has an empty `players/` (no folder at all), a fully
populated `identity_report.json` covering all 14 takes, and a full-resolution `original_annotated_video.mp4`
(3840×2160/30fps/234.13s, original AAC audio preserved) showing green boxes on every detected player/
official throughout and an "IDENTITY: unverified in this segment" panel + CUT banner note at every take —
exactly the honest, non-forced outcome CLAUDE.md's own accuracy rule requires when the evidence doesn't
support a claim, not a build failure.

> **This clip is exactly the case ADR-19 (§14) now covers.** Its escalated crops are genuinely unreadable
> (0/14 verified — a well-evidenced negative, not a bug). Under the new flow, the honest next step for a clip
> like this is: the owner watches it and drops a `Jordan Thomas Highlight Video.annotations.txt` sidecar next
> to it; the pipeline then runs from those human-observed events instead of re-deriving an impossible
> auto-identity. The `identity_report.json` "unverified" record is what tells the owner *which* takes need a
> sidecar line.

### 3.4 Third input (pending): a SoccerNet broadcast clip, to isolate "does jersey OCR/VLM work at all
on clear footage" from "this specific amateur footage has no legible numbers" (added 2026-08-27)

§3.3's 0/14 result is well-evidenced, but it's a compound answer: it doesn't separate "the ADR-15
verification *method* doesn't work" from "this *footage* (small/blurry/side-on players, wide fixed amateur
camera) genuinely has nothing to read." Owner asked (2026-08-27) to test on **clear** footage to isolate
which. Professional broadcast (SoccerNet) is the obvious control: tight camera work, players fill much more
of the frame, HD. **Blocked on the owner completing SoccerNet's NDA form** (soccer-net.org — a Google Form
tied to the requester's own identity, not something this pipeline can submit on their behalf); `.env`'s
`SOCCERNET_PASSWORD` is empty until then. `scripts/download_soccernet.py` is built and ready (see §8) —
listing available match names needs no password (the game-list JSON ships inside the `SoccerNet` pip
package itself), only the actual video download does. Default target picked and verified real via that
listing: `england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester`, `1_720p.mkv`,
trimmed to the first 5 minutes by default (§6's "validate on a short clip first," not a 45-minute half).

**Keep the SoccerNet path (owner, 2026-08-27):** the download script and its cached clip stay in the repo as
the project's **good-quality control input** — "if we need high-quality video we can take it." When the auto
flow (§14) meets a *good-quality* clip it runs the full auto jersey-read (SoccerNet-trained detector, ADR-8,
+ the ADR-15 OCR/VLM identity stack); the SoccerNet clip is what proves that path actually reads numbers when
the footage supports it, cleanly separated from the amateur-footage legibility ceiling.

**Important scope note, not yet resolved:** SoccerNet broadcast video carries no burned-in arrow (that
graphic is specific to the owner's own Veo exports, §3.2 consequence 1) and this clip's filename won't carry
a jersey number either (same filename-less path as §3.3, ADR-15 applies) — so there is no "the target
player" for this clip unless the owner supplies one via `--target-jersey` (ADR-18), only "whichever track
`heuristic_fallback_seed` locks onto." That's fine for answering the narrow legibility question this test
exists to answer, but it means this run is a capability check on the OCR/VLM stage, not a meaningful
end-to-end "did we correctly identify a named player" run — don't over-read a verified number here as
validating the *selection* heuristic, only the *reading* one.

### 3.5 Owner update 2026-08-27 (this drove ADR-19 + ADR-20 + §14 — the single auto flow)
The owner restated the end-to-end goal and added two capabilities. Verbatim intent, mapped to what already
exists vs. what ADR-19/20 add:

| Owner ask (2026-08-27) | Status |
|---|---|
| Track the target player **by jersey number**; box them with the number written on the box | ✅ existing (red box `#<N> \| TARGET`, ADR-14/§13.1); jersey from filename / `--target-jersey` / verified-ID / **now also manual-annotation sidecar** (ADR-19) |
| **Track the ball** in the same output video | ✅ existing ball marker (Stage 2 SAHI, §13.1) — solid when observed, hollow/dashed when interpolated |
| Per-player **stat card** with a **timestamped** timeline (passes at each time, then the goal) | ✅ existing `statcard.md` template (§13.2); in manual mode the timeline **is** the parsed sidecar (ADR-19) |
| **Highlight reels** per category (passes, goals, assists, …) | ✅ existing per-category compilations (§13.4) |
| Output **folder named by the player number** with highlights + stat card + overlay video | ✅ existing `output/<slug>/players/player_<N>/…` (§13.5) |
| **SoccerNet** jersey detector when quality is **good**; keep the downloaded SoccerNet video for high-quality tests | ✅ existing detector (ADR-8) + OCR/VLM identity (ADR-15); download script kept (§3.4, §8) |
| When quality is **bad**, the client supplies **manual annotations** (`Min MM:SS player #N in white …`) — run from those | 🆕 **ADR-19** — manual-annotation sidecar, authoritative event source (§14) |
| **Pass** = target passes to the **same jersey colour**; to the **other** side = not a completed pass; the pass **before a goal** = **assist**; ball into the **"white box"** = **goal** | 🆕 **ADR-20** — team-colour-aware pass/turnover/assist + human-marked goal region |
| **Everything runs auto**, correct on the first flow | 🆕 Golden Rule 9 + §14 (one `make run`, auto-branches per input) |
| **Plan with Opus, execute with Sonnet** | ✅ existing convention (§10, `opusplan`) — reinforced in the Claude Code prompt |

**Manual-mode reference input (2026-08-27):** the client's own worked example is
`https://www.youtube.com/watch?v=wzvM66c8ZmE` with the sidecar lines:
```
Min 26:56 player #2 in white takes a touch
Min 26:58 player #2 in white makes a pass
Min 27:28 player #2 in white kicks the ball out of bounds
```
Downloading YouTube video is out of scope for the pipeline itself (rights/tooling); the owner places the
downloaded file in `input/` and its `.annotations.txt` sidecar beside it. This exact three-line example is the
first acceptance test for the ADR-19 parser (§14.3).

## 4. Data contracts (`src/common/types.py`, pydantic)
Stages compose through these; intermediate artifacts cache to disk (parquet/JSON + video) so stages run independently.
- `Frame(index, t, path)`
- `Detection(bbox, cls, conf)` · `BallDetection(bbox, conf, interpolated: bool)`
- `PitchHomography(matrix, keypoints, conf)` — pixels ↔ pitch metres
- `Track(id, take_id, boxes[], team?, jersey_number?, id_confidence, path_pitch_xy[])`
- `Event(type, t_start, t_end, player_track_id?, confidence, source, evidence{})` — `source` = how it was
  derived (e.g. `scoreboard_delta`, `possession_heuristic`, `gemini_vlm`, **`manual_annotation`**,
  **`goal_region`**); `evidence` carries the full traceable trail (Golden Rule 5)
- `Clip(event_id, t_start, t_end, take_id, rank_score, path)`
- `PlayerStats(player_ref, counts{...}, confidences{...})`
- `Annotation(t, jersey_number, team_colour, action_phrase, event_type, raw_line)` — **ADR-19**; one parsed
  sidecar line; maps 1:1 to an `Event(source="manual_annotation", confidence=annotation_confidence)`
- `TakeIdentityResult(take_id, jersey_number|None, status, confidence, evidence_frames[], human_override_note?)`
Rule: no stage invents fields it can't justify; unknowns are explicit (`None` + confidence), never guessed.

## 5. Architecture (build order; models are current recommendations — verify latest before committing)
| Stage | Job | Use | Notes / license |
|---|---|---|---|
| 0 Ingest | decode, sample fps | `ffmpeg` (NVDEC), PyAV | per-stage sampling; 90min@25fps ≈ 135k frames |
| 0.5 Source profiler | detect input type | cuts (PySceneDetect) + camera-motion + resolution → run profile | `broadcast` / `single-static` / `single-panning`; broadcast path is a **superset** (single-cam = one long take) |
| 0.6 Annotation ingest *(ADR-19)* | parse manual sidecar if present | `src/annotations/parse.py` + `configs/annotations.yaml` phrase map | **if `input/<video>.annotations.txt` exists ⇒ manual-events mode: sidecar is the authoritative event source (§14.1); no new deps (stdlib `re`)** |
| 1 Temporal structure | takes, replays, shot type, clock | **PySceneDetect** (→ TransNetV2), replay + shot-type classifier, **PaddleOCR/EasyOCR** | **only when profile=`broadcast`; auto-skipped for single-camera**; shot-boundary always runs (ADR-7); protects everything downstream |
| 2 Perception | players/ball/refs, pitch | **RF-DETR** (Apache, ADR-8), **SAHI** tiling / **TrackNetV3** ball, pitch keypoints + `supervision` ViewTransformer | ball is the weak spot; homography enables speed |
| 3 Track + team | IDs within take, team-**colour** | **`supervision` ByteTrack (MIT)**, associate in pitch coords; **torso-colour CIELAB KMeans** team (ADR-12) | reset IDs at cuts; **team colour is what ADR-20's pass/turnover rule keys on** |
| 4 Identity *(Phase 2 slice: ADR-15/18/19)* | who is #N | filename / `--target-jersey` / **manual-annotation sidecar** (human-given ⇒ verified, ADR-19) → else **ViTPose**→legibility→**PARSeq/EasyOCR**→tracklet vote + **Gemini VLM** fallback; fuse #+team+ReID+position → `id_confidence` → **HITL** | human-given number is strongest evidence (GR4); ~87% per-tracklet ceiling for auto reads; route low conf to human/sidecar |
| 5 Events | actions | **manual sidecar (ADR-19, authoritative when present)**; else P1: goals (scoreboard Δ / **marked goal region, ADR-20**), shots (heuristic), sprints (homography+speed), **passes/turnovers (team-colour, ADR-20)**, touches/tackles/saves/dribbles (ADR-13), celebrations/key-moments (Gemini, ADR-14); **assist = pass-before-goal to a teammate (ADR-17/20)** | ~60 mAP@1 SOTA for ball actions; each heuristic honestly flagged |
| 6 Assemble | rank, cut, reel, stats | rules score + optional VLM; `ffmpeg` cut (snap to takes, dedupe); **per-category highlight reels cut from the annotated render** (ADR-18); stat card w/ confidence + click-to-clip | user picks clip count/length |

## 5.1 Decision log (ADRs — deviations from the original brief, with reasons)
Anything here overrides the brief's default suggestion. Add a row whenever a recommendation is changed.

> **Two-tier log.** The table below is the **working summary** — kept lean because this whole file is read at
> the start of every session. The **full verbatim rationale + run-logs** for every ADR live in
> [`docs/adr-evidence.md`](docs/adr-evidence.md), anchored by number (`#adr-15`, `#adr-17`, …). Rows that were
> condensed carry a `→ full evidence` link. **When you condense a new ADR row here, move its verbatim text into
> `docs/adr-evidence.md` in the same commit** — same discipline as keeping this file current.

| # | Decision | Why | Status |
|---|---|---|---|
| ADR-25 | **The gateway PUSHES the code to a booting pod through RunPod's proxy; a pod never calls the VPS. A pod that stops itself mid-boot is shut down and reported, never started again** — owner, 2026-09-30: a run sat at "GPU pod is running; app starting (7 min so far)" on an A100 ($1.59/h), "burning my dollars" | The first real GPU run of ADR-24. Root-caused from RunPod's own logs (`GET api.runpod.io/v2/pods/{id}/logs`) + the VPS: the pod never made a single request to the gateway (no `/pod/*` line in nginx, no half-open TCP connection on :443), and **the VPS cannot open a TCP connection to the pod's public IP either (15 s timeout) while a Mac reaches it instantly** — Hostinger's network and RunPod EUR-IS-1 simply do not connect. So ADR-24's `curl $PV_GATEWAY_URL/pod/bootstrap.sh` hung 6 × ~127 s (kernel SYN timeout → curl 28 → `--retry 5`), the pod stopped itself at 13.5 min (system log: container stopped 20:14:12), and the gateway — seeing a stopped pod while `wanted` — **started it again, which also reset the boot cap**: an endless 14-min stop-self/restart loop on the bill. Nothing else in the design needs pod→VPS: uploads, status, results all go gateway → `<id>-8000.proxy.runpod.net`. **Now:** the start command execs `docker/pod_receiver.py` (stdlib, shipped zlib+base64 in the pod env `PV_POD_RECEIVER`, ~2 KB) which waits on :8000; `PodManager._deliver_code` probes `GET /pv/code` every boot poll and POSTs the gateway's `git archive` (`main.code_bundle`, ~1 MB, cached per commit) with `X-PV-Pod-Key`; the pod writes it, runs the bundle's `runpod_bootstrap.sh` with `PV_CODE_TGZ` (same `.pv-manifest` logic as before). No code within `PV_CODE_WAIT_S` (15 min) → the receiver exits 1 → stop-self. A refused/unpackageable bundle → ERROR + shut down. **A pod this start has seen RUNNING and then finds stopped → ERROR + terminate, never `start_pod`** (the loop above). The `/pod/bootstrap.sh` + `/pod/code.tar.gz` routes, `PV_GATEWAY_URL` and the gateway's `GITHUB_TOKEN` pass-through are gone. Verified: the real start-command string run under bash on localhost with a real push (`tests/test_pod_receiver.py`), 5 state-machine tests incl. the stop-self loop (fails with the guard removed), 759 tests pass. ⚠️ Not yet exercised on a real GPU pod at commit time. | ✅ implemented 2026-09-30; real-pod boot pending |
| ADR-24 | **The GPU is the user's to switch on and off (Start GPU / Stop GPU); the gateway never starts it, only stops it (idle). Pods fetch their code from the gateway, not GitHub** — owner, 2026-09-26: "burning the RunPod dollars", "don't auto start the GPU", a 10-min video that never completed | Root-caused from the VPS journal + RunPod API: the fine-grained GitHub PAT the pods used **expired (~2026-09-24)**, so every new pod died on its first `curl`; the gateway's "boot cap exceeded → replace it" loop recreated/kept pods for **~13 GPU-hours with zero output** (one pod RunPod refused to DELETE, "Pod is locked", billed ~11 h), until the balance hit **-$0.04** — after which RunPod's "balance too low" refusal was shown as "No GPU available" and a queued 10-min job retried every 60 s for 30 h. **Now** (`apps/gateway/pod_manager.py`): `wanted` is set only by `start()` (the Start GPU button, `POST /api/pod/start`); any failed start clears it; nothing else creates/starts a pod (the old `request_wake` on results/media/frame-picker is gone, Run is refused while off, queued jobs wait for free). A boot past `boot_cap_s` (20 min) is **stopped + terminated and reported, never recreated**; `_shut_down` stops before deleting because RunPod will not delete a locked pod. Waiting for stock is capped (`max_gpu_wait_s`, 15 min). A low balance is an immediate ERROR. `stop()` (Stop GPU / `idle_check`, `PV_GPU_IDLE_MINUTES` default 10) cancels the start in progress and stops+terminates. Jobs can be cancelled, and a cancel (or Stop GPU) stops the pipeline ON THE POD too: `POST /api/jobs/{id}/cancel` raises `JobCancelled` (a BaseException, so the pipeline's broad `except Exception` blocks cannot swallow it) inside the pipeline thread via `PyThreadState_SetAsyncExc`; the video's in-flight slot is released. A pod also STOPS ITSELF when its bootstrap fetch/bootstrap/app fails (start command `fetch && bash || stop-self` via `RUNPOD_POD_ID` + `RUNPOD_API_KEY`), so a broken pod cannot crash-loop on the bill even with the gateway down. Long videos: `build_take_kit_colour` and the scoreboard-OCR goal scan now STREAM frames (they held every native frame of a take -- a single-take 90-min 1080p match was ~67 GB at 2 fps); results are pinned identical by `tests/test_long_video_memory.py`, and with no scoreboard the OCR stops decoding after the activation frames. The pod bootstrap and a `git archive` of the VPS checkout are served at `/pod/*` behind an HMAC key; `deploy_vps.sh` ships a `git bundle` over SSH — no token anywhere in the path. Long videos: the pod forwards its pipeline's INFO lines into the job log (the staged messages all fire in the first seconds, so long runs looked hung) and refuses a run with < 3 GB free. The panel shows card, $/h, auto-stop countdown and RunPod balance (GraphQL `myself`). Header/login crest: two-tone (white ring text, green ball). ⚠️ Not yet exercised on a real GPU — the account balance was negative this session; verified with unit tests + a simulated RunPod in the browser. | ✅ implemented 2026-09-26; real-pod run pending funds |
| ADR-22 | **A pass needs a verified kick, not a long flight; the target's passes are attributed by verified raw track id** — owner-reported 2026-09-19 "not counting the passes correctly" | Traced in code (no cached run was reachable — the pod was offline). Three defects, all in the PASS path: **(1) short passes were uncountable.** `pass_min_flight_s` (0.25 s, ADR-20 era flicker fix) requires the ball to sit outside BOTH players' `possession_max_dist` zones; two players 5–8 m apart have zones that nearly touch, so a real short pass is "in flight" for 1–3 frames and was dropped as flicker — and `pass_min_travel_px` (60 px, measured zone-edge to zone-edge) killed the same passes a second way. Fix: `detect_passes` now takes the per-take `ball_states` and asks whether the passer actually PLAYED the ball — `has_contact_evidence` at the end of run A (the touch detector's own measured velocity-change test) or a run A at least `possession_min_duration_s` long. A **verified kick overrides both floors** (jitter and flicker never produce a velocity change); no kick AND no possession = a **fly-by** (ball rolling past a third player on its way to the receiver, which used to credit that player with a pass) → `pass_passer_never_played_ball`, logged. Callers without `ball_states` keep the exact old behaviour. **(2) an undocumented second distance cap:** `pass_confidence`'s closeness term reached 0 at `pass_max_dist`, so `closeness × quality` fell under `min_emit_confidence` at ~550–590 px although the explicit gate is the measured 750 px. New `pass.distance_closeness_floor: 0.5` (⚠️ reasoned): distance now only lowers confidence; `pass_max_dist` alone vetoes. **(3) attribution lost the target's passes** whenever `build_take_identities` (earliest-first greedy seeding) split the target across two chains — `target_identity_id`'s majority vote kept one. `attribute_events_to_target` now judges every event that carries the acting RAW track id (pass/turnover/touch/tackle/possession/dribble evidence) against the human-verified `location_track_ids` directly, and identity-only events against EVERY chain a location fragment maps to. Nine new tests (`tests/test_possession.py`, `tests/test_aggregate.py`); 643 existing pass. ⚠️ Not yet re-run on the owner's real clips (GPU pod offline this session) — the next real run's `run_report.json` drop counters (`pass_flight_too_short`, `pass_passer_never_played_ball`) are the check. | ✅ implemented + unit-tested 2026-09-19; real-footage recheck pending |
| ADR-23 | **Product name "The Reach Vision"; brand green/black/white UI, dark only (light theme + toggle removed 2026-09-23 at owner request; header crest now sits on black, not green); user-selectable GPU tier with live RunPod pricing** | Owner (2026-09-19) supplied the College Soccer Reach crest (green `#24BB61`) and asked for green/black/white, a phone/tablet/desktop layout, white by default with a dark mode, and a choice of three GPUs with their RunPod per-hour cost. `apps/web/` rebuilt: one token stylesheet (`data-theme` on `<html>`, remembered in `localStorage`, light default — `prefers-color-scheme` deliberately ignored), mobile-first breakpoints (768/1100), 44 px targets, logo/favicon extracted from the owner's PNG into `apps/web/assets/`. The old "2D tactical pitch" card was removed: `/api/results` returns `pitch_trajectory: null` by design (ADR-6), so it only ever drew an empty pitch — replaced by an event-breakdown chart from the real timeline rows. **GPU tiers** (`apps/gateway/runpod_pods.py: GPU_TIERS`): Budget RTX 4000 Ada (~$0.28/h), Standard RTX 4090 (~$0.74/h, default), Pro A100 80 GB (~$1.59/h) — the three CUDA-12.4-compatible classes RunPod stocks in EUR-IS-1 (Blackwell excluded). Prices are never hard-coded: `GET /api/gpus` asks RunPod's GraphQL `gpuTypes` for the live Secure Cloud price and per-datacenter stock (the REST API has no catalogue endpoint), cached 5 min. A job carries `gpu_tier`; `PodManager.ensure_online(gpu_types=…)` replaces a pod that sits on a card outside the chosen tier (volume untouched) — but a plain wake-up never swaps a healthy pod. Internal identifiers (`pitchvision` pod/volume/systemd/paths) intentionally unchanged. 9 new gateway tests. | ✅ shipped 2026-09-19 |
| ADR-21 | **Jersey OCR reads a tight upper-torso NUMBER REGION, not the full-body crop — the real root cause of the owner-reported "player 8 detected as player 10"** | Owner reported repeatedly (2026-08-27 → 08-31) that jersey numbers were wrong, and specifically that "#8 is detected as #10". Root-caused 2026-09-01 by **measurement, not assumption**. **Mechanism:** PARSeq resizes every input to **128×32** (4:1, text-line shaped — `strhub`'s own `img_size`). We were handing it the **full-body player crop**, which is tall and narrow (~1:2.5 at a real 80px box), so the digits were squashed vertically into illegibility. PARSeq then returned a **confident but wrong** read — and, critically, one that was **CONSISTENT for the same track across frames**, so the existing temporal-agreement rule could not filter it (**agreement measures consistency, not correctness**). The failure was also **systematically biased toward "10"**, which is exactly why one player after another came out as the owner's target number. **Measured on real 720p broadcast** (`video2` take 0, t=40–61s, five tracks whose numbers I confirmed **by eye** from upscaled crops — #8, #4, #30, #19, #37): full-body crop **0/5 correct** (four of five collapsed onto "10"); tight number region **5/5** in isolation, **4/5** through the full production path. Scanning the take's 13 longest tracks: **before**, 12 tracks verified as just **1 distinct number** (nine of them "#10" — a physically impossible result); **after**, **10 distinct numbers across 13 tracks**. **The fix is a crop-geometry split, not a new model, not retraining, and not a threshold re-tune:** the **legibility gate keeps seeing the FULL-BODY crop** (mkoshkina's resnet34 was trained on whole-player images — measured: tight number crops pass it **0/743** vs **336/743** for full-body), while **PARSeq reads the tight number region** (`src/identity/jersey_parseq.py::number_region`, insets in `configs/identity.yaml: parseq_soccernet.number_crop`, no magic numbers in code per §10). Gate on the body, read on the number. Applied at **both** call sites — `src/identity/verify.py` (ADR-15 per-take verification) and `src/annotations/associate.py` (ADR-19 manual-mode re-identification). **This supersedes the owner's own proposed fallback of fine-tuning on the SoccerNet jersey dataset** (2026-08-31): the checkpoint in use *is already* SoccerNet-fine-tuned, so training a second one would have inherited the identical input-geometry defect and most likely reproduced the same failure at much greater cost — the model was never the problem, the crop was. **Known residual, stated honestly (Golden Rule 5):** one confirmed error survives (track 189, truly #19, read "10" at only **36%** agreement) and two numbers are still claimed by more than one track. Both are weak-plurality artifacts, addressed separately by the agreement-floor + per-take uniqueness work; the 4K amateur clips remain a genuine legibility ceiling (`clip1_43`: **0/801** crops pass the gate — an honest negative, not a regression). 4 new unit tests. | ✅ root-caused + fixed + verified on real footage 2026-09-01 |
| ADR-20 | **Team-colour-aware pass / turnover / assist, and a human-marked goal region for the owner's "ball into the white box = goal"** | Owner (2026-08-27) restated the event semantics explicitly: a **pass to the same jersey colour is a completed PASS**, the ball going to the **other** colour is **not** a completed pass, the **pass immediately before a goal (to a teammate who scores) is an ASSIST**, and the ball going into the **"white box" (goal mouth) is a GOAL**. Three of these already had seams (ADR-13 pass heuristic carries `receiver_identity`; ADR-17 assist = nearest preceding teammate PASS to the scorer); this ADR makes the **team-colour test first-class** and adds the missing pieces honestly. **What's built:** (1) **PASS vs TURNOVER** — a possession-change from the target to a next-receiver of the **same** ADR-12 colour cluster ⇒ `EventType.PASS` (counted); to a **different** colour cluster ⇒ new `EventType.TURNOVER` (**logged with evidence, NOT counted as a pass** — Golden Rule 5: a lost ball is not a completed pass, and hiding it would be a silent filter). Out-of-play losses map to `EventType.OUT_OF_BOUNDS`. (2) **ASSIST** — unchanged rule from ADR-17, now explicitly colour-gated: a `PASS` (same-colour, by definition) whose receiver becomes the credited scorer within `assist_window_seconds` ⇒ `EventType.ASSIST` to the passer. (3) **GOAL via "white box"** — deliberately **not** a new trained net/goal detector (Golden Rule 2) and **not** a fragile "ball bbox vanishes in a corner" guess (Golden Rule 5) — the two shortcuts ADR-17 already rejected. Instead, a **human-marked goal-mouth region** per static single-camera clip in `configs/goal_region.yaml` (normalised polygon, one or two regions per clip), exactly the human-in-the-loop, no-training pattern as ADR-3's 4-point calibration: the ball centroid entering a marked region within a shot window ⇒ `EventType.GOAL` candidate, `source="goal_region"`, flagged low-confidence with `goal_region_source: manual` in evidence. Goals therefore now have **three** honest sources — manual annotation (ADR-19, authoritative), scoreboard delta (ADR-17, broadcast), and marked goal region (this ADR, static cams) — and remain **`not available`** when none of the three is present for a clip (still true for the §3.1 clips until a region is marked). **Confidence stacks DOWN** the chain (goal occurrence > scoring team > scorer > assist), evidence carries the full trail. **Config (`configs/events.yaml` additions):** `pass.same_colour_required: true`, `pass.colour_cluster_margin` (reuse ADR-12 distance), `turnover.enabled: true`, `goal_region.enabled: true`, `goal_region.shot_window_s: 3.0`, `goal_region.confidence: 0.45` (below scoreboard's 0.75 — a marked region is a strong prior but not a read score). `configs/goal_region.yaml` is empty by default (⇒ region path simply inactive, honest "not available"), populated per clip the owner chooses to mark. **Verified 2026-08-28**: 40 dedicated unit tests (`tests/test_possession.py`/`tests/test_aggregate.py`/`tests/test_goals.py`) cover the PASS/TURNOVER branch (same-cluster, opposing-cluster, low-confidence-neither, config toggles), the tackle/turnover double-count suppression, and the goal-region detector (debounce, shot-window gating, no-polygon "not available"); a real cold run of `clip1 43.mp4` (own `work/clip1_43/`, 52 tracks, no cache) confirms the wiring is unchanged for the original clips — goals still report "not available" (now naming both the scoreboard scan and the goal-region check), selection/reel/stats all matched prior behaviour. ⚠️ **Not yet demonstrated on real broadcast footage**: the one thing this ADR still needs is running the colour-aware PASS/TURNOVER split against the SoccerNet clip's own real team clusters (CLAUDE.md §3.4 — the only footage in this repo where `team_confidence` is measurably usable, unlike the owner's own amateur clips). Attempted 2026-08-28 but the SoccerNet clip has no cached `work/` (the environment's cache was cleared since ADR-15/17/18 were verified) and a cold RF-DETR+SAHI run on it exceeded this session's available run time across three sized-down attempts (300s full clip, a 30s trim, a 6s trim) — **deferred per explicit owner instruction mid-session ("don't process the soccernet yet")**, not a code defect. `input/soccernet_20160924__1430_manchester_united_4__1_leicester.mp4` and `work/soccernet_...` were left untouched. | ✅ implemented + unit-tested 2026-08-28; SoccerNet real-data colour demo deferred per owner |
| ADR-19 | **Manual-annotation sidecar = authoritative event source for low-quality footage; the sidecar file winning is the auto trigger for manual mode** | Owner (2026-08-27): when footage is too poor for auto jersey/action reading (the §3.3 0/14 case is the canonical example), the client watches the video and supplies timestamped annotations — `Min 26:56 player #2 in white takes a touch` etc. — and the pipeline must run from those. **Chosen trigger (owner-confirmed):** *sidecar file wins, automatically.* If `input/<video_basename>.annotations.txt` exists next to the input, the run enters **manual-events mode**: the parsed annotations are the authoritative `Event` source (Stage 5 auto-detectors are skipped for event *generation*), and the target identity (jersey number + team colour) is taken **directly from the annotations** — a human directly watching the footage is the strongest identity evidence there is (Golden Rule 4, same logic as ADR-18's `human_confirmed_jersey`). No sidecar ⇒ full auto (§14.1). This is not a quality-score gate and not a per-video config flag — the *presence of the client's file* is the signal, so the client stays in control of which clips are manual. **Parser (`src/annotations/parse.py`):** tolerant regex over the client's natural phrasing; grammar + phrase→`EventType` map in `configs/annotations.yaml` (no magic strings in code, Golden Rule / §10). Each line → an `Annotation` → an `Event(source="manual_annotation", confidence=annotation_confidence` (default **0.95** — human-observed but transcription-fallible, per `configs/annotations.yaml`)`, evidence={raw_line, jersey_number, team_colour, action_phrase})`. **Unmapped action phrases are logged and kept** as `EventType.KEY_MOMENT` with the raw text in evidence — never silently dropped (§10). **Overlay in manual mode:** detection+tracking still run best-effort so boxes + ball marker are drawn where confident; the target's red box is associated by jersey-colour + timing to a track where possible and labelled with the annotation's number; **regardless of tracking quality, each annotated event's caption (name + timestamp + jersey #) is burned at its timestamp** so the annotated video is always faithful to the client's ground truth. The run report states plainly when a red box could not be reliably associated to a track in a given window ("event shown by caption; track association low-confidence"), never a fake box. **Statcard identity status** in manual mode reads `Human-provided (manual annotation)`. **No new dependency** (stdlib `re` + existing YAML). Same owner-authorized pull-forward pattern as ADR-13/15 — Golden Rule 7 is superseded *for inputs the client explicitly annotates*, not relaxed globally. **Verified 2026-08-28**: parser acceptance test #1 (the three canonical lines parse to exactly `[TOUCH@1616.0s, PASS@1618.0s, OUT_OF_BOUNDS@1648.0s]`, a malformed line lands in `problems`, an unmapped phrase becomes `KEY_MOMENT`), plus 8 tests for `src/annotations/associate.py`'s colour-matching (reused `src/team/classifier.py` torso/CIELAB code verbatim) and a `src/pipeline/manual_events.py` wiring smoke test (acceptance #2 — sidecar parsed, bucketed by take, grouped by jersey number, `write_player_output` called with `identity_status="Human-provided (manual annotation)"` and an explicit, non-blank `goal_reason`) with every I/O-heavy boundary mocked, per the plan's own scoping. The real end-to-end run on the owner's own annotated video is still pending — three candidate videos with real annotation-style text appeared in `input/video1-3/` mid-session, but their phrasing (`Min 0:15 — Player in white #11 takes a touch`, em-dash-separated, colour before the jersey number, multi-word colours like "light blue", some lines missing the word "player" entirely) does not match this parser's grammar 1:1 — broadening the grammar to cover that real phrasing is flagged here as follow-up work, not silently attempted mid-task. | ✅ implemented + verified 2026-08-28 (parser + wiring); real client-video grammar coverage pending |
| ADR-18 | **Spec-completeness audit against the owner's full 29-section jersey/event spec (2026-08-27) found three real gaps, fixed alongside ADR-17** | Owner asked to confirm "everything I told you in the prompt" is actually wired into the pipeline/statcard, ahead of manually confirming a jersey number for the incoming SoccerNet clip (§3.4). Audited section-by-section against the actual code (not assumed) and found: **(1) no human-confirmed-identity path existed.** `target_jersey` could only ever come from the `clip<N> <jersey>` filename regex — a filename-less video (the owner's own stated new workflow: watch the clip, then tell the system the number) had no seam to receive that. Fix: `main()` gains a `--target-jersey` CLI override for filename-less videos; `src.identity.verify` threads an optional `human_confirmed_jersey` through aggregation — a human-given number needs only ONE non-contradicting read to verify (a human directly watching the footage is stronger evidence than an independent OCR/VLM multi-frame vote, Golden Rule 4), but a read that ACTIVELY DISAGREES with the human's number is surfaced, never silently suppressed (Golden Rule 5 — disagreement is itself evidence worth keeping). **(2) highlight-category clips were cut from the RAW input, not the annotated render** — found by tracing `write_player_output`'s `video_path` argument back to `run_extended_pipeline_for_video`'s own raw-file parameter; `render_full_annotated_video` didn't even run yet at the point highlights were cut. Directly contradicts owner spec §24 ("highlight videos should also show RED/GREEN boxes, ball tracking, event name, timestamp, jersey number"). Fix: reorder so the full annotated render happens BEFORE the per-player loop, and highlight-clip cutting reads from `original_annotated_video.mp4`, not the source. **(3) green-box labels read a bare `#{track_id}`** — indistinguishable at a glance from a jersey number, undermining the owner's own explicit §7 requirement ("track ID and jersey identity are different concepts... do NOT assume Track ID 10 = Jersey #10"). Fix: green boxes now read `ID: {track_id}`; the target's red box reads `#{jersey_number} | TARGET | ID: {track_id}` — both identifiers visible, neither ambiguous. **Not a gap, verified correct:** the live stats panel already shares the exact same per-player `Event` list used to write `statcard.md`. **Verified 2026-08-27** with a full, real re-run for `jordan_thomas_highlight_video` (1123s total, exit 0). → [full evidence](docs/adr-evidence.md#adr-18) | ✅ implemented + verified 2026-08-27 |
| ADR-17 | **Finish real goal detection (scoreboard-delta) + best-effort scorer/assist attribution, self-gated on actually finding a legible scoreboard — never on the `profile` label** | Owner asked (2026-08-27) to "do something about goal and assist" rather than leave `detect_goals_scoreboard_delta` a permanent `NotImplementedError` stub. §3.2(3)/§3.4 established goals are a genuine **data ceiling** on the owner's own footage (no scoreboard exists), but the incoming SoccerNet broadcast clip (§3.4) is exactly the input this path was always meant for. **Scope of what's real, in order of confidence:** (1) **goal occurrence** — OCR-scan a small set of CANDIDATE scoreboard regions across several early sampled frames; a region only "activates" if it yields a STABLE `digit [sep] digit` pattern across multiple frames (measured self-check, not a hardcoded ROI or the `profile` label). A goal event fires on a debounced digit increment. (2) **scoring team** — credit whichever ADR-12 colour-cluster held possession in the window before the digit changed (sidesteps home/away). (3) **scorer** — the identity with the last possession run before the goal, gated to the team from (2). (4) **assist** — new `EventType.ASSIST`; nearest preceding teammate `PASS` to the credited scorer within `assist_window_seconds`. Confidence driven DOWN at each step; full `evidence` trail. **Deliberately NOT built (then):** visual goal-line/net detection (owner's "white box" framing) — needed a new trained class or panning-camera calibration. *(ADR-20 now supplies the honest version of "white box = goal": a human-marked static-camera goal region, no training.)* **Still "not available" on the §3.1 clips + `jordan_thomas_highlight_video`** because the self-gating scan finds no stable scoreboard there. **Verified 2026-08-27** against all 6 clips' cached `work/<slug>/` artifacts. → [full evidence](docs/adr-evidence.md#adr-17) | ✅ implemented + verified 2026-08-27 |
| ADR-16 | **Cut detection: fixed-threshold `ContentDetector` does not generalize across a heterogeneous multi-match compilation; per-take identity now depends on getting this right** | §3.3 measured that `Jordan Thomas Highlight Video.mp4`'s real cuts sit at content-value scores as low as ~21–27 (a 13-cut set, stable across thresholds 21.0/24.0/27.0), while clip4's three confirmed **false**-positive foreground-crossings score up to 32.0. No single fixed threshold keeps clip4's false positives out *and* catches this new video's real cuts. **Resolution:** `configs/shots.yaml: scenedetect` gains an `overrides: {<video-slug>: <threshold>}` map; `detect_takes()` looks itself up there and falls back to the global `threshold` when absent — the original 5 clips' behaviour is provably unchanged. `jordan_thomas_highlight_video: 24.0` verified by contact-sheet frames. `AdaptiveDetector` deferred. → [full evidence](docs/adr-evidence.md#adr-16) | ✅ adopted 2026-08-27 (per-video override) |
| ADR-15 | **Verified-jersey identification required when no filename jersey number exists; no verified number ⇒ no target identity ⇒ no target stats for that take** | Owner request (2026-08-27): for a video with no `clip<N> <jersey>` filename metadata, the target player's jersey number must be **read from visual evidence and confirmed**, never guessed from appearance, position, team, or the burned-in arrow alone. Owner-authorized pull-forward of a Phase-2 slice for this input only. **Pipeline addition:** per **take**, take the arrow-hint track as a **location candidate only**; crop it at native/high res where front/back-facing and large enough; run **EasyOCR** + **Gemini** as a cross-check on the same crops; require **temporal agreement across multiple frames**. Take identity record `{take_id, jersey_number|None, status, confidence, evidence_frames[]}`. **Unverified take ⇒ no red box, no target stats** (green boxes + ball still render). Output grouped by verified number, not an assumed single identity. Gemini resolved to a working model with an ordered fallback list + quota handling. **Built + run end-to-end 2026-08-27: 0/14 verified on `jordan_thomas_highlight_video`** — a genuine, well-evidenced negative (crops truly unreadable), not a quota-silenced default. *(ADR-19 is the honest recovery path for exactly this outcome: the owner annotates the clip.)* → [full evidence](docs/adr-evidence.md#adr-15) | ✅ built + run 2026-08-27 |
| ADR-14 | **Full extended-output spec: exact statcard.md format, full-resolution annotated original video, per-category highlight reels, Gemini for celebrations** | Owner supplied a complete spec (2026-08-25) — see **§13**. (1) primary output = the **original video at original resolution/fps/duration/audio**, annotated in place; (2) `statcard.md` uses the owner's exact template with `uncertain` wherever a stat can't be reliably derived; (3) category-specific highlight compilations, empty (not fabricated) when a category never occurs; (4) celebrations/key-moments via **Gemini** on cheaply pre-filtered candidate windows only (never every frame). → [full evidence](docs/adr-evidence.md#adr-14) | ✅ owner-authorized 2026-08-25 |
| ADR-13 | **Owner explicitly authorized best-effort touch/pass/tackle/save heuristics ahead of Phase 1 completion** | Asked (2026-08-25): owner chose "best-effort heuristics now". Scope: touch/pass/tackle/save-for-GK detectors, **no training, no new pretrained model** — proximity/possession-change/speed heuristics on top of existing detections+tracks, each with a low, honestly-stated confidence ceiling. Superseded by ADR-14 on celebration scope; **extended by ADR-20** on the pass/turnover colour test. → [full evidence](docs/adr-evidence.md#adr-13) | ✅ owner-authorized 2026-08-25 |
| ADR-12 | **Team assignment = torso-colour clustering, NOT SigLIP, on this footage** | Measured on `clip2`, SigLIP failed twice (48/3 then 27/2/18-NaN). Torso-band CIELAB KMeans k=2 gave 24/15 with a large L\* separation (dark-blue vs white kit). **Resolution:** `team.method: colour` (default), SigLIP retained as config-selectable fallback for real broadcast. **Cluster per take, not globally.** **ADR-20's pass/turnover/assist logic keys directly on this colour cluster** — "same jersey colour" = same cluster within the take. → [full evidence](docs/adr-evidence.md#adr-12) | ✅ adopted |
| ADR-11 | **Downstream stages must consume the *motion score*, not the static/panning *label*** | Stage 0.5 put 4 of 5 clips inside the ambiguous band (1.5–6.0 px/frame). ADR-3's homography cadence reads `motion_score` directly; recalibration interval scales continuously with measured motion, settling window excluded. | ✅ adopted |
| ADR-10 | **Expect a real domain gap; do not trust the published 0.857 mAP on this footage** | ADR-8's checkpoint was trained on professional broadcast; §3.2 footage is youth/amateur Veo. Measure mAP on *our* labeled slice; treat Roboflow CC BY 4.0 set as a second eval set. | ✅ adopted |
| ADR-9 | **⚠️ SoccerNet-derived weights are dev-only until licensing is cleared** | ADR-8's checkpoint is *declared* Apache-2.0 but trained on SoccerNet-Tracking-2023 (research/education only, §7). Use for Phase-1 dev/eval; before any commercial ship take one of: SoccerNet commercial terms, fine-tune on CC BY 4.0 Roboflow set, or fall back to RF-DETR-COCO. | ⚠️ open — revisit before ship |
| ADR-8 | **Phase-1 detector = `julianzu9612/RFDETR-Soccernet` (Apache-2.0), with RF-DETR-COCO as fallback** | The three canonical Roboflow soccer projects host no trained weights (only CC BY 4.0 data). On HF, `julianzu9612/RFDETR-Soccernet` is Apache-2.0, RF-DETR-Large, classes `ball, player, referee, goalkeeper`, mAP@50 0.857 on SoccerNet. `OrbitalLab/mova-rfdetr-soccernet-v1` (MIT) is gated → skipped. | ✅ adopted for P1 |
| ADR-7 | **Shot-boundary detection runs for every profile, not just `broadcast`** | clip4 is single-camera yet contains real cuts. Only replay/close-up classification and scoreboard OCR stay broadcast-gated. | ✅ adopted, supersedes §5 row 1 |
| ADR-6 | **Uncalibrated speed is reported as uncalibrated, never converted to fake m/s** | On this footage there are often <4 usable pitch landmarks. `Event.evidence` carries `calibrated: bool`; when false, speed is normalised-pixel units, confidence penalised, UI says "uncalibrated". | ✅ adopted |
| ADR-5 | **Python 3.11 provisioned via `uv`** (system has only 3.12) | Honors §10 without `sudo`; `uv` gives fast, reproducible locking. | ✅ adopted |
| ADR-4 | **Repo root = the existing working directory**, not a new subdir | `input/` already exists here and holds the owner's clips. | ✅ adopted |
| ADR-3 | **Phase-1 homography = assisted 4-point manual calibration for single-camera profiles** | The usual auto pitch-keypoint model is YOLOv8-pose (AGPL). For a static camera one manual calibration serves the whole clip, is more accurate, costs no VRAM, and is consistent with HITL. **ADR-20's goal region uses the same human-marked, no-training pattern.** | ✅ adopted for P1 |
| ADR-2 | **Roboflow's soccer player/pitch checkpoints are YOLOv8 → AGPL → cannot ship** | Released weights are Ultralytics YOLOv8 (AGPL-3.0). Datasets are CC BY 4.0 (eval + fine-tune material). | ✅ **resolved → see ADR-8** |
| ADR-1 | **PyAV + ffmpeg CLI for decode; drop `decord`** | `decord` 0.6.0 has no cp311/cp312 wheels and is unmaintained. PyAV + NVDEC cover it with zero build risk. | ✅ adopted |

## 6. Phases
**Phase 0 (done as part of P1 kickoff):** env check, repo scaffold, data contracts, eval harness stub, license table.

**Phase 1 — within-take core, MANUAL player pick (current):**
Ingest → **annotation-sidecar check (ADR-19)** → shot-boundary + drop replays → detection (**pretrained**
RF-DETR/soccer weights + SAHI ball; **no training**) → pitch homography → within-take tracking + **team colour**
→ events (auto: goals/shots/sprints/**passes+turnovers**/touches/tackles; **or** manual sidecar) → **user
selects a track / jersey number** → clip cut + dedupe → best-first reel export → stat card → eval (mAP + HOTA on
labeled slice) → thin API + minimal UI. **Runs entirely on the local A2000 12 GB — inference only, no paid cloud.**
Handle **any single uploaded video**: the source profiler auto-detects broadcast vs single-camera **and quality**,
and branches (§14). **Validate on a static single-camera clip first**, then broaden to broadcast.
*Definition of done:* on a sample clip + a target jersey, output a de-duplicated per-category reel set + stat
card (with confidences) + an annotated video + an eval report, reproducible via `make run`, **including the
manual-annotation path on the client's worked example (§3.5).**

**Phase 2 — player identity:** jersey OCR + cross-take re-association + human-in-the-loop confirm → follow a
chosen number automatically; per-player event attribution; passes/touches as best-effort (flagged).

**Phase 3 — hardening:** fine-tuned action spotting; tackles/saves/assists; confidence + correction UI; scale
compute; QA vs ground truth; optional provider-data adapter; active-learning loop.

## 7. Dependency license table (keep updated; ⚠️ = do not ship in closed product without action)
| Component | License | OK for closed product? |
|---|---|---|
| RF-DETR (`rfdetr`) | Apache-2.0 | ✅ |
| `supervision` (tracking/annot.) | MIT | ✅ |
| SAHI | MIT | ✅ |
| PySceneDetect | BSD-3 | ✅ |
| PyAV | BSD-3 | ✅ |
| MMPose / MMOCR (ViTPose) | Apache-2.0 | ✅ |
| PARSeq | Apache-2.0 | ✅ |
| PaddleOCR / EasyOCR | Apache-2.0 | ✅ |
| `requests` (ADR-15 Gemini HTTP) | Apache-2.0 | ✅ |
| `reportlab` (Plan Stage 3, 2026-09-14, `statcard.pdf` export, `api` extra) | BSD-3-Clause | ✅ |
| **manual-annotation parser (ADR-19)** | stdlib `re` + PyYAML (BSD/MIT) | ✅ **no new dependency** |
| `cloudflared` (host binary on the RunPod pod only, docs/DEPLOY.md; not a Python dep, not shipped in the product) | Apache-2.0 | ✅ |
| `SoccerNet` (pip package, `scripts/download_soccernet.py`) | MIT | ✅ package code only — the video *data* it downloads is the separate, NDA-gated row below |
| SigLIP weights (via `transformers`/`timm`) | Apache-2.0 | ✅ (verify weight card) |
| TransNetV2 | MIT | ✅ |
| TrackEval | MIT | ✅ |
| Ultralytics YOLO | **AGPL-3.0** | ⚠️ avoid — use RF-DETR |
| **Roboflow soccer player/field *weights* (YOLOv8)** | **AGPL-3.0** | ⚠️ **do not ship** — see ADR-2 |
| `roboflow/sports` (code only) | MIT | ✅ code ok; weights are the problem |
| **`julianzu9612/RFDETR-Soccernet` weights** | Apache-2.0 *(declared)* | ⚠️ **dev/eval only** — trained on SoccerNet; see **ADR-9** before shipping |
| Roboflow `football-*-detection` **datasets** | **CC BY 4.0** | ✅ commercial OK **with attribution** — our eval + fine-tune source |
| `OrbitalLab/mova-rfdetr-soccernet-v1` | MIT | ⛔ gated on HF (access request) — not used |
| BoxMOT | **AGPL-3.0** | ⚠️ avoid — use `supervision` tracker |
| **`mkoshkina/jersey-number-pipeline` CHECKPOINTS ONLY** — `legibility_resnet34_soccer_20240215.pth` + `parseq_epoch=24-...ckpt` (SoccerNet-fine-tuned) | **CC BY-NC 3.0** | ⛔ **non-commercial only — owner-authorized exception, 2026-08-31.** Generic EasyOCR+Gemini reads jersey numbers unreliably on real broadcast footage (~70-90px player crops); these are the actual purpose-built SoccerNet-fine-tuned weights for exactly this problem, verified directly against the repo (not assumed) — real GDrive-hosted checkpoints, not a research claim with no artifact. Scope of the NC encumbrance is deliberately minimized to the two downloaded **weight files only**: the legibility classifier is a standard ResNet34 + one linear layer, simple enough to be an independently-authored model definition in this project's own (clean) code, loading only mkoshkina's checkpoint into it — never their `networks.py` vendored verbatim. The PARSeq **architecture code** stays genuinely Apache-2.0 (`baudm/parseq` upstream, unmodified) — only the *fine-tuned checkpoint* mkoshkina trained is NC. Owner explicitly authorized using these two weight files anyway, stated reason: personal/internal use, not commercial distribution. **This is a live constraint, not a formality**: this pipeline's output must not be sold, commercially distributed, or shipped in a closed/paid product while these two checkpoints are in use, without either removing them or resolving their licensing first (same "flag, don't ship silently" discipline as ADR-9's SoccerNet-derived detector weights). |
| PARSeq (`baudm/parseq`) **architecture/code**, used as the loader for the checkpoint above | Apache-2.0 | ✅ clean — see the NC row above for why the *checkpoint* it loads is still restricted |
| SoccerNet data | **Research/education** | ⚠️ email for commercial terms |
| Broadcast / YouTube footage | Rights-holder owned | ⚠️ legal review before commercial use — pipeline does **not** download YouTube; owner supplies the file + sidecar (§3.5) |
Audit each new dependency's license here **before** adding it.

## 8. Commands (implement these; keep accurate)
```
make setup     # create env, install deps, download model weights
make run       # AUTO FLOW (§14): process EVERY video in input/ → per input, auto-branch on
               #   (a) annotation sidecar present? -> manual-events mode (ADR-19)
               #   (b) else profile + quality -> full auto (detector/OCR/VLM + colour events, ADR-20)
               # -> annotated video + per-player stat card + per-category highlight reels + report
               #    in output/<slug>/ (resumable via work/)
make run VIDEO=input/"clip2 77.mp4"   # process a single video
make eval      # detection mAP + tracking HOTA (+ action mAP@1 later) on data/eval/
make test      # unit tests (contracts, ANNOTATION PARSER (ADR-19), colour pass/turnover (ADR-20),
               #             ranking, dedupe, speed calc)
make lint      # ruff + black
make models    # fetch + checksum-verify the three model checkpoints into models/ (idempotent)
make serve     # web UI + API on :8000 (PV_DEV=1 = auto-reload for local editing only; login gate
               #  is ON BY DEFAULT -- default password `admin1122`, PV_ACCESS_PASSWORD=... to
               #  change it, PV_ACCESS_PASSWORD=off to disable). HOSTING: docs/DEPLOY.md -- RunPod
               #  pod + network volume bootstrapped by docker/runpod_bootstrap.sh; an ALWAYS-ON
               #  GATEWAY on the Hostinger VPS (apps/gateway, `bash docker/deploy_vps.sh`) serves
               #  the frontend + uploads with no pod; the GPU starts ONLY on the user's Start GPU
               #  and stops on Stop GPU / idle (ADR-24); the gateway PUSHES each pod its code (ADR-25).
               #  Uploads are chunked (apps/api/uploads.py, shared by pod + gateway).
scripts/download_soccernet.py   # --list (no password) or pulls one broadcast clip into input/
                                 # (needs SOCCERNET_PASSWORD in .env, from the NDA form at soccer-net.org)
                                 # -- kept as the GOOD-QUALITY control input (§3.4); "high-quality when needed"
scripts/pull_roboflow.py        # pulls datasets + pretrained soccer weights (needs ROBOFLOW_API_KEY)
scripts/prepare_eval.py         # ingest labeled slices into data/eval/
scripts/make_annotation_template.py  # (ADR-19) emit a blank <video>.annotations.txt next to a video,
                                      #  prefilled with its identity_report "unverified" takes as TODO lines
```
Secrets in `.env` (`HF_TOKEN`, `ROBOFLOW_API_KEY`, `SOCCERNET_PASSWORD`, optional `GEMINI_API_KEY`,
`TWELVELABS_API_KEY`); never commit them.

Per-input sidecars in `input/` (never committed, gitignored with the rest of `input/`):
- `input/<video>.annotations.txt` — manual events (ADR-19, §14.3). Presence ⇒ manual mode.
- `configs/goal_region.yaml` — human-marked goal polygons per video slug (ADR-20). Empty ⇒ region path off.

## 9. Evaluation (metrics + first targets — refine against your labeled set)
- **Detection:** mAP@50 on your labeled slice (players first, ball tracked separately — expect ball lower).
- **Tracking:** **HOTA** via **TrackEval** (report IDF1 + ID switches too), measured **within takes**.
- **Actions (Phase 2+):** **mAP@1s** (SoccerNet ball-action style).
- **Jersey (Phase 2):** tracklet accuracy.
- **Manual-annotation parser (ADR-19):** unit-tested round-trip — the §3.5 three-line example must parse to
  exactly `[TOUCH@26:56 #2 white, PASS@26:58 #2 white, OUT_OF_BOUNDS@27:28 #2 white]`; malformed lines are
  reported, never silently dropped.
Wire these before optimizing anything. Log a run report (numbers + config hash) per run.

### 9.1 Eval datasets on disk
| Set | Path | Content | Licence | What it measures |
|---|---|---|---|---|
| **Roboflow football-players v20** | `data/eval/roboflow-football-players-v20/` | 372 imgs / 8,905 anns; classes `ball, goalkeeper, player, referee` | **CC BY 4.0** — attribution required | Detection mAP **baseline** |
| **Our Veo slice** | `data/eval/veo-labeled/` | ⛔ **not yet created** — must be hand-labeled from `input/` | own footage | Detection mAP + HOTA **that actually counts** |

> ⚠️ **The Roboflow set is a sanity baseline, not our number** (576×576 broadcast crops). Per **ADR-10**,
> results there do **not** transfer to youth/amateur Veo footage. Report both numbers side by side.

## 10. Working conventions
- Python 3.11 (via `uv`, see ADR-5). One `configs/*.yaml` per stage; **no magic numbers in code** — this now
  explicitly includes the **annotation phrase→EventType map** (ADR-19) and the **colour/goal-region thresholds**
  (ADR-20), which live in `configs/annotations.yaml`, `configs/events.yaml`, and `configs/goal_region.yaml`.
- Cache each stage's output to disk; stages must be independently runnable/debuggable.
- After each stage: a **smoke test** on a short clip + an **annotated overlay video** artifact.
- **Log everything dropped** (replays, close-ups, low-confidence, **unmapped annotation lines**, **turnovers**).
  Silent filtering = hidden bugs.
- Commit per stage; update this file when decisions change; verify with eval numbers before "done".
- **I/O contract:** input video (+ optional sidecars) in `input/` (auto-detected); cached artifacts in `work/`
  (resumable, chunked); final annotated video + stat card + highlight reels + run report in `output/`.
  Gitignore `input/`, `work/`, `output/`, `.env`, `models/`, `data/`.
- **Secrets: ask, never assume.** Load from `.env`; when a token/decision is missing, STOP and ask the user —
  never mock data or skip a stage silently.
- **Model workflow:** the user runs Claude Code in **`opusplan`** — **Opus plans (Plan Mode), Sonnet executes.**
  Plan thoroughly; keep execution incremental + resumable. Provide `.claude/commands/process-match.md` to run
  the flow. (See the delivered Claude Code prompt for the exact plan→execute contract.)

## 11. Hardware profile & compute budget
**Single machine: NVIDIA RTX A2000 12 GB** (Ampere; NVENC/NVDEC). **LOCAL ONLY — no paid cloud.** The entire
pipeline, including full 90-min matches, runs on this GPU; long runtimes are acceptable (design for "slow but
completes"). Phase 1 is inference-only with pretrained models, so 12 GB is comfortable. Any fine-tuning is
deferred/optional (overnight on the A2000 or a free Colab/Kaggle T4/P100). Never gate progress on paid GPUs.

### 11.1 Measured environment (2026-08-24)
| Item | Value |
|---|---|
| GPU | NVIDIA RTX A2000 12 GB (12282 MiB), Ampere |
| Driver / CUDA | 580.173.02 / CUDA 13.0 runtime; `nvcc` 12.0 |
| ffmpeg | 6.1.1 — `h264_cuvid` NVDEC ✅, `h264_nvenc` NVENC ✅, `-hwaccel cuda` ✅ |
| Python | system 3.12.3; **project uses 3.11 via `uv`** (ADR-5) |
| RAM / disk | 125 GB / 124 GB free |

> ⚠️ **VRAM contention:** an unrelated process was holding 6.0 GB at scaffold time. `configs/hardware.yaml`
> defaults to a **5 GB budget**; the pipeline runs a **VRAM preflight** that warns (not fails) below budget.

Fit-in-12 GB rules (knobs in `configs/hardware.yaml`):
- **Run stages sequentially, cache to disk** — never hold detector + pose + SigLIP + OCR in VRAM at once.
- **Chunk the match** and make every run **resumable** from cached artifacts.
- **Mixed precision (AMP/FP16)** everywhere; add an **ONNX / TensorRT** path.
- **Small variants:** RF-DETR-base/small, ViTPose-small, ByteTrack. Batch 1–2.
- **VLM via API** (Gemini free tier), not a local VLM.
- **Decode with NVDEC**, sample frames per stage, skip replays/close-ups.
- **Downscale 4K before detection** (§3.1). The manual-annotation path (ADR-19) still decodes at native res for
  the overlay render but does **no** heavy per-frame OCR/VLM (events come from the sidecar) — so it is cheaper
  than the full auto path, which matters because bad-quality clips are exactly where auto would waste the most VLM budget.

Per-stage fps sampling defaults: shot-boundary/scoreboard **1–2 fps** · detection+tracking **5–12 fps** ·
ball + action windows **full fps only where needed** · team/jersey **1–2 fps per tracklet**.

Estimate runtime honestly: **time a 2-min clip end-to-end, then multiply (~45× for 90 min).** Print a per-stage
VRAM + wall-clock report. If a stage OOMs: batch→1, lower image size, or switch to a smaller variant.

## 12. Don'ts
❌ Train from scratch. ❌ Ship YOLO/BoxMOT (AGPL). ❌ Track across cuts. ❌ Emit untraceable stats.
❌ Build everything at once. ❌ Hard-code params (incl. annotation phrases + colour/goal-region thresholds).
❌ Over-promise accuracy in UI copy. ❌ Feed raw 4K to a detector. ❌ Download YouTube in-pipeline (owner
supplies the file + sidecar). ❌ Count a turnover as a completed pass. ❌ Draw a red target box you can't
associate to a real track (caption the event instead).
> ❌ ~~Pass/touch/tackle stats in Phase 1~~ — **owner-authorized, ADR-13/14/20, §13.** Best-effort heuristics,
> never a fine-tuned model, always a low honestly-stated confidence.
> ❌ ~~Auto-ID player in Phase 1~~ — **owner-authorized exceptions:** ADR-15 (verified OCR/VLM on legible
> footage) and ADR-19 (**manual-annotation sidecar** on illegible footage). Identity is still either *verified*
> or *human-provided*, never inferred from appearance/position/arrow alone. Filename-metadata §3.1 clips keep
> the human-selects-the-track flow.
> ❌ ~~Goal detection on non-broadcast footage~~ — **owner-authorized, ADR-20:** a **human-marked** goal region
> (no training) or a **manual annotation** may source a goal; a guessed "ball vanished near a corner" may not.

## 13. Extended output spec (owner-authorized 2026-08-25 — ADR-14; revised 2026-08-27 — ADR-15/18;
extended 2026-08-27 — ADR-19/20; supersedes prior output shape)

> **Identity gating (ADR-15/18/19):** everything that names "the target player" is conditioned on that take's
> identity being **known** — either **verified** (OCR/VLM temporal agreement, ADR-15), **human-confirmed**
> (`--target-jersey` / filename, ADR-18), or **human-provided by sidecar** (ADR-19). An unknown-identity take
> still gets full green-box + ball-tracking output; it just contributes no red box and no target-player stats,
> and the run report says why.

### 13.1 Primary output: full-resolution annotated original video
**`output/<slug>/original_annotated_video.mp4`** — the original input video, unchanged resolution/fps/
duration, **original audio preserved**, with detections/tracking/events/stats drawn directly on top. Draw:
- **Red** box, thick, persistent label, around the **target player**, labelled with that take's jersey number
  and both identifiers (`#<N> | TARGET | ID: <track_id>`) — never a bare track ID standing in for a jersey
  identity (ADR-18). ID must not visibly reset when the target overlaps another player or leaves/re-enters
  frame **within the same take** (best-effort within-take stitching; flagged approximate). A take with no
  known identity gets **no red box** — every player renders green, plus an "IDENTITY: unverified in this
  segment" note by that take's "CUT — take N" banner. **In manual mode (ADR-19), the target's number/colour
  come from the sidecar**; where the red box can't be reliably associated to a track in an event window, the
  event is shown by **caption** (name + timestamp + jersey #) rather than a fabricated box.
- **Green** boxes, thin, around every other detected player, labelled `ID: <track_id>`.
- A small distinct **ball marker** — solid when observed, visually distinct (hollow/dashed) when interpolated
  across a gap (Golden Rule 5).
- A **live stats panel** (semi-transparent, corner-anchored) that accumulates as the video plays: every
  category in §13.3, each showing its real running count/value, or a static "not detected"/"uncertain" line
  for whatever the current build genuinely cannot produce — never a fabricated or frozen-fake number.
- A brief **"CUT — take N"** banner at every take boundary.
- **Event captions** (ADR-19/20): at each event's timestamp, a brief on-screen caption naming the event, the
  jersey number, and the timestamp — this is what guarantees the annotated video is faithful to a manual
  sidecar even when tracking is weak, and it doubles as the caption burned into the highlight clips (§13.4).
4K NVENC encode + audio mux is expensive (multi-minute per clip, large files) — an accepted cost, not a reason
to substitute a downscaled render.

### 13.2 `statcard.md` — exact template (owner-specified, revised 2026-08-27 — ADR-15/19/20)
One file per **known** jersey number (§13.5), never one file blending two different numbers. **Identity Status**
reads `Verified` (the existing ADR-15 auto path — covers both an OCR/VLM-confirmed read and a
`--target-jersey`/filename-confirmed run, since `TakeIdentityResult.status` only has two literal values,
`verified`/`unverified` — a third UI-only tier is not worth a type change) or `Human-provided (manual
annotation)` (ADR-19, the new `identity_status` param on `render_statcard_markdown`):
```markdown
# Player Statistics

## Player #<jersey_number>

**Identity Status:** Verified | Human-provided (manual annotation)

**Touches:** [actual value]
**Passes:** [actual value]           # completed passes only (same-colour, ADR-20)
**Turnovers:** [actual value]        # passes/losses to the other colour — logged, NOT added to Passes
**Sprints/Runs:** [actual value]
**Goals:** [actual value]
**Assists:** [actual value]
**Shots:** [actual value]
**Tackles:** [actual value]
**Saves:** [actual value]
**Dribbles:** [actual value]
**Possession Time:** [actual value]
**Distance Covered:** [actual value]

## Event Timeline

| Time | Event | Confidence |
|------|-------|------------|
| [Timestamp] | Touch | [0-1] |
| [Timestamp] | Pass | [0-1] |
| [Timestamp] | Turnover | [0-1] |
| [Timestamp] | Sprint | [0-1] |
| [Timestamp] | Dribble | [0-1] |
| [Timestamp] | Shot | [0-1] |
| [Timestamp] | Tackle | [0-1] |
| [Timestamp] | Assist | [0-1] |
| [Timestamp] | Goal | [0-1] |
| [Timestamp] | Out of bounds | [0-1] |
| [Timestamp] | Celebration | [0-1] |
| [Timestamp] | Other Key Moment | [0-1] |
```
Only list events that actually fired; **never pad the timeline to look complete.** Wherever a value can't be
reliably derived, write `uncertain` (owner's own words) rather than a number. `Distance Covered` and any
speed-derived value are suffixed `(uncalibrated)` per ADR-6. `Goals`/`Assists` read `not available` when no
scoreboard, no marked goal region (ADR-20), and no manual annotation exists to source them (re-check per video).
**In manual mode the confidence column shows the annotation confidence** (default 0.95, `configs/annotations.yaml`)
and the timeline is exactly the parsed sidecar. If a take's identity is unknown (ADR-15), no `statcard.md` is
written for it; the run report says: *"Target player could not be reliably identified. Statistics were not
generated because jersey identity could not be verified"* — and points to the sidecar path (ADR-19) as the fix.

### 13.3 Event categories and their real status on this footage
| Category | Status | Method |
|---|---|---|
| Sprint, Shot | ✅ built (Stage 4/5) | speed/direction heuristic |
| Touch, Tackle, Save (GK) | ✅ built + wired 2026-08-27 (ADR-13) | ball-proximity / possession-change heuristic, no training, low confidence ceiling |
| **Pass, Turnover** | ✅ ADR-13 base + 🆕 **ADR-20 colour test** | possession-change to a **same-colour** receiver ⇒ **Pass** (counted); to a **different-colour** receiver ⇒ **Turnover** (logged, not counted); reuses ADR-12 clusters per take |
| Dribble, Ball-possession, Possession duration | ✅ built + wired 2026-08-27 (ADR-14) | extension of the same possession-heuristic layer |
| Distance covered | ✅ built + wired 2026-08-27 (ADR-14) | normalised-pixel-unit speed integration (ADR-6), always `(uncalibrated)`; summed one take at a time |
| Celebration, Other key moment | ✅ built + wired 2026-08-27 (ADR-14) | cheap motion pre-filter → **Gemini** classifies only surviving candidate windows; emits only on explicit "yes" |
| **Goal, Assist** | ✅ ADR-17 base + 🆕 **ADR-20 goal region** + 🆕 **ADR-19 annotation** | **Goal** occurrence from any of: manual annotation (authoritative), scoreboard-OCR delta (broadcast, self-gated), or **human-marked goal region** (static cam, ball enters region within a shot window). **Assist** = nearest preceding **same-colour** `PASS` to the credited scorer within `assist_window_seconds`. Still `not available` when none of the three goal sources exists for a clip |
| **All of the above, manual mode** | 🆕 **ADR-19** | when a sidecar exists, the timeline **is** the parsed annotations (`source="manual_annotation"`); auto-detectors are skipped for event generation but detection/tracking still run for the overlay |

### 13.4 Highlight compilations (per clip, category-specific — not the single ranked reel)
`output/<slug>/players/player_<N>/highlights/{ball_possession,passes,turnovers,dribbles,assists,goals,key_moments}.mp4`
— each a concatenation of every event in that category for **that known player**, a few seconds of context
before/after each event (`pre_seconds`/`post_seconds`). **An empty/absent file, not a fabricated one, when a
category has zero real events.** Each highlight clip carries the same red/green/ball overlay plus an
event-name + timestamp + jersey-number caption — it is a cut of the **annotated** video (ADR-18), not the raw
footage. (`passes.mp4` is the owner's explicitly requested pass reel; `turnovers.mp4` is its ADR-20 counterpart
-- a lost possession never counted toward `passes.mp4`; `goals.mp4`/`assists.mp4` are usually empty on
amateur footage unless a goal region is marked or a sidecar supplies them.)

### 13.5 Required directory layout (per input video)
**When the jersey number is known from the filename (§3.1 clips):** one flat `output/<slug>/` per clip.

**When the number is verified (ADR-15) or human-provided (ADR-18 `--target-jersey` / ADR-19 sidecar):**
identity is per-*known-number*, so a compilation can contain more than one, or none:
```
output/<slug>/
  original_annotated_video.mp4     # §13.1 — ONE file, whole video, red box only in known-identity takes
  players/
    player_<N>/                    # one folder per DISTINCT known jersey number
      statcard.md                  # §13.2 — exact owner template, this player's real values only
      highlights/
        ball_possession.mp4
        passes.mp4                 # owner's requested pass reel
        turnovers.mp4              # ADR-20: never double-counted into passes.mp4
        dribbles.mp4
        assists.mp4                # empty unless a goal source + preceding same-colour pass exist
        goals.mp4                  # empty unless annotation / scoreboard / marked goal region sources one
        key_moments.mp4            # celebrations / other key moments, Gemini-classified (§13.3)
      events/
        event_timeline.json        # machine-readable timeline incl. confidence + source
  identity_report.json             # every take's {take_id, jersey_number|null, status, confidence,
                                    # evidence_frames} — including UNVERIFIED takes (so "why is there no
                                    # player_7 folder for take 3" is always answerable, and which takes need
                                    # a manual sidecar line is obvious)
  annotation_report.json           # (ADR-19, manual mode) parsed sidecar + any malformed/unmapped lines
```
The pre-existing `reel.mp4` / `stat_card.json` / `stat_card.md` / `run_report.json` / `clips/` artifacts are
kept alongside (other tooling/tests depend on them) — §13.5 is additive.

## 14. The single auto flow (Golden Rule 9 — one command, first-flow-correct)
`make run` iterates every video in `input/` and, **per video, auto-selects a branch with no human step**:

### 14.1 Branch selection (deterministic, in this order)
1. **Sidecar present?** — if `input/<video_basename>.annotations.txt` exists ⇒ **MANUAL-EVENTS MODE**
   (ADR-19). The parsed annotations are the authoritative event source; identity = the sidecar's jersey
   number + colour; Stage-5 auto event-detectors are skipped; detection/tracking still run for the overlay.
   *This is the owner-confirmed rule: the client's file winning is the trigger — not a quality score.*
2. **Else — full AUTO MODE**, per the existing `src.pipeline.run.main()` branch (unchanged by this work):
   - **Jersey number available?** filename `clip<N> <jersey>` (§3.1) ⇒ the original Phase-1
     `run_pipeline_for_video` flow (human-selects-the-track, `--track-id`).
   - **No filename number** ⇒ ADR-15's extended pipeline (`run_extended_pipeline_for_video`) always runs —
     there is no separate quality pre-check, because the OCR/VLM verification step IS the honest quality
     check: on clear footage (e.g. SoccerNet) it verifies takes; on illegible footage (e.g.
     `jordan_thomas_highlight_video`) it correctly verifies none. `--target-jersey` (ADR-18), when given,
     is threaded in as a human-confirmed number that needs only one supporting read. Whichever takes come
     back unverified are named in `identity_report.json`, which is exactly what
     `scripts/make_annotation_template.py` reads to build a starter sidecar for a re-run.
3. **Events, both modes:** colour-aware pass/turnover/assist (ADR-20) and the three goal sources (ADR-19/20/17)
   apply in auto mode; in manual mode the sidecar already carries them.

### 14.2 Why this is safe to run unattended
Every branch has an **honest terminal state**: a value, or `not available`/`unverified`/`uncertain` with a
reason in the run report (Golden Rules 5 + 9). "Runs auto and correct on the first flow" is satisfied by never
letting a branch fabricate to look complete — a correctly-empty `goals.mp4` on amateur footage is a *pass*,
not a failure. Resumability (§10) means a re-run after dropping a sidecar reuses all cached detection/tracking.

### 14.3 Manual annotation format (ADR-19 — the client's own phrasing, parsed tolerantly)
Sidecar file: `input/<video_basename>.annotations.txt` (UTF-8). Lines:
```
# comments and a source URL line are allowed and ignored by the parser
https://www.youtube.com/watch?v=wzvM66c8ZmE
Min 26:56 player #2 in white takes a touch
Min 26:58 player #2 in white makes a pass
Min 27:28 player #2 in white kicks the ball out of bounds
```
**Grammar (one event per line):** `[Min ]<TIME> player #<N> in <COLOUR> <ACTION PHRASE>`
- `TIME` = `MM:SS` or `H:MM:SS`/`HH:MM:SS`; the literal `Min ` prefix is optional; seconds are converted to
  a float offset from video start.
- `#<N>` = target jersey number (integer). `<COLOUR>` = a free word (matched to the ADR-12 team cluster).
- `<ACTION PHRASE>` = free text, mapped to an `EventType` via the **phrase map in `configs/annotations.yaml`**
  (no magic strings in code). Starter map: `touch→TOUCH`, `pass→PASS`, `out of bounds→OUT_OF_BOUNDS`,
  `shot|shoots→SHOT`, `goal|scores→GOAL`, `assist→ASSIST`, `tackle→TACKLE`, `save→SAVE`, `sprint|run→SPRINT`,
  `dribble→DRIBBLE`, `celebrat→KEY_MOMENT` (rendered "Celebration" in the timeline — `_key_moment_label` reads
  the annotation's own `action_phrase`, the same text heuristic already used for Gemini-classified key
  moments, ADR-14 — deliberately NOT a new `CELEBRATION` `EventType`, to avoid two competing
  representations of the same concept). An **unmapped phrase** ⇒ `EventType.KEY_MOMENT` with the raw text
  kept in `evidence`, and it is **reported** in `annotation_report.json` (never silently dropped, §10).
- Each parsed line → `Event(source="manual_annotation", confidence=annotations.default_confidence)`.
- **Acceptance test (§9):** the three lines above parse to exactly `TOUCH@26:56 #2 white`,
  `PASS@26:58 #2 white`, `OUT_OF_BOUNDS@27:28 #2 white`, and drive a `player_2/` output folder with a
  `statcard.md` (Identity Status: `Human-provided (manual annotation)`), a `passes.mp4` containing the 26:58
  event, and an `original_annotated_video.mp4` whose captions fire at all three timestamps.