# AGENTS.md — AWARE missing-person search (for coding agents)

Read this first. It describes the project, what has already been changed, and
how to work safely in this repository.

## What the project is

AWARE: a drone patrols a simulated crowded expo venue (Gazebo + PX4 + ROS 2)
and searches for a missing person from a text description (default: "a person
wearing a red top and white trousers"). An operator confirms (c) or rejects (r)
each possible match.

Repository layout:

- `rasid_video_pipeline/` — the AI search pipeline (Python, `ai_venv`)
- `simulation/` — Gazebo world, PX4 drone, patrol script, launch scripts
  (maintained by a teammate; see `simulation/README.md`)

## Dashboard goals

The dashboard has two sections. (A) Crowd analysis: a live snapshot of the venue from the drone camera, and below it statistics such as total people and people per zone. (B) Missing person: a description field and a Start search button. When a candidate is found, the drone's camera image of them appears; the operator clicks Confirm or Reject. Confirm: the dashboard shows the person's location. Reject: the drone keeps searching for another candidate.

## Machine / environment

- Ubuntu **22.04**, ROS 2 **Humble**, Gazebo Harmonic, PX4 (main).
  The original setup targets 24.04/Jazzy; `simulation/setup/*.sh` and
  `simulation/launch/_common.sh` were adapted to pick Humble on 22.04.
- **No NVIDIA GPU** on this laptop: Grounding DINO runs on CPU, ~6–12 s per
  frame. Tracking behaviour on this machine is much worse than on a GPU
  (a teammate's RTX 4060 laptop). Do not "fix" slowness by hiding errors.
- PX4 `main` writes `gz_env.sh` to `build/px4_sitl_default/`; a symlink was
  added at `build/px4_sitl_default/rootfs/gz_env.sh`.
- Never read, print or commit `rasid_video_pipeline/.env` (OpenAI key).

## How to run (two terminals)

```bash
# terminal 1: simulation + patrol, no AI
aware_run --no-ai --scenario missing_person --scale 0.5 --loops 4
# then type `commander takeoff` in the PX4 tab when the PATROL tab asks

# terminal 2: the AI, with a log file
aware_ai
python3 -u run_live.py 2>&1 | tee ~/ai_log.txt
```

`aware_stop` stops everything. Quick import check:
`aware_ai && python3 -c "import pipeline_runner; print('OK')"`.

## Architecture (CloudTrack design — keep it)

The team chose to keep the CloudTrack design (big model occasionally, small
tracker in between). Do not replace Grounding DINO with another detector
unless explicitly asked.

```
camera (ROS /aware/camera/image) -> ros_frame_source.py (newest frame only)
  -> pipeline_runner.run_on_video -> SearchController.process_frame
     -> CloudTrack.forward (cloud_track/pipeline/cloud_track.py)
        SEARCHING: DetectorVlmPipeline.run_inference
           Grounding DINO -> ByteTrack -> select_candidates (colour filter,
           rejected places) -> GPT verification (cache, budget, timeout)
        TRACKING: OpenCV "nano" tracker every frame
           + periodic re-detection (Grounding DINO only, no GPT)
```

Key files (`rasid_video_pipeline/`):

| File | Role |
|---|---|
| `pipeline_runner.py` | SearchController, events, main loop; publishes target position |
| `run_live.py` | terminal UI on the live camera (c / r / q) |
| `backend.py` | FastAPI dashboard backend (same events) |
| `drone_state.py` | reads `/aware/drone/state` + camera_info; `TargetPublisher` |
| `cloud_track/pipeline/cloud_track.py` | CloudTrack: search, tracking, re-detection, verified target |
| `cloud_track/foundation_model_wrappers/detector_vlm_pipeline.py` | detector + ByteTrack + GPT, cache, rejected places |
| `cloud_track/foundation_model_wrappers/aware_candidates.py` | candidate selection, colour scoring, `verify_target`, appearance signature |
| `cloud_track/foundation_model_wrappers/aware_geo.py` | box -> ground lat/lon (uses drone tilt), `RejectedPlaces` |
| `cloud_track/foundation_model_wrappers/gpt_four_wrapper.py` | OpenAI calls: timeouts, retries, raises `VlmRequestError` |
| `simulation/aware_sim/scripts/patrol.py` | MAVSDK patrol; holds/flies to look at candidates; publishes drone state |

ROS topics (domain 42): `/aware/camera/image`, `/aware/camera/camera_info`,
`/aware/search_state` (AI -> patrol), `/aware/drone/state` (patrol -> AI,
JSON: lat, lon, alt, heading, roll, pitch), `/aware/target` (AI -> patrol,
JSON: state, lat, lon), `/aware/ground_truth*` (scoring only, NEVER AI input).

## Changes already made (in order)

1. GPT wrapper: request timeouts, one retry on 429/5xx, JPEG upload, raise on failure.
2. Pipeline: per-track-ID answer cache (time-based TTL), max 3 parallel GPT calls per frame.
3. Rejected track IDs never take candidate slots.
4. No blocking wait for the operator: the candidate keeps being tracked; sim timestamps passed through.
5. Drone GPS published by the patrol; dashboard `location` filled on confirm.
6. Periodic re-detection while tracking (every 3 s) to catch tracker drift.
7. Rejected people remembered by ground PLACE + clothing look (radius 4 m, 3 min).
8. Patrol flies to look at / follows the candidate (`goto_location`).
9. Rejection and drone steering use the last VERIFIED target position (not the drifting tracker box); late "r" (within 20 s of a loss) still counts.
10. Drone roll/pitch included in the ground-position math; "lost" only after 30 s unseen; re-detection misses log a reason.

## Known open questions / next steps

- Is the drone camera body-fixed or gimbal-stabilised? `aware_geo.CAMERA_STABILIZED`
  assumes body-fixed. Check `simulation/aware_sim/scripts/make_drone.py`.
- Measure ground-position accuracy against `/aware/ground_truth/missing_person`
  (scoring only).
- Whether `ActionAsync.goto_location` exists in MAVSDK v4 asyncio (patrol prints
  "could not fly to the person" if not).
- Re-detection could run on a crop around the target (faster on CPU, better for
  small far-away people).

## Rules for changes
- After editing, run at least the import check above. Prefer adding unit-style
  tests with synthetic data (no GPU, no ROS) for pure logic
  (`aware_geo.py`, `aware_candidates.py`).
- Don't change the dashboard protocol in `backend.py` without saying so.

- When analysing a run, read `~/ai_log.txt` and the PATROL tab output; base conclusions on log lines, not guesses.