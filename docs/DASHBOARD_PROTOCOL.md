# Dashboard WebSocket protocol proposal

7 October 2026. Based on AGENTS.md and INVENTORY.md. Endpoint: `/ws`; browser commands remain WebSocket commands. `POST /start` and `POST /confirm` remain supported and use the same search state and callbacks. This document specifies the complete proposed interface. Missing-person and live crowd messages are now implemented. Phase 3c clarifies geometric coverage, occupancy gating and snapshot scheduling below.

## Common rules

- UTF-8 JSON objects with a `type` discriminator. Server messages always include `sim_time`: camera-frame simulation seconds, or null for connection/command events without a frame. For a recorded video it can be null; never substitute browser time for an observation time.
- Commands may include `request_id` (string) for correlation; responses caused by validation/queuing echo it. Existing commands without it remain accepted.
- `candidate_id` is a server-generated string unique to one offered candidate, distinct from the possibly temporary numeric `track_id`. Browser decisions include it; old HTTP clients may omit it. A supplied stale ID is rejected.
- Images are complete JPEG data URLs: `data:image/jpeg;base64,<JPEG bytes>`. In examples below `/9j/EXAMPLE` abbreviates real base64; it is not a valid test image.
- No fabricated confidence field. GPT justification is text, not a probability.
- A queued decision is not a completed decision. “Confirmed” appears only after `search_status.state="confirmed"`, issued when the pipeline emits ConfirmedEvent. A clicked button or HTTP acceptance alone is insufficient.
- Unknown or unavailable geographic positions are null. Never substitute camera-center estimates for a person. `position_sim_time` identifies the verified sighting used for the coordinates; `sim_time` identifies the event frame. The position may therefore be older than the event.
- Only one search is active per backend. These messages describe shared state for all connected dashboards; no per-user search sessions are added.

## Browser → backend

### start_person_search

`description` required nonempty string; `video_path` optional, defaults to `ros`; `category` optional, defaults to `person`. The existing browser command that contains only type and description remains valid.

```json
{"type":"start_person_search","request_id":"start-1","description":"a person wearing a red top and white trousers","video_path":"ros","category":"person"}
```

Success produces `search_status` with state `searching`. Rejection produces `error`. The WebSocket handler invokes the same start implementation as HTTP `/start`.

### confirm_candidate

```json
{"type":"confirm_candidate","request_id":"confirm-2","candidate_id":"offer-2"}
```

Backend first reports `confirming`; after actual pipeline confirmation, reports `confirmed` and `person_location`. An absent/stale candidate or a second pending decision produces `error`.

### reject_candidate

```json
{"type":"reject_candidate","request_id":"reject-1","candidate_id":"offer-1"}
```

Backend first reports `resuming_search`; after actual rejection, emits `rejected` then `search_status` with state `searching`. No client-side assumption that the decision was already consumed.

## Backend → browser: missing person

### search_status

States: `idle`, `searching`, `awaiting_decision`, `confirming`, `resuming_search`, `confirmed`, `finished`. Text is explanatory; clients use state for behavior. `gpt_verification_off` is a boolean describing the current search instance; false resets an old warning after a fresh search. On socket connection, send the last status; reconnect may also replay the pending candidate, most recent confirmed position, and verification-off warning.

```json
{"type":"search_status","sim_time":null,"state":"searching","message":"Searching the drone camera for matching people.","request_id":"start-1"}
```

Actual confirmation acknowledgement:

```json
{"type":"search_status","sim_time":103.028,"state":"confirmed","message":"Person confirmed. Tracking continues.","request_id":null}
```

Completion of a finite source uses the same message, with state `finished` and optional `output_video_path` filesystem path. It is not a download URL.

### candidate

Required `candidate_id`, nullable numeric `track_id`, integer `frame_index`, nonempty `justification`, `image` (crop), and `frame` (full camera image with box). No confidence/score field.

```json
{"type":"candidate","sim_time":71.812,"candidate_id":"offer-1","track_id":-3,"frame_index":4,"image":"data:image/jpeg;base64,/9j/EXAMPLE","frame":"data:image/jpeg;base64,/9j/EXAMPLE","justification":"The person is wearing a red top and white trousers."}
```

### tracking_update

Preview of the current candidate or confirmed target. `confirmed` is factual pipeline state; it is not an acknowledgement of a newly clicked button. Throttle to at most one message per wall second; publish a preview on initial candidate/confirmation as well. No match-confidence score.

```json
{"type":"tracking_update","sim_time":110.28,"candidate_id":"offer-2","track_id":-7,"frame_index":11,"confirmed":true,"image":"data:image/jpeg;base64,/9j/EXAMPLE"}
```

### person_location

Sent after confirmation and on later confirmed tracking updates. `location` is `{lat,lon}` from the last detector-verified target ground position, or null. `drone` is the drone telemetry associated with the current frame, or null. Unknown drone values may be null. `confirmed` is true. The frontend shows target and drone separately, identifies the verified position time, and displays `seen N s ago` using `max(0, sim_time - position_sim_time)` rounded to simulation seconds. Missing observation times show age unavailable. The existing `position_sim_time` field is required on every location message; its value is null if no verified observation time exists. Confirmation is acknowledged immediately; detector re-verification is scheduled for the next frame, without waiting for the normal 3 s interval.

```json
{"type":"person_location","sim_time":103.028,"position_sim_time":87.652,"candidate_id":"offer-2","track_id":-7,"confirmed":true,"location":{"lat":24.7134169857,"lon":46.6754285135},"drone":{"lat":24.7134,"lon":46.6752,"alt_rel_m":10.0,"heading_deg":116.0,"roll_deg":1.0,"pitch_deg":-2.0}}
```

Coordinates must come from `CloudTrack.verified_target()` through an explicit runner callback, not `DroneStateReader.location()` or a fresh calculation from the drifting local-tracker box. Current drone pose remains a separate measurement.

### rejected

```json
{"type":"rejected","sim_time":87.32,"candidate_id":"offer-1","track_id":-3,"frame_index":6,"status":"searching"}
```

This is the rejection acknowledgement. Remove the candidate card, enable future candidate decisions, then continue searching.

### lost

```json
{"type":"lost","sim_time":140.0,"candidate_id":"offer-2","track_id":-7,"frame_index":15,"was_confirmed":true,"status":"searching"}
```

Clear old candidate/confirmed cards and stale location; new candidates may follow. This is conditional: a healthy test need not produce it.

### error

`code` is machine-readable; `message` is display text; `recoverable=true` means the existing search can continue. Validation errors are sent only to the requesting socket; worker failures are broadcast. Invalid JSON does not close an otherwise healthy connection.

```json
{"type":"error","sim_time":null,"code":"invalid_command","message":"Unknown WebSocket command.","request_id":"bad-1","recoverable":true}
```

Other codes: `invalid_request`, `command_rejected`, `search_failed`. Recoverable command errors do not assert that an active search has stopped.

### gpt_verification_off

Permanent quota failure; shown persistently in the missing-person section. Search may still inspect camera frames, but cannot verify new candidates. This does not automatically confirm anyone. Starting a new search instance resets verification state; restoring credits requires a fresh instance to retry.

```json
{"type":"gpt_verification_off","sim_time":56.476,"code":"insufficient_quota","message":"OpenAI API has no credits: GPT verification is OFF"}
```

## Backend → browser: crowd section

### crowd_snapshot

Annotated camera photograph paired with every crowd_analytics update, including while no missing-person search is active. It is a JPEG data URL scaled to at most 640 px wide. Nominal idle analysis cadence is 10 wall seconds, limited by CPU speed. During search, reuse raw detections; extra tracking-frame inference is limited to once per 15 wall seconds and deferred if it would overrun search re-detection. No new message when no camera image exists. `sim_time` is the image timestamp, matching the paired analytics, not sending time.

```json
{"type":"crowd_snapshot","sim_time":120.5,"image":"data:image/jpeg;base64,/9j/EXAMPLE"}
```

### crowd_analytics

`total_people_in_view` is people in this analyzed camera image, not all people in the venue. `zones` is keyed by the IDs from zones.yaml. Each value includes `id`, `type`, `display_name`, `capacity`, and `booth` (the booth ID for booth fronts; null for areas). All metadata is loaded from YAML once at backend startup, including readable names such as "Booth A front"; the dashboard displays `display_name`. `count` is the visible-part count in this image; unseen/unknown zones have null count, not invented zeros. `coverage_fraction` is the geometric area fraction in view (0–1), or null without usable pose. `coverage` is `full` at fraction ≥0.8, `partial` for 0<fraction<0.8, or `unseen` at zero/unknown coverage. `occupancy_percent` is `100 * count / capacity` only at coverage ≥0.8; otherwise null. `crowded` is occupancy ≥80%, or null when occupancy is unknown. No extrapolation of partial counts. `last_seen_sim_time` is when the zone had positive geometric coverage and retains its prior timestamp when it leaves view; null if never observed. UI displays people in view, partly visible, and seen N s ago. Do not reinterpret percentages ≤1 as fractions. Bars may cap at 100%, but the numeric value remains uncapped. No footfall or dwell fields/cards in this phase.
```json
{
  "type":"crowd_analytics",
  "sim_time":120.5,
  "total_people_in_view":18,
  "zones":{
    "entrance":{"id":"entrance","type":"area","display_name":"Entrance","booth":null,"count":4,"capacity":60,"occupancy_percent":6.67,"crowded":false,"last_seen_sim_time":120.5,"coverage_fraction":0.95,"coverage":"full"},
    "walkway":{"id":"walkway","type":"area","display_name":"Walkway","booth":null,"count":7,"capacity":120,"occupancy_percent":null,"crowded":null,"last_seen_sim_time":120.5,"coverage_fraction":0.45,"coverage":"partial"},
    "plaza":{"id":"plaza","type":"area","display_name":"Plaza","booth":null,"count":null,"capacity":250,"occupancy_percent":null,"crowded":null,"last_seen_sim_time":null,"coverage_fraction":0.0,"coverage":"unseen"},
    "booth_a_front":{"id":"booth_a_front","type":"booth_front","display_name":"Booth A front","booth":"booth_a","count":null,"capacity":25,"occupancy_percent":null,"crowded":null,"last_seen_sim_time":null,"coverage_fraction":0.0,"coverage":"unseen"},
    "booth_b_front":{"id":"booth_b_front","type":"booth_front","display_name":"Booth B front","booth":"booth_b","count":null,"capacity":25,"occupancy_percent":null,"crowded":null,"last_seen_sim_time":null,"coverage_fraction":0.0,"coverage":"unseen"},
    "booth_c_front":{"id":"booth_c_front","type":"booth_front","display_name":"Booth C front","booth":"booth_c","count":null,"capacity":25,"occupancy_percent":null,"crowded":null,"last_seen_sim_time":null,"coverage_fraction":0.0,"coverage":"unseen"},
    "booth_d_front":{"id":"booth_d_front","type":"booth_front","display_name":"Booth D front","booth":"booth_d","count":null,"capacity":25,"occupancy_percent":null,"crowded":null,"last_seen_sim_time":null,"coverage_fraction":0.0,"coverage":"unseen"}
  }
}
```

Zones do not cover the whole floor, so zone counts need not sum to `total_people_in_view`. People outside all zones still count in view. Pixel-to-world projection and camera-footprint coverage are needed to distinguish absent people from an unseen zone. Ground truth is scoring only and must not populate these model estimates.

## HTTP compatibility

- `POST /start`: existing JSON body with required `video_path`, required `description`, optional `category`. Keep existing success keys `ok`, `status`, `output_video_path`, and error JSON responses.
- `POST /confirm`: existing `decision` (`confirm`/`reject`) retained; optional `candidate_id` enables stale-decision protection. Keep existing success keys `ok`, `decision`, `status`, and error JSON responses.
- HTTP and WebSocket commands affect the same pipeline and produce the same broadcast events. A queued HTTP decision is not yet a confirmation acknowledgement.
- Do not silently alias old outgoing event names in the new UI. Migrate both ends together.

## Files that must change

### Missing-person implementation now

1. `rasid_video_pipeline/backend.py`: WebSocket command dispatch, shared HTTP handlers, candidate IDs, acknowledged state changes, data URLs, new message names, verification-off callback, verified-position callback and location events.
2. `rasid_video_pipeline/dashboard.html`: handle canonical messages, render justification instead of confidence, pass candidate ID, wait for acknowledgement, display separate target/drone coordinates, errors and verification-off state, show ongoing preview.
3. `rasid_video_pipeline/pipeline_runner.py`: optional target-position observer that exposes the existing verified anchor and its time, plus frame drone pose, before event callbacks. Existing callers remain compatible.
4. Tests and test tools: mocked WebSocket/HTTP regression tests for rare failure paths; a real WebSocket client for the live reject/confirm flow, image/field validation, separate ground-truth scoring, bounded run and GPT usage reporting.

### Crowd implementation (Phase 3c)

1. `backend.py` and `analytics/crowd_service.py`: one startup detector, independent idle camera worker, search-priority reuse, paired snapshots and canonical messages.
2. `analytics/moving_crowd.py` and `analytics/shared_detector.py`: venue polygon assignment using projected feet, YAML capacities, sampled coverage/last-seen bookkeeping, and shared serialized inference. The old tracking/footfall demo is not used.
3. `dashboard.html`: replace symbolic snapshot with camera photograph; use `total_people_in_view` and canonical zone fields, show unknown/stale/partial zones; remove footfall, dwell and ranking cards.
4. `simulation/aware_sim/config/zones.yaml`: read existing definitions; no zone redesign is needed.
5. Crowd evaluation/tests: compare camera-visible estimates with separate ground truth at matching simulation time, including outside-zone people and partial coverage.

`drone_state.py`'s camera-center report may remain for other callers, but must no longer supply person_location. No model, patrol behavior, or rejection-memory redesign is part of this protocol change.

## Phase 3c implementation files

backend.py owns CrowdService and one injected DINO detector. analytics/crowd_service.py schedules counts and maps fields; analytics/shared_detector.py serializes inference and shares raw results. SearchController/get_vlm_pipeline/run_on_video accept detector injection and a post-search frame observer. dashboard.html renders snapshots and coverage-aware counts. HTTP routes are retained.
