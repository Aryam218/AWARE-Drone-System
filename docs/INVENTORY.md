# Dashboard and crowd-analysis inventory

Checked on 7 October 2026 in `/home/aryam/Downloads/rasid_video_pipeline`.

This is a source-code inventory. No dashboard, crowd-model inference, simulation, or GPT calls were started for this audit. The crowd module import was checked without constructing the model. Earlier headless Gazebo tests exercised the missing-person pipeline and patrol, **not** the HTML dashboard or the crowd-analytics class. Source support is not evidence of an end-to-end dashboard test.

## Dashboard goals

The dashboard has two sections. (A) Crowd analysis: a live snapshot of the venue from the drone camera, and below it statistics such as total people and people per zone. (B) Missing person: a description field and a Start search button. When a candidate is found, the drone's camera image of them appears; the operator clicks Confirm or Reject. Confirm: the dashboard shows the person's location. Reject: the drone keeps searching for another candidate.

## Main findings

1. There is one dashboard frontend in this repository: `rasid_video_pipeline/dashboard.html`, a single HTML/CSS/JavaScript file with no build step.
2. It points to the FastAPI backend at `ws://localhost:8000/ws`, but **its protocol does not match `backend.py`**. The frontend sends search commands over WebSocket; the backend reads and discards incoming WebSocket text and expects HTTP requests. The frontend and backend also use different outgoing event names and image formats.
3. `backend.py` only runs missing-person search. It never constructs `CrowdAnalytics` and emits no crowd-analytics messages. There is no live crowd snapshot route or crowd worker attached to it.
4. Crowd analytics is real detector/tracker code, not a density-map model. Its default zones are hard-coded image thirds, not the seven polygons in the Gazebo venue.
5. Dashboard confirmation location is currently an approximate **camera-center** ground point, not the confirmed person's box/verified ground position. The headless pipeline's successful position measurements do not validate this separate dashboard location calculation.

## Files and backends

| File | Responsibility / relationship |
| --- | --- |
| `rasid_video_pipeline/dashboard.html` | Actual browser frontend; inline styles and JavaScript; only network connection is `ws://localhost:8000/ws`. |
| `rasid_video_pipeline/backend.py` | FastAPI missing-person dashboard service: HTTP `/start`, HTTP `/confirm`, WebSocket `/ws`. Runs `pipeline_runner.run_on_video()` in a background thread. |
| `rasid_video_pipeline/pipeline_runner.py` | Search controller and callbacks; `video_path="ros"` selects the ROS camera. Publishes verified target positions to the patrol separately from dashboard location reporting. |
| `rasid_video_pipeline/aware_status.py` | Publishes ROS search state, not browser WebSocket messages. Backend callbacks publish candidate/confirmed/rejected/lost state for the patrol. |
| `rasid_video_pipeline/drone_state.py` | Reads ROS drone state/camera calibration. Its `location()` method supplies the backend's current confirmation payload. |
| `rasid_video_pipeline/analytics/crowd_analytics.py` | Independent crowd detector/tracker/statistics class. No browser connection or HTTP service. |
| `rasid_video_pipeline/analytics/test_crowd.py` | Standalone recorded-video crowd demo with an OpenCV window and console statistics. |
| `rasid_video_pipeline/run_backend.py` | A different model RPC server launcher, default port 3000; not the dashboard FastAPI service. |
| `rasid_video_pipeline/cloud_track/api_functions/run_backend.py` | Builds a detector/VLM pipeline and starts the RPC server. |
| `rasid_video_pipeline/cloud_track/rpc_communication/rpc_server.py` | WSGI HTTP transport carrying binary MessagePack RPC; exposes public RPC `run_inference`, not dashboard JSON routes or WebSocket messages. Default launcher selects `sam_hq` and LLaVA. No reference to it in the dashboard. |
| `simulation/launch/aware_run.sh` | `--dashboard` launches `uvicorn backend:app --host 0.0.0.0 --port 8000` by default. |

The backend does not serve `dashboard.html`; open it separately or serve it with a static file server. `rasid_video_pipeline/README.md` instead documents port **8001** and a video-path input; current HTML uses **8000** and has no video-path field. Those README instructions describe an older interface. The hard-coded `localhost` also means the browser expects the backend on the browser's own machine.

Sources: [frontend configuration](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/dashboard.html:1975), [backend](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/backend.py:276), [older launch instructions](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/README.md:38), [RPC launcher](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/run_backend.py:1).

## Every HTTP endpoint the frontend uses

**None.** No `fetch`, XMLHttpRequest, HTTP form submission, or HTTP API request is present in the frontend. Candidate images are assigned to `<img src>` from backend-supplied strings (`image`, `image_url`, or `frame`); those could cause ordinary image-resource requests if a sender provided URLs, but no fixed image endpoint exists in this frontend or backend.

## Every WebSocket message the frontend uses

One socket: `ws://localhost:8000/ws`. It parses incoming text as JSON. On disconnect it attempts reconnection after two seconds. Connection success changes the label to “System connected”; that only proves socket connection, not working inference or protocol compatibility. There is no heartbeat or ping message implemented in the frontend.

### Browser → server

| Action | Exact payload | Current backend treatment |
| --- | --- | --- |
| Start search | `{"type":"start_person_search","description":"<entered text>"}` | Received as text and ignored. Does not call `/start`. No `video_path` is supplied. |
| Confirm | `{"type":"confirm_candidate"}` | Ignored. Does not call `/confirm`. |
| Reject | `{"type":"reject_candidate"}` | Ignored. Does not call `/confirm`. |

### Server → browser: messages the frontend recognizes

| Frontend discriminator | Fields read | UI effect / compatibility |
| --- | --- | --- |
| `type="crowd_analytics"`, **or any message with truthy `analytics`** | Uses `analytics` object if present, otherwise whole message; reads crowd fields listed below. | Updates crowd cards. Backend never emits it. |
| `type="candidate"` | `confidence` or `score`; `image`, `image_url`, or `frame`. | Displays candidate and confidence. Backend emits `candidate_found` with raw base64 in `crop_jpeg_b64`, so neither discriminator nor image field matches. |
| `type="search_status"` | `message`, default “Searching...”. | Updates searching text. Backend never emits it. |
| `type="person_location"` | `location` or `coordinates`, default “Location received”; formats `latitude/longitude`, `lat/lon`, string, or object JSON. | Shows confirmed card/location. Backend emits `target_confirmed`, which is ignored. Its location is also a nested object. |

Crowd input aliases read by the frontend:

- Zone collection: `zones`, `zone_analytics`, or `areas`; object map or array. Array labels use `name`, `zone`, `area`, or generated `Area N`.
- Total people: `total_people` or `people`; otherwise sum zone `current_people`, `people`, or `count`.
- Total footfall: `total_footfall` or `footfall`; otherwise sum zone `unique_footfall` or `footfall`.
- Zone count: `current_people`, `people`, or `count`. **Does not read the crowd model's `current_count`.**
- Capacity: `capacity`. Occupancy: `occupancy` or `occupancy_percent`; otherwise count/capacity. Positive occupancy values ≤1 are treated as fractional ratios and multiplied by 100; displayed bars are capped at 100%.
- Crowded flag: `crowded` or `is_crowded`, or occupancy ≥80%.
- Dwell: `average_dwell`, `avg_dwell`, or `dwell_time`. **Does not read the model's `average_dwell_seconds`.**
- Timestamp: `timestamp` or `last_analyzed`; otherwise browser current time.
- Ranking is rebuilt locally from zone footfall. It does not consume the model's `footfall_ranking` list.

Sources: [message router](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/dashboard.html:2093), [crowd inputs](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/dashboard.html:2143), [outgoing controls](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/dashboard.html:2846).

## Every HTTP endpoint `backend.py` offers

| Method / path | Request | Response / effect | Frontend uses it? |
| --- | --- | --- | --- |
| `POST /start` | JSON: required `video_path`, required `description`, optional `category` default `person`. Whitespace stripped; `video_path="ros"` selects live camera. | Starts one background worker, returns `{"ok":true,"status":"searching","output_video_path":"outputs/annotated_...mp4"}`. Empty fields or an already-running worker return `{"error":"..."}`. | No. |
| `POST /confirm` | JSON: `{"decision":"confirm"}` or `{"decision":"reject"}`. | Queues operator action, returns `ok`, `decision`, and `status` (`confirming_target` or `resuming_search`). Invalid decision or no waiting candidate returns `{"error":"..."}`. | No. |
| `GET / HEAD /openapi.json` | None | FastAPI-generated API schema. | No. |
| `GET / HEAD /docs` | None | Default FastAPI Swagger UI. | No. |
| `GET / HEAD /docs/oauth2-redirect` | None | Default Swagger OAuth redirect helper; no application authentication flow is defined here. | No. |
| `GET / HEAD /redoc` | None | Default FastAPI ReDoc documentation. | No. |

The last four are implicit FastAPI defaults, not explicitly decorated application routes. Their GET and HEAD methods were checked against the locally installed FastAPI; the three application routes were checked directly from the backend syntax tree. Pydantic rejects missing/wrongly typed bodies with HTTP 422. Application validation failures above return an `error` JSON object using the default HTTP 200 response. CORS middleware also answers eligible preflight `OPTIONS` requests; this is not another application endpoint.

There is no application `GET /`, crowd endpoint, image stream/snapshot endpoint, stop-search endpoint, current-state endpoint, video-download endpoint, or output-file static mount. The returned annotated video path is a filesystem path, not a served download URL.

Sources: [request models](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/backend.py:90), [/start](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/backend.py:542), [/confirm](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/backend.py:640).

## Every WebSocket message `backend.py` offers

`/ws` accepts a socket and broadcasts JSON text to connected clients. Incoming text is read without parsing or dispatch. It sends no initial state, command acknowledgements, heartbeat responses, or replay after reconnect.

| Backend `type` | All message-specific fields | Trigger | Frontend recognizes it? |
| --- | --- | --- | --- |
| `candidate_found` | `crop_jpeg_b64`, `justification`, `frame_index` | New candidate; sets awaiting-decision flag. | No. |
| `tracking_update` | `frame_jpeg_b64`, `score`, `frame_index`, `confirmed` | Tracking event, only when `frame_index % 10 == 0`. | No. |
| `target_confirmed` | `frame_jpeg_b64`, `crop_jpeg_b64`, `frame_index`, `location` (object or null) | Operator confirmation consumed. | No. |
| `candidate_rejected` | `frame_index`, `status="searching"` | Operator rejection consumed. | No. |
| `track_lost` | `frame_index`, `was_confirmed`, `status="searching"` | Candidate/confirmed track lost. | No. |
| `finished` | `output_video_path` | Video/source loop finishes normally. | No. |
| `error` | `message` | Exception escapes the search worker. | No. |

All image fields are raw JPEG base64, without a `data:image/jpeg;base64,` prefix. Backend messages do not include event simulation timestamps or track IDs. `score` in a tracking update is not a GPT match probability, and `candidate_found` contains no confidence field.

The recently added insufficient-quota warning is connected to `run_live.py`, but `backend.py` does not pass `on_verification_error` to the runner. Therefore its generic `error` message does not currently expose that handled permanent verification failure to the dashboard.

The `target_confirmed.location` object contains `drone: {lat,lon,alt_rel_m,heading_deg}`, `target_estimate: {lat,lon,meters_ahead_of_drone}` or null, and a `note`. `DroneStateReader.location()` computes the view-center offset from altitude, heading, and fixed camera tilt. It does not use the candidate bbox or roll/pitch; it is not the verified target location that the patrol receives.

Sources: [candidate and tracking callbacks](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/backend.py:317), [confirmation location](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/backend.py:381), [WebSocket receive loop](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/backend.py:718), [location calculation](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/drone_state.py:174).

## Status definitions

- **works**: a self-contained local UI behavior supported by inspection, or a specifically identified behavior proven in the earlier headless test. It does not mean this dashboard was tested end to end.
- **exists but untested with the Gazebo camera**: supporting implementation exists, but that particular dashboard/model route has not been exercised with the camera.
- **missing**: required data path, handler, or integration is absent. Existing widgets do not make an absent connection work.
- **shows fake or hard-coded data**: value or region is a synthetic default, misleading fallback, or local optimistic state rather than measured/acknowledged data. Waiting dashes are placeholders, not fabricated measurements.

### A. Crowd analysis

| Dashboard shows/does | Intended/current source | Status | Evidence / limitation |
| --- | --- | --- | --- |
| Live venue camera snapshot | Would need camera JPEG/frame endpoint or WS message; none exists for crowd. | missing | “Crowd Analysis Snapshot” currently contains zone bars, not an image element or camera photograph. |
| Area overview cards/bars | Expected WS `crowd_analytics` / `analytics` → `zones`. | missing | Renderer exists, but no backend crowd producer. |
| Total people in current image | Expected `total_people`; model emits it in `get_analytics()`. | missing | No model-to-dashboard transport. This count is visible tracks in one frame, not total people throughout the venue. |
| People per area | Expected zone `current_people/people/count`; model emits `current_count`. | missing | No producer plus a field mismatch; passing model output unchanged displays zero per zone. |
| Venue-zone assignments | No current dashboard/model source for venue polygons. | missing | Model does not load `zones.yaml` or project camera boxes into world coordinates. |
| Model's default area names and capacities | Crowd class defaults: Zone A/B/C image thirds, capacity 10 each. | shows fake or hard-coded data | Synthetic test regions and capacities, not physical venue zones. Not currently transmitted to the page. |
| Occupancy percentage and bars | Expected `occupancy/occupancy_percent`, or count/capacity; model provides `occupancy_percent`. | missing | Computation/renderer exists, no producer. Capacity defaults are synthetic unless supplied. |
| Total footfall | Expected top-level `total_footfall/footfall`, else sum zone unique visitors. | missing | Model emits per-zone `unique_footfall`, no total. Summing zone visitors can count the same person multiple times across areas. |
| Crowded-area count | Frontend derives from `crowded/is_crowded` or ≥80% occupancy. | missing | No crowd message. Default model threshold is also 80% capacity, not people per square metre. |
| Overall Low/Moderate/High level | Frontend averages zone occupancy; hard-coded thresholds 50%/80%. | missing | Formula exists, no measured input; equal-weight average is not whole-venue occupancy. |
| Per-zone average dwell | Expected `average_dwell/avg_dwell/dwell_time`; model emits `average_dwell_seconds`. | missing | No producer plus a field mismatch. Model output unchanged renders a dash. |
| Most visited areas ranking | Frontend sorts zone `unique_footfall/footfall`. | missing | Renderer exists; no producer. Model also offers `footfall_ranking`, which frontend ignores. |
| “Last analyzed” time | Expected `timestamp/last_analyzed`; model emits neither. | shows fake or hard-coded data | If a payload omits time, frontend substitutes browser current time; this is receipt/display time, not camera observation time. Initially shows “No analysis received yet.” |
| Crowd model on camera frames | `CrowdAnalytics.analyze_frame(BGR_frame, timestamp)` accepts an image. | exists but untested with the Gazebo camera | Class is frame-source agnostic; existing executable demo reads only recorded video. No camera worker currently wires it to the dashboard. |

### B. Missing person and shared connection

| Dashboard shows/does | Backend message/endpoint or local behavior | Status | Evidence / limitation |
| --- | --- | --- | --- |
| Enter description / reject empty input | Local textarea and `startSearch()` validation. | works | Local UI only; blank description triggers an alert. |
| Socket connected/disconnected indicator and reconnect | WS `/ws`, browser `onopen/onclose`, two-second retry. | exists but untested with the Gazebo camera | Connection does not prove protocol compatibility or a running search. |
| Start search from button | Frontend sends WS `start_person_search`; backend requires `POST /start`. | missing | Incoming WS command is discarded. Backend also requires `video_path`, which HTML does not provide. |
| Start ROS search through the backend API directly | `POST /start` with `video_path="ros"`. | exists but untested with the Gazebo camera | ROS support is in the runner; earlier headless test bypassed this FastAPI worker. |
| Searching/progress message | Frontend expects WS `search_status.message`. | shows fake or hard-coded data | Start button immediately claims it is analyzing the drone feed, even if command was ignored/disconnected. No server `search_status` message exists. |
| Candidate camera image | Frontend expects WS `candidate.image/image_url/frame`; backend offers `candidate_found.crop_jpeg_b64`. | missing | Name, field, and base64-as-image-source conventions mismatch. Backend cropping exists but dashboard cannot consume it. |
| GPT explanation | Backend `candidate_found.justification`. | missing | Current frontend does not render justification. |
| Match confidence | Frontend `candidate.confidence/score`, default zero. | shows fake or hard-coded data | No candidate-confidence value is sent. If a compatible candidate message lacks it, frontend shows 0.0%, not a measured probability. |
| Confirm button actually confirms target | Frontend WS `confirm_candidate`; backend `POST /confirm`, decision `confirm`. | missing | Command ignored. Page immediately shows “Person Confirmed” without backend acknowledgement. |
| Reject button actually resumes search | Frontend WS `reject_candidate`; backend `POST /confirm`, decision `reject`, then `candidate_rejected`. | missing | Command ignored, despite local “Searching for another match...” text. |
| Backend confirm/reject callbacks and patrol reaction | HTTP actions → runner callbacks → ROS `publish_status`. | exists but untested with the Gazebo camera | Underlying runner/patrol flow worked in the earlier headless test; browser-to-HTTP integration did not participate. |
| Confirmed person's location | Frontend `person_location`; backend `target_confirmed.location`. | missing | Event mismatch. Existing backend estimate is camera center, not the person's verified position; null when drone telemetry unavailable. |
| Continuing live tracking preview | Backend `tracking_update.frame_jpeg_b64` every 10 processed frames. | missing | Frontend ignores it; this is also far slower than camera FPS on CPU. |
| Track-lost/reset/search-again feedback | Backend `track_lost`, `candidate_rejected`. | missing | Frontend handles neither. |
| Search finished / annotated recording link | Backend `finished.output_video_path`. | missing | Frontend ignores it; no route serves the output file. |
| Worker failures | Backend `error.message`. | missing | Frontend ignores `error`; only developer console catches malformed JSON. |
| “GPT verification OFF” warning | Runner's optional `on_verification_error`. | missing | `run_live.py` subscribes; `backend.py` does not. No dashboard message/card exists. |
| Stop search | No endpoint/button/runner stop control in dashboard path. | missing | Live source is indefinite unless source ends or process is stopped; `/start` refuses a second active worker. |

## Crowd-analysis model: simple explanation

**What it is.** `CrowdAnalytics` uses **Grounding DINO Tiny**, loaded as `IDEA-Research/grounding-dino-tiny`, to find people in a picture. It asks the detector for `person`. ByteTrack links boxes between processed pictures, and additional bookkeeping tries to preserve a logical person ID across brief ID switches. Counting, footfall, dwell, and crowded flags are ordinary calculations around these detections, not another neural density/counting model. GPT is not used by this crowd class.

**What goes in.** One OpenCV BGR image array at a time, optionally with its observation timestamp in seconds. A video must be decoded into frames by a separate caller. Default inference width is 1280 pixels; wider images are resized for the detector and boxes scaled back. Detection uses RGB/PIL internally. CUDA is selected if available, otherwise CPU. The constructor loads model weights and processor; first use can need downloaded/cache assets.

**What comes out.** `analyze_frame()` returns `(annotated_frame, people_count, boxes, scores)`. Boxes are person detections after duplicate-overlap suppression, while `people_count` is the number of current logical people represented by ByteTrack, so these numbers need not match. The annotated image draws zones, tracked boxes/IDs, and a total count. `get_analytics()` returns:

```text
{
  total_people,
  zones: {
    zone_name: {
      current_count, capacity, occupancy_ratio, occupancy_percent,
      crowded, unique_footfall, average_dwell_seconds
    }
  },
  footfall_ranking: [{zone, unique_footfall}, ...]
}
```

There is **no density map, segmentation mask, per-square-metre density, geographic coordinates, or JSON list of tracked-person boxes/IDs in `get_analytics()`**. Track state and boxes exist inside/alongside the class; the statistical getter is not a camera/zone projection API. Total count excludes stale tracks; older track memory is retained briefly for continuity and dwell/footfall accounting. Visibility loss, moving camera viewpoint, and imperfect identities can affect footfall and dwell.

**How to run the existing demo.** From the AI folder, with the existing AI environment:

```bash
cd /home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline
source ../simulation/ai_venv/bin/activate
PYTHONPATH="$PWD" python3 analytics/test_crowd.py
```

The script opens the hard-coded relative file `7415436-uhd_3840_2160_25fps.mp4` (present in this checkout), processes selected frames, prints statistics, and opens an OpenCV window. Press Q to exit. It targets five analytics updates per video second by frame skipping; that is not a guaranteed five wall-clock inferences per second on this CPU. It passes the recorded-video timestamp for dwell calculations. It is a demo, not a small unit test, and running it constructs the detector.

To use Gazebo later, a separate caller must read frames from `RosFrameSource`, pass each BGR image and `last_stamp` to `analyze_frame()`, and transmit the image/statistics to the dashboard with an agreed message schema. That wiring does not exist. Reading actual venue zones also requires camera-pixel-to-ground projection plus polygon membership; simply substituting the YAML polygons into normalized rectangles will not work.

Sources: [crowd constructor/default zones](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/analytics/crowd_analytics.py:68), [image-zone membership](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/analytics/crowd_analytics.py:614), [analytics output](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/analytics/crowd_analytics.py:1185), [frame API](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/analytics/crowd_analytics.py:1587), [model identity](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/cloud_track/foundation_model_wrappers/grounding_dino_huggingface_wrapper.py:17), [demo](/home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline/analytics/test_crowd.py:1).

## Does it know about zones?

Yes, but only **normalized image rectangles** supplied to the constructor. It tests the bottom-center of each person's box against these rectangles. Defaults are left/middle/right strips called Zone A/B/C. They move with the camera's view and are not stable geographic areas. Their default capacities are 10 each.

The real venue zones are defined separately in `simulation/aware_sim/config/zones.yaml`, using polygons in Gazebo world metres: x East, y North, origin at the floor center. Floor size is 80 × 50 m. Each zone has an ID, type, capacity and polygon; booth-front zones also identify a booth.

| Venue zone ID | Type | Capacity |
| --- | --- | --- |
| `entrance` | `area` | 60 |
| `walkway` | `area` | 120 |
| `plaza` | `area` | 250 |
| `booth_a_front` | `booth_front` | 25 |
| `booth_b_front` | `booth_front` | 25 |
| `booth_c_front` | `booth_front` | 25 |
| `booth_d_front` | `booth_front` | 25 |

These polygons cover selected regions, not the entire floor: a person can legitimately be outside every zone. World latitude/longitude and elevation are also defined in the YAML. Crowd analytics does not import/read that YAML, use those capacities, or use drone GPS/attitude. The “crowd analytics (later)” comment in the YAML describes intended integration, not current functionality.

Source: [venue blueprint](/home/aryam/Downloads/rasid_video_pipeline/simulation/aware_sim/config/zones.yaml:1).

## Ground truth and future count validation

**Yes: `/aware/ground_truth` includes every person's `zone`.** `GroundTruth.zone_of(x,y)` checks the YAML polygons and assigns the first matching zone ID, or null outside all zones. Each `people` element includes `name`, `role`, `outfit`, `looks`, world `x/y`, `lat/lon`, and `zone`. Top-level data includes `sim_time`, `scenario`, and `zone_counts`, with a count for every configured zone.

The default publisher runs at two wall-clock updates per second using the latest `/clock`. Positions come from interpolating the generated actors' scripted paths; this is a scripted reference, not an independent visual detector. `zone_counts` counts all scripted people in those polygons, including people outside the drone image, behind occluders, or otherwise not detectable. Therefore compare visible-image counts against camera-visible ground truth, or explicitly distinguish whole-venue counts from visible counts. Comparing one partial camera frame directly with all zone occupants would be misleading.

`/aware/ground_truth/missing_person` is a separate `geometry_msgs/PointStamped` topic: x East/y North in metres, stamped with simulation time. It has no zone label. The full ground-truth topic is `std_msgs/String` containing JSON, not a frontend HTTP/WS feed. Neither crowd analytics nor the dashboard currently subscribes to it.

Start the reference publisher alongside world/bridge with:

```bash
bash /home/aryam/Downloads/rasid_video_pipeline/simulation/launch/ground_truth.sh
```

Inspect it in a ROS-sourced terminal on ROS domain 42 with `ros2 topic echo /aware/ground_truth`. Keep these reference answers in scoring/evaluation only, never as the AI's input or as supposedly measured dashboard counts.

Sources: [zone assignment and JSON fields](/home/aryam/Downloads/rasid_video_pipeline/simulation/aware_sim/scripts/ground_truth.py:124), [publisher topics](/home/aryam/Downloads/rasid_video_pipeline/simulation/aware_sim/scripts/ground_truth.py:109).

## Scope of this audit

Only documentation was changed: `AGENTS.md` and this inventory. No application code, dashboard protocol, model weights, or configuration was changed. No live dashboard/crowd inference test was performed. The implementation gaps above are recorded for planning, not repaired in this task.
