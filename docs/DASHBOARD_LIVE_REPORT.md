# Live dashboard WebSocket report

Run: `20261007_142739_dashboard`. Test logs: `/home/aryam/aware_test_logs/20261007_142739_dashboard`.

## What was tested

Headless Gazebo + PX4 + ROS bridge + ground-truth publisher + patrol (`--start auto --loops 4`), using `missing_person --scale 0.5`. The automatic client connected to `/ws`, sent `start_person_search` with `video_path="ros"`, rejected the first candidate after about 3 wall seconds, then confirmed the second after about 3 seconds. The client ran for 480 wall seconds, including model startup. Each image data URL was decoded as a JPEG; expected fields and acknowledgement ordering were checked. Ground truth was read only by the scoring client; the backend/AI received no reference positions.

**WebSocket result: PASS.** Errors: [].

## Candidates and positions

“Within 3 m” is the requested position-based missing-person label, not independent visual identity proof. Both crops show a person in a red top and light trousers. Candidate one is only 0.07 m outside the requested 3 m threshold, so its false label is borderline and does not establish a different person. Ground positions are scored against truth at the verified sighting time; a separate current-error column exposes old positions while the person moves.

| Event | Sim time | Track ID | Error at verified sighting | Within 3 m | Error at event time |
|---|---:|---:|---:|---|---:|
| Candidate 1 (rejected) | 67.192 | -2 | 3.07 m | False | 3.07 m |
| Candidate 2 (confirmed) | 82.900 | 46 | 2.037 m | True | 2.037 m |
| ConfirmedEvent | 99.664 | 46 | 2.037 m | True | 9.742 m |
| RejectedPlace | 82.568 | — | 13.957 m | False | 13.957 m |

Rejected-place exclusion was recorded 1 time(s); a later place-filter skip was observed 0 time(s).
The first candidate ID was not offered again during this run; a new ID does not prove that a different physical person was selected. No later search detection hit the place filter before the second candidate was confirmed, so this run cannot prove that the moving rejected person would always be skipped later. Its temporary track ID cannot by itself persist across frames; the existing ground-place memory was recorded.

## Patrol and tracking

Patrol evidence:

- `AI: CANDIDATE -> pausing the patrol to look at the person`
- `AI: flying to look at the person (15 m away), facing 125 deg`
- `AI: REJECTED -> RESUMING the patrol`
- `AI: CANDIDATE -> pausing the patrol to look at the person`
- `AI: flying to look at the person (45 m away), facing 121 deg`
- `AI: CONFIRMED -> keep following (Ctrl+C = return and land)`

Confirmed tracking previews: 48. Location messages: 48. Lost messages: 0.
Verified-position error for location updates: median 0.526 m; last 0.547 m; maximum 2.525 m. Final observation time: 482.660 sim seconds.

## Protocol checks

| Message | Received |
|---|---:|
| candidate | 2 |
| error | 1 |
| person_location | 48 |
| rejected | 1 |
| search_status | 8 |
| tracking_update | 52 |

The error message was intentionally tested with an unknown command; the socket remained usable. `lost`, `gpt_verification_off`, and worker-failure paths were exercised by mocked WebSocket tests, not forced into the healthy live run. Finite-source `finished` was tested with mocks; the live server was externally stopped at the time limit. Three dashboard Python regressions and ten existing geometry/quota/loss tests passed. A Node test executed the actual page script and verified that clicking Confirm does not show Confirmed until the backend acknowledgement, separate target/drone locations, loss reset, and persistent quota warning. The ai_venv import check passed. The actual page was not opened in a browser during this headless test.

## GPT usage

HTTP attempts: 4. Completed responses: 4. Successful responses: 4. These are observed wrapper HTTP calls, including retries if any; usage comes from the API response.

| Call | Model | HTTP/error | Input tokens | Output tokens | Total |
|---|---|---|---:|---:|---:|
| 1 | gpt-4o-mini | 200 | 8852 | 32 | 8884 |
| 2 | gpt-4o-mini | 200 | 8852 | 34 | 8886 |
| 3 | gpt-4o-mini | 200 | 8852 | 31 | 8883 |
| 4 | gpt-4o-mini | 200 | 8852 | 25 | 8877 |

Total reported tokens: 35,530. No keys, request headers, or API request contents were recorded.

## Warnings and cleanup

- `2026-10-07 14:29:16.564 | WARNING  | cloud_track.pipeline.cloud_track:_redetect:961 - AWARE re-detection: no matching person at the tracked box (miss 1/2): missing: 32 people in view; 0 near last position with wrong colors.`
- `2026-10-07 14:29:16.592 | WARNING  | cloud_track.foundation_model_wrappers.detector_vlm_pipeline:reject_track_id:279 - AWARE: track ID -2 is temporary (not confirmed by ByteTrack); rejection cannot be remembered across frames.`
- `2026-10-07 14:29:34.279 | WARNING  | cloud_track.pipeline.cloud_track:_redetect:961 - AWARE re-detection: no matching person at the tracked box (miss 1/2): missing: 38 people in view; 0 near last position with wrong colors.`
- `2026-10-07 14:29:48.668 | WARNING  | cloud_track.pipeline.cloud_track:_redetect:939 - AWARE re-detection: tracker had slipped off the target; re-found the person nearby at [473, 238, 490, 279]`
- `2026-10-07 14:29:55.922 | WARNING  | cloud_track.pipeline.cloud_track:_redetect:939 - AWARE re-detection: tracker had slipped off the target; re-found the person nearby at [595, 238, 609, 277]`
- `2026-10-07 14:30:10.500 | WARNING  | cloud_track.pipeline.cloud_track:_redetect:939 - AWARE re-detection: tracker had slipped off the target; re-found the person nearby at [621, 328, 639, 379]`
- `2026-10-07 14:30:18.232 | WARNING  | cloud_track.pipeline.cloud_track:_redetect:939 - AWARE re-detection: tracker had slipped off the target; re-found the person nearby at [641, 403, 663, 458]`

PX4 logged `LED: open /dev/led0 failed (22)` in simulation; takeoff and patrol still succeeded. The ground-truth publisher printed `rclpy.executors.ExternalShutdownException` when its process group was stopped during cleanup; its reference data was available throughout the test. No OpenAI API failures or unexpected backend exceptions occurred.

Camera rate was checked before AI startup (`camera_hz.log`, about 13 Hz).
Cleanup called the standard `aware_stop` implementation (`simulation/launch/stop.sh`), then terminated the test server/client and ground-truth process groups. See `aware_stop.log` and `processes_after.log`.

## Most important next fix

Obtain a fresh detector-verified position when applying Confirm. The existing confirmation can report an older, correctly measured anchor while a moving person is already several metres away. Keep the operator decision pending while refreshing that sighting, then acknowledge it with the fresh position. This recommendation is not implemented here.

## Changes in this task

`backend.py` now handles existing WebSocket commands, shares them with the HTTP routes, uses canonical messages and JPEG data URLs, remembers offered candidate IDs, rejects stale/pending decisions, reports errors/quota state, and uses verified target locations. `dashboard.html` displays GPT justification, waits for acknowledgements, shows target/drone coordinates and tracking previews, and handles rejection, loss and errors. `pipeline_runner.py` adds an optional verified-position observer before event callbacks. Crowd messages remain a proposal. No crowd/model/patrol/rejection behavior was changed.

## Run the automatic client again

With the simulation/bridge/ground truth/patrol running, from `rasid_video_pipeline/` in the ROS-sourced ai_venv, use two terminals (same log directory):

```bash
python3 -u tools/dashboard_test_server.py --log-dir /path/to/new/logs
python3 -u tools/auto_dashboard_ws_test.py --log-dir /path/to/new/logs --time-limit 480
```

Create the log directory first. The test-only server records GPT usage and verified-anchor events; the client alone reads ground truth. Stop the server and simulation after the bounded client exits. Regression checks: `python3 -m unittest test_dashboard_protocol test_aware_geo test_aware_quota test_item10 -v`; UI check: `node tools/test_dashboard_ui.js dashboard.html`.
