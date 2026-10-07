# Dashboard follow-up live report

Run `20261007_144607_dashboard`; logs: `/home/aryam/aware_test_logs/20261007_144607_dashboard`. Compared with [the previous report](DASHBOARD_LIVE_REPORT.md).

The dashboard now shows `seen N s ago`, calculated from camera simulation timestamps. Every `person_location` already included `position_sim_time`; tests now explicitly check it. Confirm remains immediate and schedules one forced detector check on the next frame. `get_vlm` now uses `image_detail="low"`. The ROS default already existed; this live client omitted `video_path` to exercise it.

**All checks passed:** 16 Python regressions, the actual page-script test, and the ai_venv import check. The eight-minute WebSocket test validated 102 messages: two candidates, one rejection, 43 confirmed previews and 43 location messages. No lost event occurred. The intentional unknown-command error left the socket usable.

## GPT comparison

| | Previous: auto detail | This run: low detail |
|---|---:|---:|
| Successful calls / attempts | 4 / 4 | 4 / 4 |
| Average tokens per call | 8,882.5 | 3,212 |
| Total tokens | 35,530 | 12,848 |
| MATCH / NO_MATCH | 2 / 2 | 2 / 2 |

**Tokens per call fell 63.8%.** Both runs used gpt-4o-mini. The flights produced different crops, so these results do not establish equal accuracy on identical images.

| Call | Input tokens | Output tokens | Total | GPT decision |
|---|---:|---:|---:|---|
| 1 | 3185 | 31 | 3216 | NO_MATCH: no matching outfit visible |
| 2 | 3185 | 30 | 3215 | NO_MATCH: black trousers |
| 3 | 3185 | 23 | 3208 | MATCH: red top, white trousers |
| 4 | 3185 | 24 | 3209 | MATCH: red top, white trousers |

Previously, the negative replies described grey trousers and no matching person; both positive replies described red tops and white trousers. Full replies and API usage are in `server_events.jsonl`.

## Positions and confirmation

| Observation | Error against ground truth |
|---|---:|
| First candidate, rejected | 2.676 m |
| Second candidate, confirmed | 0.160 m |
| First refreshed position | 0.549 m |
| Final confirmed position | 0.488 m |

Both candidates were within the requested 3 m threshold. Changing IDs does not prove they were different people. Rejected-place memory was recorded, but no later place-filter skip occurred. Ground truth was scoring-only.

Confirmation acknowledged a **15.84-second-old position**. The forced check ran on the next frame, refreshed its observation time, and finished **7.39 wall seconds later**. The refreshed message showed `seen 0 s ago`. On this CPU the ordinary interval was already overdue; synthetic tests prove that forcing also works below the 3-second interval. Detector latency still limits refresh speed.

The patrol flew to inspect both candidates, resumed after rejection, and followed after confirmation. Median confirmed-position error was 0.438 m. Six backend warnings covered temporary IDs, brief misses and recovered tracker slips; no API failure or unexpected backend exception occurred. PX4’s LED warning and ground truth’s cleanup exception did not prevent the test.

Camera publishing was checked before AI startup. Scenario: `missing_person --scale 0.5`; patrol: `--start auto --loops 4`. `aware_stop` and process-group cleanup stopped everything cleanly. Full test output is in `regression_tests.log`; images, scoring and messages are retained in the run directory. The browser window was not opened; page behavior was tested through its script.
