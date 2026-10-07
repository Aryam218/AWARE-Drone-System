# Dependency compatibility verification

7 October 2026. Tested on Ubuntu 22.04 / ROS 2 Humble with `simulation/ai_venv`.

## What changed

- Typer: 0.13.1 → 0.27.3. This version uses its own embedded Click implementation.
- Click: 8.0.4 → 8.4.2. Installed Hugging Face Hub 1.33.0 requires Click >=8.4.2,<9.
- setuptools: 84.0.0 → 78.1.1. Installed PyTorch 2.14.1 requires >=77.0.3 and
  ROS colcon-core 0.21.3 requires >=30.3.0,<80. The initially proposed 75.6.0
  pin does not satisfy PyTorch and was not applied to the working environment.

`pyproject.toml` declares these pins for editable installs and uses the same
setuptools version for builds. `constraints-ai.txt` applies them to requirements
and installer commands. The installer now runs `pip check` and fails on a conflict.
Two CLI tests check option parsing/dispatch and subcommand help after the Typer change.

## Results

Trial dependencies were installed under `/tmp` before modifying `ai_venv`.
The actual environment then passed:

- `python3 -m pip check`: **No broken requirements found**.
- **39** search, geometry, quota, dashboard protocol and crowd regression tests.
- **5** CLI tests, including backend option dispatch and all subcommand help pages.
- Dashboard JavaScript tests for acknowledgements, locations, errors, crowd snapshots,
  partial coverage, occupancy and observation ages.
- Imports of pipeline_runner, backend and rclpy.
- A real backend startup with its cached Grounding DINO model and ROS camera worker,
  one CPU inference on a synthetic black image, validated crowd_analytics/crowd_snapshot
  WebSocket messages, and clean worker/backend shutdown (8.55 s total).
- Installer shell syntax and Git whitespace checks.

The runtime smoke test used ROS domain 43 and offline cached model files. It made
no GPT requests and did not start a Gazebo flight. A black image correctly returned
zero people. These checks establish the tested paths work; they cannot guarantee
that every possible runtime situation is crash-free.

Model libraries remained unchanged: torch 2.14.1, torchvision 0.29.1,
transformers 5.19.0, huggingface-hub 1.33.0, numpy 1.26.4 and
opencv-contrib-python 4.9.0.80. This is a targeted dependency fix, not a complete
lock of all transitive packages or a fresh OS installation test.

## Repeat the checks

From the repository root:

```bash
source /opt/ros/humble/setup.bash
source simulation/ai_venv/bin/activate
python3 -m pip check
cd rasid_video_pipeline
python3 -m unittest test_crowd_service test_moving_crowd test_crowd_projection test_crowd_tracking_sampling test_dashboard_followups test_dashboard_protocol test_aware_geo test_aware_quota test_item10
python3 -m unittest discover -s tests -p test_ci.py
node tools/test_dashboard_ui.js dashboard.html
python3 -c "import pipeline_runner; import backend; import rclpy"
```
