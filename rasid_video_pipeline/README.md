# Rasid — video-in search pipeline (built directly on CloudTrack)

This is the "input a video, get the solution" version: detection + VLM
verification + local tracking + a confirm/reject dashboard, all driven by a
video file or the live drone camera, with continuous crowd counting on the dashboard.

Almost none of this reimplements CloudTrack — `pipeline_runner.py` mostly
just calls their real classes (`CloudTrack`, `get_vlm_pipeline`,
`OpenCVWrapper`, `VideoStreamer`) and adds callbacks so a dashboard can react
to what they're already doing internally.

## Setup

Use the simulation installer described in [simulation/README.md](../simulation/README.md).
It creates `simulation/ai_venv` with access to ROS 2 and installs the CloudTrack
implementation included in this repository; do not install a separate upstream copy.

To update an existing AI environment, run from the repository root:

```bash
source /opt/ros/humble/setup.bash  # use jazzy instead on Ubuntu 24.04
source simulation/ai_venv/bin/activate
python3 -m pip install -c rasid_video_pipeline/constraints-ai.txt -r rasid_video_pipeline/requirements.txt -e rasid_video_pipeline
python3 -m pip check
```

The tested compatibility pins are Typer 0.27.3, Click 8.4.2 and setuptools 78.1.1.
The installer applies the same constraints and checks package compatibility.
See [dependency verification](../docs/DEPENDENCY_FIX.md) for the reasons and tests.
Grounding DINO selects CPU automatically when CUDA is unavailable.

Set `OPENAI_API_KEY` in your environment or the ignored local `.env` file. Never
commit credentials.

## Run it

With the simulation and camera running, use the AI environment above, then:

```bash
cd rasid_video_pipeline
uvicorn backend:app --host 127.0.0.1 --port 8000
```

Open `dashboard.html` directly in a browser. The page connects to
`ws://localhost:8000/ws`. Describe the missing person and press **Start search**;
the default video source is the live ROS camera. Confirm or reject candidates
on the page. Crowd counting runs in the background with the same detector.
The backend now requires ROS at startup for its continuous camera worker.

Annotated search output is written to `outputs/`.

## Why tracking doesn't wait for your confirmation

This isn't something we built — it's how CloudTrack's own `forward()` method
already works (`cloud_track/pipeline/cloud_track.py`): the moment the
backend (detector + VLM) approves a match, `frontend_tracker.init()` runs
immediately, before any notion of a "parent" exists in their code. Our
dashboard just listens for that transition (`SearchController.process_frame`
in `pipeline_runner.py` — compares `tracker_initialized` before and after
each `forward()` call) and shows it to you. If you click **reject**,
`pipeline.reset()` is called — CloudTrack's own method for dropping the
current lock and going back to full search.

## What's real vs. what needs your GPU/API key to verify

I wrote and syntax-checked every file here, and read CloudTrack's actual
source to make sure the class names, method signatures, and return values
match what's really in their repo — but I could not run a live inference
pass myself (no GPU, and this sandbox can't reach the HuggingFace Hub or the
OpenAI API to download the model / make VLM calls). Test the first run on a
short video with 1–2 people in frame before trusting it on anything longer.

## The seam for the simulator, when your EE teammate is ready

Everything video-specific lives in exactly one place: the call to
`run_on_video(...)` inside `backend.py`'s `_worker()` function. When the
Gazebo/ROS2 feed exists, write a `run_on_stream(frame_generator, ...)` in
`pipeline_runner.py` with the same callback signature, and swap that one
call — `SearchController`, the dashboard, and the WebSocket protocol don't
need to change at all.
