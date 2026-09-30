# Rasid — video-in search pipeline (built directly on CloudTrack)

This is the "input a video, get the solution" version: detection + VLM
verification + local tracking + a confirm/reject dashboard, all driven by a
video file today, ready to swap for the live drone feed later.

Almost none of this reimplements CloudTrack — `pipeline_runner.py` mostly
just calls their real classes (`CloudTrack`, `get_vlm_pipeline`,
`OpenCVWrapper`, `VideoStreamer`) and adds callbacks so a dashboard can react
to what they're already doing internally.

## Setup

1. Clone CloudTrack somewhere and install it so `import cloud_track...` works:
   ```
   git clone https://github.com/yblei/CloudTrack.git
   pip install -e CloudTrack
   ```
   (it has a `pyproject.toml`, so editable-install works directly)

2. Install this project's own requirements:
   ```
   pip install -r requirements.txt
   ```

3. Patch the CUDA-only line if you don't have an NVIDIA GPU — same fix as
   before, in CloudTrack's own file:
   `CloudTrack/cloud_track/foundation_model_wrappers/grounding_dino_huggingface_wrapper.py`
   ```python
   self.device = "cuda"  ->  self.device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

4. Set your OpenAI key (used for VLM verification via `gpt-4o-mini`):
   ```
   export OPENAI_API_KEY=sk-...
   ```

## Run it

```
uvicorn backend:app --port 8001 --reload
```

Then open `dashboard.html` directly in a browser (no build step — it's one
self-contained file). Type in a path to a video file that's on the same
machine as the backend, describe who you're looking for, hit **Start
search**. When a candidate is found, confirm or reject it right in the
dashboard; while you're deciding, the drone (in this version: the video
loop) never stops tracking — see the note below.

The annotated output video is written to `outputs/annotated_<name>.mp4`
once the video finishes.

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
