# Connecting the AI search pipeline to the live simulation

The AI pipeline (`pipeline_runner.py` + `cloud_track`) is in `../rasid_video_pipeline/`.
These three files connect it to the simulated drone camera:

| File | What it does | Goes into |
|---|---|---|
| `ros_frame_source.py` | `RosVideoStreamer`: the live camera, used **exactly like a video file** (`for frame in stream:`) | next to `pipeline_runner.py` |
| `run_live.py` | Runs the search on the live camera from a terminal, with a video window. You play the operator: **c** = confirm, **r** = reject, **q** = quit | next to `pipeline_runner.py` |
| `aware_candidates.py` | Chooses **which** detected people GPT checks: drops the drone's own legs and whole-frame boxes, ranks people by the colours in the description, sends the best 3 | `cloud_track/foundation_model_wrappers/` |

## 1. The AI's Python environment (once)

The AI environment must be able to see ROS 2 and must use **NumPy 1.x**:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages ~/ai_venv          # --system-site-packages lets it see ROS
source ~/ai_venv/bin/activate
cd ~/AWARE-Drone-System/rasid_video_pipeline
pip install -r requirements.txt "numpy<2"
pip install -e . "numpy<2"
pip install supervision python-dotenv openai "numpy<2"
```

Check:
```bash
python3 -c "import rclpy, cloud_track, supervision; print('OK')"
python3 -c "import numpy; print(numpy.__version__)"        # must be 1.x
```

The OpenAI key goes in a file named `.env` in `rasid_video_pipeline/`:
```
OPENAI_API_KEY=sk-...
```
**Never commit `.env` to git.** This repository is **public**: anyone could use the key, billed to you.

## 2. Changes in `pipeline_runner.py`

**a. Import** (after the other imports, never before `from __future__ ...`):
```python
from ros_frame_source import RosVideoStreamer
```

**b. Choose the source** (module level, not inside a class):
```python
def open_stream(video_path: str):
    """'ros' = live Gazebo camera; anything else = a video file, as before."""
    if video_path == "ros":
        return RosVideoStreamer("/aware/camera/image")
    return VideoStreamer(video_path)
```
and in `run_on_video`: `stream = open_stream(video_path)`.

**c. Don't re-open a live stream** (in the output-video block):
```python
        if video_path != "ros":
            stream = VideoStreamer(video_path)
```

**d. Live view callback**: add the parameter to `run_on_video(...)`:
```python
    on_frame: Optional[
        Callable[[np.ndarray, object], None]
    ] = None,
```
and at the **end of the frame loop**, just before the "OPTIONAL VIDEO OUTPUT" block:
```python
        if on_frame is not None:
            on_frame(annotated, event)
```

**e. Release the camera** at the end of `run_on_video`:
```python
    if hasattr(stream, "release"):
        stream.release()
```

## 3. Change in `cloud_track/foundation_model_wrappers/detector_vlm_pipeline.py`

**Why:** ByteTrack confirms a person only after matching them in consecutive
frames. Live, frames are processed seconds apart while the drone moves, so
almost nobody got confirmed and the person in red was never checked.

Import near the top:
```python
from cloud_track.foundation_model_wrappers.aware_candidates import select_candidates
```
Right after `tracked_people = self.person_tracker.update(boxes_filt, scores)`:
```python
        # AWARE: choose WHICH people the VLM checks (see aware_candidates.py)
        tracked_people = select_candidates(
            image,
            boxes_filt,
            scores,
            tracked_people,
            verbal_description,
        )
```

## 4. Run

With the simulation running (`aware_start --no-gui` is lighter):
```bash
source ~/ai_venv/bin/activate
cd <the folder with pipeline_runner.py>
python3 ros_frame_source.py           # test: a window with the live camera (q to quit)
python3 run_live.py                   # the live search
python3 run_live.py --description "a person wearing a red top and white trousers"
```
Then start the patrol (`aware_patrol`, then `commander takeoff` in the PX4 tab).

In the log, each frame prints a line like:
```
AWARE candidates: 151 detections, 2 implausible, colors top=red bottom=white, best color scores=[0.62]
```
`best color scores=[]` means nobody in the view matches the colours, so no GPT call is made.

## Tips

- The AI and the simulation share the GPU. Run the simulation **without** the
  3D window, and check the speed (`real_time_factor` in README section 8).
- Lower altitude (`patrol.yaml`, e.g. 10 m) = bigger people in the image.
- Settings for the candidate filter are at the top of `aware_candidates.py`
  (`MAX_CANDIDATES`, `MIN_COLOR_SCORE`, ...).
