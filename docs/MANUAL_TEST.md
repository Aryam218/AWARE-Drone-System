# Run the dashboard and simulated drone

Use three terminals. Start the simulator first: its launcher stops an existing
backend while cleaning up the previous run. Commands below match this computer
(Ubuntu 22.04 / ROS 2 Humble).

## 1. Simulator and drone — terminal 1

```bash
export ROS_DOMAIN_ID=42
bash /home/aryam/Downloads/rasid_video_pipeline/simulation/launch/aware_run.sh --no-ai --scenario missing_person --scale 0.5 --loops 4 --auto-takeoff
```

Wait for the launcher's WORLD, PX4, BRIDGE and PATROL tabs to start. The drone
will take off automatically and fly four patrol rounds. `--no-ai` prevents a
second AI process: the backend below provides the AI. Add `--gui` if you want
the Gazebo 3D window. Four rounds are finite; the patrol is not an endless flight.

## 2. Backend — terminal 2

Start after the simulator launcher has completed startup.

```bash
cd /home/aryam/Downloads/rasid_video_pipeline
source /opt/ros/humble/setup.bash
source simulation/ai_venv/bin/activate
export ROS_DOMAIN_ID=42
cd rasid_video_pipeline
python3 -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`. The real detector loads at startup.
Use `simulation/ai_venv`, not the root `.venv`. Run without `--reload`: watching
PX4's large source tree can exhaust Linux's file-watch limit. The dashboard
expects backend port 8000.

## 3. Frontend — terminal 3

If an old frontend server is still running, reuse it or stop it with Ctrl+C
before starting another server on the same port.

```bash
python3 -m http.server 8002 --bind 127.0.0.1 --directory /home/aryam/Downloads/rasid_video_pipeline/rasid_video_pipeline
```

Open **http://127.0.0.1:8002/dashboard.html**. After dashboard changes, refresh
with Ctrl+Shift+R. Port 8002 serves the page; port 8000 provides its WebSocket
backend. Opening the backend's root URL does not serve the dashboard.

## 4. Try the features

Before starting a search, the missing-person section should say Ready to search.
Crowd snapshots update independently, roughly every 10 seconds on this CPU,
and can take longer under load. They are periodic snapshots, not live video.

Enter a description (for example, a person wearing a red top and white trousers)
and click Start Search. Wait for a candidate; use Confirm or Reject. Confirmation
is shown after the backend applies it, followed by verified position updates.
CPU detector work can take several seconds per frame.

## 5. Check a frozen view

In another terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
ros2 topic hz /aware/camera/image
```

A reported rate means camera frames are arriving. Press Ctrl+C, then check:

```bash
ros2 topic echo /aware/drone/state --once
```

If drone state never arrives, check the PATROL tab for its final message. The
patrol may have finished its rounds or exited with an error. Also check the
backend terminal: a camera feed alone does not prove that the drone is patrolling.

If the simulator is still running and the previous patrol has ended, launch a
new patrol in another terminal (do not run two patrols together):

```bash
export ROS_DOMAIN_ID=42
bash /home/aryam/Downloads/rasid_video_pipeline/simulation/launch/5_patrol.sh --start auto --loops 4
```

## 6. Stop or repeat

```bash
bash /home/aryam/Downloads/rasid_video_pipeline/simulation/launch/stop.sh
```

This stops the simulation, patrol and backend. Stop the frontend HTTP server
with Ctrl+C separately. To repeat a clean test, start again in the order above.
