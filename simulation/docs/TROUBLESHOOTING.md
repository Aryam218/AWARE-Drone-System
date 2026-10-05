# Troubleshooting (detailed)

Always start with `aware_stop`. Then find your symptom below.

## Installation

### `Conflicting values set for option Signed-By ... packages.ros.org`
Two entries for the ROS package source exist (an old and a new method).
```bash
ls /etc/apt/sources.list.d/ | grep -i ros
```
Keep `ros2.sources` and remove the old one:
```bash
sudo rm /etc/apt/sources.list.d/ros2.list
sudo rm -f /usr/share/keyrings/ros-archive-keyring.gpg
```
Then run `bash setup/install.sh` again.

### `error: externally-managed-environment` when using `pip`
Ubuntu protects its own Python. Never install with plain `pip` outside an
environment. Use the project's environment: `aware`, then `pip install ...`.

### `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
A NumPy 2 got into your personal folder (`~/.local`). Remove it (outside any venv):
```bash
python3 -m pip uninstall -y numpy --break-system-packages
python3 -c "import numpy; print(numpy.__version__)"     # must print 1.x
```

## Starting

### Gazebo window blank (`N/A` at the bottom) or black
1. `aware_stop`
2. Check nothing is left: `ps aux | grep -E "gz sim|px4" | grep -v grep` (should print nothing)
3. `aware_start` and **wait**: 248 people take up to a minute to load.

### PX4 tab: `Arming denied` / preflight fails
In the `pxh>` prompt type `commander check` to see the exact reason.
- "No connection to the GCS": start `aware_patrol` first (it is a ground station).
- GPS / position not ready: wait 20 seconds and try again.

### The 3D view keeps following the drone
That happens only if PX4 opens the window itself. Use our scripts (`aware_start`
or `launch/3_gui.sh`), which open it separately.

## Performance

### Check which GPU is used
```bash
nvidia-smi
```
While the simulation runs, `gz sim` processes should be listed. If not, the
integrated graphics is doing the work: install the NVIDIA driver
(`sudo ubuntu-drivers install`, restart).

### Measure the simulation speed
```bash
gz topic -e -t /world/aware_expo/stats | grep --line-buffered real_time_factor
```
1.0 = real time. To speed it up: close the 3D window, generate a smaller crowd
(`--scale 0.5`), reduce outfits, keep `far_clip_m` small in `drone.yaml`, plug in
the charger and set the power mode to Performance.

## Known rules (learned the hard way)

| Rule | Why |
|---|---|
| Every actor must loop (`<loop>true</loop>`) | A non-looping actor froze the whole simulation |
| Standers use `talk_b.dae` as skin **and** animation | Walk animation on a standing person plays "walking in place"; mixing files breaks the skeleton |
| Server first, window second | A window opened too early stays blank |
| Don't edit PX4's own files | We add our drone and world next to them (symlinks), so PX4 updates don't erase our work |
