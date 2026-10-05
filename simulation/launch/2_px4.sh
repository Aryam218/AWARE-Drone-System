#!/usr/bin/env bash
# STEP 2: start PX4 (the autopilot) and spawn our drone on the launch pad.
# Start it AFTER step 1 has finished loading. You get a "pxh>" prompt:
# that's PX4's own command line (e.g. "commander takeoff", "commander land").
source "$(dirname "$0")/_common.sh"
title "2 PX4 (autopilot)"
MODEL="$(drone_model)"
POSE="$(python3 -c "import yaml;p=yaml.safe_load(open('$AWARE_SIM/config/zones.yaml'))['launch_pad']['pose'];print(f'{p[0]},{p[1]},0.3,0,0,{p[2]}')")"
cd "$PX4_DIR" || exit 1
PX4_SYS_AUTOSTART=4014 \
PX4_SIM_MODEL="gz_${MODEL}" \
PX4_GZ_MODEL_POSE="$POSE" \
PX4_GZ_STANDALONE=1 \
PX4_GZ_WORLD=aware_expo \
./build/px4_sitl_default/bin/px4
