#!/usr/bin/env bash
# STEP 5: autonomous patrol. The script prepares everything, then WAITS:
# when it prints "waiting for YOU", type "commander takeoff" in the PX4 terminal.
# Extra options are passed on, e.g.:  launch/5_patrol.sh --loops 2
source "$(dirname "$0")/_common.sh"
title "5 PATROL"
# shellcheck disable=SC1091
source "$AWARE_ROOT/aware_venv/bin/activate" || { echo "Run setup/install.sh first"; exit 1; }
python3 "$AWARE_SIM/scripts/patrol.py" "$@"
