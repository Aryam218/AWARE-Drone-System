#!/usr/bin/env bash
# Loaded by ~/.bashrc in every new terminal (added by setup/install.sh).
# Gives you short commands; see README.md "Shortcuts".
export AWARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
[ -f /opt/ros/jazzy/setup.bash ] && source /opt/ros/jazzy/setup.bash

alias aware='cd "$AWARE_ROOT" && source "$AWARE_ROOT/aware_venv/bin/activate"'
alias aware_start='"$AWARE_ROOT/launch/start_all.sh"'
alias aware_stop='"$AWARE_ROOT/launch/stop.sh"'
alias aware_patrol='"$AWARE_ROOT/launch/5_patrol.sh"'
alias aware_camera='"$AWARE_ROOT/launch/camera_view.sh"'
alias aware_check='bash "$AWARE_ROOT/setup/check.sh"'
