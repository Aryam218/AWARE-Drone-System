#!/usr/bin/env bash
# Loaded by ~/.bashrc in every new terminal (added by setup/install.sh).
# Gives you short commands; see README.md "Shortcuts".
export AWARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AWARE_AI_DIR="${AWARE_AI_DIR:-$(cd "$AWARE_ROOT/.." && pwd)/rasid_video_pipeline}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
# ROS 2: Jazzy on Ubuntu 24.04, Humble on Ubuntu 22.04 (whichever is installed)
for _d in jazzy humble; do
  if [ -f "/opt/ros/$_d/setup.bash" ]; then source "/opt/ros/$_d/setup.bash"; break; fi
done
unset _d

alias aware_run='"$AWARE_ROOT/launch/aware_run.sh"'        # everything, one command
alias aware_start='"$AWARE_ROOT/launch/start_all.sh"'      # simulation only (world, PX4, GUI, bridge)
alias aware_stop='"$AWARE_ROOT/launch/stop.sh"'
alias aware_kill='"$AWARE_ROOT/launch/stop.sh"'
alias aware_patrol='"$AWARE_ROOT/launch/5_patrol.sh"'
alias aware_camera='"$AWARE_ROOT/launch/camera_view.sh"'
alias aware_check='bash "$AWARE_ROOT/setup/check.sh"'
alias aware='cd "$AWARE_ROOT" && source "$AWARE_ROOT/aware_venv/bin/activate"'          # simulation Python
alias aware_ai='source "$AWARE_ROOT/ai_venv/bin/activate" && cd "$AWARE_AI_DIR"'         # AI Python