#!/usr/bin/env bash
# _common.sh: shared settings for every launch script (sourced, not run).
# You normally never need to edit this file.

# Repository root = the folder that contains launch/ (works wherever you cloned it)
export AWARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AWARE_SIM="$AWARE_ROOT/aware_sim"
export PX4_DIR="$AWARE_ROOT/PX4-Autopilot"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"     # "radio channel" so AWARE doesn't mix with other ROS projects

AWARE_ROS=jazzy; [ -f /opt/ros/jazzy/setup.bash ] || AWARE_ROS=humble
# ROS 2 Jazzy (installed system-wide by setup/install.sh)
if [ -f /opt/ros/$AWARE_ROS/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/$AWARE_ROS/setup.bash
else
  echo "ERROR: ROS 2 Jazzy not found. Run setup/install.sh first." >&2
  exit 1
fi

# nv <command>: run a command on the NVIDIA GPU of a laptop with hybrid graphics.
# On machines without NVIDIA it simply runs the command unchanged.
nv() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
     && [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    __NV_PRIME_RENDER_OFFLOAD=1 \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    "$@"
  else
    "$@"
  fi
}

# PX4's Gazebo paths (models, plugins, server config). Created when PX4 is built.
px4_gz_env() {
  local f="$PX4_DIR/build/px4_sitl_default/rootfs/gz_env.sh"
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found. PX4 is not built yet: run setup/install.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$f"
}

# Model name of our drone (from config/drone.yaml), used in Gazebo topic names
drone_model() {
  python3 -c "import yaml;print(yaml.safe_load(open('$AWARE_SIM/config/drone.yaml'))['model_name'])"
}

title() { printf '\033]0;%s\007' "$1"; echo -e "\n\033[1;36m=== $1 ===\033[0m"; }
