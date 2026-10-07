#!/usr/bin/env bash
# =====================================================================
# AWARE simulation: one-time installer for Ubuntu 24.04 (ROS 2 Jazzy)
#                   or Ubuntu 22.04 (ROS 2 Humble)
#
#   cd <this repository>
#   bash setup/install.sh
#
# Safe to run again: steps that are already done are skipped.
# Takes 30-90 minutes the first time (large downloads + building PX4).
# You will be asked for your password (sudo) a few times.
# =====================================================================
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/setup/versions.env"
LOG="$ROOT/setup/install.log"
exec > >(tee -a "$LOG") 2>&1          # everything is also saved to setup/install.log

step() { echo -e "\n\033[1;36m==== [$1] $2 ====\033[0m"; }
ok()   { echo -e "\033[32m  OK:\033[0m $1"; }
warn() { echo -e "\033[33m  NOTE:\033[0m $1"; }
fail() { echo -e "\033[31m  ERROR:\033[0m $1"; echo "Full log: $LOG"; exit 1; }
trap 'fail "the step above failed (line $LINENO). Send setup/install.log to the team."' ERR

# ---------------------------------------------------------------- 0
step 0 "Checking the computer"
. /etc/os-release
case "$VERSION_ID" in
  24.04) AWARE_ROS=jazzy ;;
  22.04) AWARE_ROS=humble ;;
  *) fail "This needs Ubuntu 22.04 or 24.04 (found $PRETTY_NAME)." ;;
esac
case "$ROOT" in *" "*) fail "The folder path contains a space: $ROOT. Move the repository to a path without spaces.";; esac
ok "Ubuntu $VERSION_ID (ROS 2 $AWARE_ROS), repository at $ROOT"
FREE_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
[ "$FREE_GB" -ge 25 ] || warn "Only ${FREE_GB} GB free; about 25 GB is recommended."

# ---------------------------------------------------------------- 1
step 1 "ROS 2 $AWARE_ROS (robot software framework)"
if [ -f "/opt/ros/$AWARE_ROS/setup.bash" ]; then
  ok "already installed"
else
  sudo apt update
  sudo apt install -y software-properties-common curl
  sudo add-apt-repository -y universe
  if ! ls /etc/apt/sources.list.d/ 2>/dev/null | grep -q "^ros2"; then
    V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F'"' '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${V}/ros2-apt-source_${V}.$(. /etc/os-release && echo "$VERSION_CODENAME")_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb
  fi
  sudo apt update
  sudo apt install -y "ros-$AWARE_ROS-desktop" ros-dev-tools
fi
sudo apt install -y "ros-$AWARE_ROS-rqt-image-view" mesa-utils git lsb-release \
                    python3-yaml python3-jinja2 python3-matplotlib python3-venv
if [ "$AWARE_ROS" = "jazzy" ]; then
  # Jazzy's standard bridge is built for Gazebo Harmonic.
  sudo apt install -y ros-jazzy-ros-gz
  ok "ROS 2 Jazzy + Gazebo bridge + tools"
else
  # Humble's standard bridge is built for an older Gazebo (Fortress), so the
  # Harmonic version (ros-humble-ros-gzharmonic) is installed after step 3.
  ok "ROS 2 Humble + tools (Gazebo bridge comes after step 3)"
fi

# ---------------------------------------------------------------- 2
step 2 "PX4 autopilot source (version: $PX4_REF)"
if [ -d "$ROOT/PX4-Autopilot/.git" ]; then
  ok "already downloaded"
else
  git clone https://github.com/PX4/PX4-Autopilot.git "$ROOT/PX4-Autopilot"
  git -C "$ROOT/PX4-Autopilot" checkout "$PX4_REF"
  git -C "$ROOT/PX4-Autopilot" submodule update --init --recursive
fi

# ---------------------------------------------------------------- 3
step 3 "PX4 dependencies + Gazebo Harmonic (PX4's own setup script)"
if command -v gz >/dev/null && gz sim --version 2>/dev/null | grep -q "version 8\."; then
  ok "Gazebo Harmonic already installed"
else
  bash "$ROOT/PX4-Autopilot/Tools/setup/ubuntu.sh" --no-nuttx
fi
# Known problem: PX4's script can put NumPy 2 into ~/.local, which breaks ROS tools.
if python3 -c "import numpy,sys; sys.exit(0 if numpy.__version__.startswith('2') and '.local' in numpy.__file__ else 1)" 2>/dev/null; then
  warn "Removing NumPy 2 from ~/.local (conflicts with ROS)"
  python3 -m pip uninstall -y numpy --break-system-packages 2>/dev/null \
    || python3 -m pip uninstall -y numpy          # older pip (Ubuntu 22.04)
fi
ok "NumPy for the system: $(python3 -c 'import numpy;print(numpy.__version__)')"

# Humble only: the Gazebo-ROS bridge built for Gazebo Harmonic.
# It comes from the Gazebo package server (normally added by PX4's script above).
if [ "$AWARE_ROS" = "humble" ]; then
  sudo apt update
  if ! apt-cache show ros-humble-ros-gzharmonic >/dev/null 2>&1; then
    warn "Adding the Gazebo package server"
    sudo curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
    sudo apt update
  fi
  sudo apt install -y ros-humble-ros-gzharmonic
  ok "Gazebo-ROS bridge for Humble + Harmonic"
fi

# ---------------------------------------------------------------- 4
step 4 "Building PX4 for simulation (10-30 minutes the first time)"
if [ -x "$ROOT/PX4-Autopilot/build/px4_sitl_default/bin/px4" ]; then
  ok "already built"
else
  (cd "$ROOT/PX4-Autopilot" && make px4_sitl)
fi

# ---------------------------------------------------------------- 5
step 5 "Python environment for AWARE (aware_venv)"
if [ ! -d "$ROOT/aware_venv" ]; then
  # shellcheck disable=SC1091
  source "/opt/ros/$AWARE_ROS/setup.bash"
  python3 -m venv --system-site-packages "$ROOT/aware_venv"
fi
# shellcheck disable=SC1091
source "$ROOT/aware_venv/bin/activate"
pip install --upgrade "mavsdk>=4" "numpy<2"
deactivate
ok "aware_venv ready (MAVSDK v4)"

# ---------------------------------------------------------------- 6
step 6 "Downloading the 3D people models (Gazebo Fuel)"
gz fuel download -u "https://fuel.gazebosim.org/1.0/Mingfei/models/actor" -v 1 || warn "download 1 failed (retry later)"
gz fuel download -u "https://fuel.gazebosim.org/1.0/OpenRobotics/models/actor - relative paths" -v 1 || warn "download 2 failed (retry later)"

# ---------------------------------------------------------------- 7
step 7 "Building the AWARE drone, outfits and world"
export AWARE_ROOT="$ROOT"
cd "$ROOT/aware_sim"
python3 scripts/make_outfits.py
python3 scripts/make_drone.py
python3 scripts/generate_world.py --scenario missing_person

# ---------------------------------------------------------------- 8
step 8 "Shortcuts in your terminal (~/.bashrc)"
LINE="source \"$ROOT/setup/aware_env.sh\""
if grep -qF "$LINE" ~/.bashrc; then
  ok "already added"
else
  printf '\n# AWARE simulation shortcuts\n%s\n' "$LINE" >> ~/.bashrc
  ok "added (new terminals will have them)"
fi

# ---------------------------------------------------------------- 9
step 9 "AI environment (ai_venv) for ../rasid_video_pipeline"
AI_DIR="$(cd "$ROOT/.." && pwd)/rasid_video_pipeline"
if [ ! -d "$AI_DIR" ]; then
  warn "AI folder not found ($AI_DIR): skipping. The simulation works without it."
else
  if [ ! -d "$ROOT/ai_venv" ]; then
    # shellcheck disable=SC1091
    source "/opt/ros/$AWARE_ROS/setup.bash"
    python3 -m venv --system-site-packages "$ROOT/ai_venv"     # sees ROS (rclpy)
  fi
  # shellcheck disable=SC1091
  source "$ROOT/ai_venv/bin/activate"
  [ -f "$AI_DIR/requirements.txt" ] && pip install -c "$AI_DIR/constraints-ai.txt" -r "$AI_DIR/requirements.txt" "numpy<2"
  [ -f "$AI_DIR/pyproject.toml" ] && pip install -c "$AI_DIR/constraints-ai.txt" -e "$AI_DIR" "numpy<2"
  pip install -c "$AI_DIR/constraints-ai.txt" supervision python-dotenv openai "numpy<2"
  python3 -m pip check
  python3 -c "import rclpy, cloud_track, supervision; print('AI imports OK')" || warn "an AI import failed (see above)"
  deactivate
  if [ ! -f "$AI_DIR/.env" ]; then
    warn "No OpenAI key yet. Create $AI_DIR/.env containing:  OPENAI_API_KEY=sk-...  (never commit it)"
  fi
  ok "ai_venv ready"
fi

trap - ERR
echo -e "\n\033[1;32m==== INSTALLATION COMPLETE ====\033[0m"
echo "1. RESTART the computer once (PX4's setup changed your user groups)."
echo "2. Then check everything:   bash setup/check.sh"
echo "3. Then start everything:     aware_run      (see README.md)"