#!/usr/bin/env bash
# check.sh: is everything installed correctly?   bash setup/check.sh
# Every line is PASS / WARN / FAIL. A FAIL line tells you how to fix it.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pass=0; fail=0; warn=0
ok()   { echo -e "  \e[32m[PASS]\e[0m $1"; pass=$((pass+1)); }
bad()  { echo -e "  \e[31m[FAIL]\e[0m $1\n         fix: $2"; fail=$((fail+1)); }
note() { echo -e "  \e[33m[WARN]\e[0m $1\n         note: $2"; warn=$((warn+1)); }
section() { echo -e "\n\e[1m== $1 ==\e[0m"; }

section "Computer"
. /etc/os-release
[ "$VERSION_ID" = "24.04" ] && ok "Ubuntu 24.04" || bad "Ubuntu $VERSION_ID" "this project needs Ubuntu 24.04"
case "$ROOT" in *" "*) bad "path has spaces: $ROOT" "move the repository to a path without spaces";; *) ok "repository: $ROOT";; esac

section "ROS 2 + Gazebo"
if [ -f /opt/ros/jazzy/setup.bash ]; then
  source /opt/ros/jazzy/setup.bash; ok "ROS 2 Jazzy"
  ros2 pkg prefix ros_gz_bridge >/dev/null 2>&1 && ok "Gazebo-ROS bridge" || bad "ros_gz missing" "sudo apt install ros-jazzy-ros-gz"
else
  bad "ROS 2 Jazzy missing" "bash setup/install.sh"
fi
if command -v gz >/dev/null && gz sim --version 2>/dev/null | grep -q "version 8\."; then
  ok "Gazebo Harmonic ($(gz sim --version | head -1))"
else
  bad "Gazebo Harmonic missing" "bash setup/install.sh"
fi

section "Graphics"
if command -v glxinfo >/dev/null; then
  r=$(glxinfo -B 2>/dev/null | grep "OpenGL renderer" | cut -d: -f2 | xargs)
  [[ "$r" == *llvmpipe* ]] && bad "software rendering only ($r)" "install the GPU driver: sudo ubuntu-drivers install, then restart" || ok "renderer: $r"
fi
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then ok "NVIDIA GPU available"
else note "no NVIDIA GPU detected" "works, but 248 people may be slow; use --scale 0.5 when generating the world"; fi

section "PX4"
[ -x "$ROOT/PX4-Autopilot/build/px4_sitl_default/bin/px4" ] && ok "PX4 built" || bad "PX4 not built" "bash setup/install.sh"
[ -L "$ROOT/PX4-Autopilot/Tools/simulation/gz/models/aware_x500" ] && ok "AWARE drone linked into PX4" \
  || bad "AWARE drone not built" "cd aware_sim && AWARE_ROOT=$ROOT python3 scripts/make_drone.py"

section "Python"
if [ -x "$ROOT/aware_venv/bin/python" ]; then
  "$ROOT/aware_venv/bin/python" -c "import mavsdk" 2>/dev/null && ok "aware_venv + MAVSDK" || bad "MAVSDK missing" "bash setup/install.sh"
  "$ROOT/aware_venv/bin/python" -c "import rclpy" 2>/dev/null && ok "aware_venv can use ROS" || bad "venv cannot see ROS" "delete aware_venv and run setup/install.sh again"
else
  bad "aware_venv missing" "bash setup/install.sh"
fi
v=$(python3 -c "import numpy;print(numpy.__version__)" 2>/dev/null)
[[ "$v" == 1.* ]] && ok "system NumPy $v" || bad "system NumPy is '$v' (must be 1.x)" "python3 -m pip uninstall -y numpy --break-system-packages"

section "AWARE files"
ls ~/.gz/fuel/fuel.gazebosim.org/mingfei/models/actor >/dev/null 2>&1 && ok "people models downloaded" || bad "people models missing" "bash setup/install.sh (step 6)"
n=$(ls "$ROOT/aware_sim/models/aware_people/meshes" 2>/dev/null | wc -l)
[ "$n" -gt 0 ] && ok "outfits built ($n files)" || bad "outfits not built" "cd aware_sim && python3 scripts/make_outfits.py"
[ -f "$ROOT/aware_sim/worlds/aware_expo.sdf" ] && ok "world generated" || bad "world not generated" "cd aware_sim && python3 scripts/generate_world.py --scenario missing_person"
grep -qF "setup/aware_env.sh" ~/.bashrc && ok "terminal shortcuts installed" || note "shortcuts not in ~/.bashrc" "bash setup/install.sh (step 8)"

echo -e "\n\e[1m== Result ==\e[0m  PASS: $pass   WARN: $warn   FAIL: $fail"
[ $fail -eq 0 ] && echo "Ready! Continue with 'Running the simulation' in README.md." || echo "Fix the FAIL lines (top to bottom), then run this check again."
exit $fail
