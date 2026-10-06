#!/usr/bin/env bash
# =====================================================================
# aware_run: start the WHOLE AWARE run with one command.
# Opens one terminal tab per part, in the right order, waiting for each.
#
#   aware_run                          world + PX4 + bridge + AI window + patrol
#   aware_run --gui                    also open the Gazebo 3D window
#   aware_run --dashboard              AI via the dashboard backend instead of the AI window
#   aware_run --scenario normal        regenerate the world first (normal / crowded_booth / missing_person)
#   aware_run --scenario missing_person --scale 0.5    ...with half the people (weaker computers)
#   aware_run --loops 3                patrol rounds (default 2)
#   aware_run --auto-takeoff           take off automatically (default: YOU type "commander takeoff")
#   aware_run --no-ai                  simulation + patrol only
#
# Stop everything:  aware_stop
# =====================================================================
source "$(dirname "$0")/_common.sh"
LAUNCH="$AWARE_ROOT/launch"
AI_DIR="${AWARE_AI_DIR:-$(cd "$AWARE_ROOT/.." && pwd)/rasid_video_pipeline}"
AI_VENV="$AWARE_ROOT/ai_venv/bin/activate"
BACKEND_CMD="${AWARE_BACKEND_CMD:-uvicorn backend:app --host 0.0.0.0 --port 8000}"

GUI=0; DASHBOARD=0; AI_ON=1; SCENARIO=""; SCALE=""; LOOPS=2; START="takeoff"
while [ $# -gt 0 ]; do
  case "$1" in
    --gui) GUI=1 ;;
    --dashboard) DASHBOARD=1 ;;
    --no-ai) AI_ON=0 ;;
    --scenario) SCENARIO="$2"; shift ;;
    --scale) SCALE="$2"; shift ;;
    --loops) LOOPS="$2"; shift ;;
    --auto-takeoff) START="auto" ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1  (see: aware_run --help)"; exit 1 ;;
  esac
  shift
done

say()  { echo -e "\033[1;36m>>> $1\033[0m"; }
fail() { echo -e "\033[31mERROR: $1\033[0m"; exit 1; }
tab()  { gnome-terminal --tab --title="$1" -- bash -ic "$2; echo; echo '[$1 finished] press Enter to close'; read"; }
wait_for() {  # wait_for "<what>" <seconds> <command...>
  local what="$1" limit="$2"; shift 2
  echo -n "    waiting for $what"
  for _ in $(seq 1 $((limit / 2))); do
    if "$@" >/dev/null 2>&1; then echo " ready"; return 0; fi
    echo -n "."; sleep 2
  done
  echo " timed out"; return 1
}
MODEL="$(drone_model)_0"

# ---------------------------------------------------------------- 0
say "0/6 Stopping old processes"
"$LAUNCH/stop.sh" >/dev/null
if [ -n "$SCENARIO" ]; then
  say "Generating the world: scenario '$SCENARIO'${SCALE:+, scale $SCALE}"
  (cd "$AWARE_SIM" && python3 scripts/generate_world.py --scenario "$SCENARIO" ${SCALE:+--scale "$SCALE"}) \
    || fail "world generation failed"
fi
if [ $AI_ON = 1 ]; then
  [ -d "$AI_DIR" ] || fail "AI folder not found: $AI_DIR (or use --no-ai)"
  [ -f "$AI_VENV" ] || fail "AI environment missing: run 'bash setup/install.sh' (step 9) or use --no-ai"
fi

# ---------------------------------------------------------------- 1
say "1/6 WORLD (Gazebo server)"
tab "1 WORLD" "'$LAUNCH/1_world.sh'"
wait_for "the world to load" 240 bash -c "gz service -l | grep -q /world/aware_expo/control" \
  || fail "the world did not start: look at the WORLD tab"
sleep 5

# ---------------------------------------------------------------- 2
say "2/6 PX4 (drone $MODEL on the launch pad)"
tab "2 PX4" "'$LAUNCH/2_px4.sh'"
wait_for "the drone to spawn" 120 bash -c "gz topic -l | grep -q /model/$MODEL/" \
  || fail "the drone did not spawn: look at the PX4 tab"

# ---------------------------------------------------------------- 3
if [ $GUI = 1 ]; then say "3/6 GUI (Gazebo window)"; tab "3 GUI" "'$LAUNCH/3_gui.sh'"
else say "3/6 GUI skipped (add --gui to open it)"; fi

# ---------------------------------------------------------------- 4
say "4/6 BRIDGE (camera + clock to ROS 2)"
tab "4 BRIDGE" "'$LAUNCH/4_bridge.sh' $MODEL"
wait_for "camera images on ROS 2" 60 bash -c "ros2 topic list | grep -q /aware/camera/image" \
  || fail "the camera topic is missing: look at the BRIDGE tab"

# ---------------------------------------------------------------- 5
if [ $AI_ON = 1 ]; then
  if [ $DASHBOARD = 1 ]; then
    say "5/6 AI via the DASHBOARD backend"
    tab "5 AI BACKEND" "source '$AI_VENV'; cd '$AI_DIR'; $BACKEND_CMD"
    echo "    Start the search from the dashboard with video_path = ros"
  else
    say "5/6 AI search window"
    tab "5 AI" "source '$AI_VENV'; cd '$AI_DIR'; python3 run_live.py"
  fi
  echo "    loading the AI models (about 20 s)..."; sleep 20
else
  say "5/6 AI skipped (--no-ai)"
fi

# ---------------------------------------------------------------- 6
say "6/6 PATROL ($LOOPS loops)"
tab "6 PATROL" "'$LAUNCH/5_patrol.sh' --loops $LOOPS --start $START"

echo
echo -e "\033[1;32m=== Everything is starting (see the tabs) ===\033[0m"
[ "$START" = "takeoff" ] && echo "When the PATROL tab says 'waiting for YOU', type in the PX4 tab:  commander takeoff"
echo "AI window keys: c = confirm, r = reject, q = quit"
echo "Stop everything: aware_stop"
