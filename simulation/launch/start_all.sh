#!/usr/bin/env bash
# Start steps 1-4 automatically, each in its own terminal tab, in the right order.
# Then start the patrol yourself:  launch/5_patrol.sh
#   launch/start_all.sh            with the Gazebo window
#   launch/start_all.sh --no-gui   without it (lighter, recommended with the AI)
DIR="$(cd "$(dirname "$0")" && pwd)"
"$DIR/stop.sh"
GUI=1; [ "$1" = "--no-gui" ] && GUI=0

open_tab() {   # open_tab "<title>" "<script>"
  gnome-terminal --tab --title="$1" -- bash -ic "$2; echo; echo '[finished] press Enter to close'; read"
}
wait_for_world() {
  echo -n "Waiting for the world to load"
  for _ in $(seq 1 120); do
    if gz service -l 2>/dev/null | grep -q "/world/aware_expo/control"; then echo " ready."; return 0; fi
    echo -n "."; sleep 2
  done
  echo " timed out (check the WORLD tab)."; return 1
}

open_tab "1 WORLD" "$DIR/1_world.sh"
wait_for_world || exit 1
sleep 5
open_tab "2 PX4" "$DIR/2_px4.sh"
sleep 8
[ $GUI = 1 ] && open_tab "3 GUI" "$DIR/3_gui.sh"
open_tab "4 BRIDGE" "$DIR/4_bridge.sh"
echo
echo "Started: WORLD, PX4$([ $GUI = 1 ] && echo ', GUI'), BRIDGE (see the tabs)."
echo "Next:  launch/5_patrol.sh   then type 'commander takeoff' in the PX4 tab when asked."
