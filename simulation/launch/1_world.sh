#!/usr/bin/env bash
# STEP 1: start the simulated world (Gazebo server, no window).
# Leave this terminal open. Stop with Ctrl+C (or launch/stop.sh).
source "$(dirname "$0")/_common.sh"
title "1 WORLD (Gazebo server)"
WORLD="$AWARE_SIM/worlds/aware_expo.sdf"
if [ ! -f "$WORLD" ]; then
  echo "World not generated yet. Generating the 'missing_person' scenario..."
  python3 "$AWARE_SIM/scripts/generate_world.py" --scenario missing_person || exit 1
fi
px4_gz_env
echo "Loading the world (248 people: the first start can take a minute)..."
nv gz sim -v 4 -s -r "$WORLD"
