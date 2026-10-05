#!/usr/bin/env bash
# STEP 3 (optional): open the Gazebo window to watch the world.
# Closing this window does NOT stop the simulation. Skip it when running the AI
# on a weak GPU: it is a second renderer drawing all the people.
source "$(dirname "$0")/_common.sh"
title "3 GUI (Gazebo window)"
px4_gz_env
nv gz sim -g
