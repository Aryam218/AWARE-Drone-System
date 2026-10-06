#!/usr/bin/env bash
# Stop EVERYTHING (Gazebo, PX4). Always run this before starting again:
# leftover "ghost" processes from a previous run cause blank or frozen windows.
echo "Stopping Gazebo, PX4, the bridge and the AI..."
pkill -f "gz sim"; pkill -f "px4"; pkill -f "parameter_bridge"
pkill -f "run_live.py"; pkill -f "uvicorn backend:app"; pkill -f "patrol.py"; sleep 3
if pgrep -f "gz sim|px4" >/dev/null; then
  echo "Some processes did not stop politely, forcing them..."
  pkill -9 -f "gz sim"; pkill -9 -f "px4"; sleep 1
fi
if pgrep -f "gz sim|px4" >/dev/null; then
  echo "Still running:"; ps aux | grep -E "gz sim|px4" | grep -v grep
else
  echo "All stopped. Clean."
fi
