#!/usr/bin/env bash
# STEP 4: connect Gazebo to ROS 2: drone camera + simulation clock.
# Publishes /aware/camera/image, /aware/camera/camera_info and /clock.
source "$(dirname "$0")/_common.sh"
title "4 BRIDGE (Gazebo -> ROS 2)"
MODEL="${1:-$(drone_model)_0}"
CAM="/world/aware_expo/model/$MODEL/link/camera_link/sensor/camera"
echo "Bridging camera of model: $MODEL"
ros2 run ros_gz_bridge parameter_bridge \
  /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock \
  "$CAM/image@sensor_msgs/msg/Image[gz.msgs.Image" \
  "$CAM/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo" \
  --ros-args -r "$CAM/image:=/aware/camera/image" -r "$CAM/camera_info:=/aware/camera/camera_info"
