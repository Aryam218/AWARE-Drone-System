#!/usr/bin/env bash
# Show what the drone camera sees (needs step 4 running).
# In the window, pick /aware/camera/image from the drop-down list.
source "$(dirname "$0")/_common.sh"
title "CAMERA VIEW"
ros2 run rqt_image_view rqt_image_view /aware/camera/image
