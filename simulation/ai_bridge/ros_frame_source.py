#!/usr/bin/env python3
"""
ros_frame_source.py: use the LIVE drone camera wherever code expects a video file.

It behaves like OpenCV's cv2.VideoCapture AND can be iterated like AWARE's
VideoStreamer (`for frame in stream:`), so AI code that reads a video can switch
to the live simulation with a one-line change:

    # before (recorded video)
    cap = cv2.VideoCapture("patrol.mp4")
    # after (live Gazebo camera via ROS 2)
    from ros_frame_source import RosFrameSource
    cap = RosFrameSource()                      # /aware/camera/image

    while True:
        ok, frame = cap.read()                  # frame = BGR numpy array, like OpenCV
        if not ok:
            break
        ...                                     # unchanged AI code
    cap.release()

Extra (not in OpenCV): cap.last_stamp = simulation time of the frame (seconds),
needed later to compare AI results with ground truth at the same moment.

Requirements: ROS 2 sourced; a Python env that sees rclpy (venv created with
--system-site-packages) and NumPy < 2.

Self-test (shows the live camera + frame rate):
    python3 ros_frame_source.py
"""
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def ros_image_to_bgr(msg):
    """sensor_msgs/Image -> OpenCV BGR array (no cv_bridge, so no NumPy conflicts)."""
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 4
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    conv = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR, "bgra8": cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(img, conv[msg.encoding]) if msg.encoding in conv else img.copy()


class RosFrameSource:
    """A VideoCapture look-alike fed by a ROS 2 image topic.

    A background thread receives images and keeps only the NEWEST one, so slow
    AI code always gets the current view instead of a growing backlog
    (live TV, not a recording)."""

    def __init__(self, topic="/aware/camera/image", timeout_s=10.0):
        self.timeout_s = timeout_s
        self._lock = threading.Condition()
        self._msg = None
        self._new = False
        self._open = True
        self.last_stamp = None
        self.width = self.height = 0
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("ros_frame_source")
        self._node.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._open and rclpy.ok():
            self._exec.spin_once(timeout_sec=0.1)

    def _on_image(self, msg):
        with self._lock:
            self._msg, self._new = msg, True
            self._lock.notify_all()

    # ---- the cv2.VideoCapture-like interface ----
    def isOpened(self):
        return self._open

    def read(self):
        """Wait for the next new frame. Returns (True, frame) or (False, None) on timeout."""
        with self._lock:
            if not self._lock.wait_for(lambda: self._new or not self._open, timeout=self.timeout_s):
                return False, None
            if not self._open:
                return False, None
            msg, self._new = self._msg, False
        self.last_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.width, self.height = msg.width, msg.height
        return True, ros_image_to_bgr(msg)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return 15.0                      # our camera's update_rate (drone.yaml)
        return 0.0

    def __iter__(self):
        """Iterate like a video streamer: `for frame in source:` yields BGR frames
        until no frame arrives for timeout_s (simulation stopped)."""
        while self._open:
            ok, frame = self.read()
            if not ok:
                return
            yield frame

    def release(self):
        self._open = False
        with self._lock:
            self._lock.notify_all()
        self._thread.join(timeout=1.0)
        self._exec.remove_node(self._node)
        self._node.destroy_node()


# Same object, named to match AWARE's VideoStreamer (iterate it: `for frame in stream`)
RosVideoStreamer = RosFrameSource


if __name__ == "__main__":
    cap = RosFrameSource()
    print("Waiting for /aware/camera/image ... (q to quit)")
    n, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame for 10 s: is aware_bridge running?")
                break
            n += 1
            fps = n / max(time.time() - t0, 1e-6)
            cv2.putText(frame, f"sim t={cap.last_stamp:.1f}s  {fps:.1f} fps", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("RosFrameSource self-test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
