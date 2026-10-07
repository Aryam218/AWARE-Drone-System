#!/usr/bin/env python3
"""
drone_state.py: the drone's latest GPS position, for the AI side.

The patrol (simulation/aware_sim/scripts/patrol.py) publishes the drone's
position about twice per second on /aware/drone/state (JSON):

    {"lat": ..., "lon": ..., "alt_abs_m": ..., "alt_rel_m": ...,
     "heading_deg": ..., "time": ...}

DroneStateReader listens in a background thread and keeps only the newest
message, like ros_frame_source.py does for camera frames. It also reads the
camera's lens data (/aware/camera/camera_info), so frame_pose() gives
everything needed to turn a box into a ground position (see aware_geo.py).

    from drone_state import DroneStateReader
    drone = DroneStateReader()
    ...
    drone.location()   # dict for the dashboard, or None if unknown

About accuracy:
    The camera looks FORWARD and DOWN (pitch_deg in drone.yaml, 45 degrees),
    so the person is not under the drone: the centre of the image is about
    "altitude / tan(pitch)" metres ahead of it (15 m at 15 m altitude).
    location() therefore gives:
      - "drone":            where the drone is (exact)
      - "target_estimate":  the point at the CENTRE of the camera image
                            (approximate: the person may be off-centre)
    A precise person position (using where the box is in the image) is a
    later step.

TargetPublisher (the other direction): the AI publishes where the current
candidate / confirmed person stands on /aware/target (JSON), so the patrol
can fly to look at them and follow them:

    {"state": "candidate" | "confirmed" | "none",
     "lat": ..., "lon": ..., "track_id": ..., "time": ...}

Self-test (prints the drone position while the patrol is running):
    python3 drone_state.py
"""
import json
import math
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

DEFAULT_TOPIC = "/aware/drone/state"
CAMERA_INFO_TOPIC = "/aware/camera/camera_info"
TARGET_TOPIC = "/aware/target"
CAMERA_PITCH_DEG = 45.0          # must match pitch_deg in aware_sim/config/drone.yaml
METERS_PER_DEG_LAT = 111_320.0

_reader_count = 0                # gives each reader its own ROS node name


def _ensure_rclpy():
    if not rclpy.ok():
        # Inside a server (e.g. the FastAPI backend) this runs in a background
        # thread: ROS must NOT install its own Ctrl+C handler there.
        if threading.current_thread() is threading.main_thread():
            rclpy.init()
        else:
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)


class TargetPublisher:
    """Publishes the candidate's ground position on /aware/target."""

    def __init__(self, topic=TARGET_TOPIC):
        global _reader_count
        _ensure_rclpy()
        _reader_count += 1
        self._node = Node(f"aware_target_publisher_{_reader_count}")
        self._pub = self._node.create_publisher(String, topic, 10)

    def publish(self, state, latlon=None, track_id=None):
        data = {"state": state, "time": time.time(), "track_id": track_id,
                "lat": None, "lon": None}
        if latlon is not None:
            data["lat"], data["lon"] = float(latlon[0]), float(latlon[1])
        msg = String()
        msg.data = json.dumps(data)
        self._pub.publish(msg)

    def release(self):
        self._node.destroy_node()


class DroneStateReader:
    """Keeps the newest drone position from /aware/drone/state."""

    def __init__(self, topic=DEFAULT_TOPIC, max_age_s=5.0,
                 camera_info_topic=CAMERA_INFO_TOPIC):
        global _reader_count
        self.max_age_s = max_age_s
        self._lock = threading.Lock()
        self._latest = None
        self._received_at = None
        self._lens = None                # {"fx", "fy", "cx", "cy"} from camera_info
        self._open = True

        _ensure_rclpy()

        _reader_count += 1
        self._node = Node(f"aware_drone_state_{_reader_count}")
        self._node.create_subscription(String, topic, self._on_msg, 10)
        self._node.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, 10)
        self._exec = SingleThreadedExecutor()
        self._exec.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        from rclpy.executors import ExternalShutdownException
        try:
            while self._open and rclpy.ok():
                self._exec.spin_once(timeout_sec=0.1)
        except (ExternalShutdownException, KeyboardInterrupt):
            pass

    def _on_msg(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        with self._lock:
            self._latest = data
            self._received_at = time.monotonic()

    def _on_camera_info(self, msg):
        k = list(msg.k)                  # 3x3 lens matrix, row by row
        if len(k) == 9 and k[0] > 0 and k[4] > 0:
            with self._lock:
                self._lens = {"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5]}

    def frame_pose(self):
        """Drone position + camera lens for aware_geo.box_to_ground(),
        or None if either is unknown. Call it when a camera frame is
        read, so the pose matches that frame."""
        s = self.latest()
        with self._lock:
            lens = dict(self._lens) if self._lens else None
        if s is None or lens is None:
            return None
        pose = {
            "lat": s.get("lat"),
            "lon": s.get("lon"),
            "alt_rel_m": s.get("alt_rel_m"),
            "heading_deg": s.get("heading_deg"),
            "roll_deg": s.get("roll_deg", 0.0),
            "pitch_deg": s.get("pitch_deg", 0.0),
        }
        pose.update(lens)
        return pose

    def latest(self):
        """The newest raw message, or None if nothing arrived recently."""
        with self._lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._received_at > self.max_age_s:
                return None
            return dict(self._latest)

    def location(self):
        """Location report for the dashboard, or None if the drone position
        is unknown (patrol not running, or no message for max_age_s)."""
        s = self.latest()
        if s is None:
            return None

        lat, lon = s.get("lat"), s.get("lon")
        alt = s.get("alt_rel_m")
        heading = s.get("heading_deg")

        report = {
            "drone": {
                "lat": lat,
                "lon": lon,
                "alt_rel_m": alt,
                "heading_deg": heading,
            },
            "target_estimate": None,
            "note": "target_estimate = centre of the camera view (approximate)",
        }

        if None not in (lat, lon, alt, heading) and alt > 0:
            ahead_m = alt / math.tan(math.radians(CAMERA_PITCH_DEG))
            h = math.radians(heading)
            dlat = ahead_m * math.cos(h) / METERS_PER_DEG_LAT
            dlon = ahead_m * math.sin(h) / (METERS_PER_DEG_LAT * math.cos(math.radians(lat)))
            report["target_estimate"] = {
                "lat": lat + dlat,
                "lon": lon + dlon,
                "meters_ahead_of_drone": round(ahead_m, 1),
            }

        return report

    def release(self):
        self._open = False
        self._thread.join(timeout=1.0)
        self._exec.remove_node(self._node)
        self._node.destroy_node()


if __name__ == "__main__":
    reader = DroneStateReader()
    print(f"Waiting for {DEFAULT_TOPIC} ... (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1.0)
            print(reader.location() or "no drone position yet (is the patrol flying?)")
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        if rclpy.ok():
            rclpy.shutdown()