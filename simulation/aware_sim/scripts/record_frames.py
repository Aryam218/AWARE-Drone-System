#!/usr/bin/env python3
"""
record_frames.py: save high-quality training frames from the drone camera.

Saves one frame every --interval seconds of SIMULATION time (so a slow
simulation doesn't change the spacing), as lossless PNG at full resolution,
and rejects frames that would hurt training:
  - "blank"     almost uniform image (black/grey while the world is loading)
  - "duplicate" nearly identical to the last saved frame (drone hovering/parked)
  - "soft"      too little detail/sharpness

Output (one folder per recording session):
  aware_sim/dataset/raw/<date>_<scenario>_alt<A>_pitch<P>/
      frames/000001.png ...      the images
      frames.csv                 per-frame: file, sim time, quality numbers
      session.yaml               everything needed to reproduce / label later:
                                 scenario, seed, altitude, camera pitch, intrinsics

Run (aware venv, ROS sourced, aware_bridge running):
    python3 record_frames.py                     # 1 frame / s
    python3 record_frames.py --interval 0.5      # 2 frames / s
Stop with Ctrl+C. The summary shows how many frames were kept / rejected.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

PKG_DIR = Path(__file__).resolve().parent.parent


def ros_image_to_bgr(msg):
    """sensor_msgs/Image -> OpenCV BGR array (no cv_bridge, so no NumPy conflicts)."""
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 4
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    conv = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR, "bgra8": cv2.COLOR_BGRA2BGR}
    return cv2.cvtColor(img, conv[msg.encoding]) if msg.encoding in conv else img.copy()


def quality(gray):
    """Two simple quality numbers:
    contrast  = standard deviation of brightness (0 = flat single color)
    sharpness = variance of the Laplacian (edge strength; low = blurry/empty)"""
    return float(gray.std()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_yaml(rel):
    p = PKG_DIR / rel
    return yaml.safe_load(p.read_text()) if p.exists() else {}


class Recorder(Node):
    def __init__(self, args, out):
        super().__init__("aware_frame_recorder")
        self.args, self.out = args, out
        self.frames_dir = out / "frames"
        self.frames_dir.mkdir(parents=True)
        self.csv = open(out / "frames.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow(["file", "sim_time_s", "contrast", "sharpness", "diff_to_prev"])
        self.info = None
        self.last_t = None
        self.last_small = None
        self.saved = 0
        self.rejected = {"blank": 0, "duplicate": 0, "soft": 0}
        self.create_subscription(CameraInfo, args.info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, args.topic, self.on_image, qos_profile_sensor_data)

    def on_info(self, msg):
        if self.info is None:
            self.info = msg
            self.write_session()

    def write_session(self):
        drone = load_yaml("config/drone.yaml")
        patrol = load_yaml("config/patrol.yaml")
        truth = load_yaml("worlds/aware_expo.actors.yaml")
        crowd = load_yaml("config/crowd.yaml")
        scen = truth.get("scenario", "unknown")
        session = dict(
            created=datetime.now().isoformat(timespec="seconds"),
            scenario=scen,
            seed=crowd.get("scenarios", {}).get(scen, {}).get("seed"),
            actors=len(truth.get("actors", [])),
            patrol_altitude_m=patrol.get("altitude_m"),
            camera=dict(
                model=drone.get("model_name", "unknown"),
                pitch_deg=drone.get("camera", {}).get("pitch_deg"),
                width=self.info.width, height=self.info.height,
                k=[float(v) for v in self.info.k],      # intrinsics: fx 0 cx / 0 fy cy / 0 0 1
            ),
            recording=dict(interval_s=self.args.interval, format="png (lossless)",
                           min_contrast=self.args.min_contrast,
                           min_sharpness=self.args.min_sharpness,
                           min_diff=self.args.min_diff),
            labels="none yet: will be generated from simulator ground truth",
        )
        (self.out / "session.yaml").write_text(yaml.safe_dump(session, sort_keys=False))
        self.get_logger().info(f"camera {self.info.width}x{self.info.height}; recording to {self.out}")

    def on_image(self, msg):
        if self.info is None:
            return                                   # wait for intrinsics first
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_t is not None and t - self.last_t < self.args.interval:
            return                                   # not time for the next frame yet
        self.last_t = t

        img = ros_image_to_bgr(msg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast, sharp = quality(gray)
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = float(np.abs(small - self.last_small).mean()) if self.last_small is not None else 255.0

        if contrast < self.args.min_contrast:
            self.rejected["blank"] += 1; return
        if diff < self.args.min_diff:
            self.rejected["duplicate"] += 1; return
        if sharp < self.args.min_sharpness:
            self.rejected["soft"] += 1; return

        self.saved += 1
        name = f"{self.saved:06d}.png"
        cv2.imwrite(str(self.frames_dir / name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        self.writer.writerow([name, f"{t:.3f}", f"{contrast:.1f}", f"{sharp:.1f}", f"{diff:.2f}"])
        self.csv.flush()
        self.last_small = small
        if self.saved % 10 == 0:
            self.get_logger().info(f"saved {self.saved} frames  (rejected {self.rejected})")

    def close(self):
        self.csv.close()
        print(f"\nSaved {self.saved} frames to {self.frames_dir}")
        print(f"Rejected: {self.rejected}")


def main():
    ap = argparse.ArgumentParser(description="Record training frames from the drone camera")
    ap.add_argument("--topic", default="/aware/camera/image")
    ap.add_argument("--info-topic", default="/aware/camera/camera_info")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds of SIM time between frames")
    ap.add_argument("--min-contrast", type=float, default=3.0, help="below = blank frame")
    ap.add_argument("--min-sharpness", type=float, default=20.0, help="below = too soft")
    ap.add_argument("--min-diff", type=float, default=2.0, help="below = duplicate of last frame")
    ap.add_argument("--name", default="", help="extra text in the session folder name")
    args = ap.parse_args()

    drone = load_yaml("config/drone.yaml")
    patrol = load_yaml("config/patrol.yaml")
    truth = load_yaml("worlds/aware_expo.actors.yaml")
    tag = "_".join(filter(None, [
        datetime.now().strftime("%Y%m%d_%H%M%S"), truth.get("scenario", "unknown"),
        f"alt{patrol.get('altitude_m', 'x')}",
        f"pitch{drone.get('camera', {}).get('pitch_deg', 'x')}", args.name]))
    out = PKG_DIR / "dataset" / "raw" / tag

    rclpy.init()
    node = Recorder(args, out)
    print(f"Recording {1 / args.interval:.1f} frame(s) per sim-second -> {out}\nCtrl+C to stop")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
