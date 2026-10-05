#!/usr/bin/env python3
"""
ground_truth.py: publish where every actor REALLY is (the "director's script").

The world generator wrote every actor's route to worlds/aware_expo.actors.yaml:
straight lines between waypoints at known times, looping. This node replays
that script against the SIMULATION clock (/clock), so it knows each person's
true position without asking Gazebo.

Publishes (2 Hz by default):
  /aware/ground_truth                 std_msgs/String, JSON:
      {sim_time, scenario, zone_counts{zone: n}, people[{name, role, outfit,
       looks, x, y, lat, lon, zone}]}
  /aware/ground_truth/missing_person  geometry_msgs/PointStamped (world frame,
                                      x = East, y = North, meters)

USE IT FOR EVALUATION ONLY. The dashboard's "missing person" must come from the
AI pipeline; ground truth is the answer key used to SCORE it.

Run (aware venv, ROS sourced, aware_bridge running: it bridges /clock):
    python3 ground_truth.py
    python3 ground_truth.py --snapshot     # also draw docs/gt_snapshot.png every 5 s
"""
import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crowd import point_in_polygon  # noqa: E402
from patrol_plan import enu_to_geo  # noqa: E402

PKG_DIR = Path(__file__).resolve().parent.parent


class Track:
    """One actor's script: position at any time by linear interpolation."""

    def __init__(self, a):
        self.name, self.role, self.loop = a["name"], a["role"], a["loop"]
        self.outfit, self.looks = a.get("outfit", ""), a.get("looks", "")
        wps = a["waypoints"]
        self.t = [w[0] for w in wps]
        self.xy = [(w[1], w[2]) for w in wps]
        self.duration = self.t[-1]

    def position(self, sim_t):
        t = sim_t % self.duration if (self.loop and self.duration > 0) else min(sim_t, self.duration)
        i = bisect_right(self.t, t) - 1          # last waypoint at or before t
        if i >= len(self.t) - 1:
            return self.xy[-1]
        t0, t1 = self.t[i], self.t[i + 1]
        f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        (x0, y0), (x1, y1) = self.xy[i], self.xy[i + 1]
        return x0 + f * (x1 - x0), y0 + f * (y1 - y0)


def load_world():
    truth = yaml.safe_load((PKG_DIR / "worlds" / "aware_expo.actors.yaml").read_text())
    venue = yaml.safe_load((PKG_DIR / "config" / "zones.yaml").read_text())
    if truth["actors"] and "waypoints" not in truth["actors"][0]:
        sys.exit("aware_expo.actors.yaml has no waypoints: regenerate the world with the "
                 "current generate_world.py (and restart the simulation).")
    return truth, venue


def snapshot(people, venue, sim_t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fx, fy = venue["venue"]["floor_size"]
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.add_patch(Rectangle((-fx / 2, -fy / 2), fx, fy, fc="#e6e6e0", ec="k"))
    for z in venue["zones"]:
        ax.add_patch(Polygon(z["polygon"], fc="none", ec="gray", ls="--"))
    for b in venue["booths"]:
        x, y, _ = b["pose"]; d, w = b["size"]
        ax.add_patch(Rectangle((x - d / 2, y - w / 2), d, w, fc=b["color"], ec="k"))
    for p in people:
        if p["role"] == "missing_person":
            ax.plot(p["x"], p["y"], "*", color="red", ms=18, mec="k", zorder=5)
            ax.annotate("missing person", (p["x"], p["y"]), xytext=(8, 8), textcoords="offset points")
        else:
            ax.plot(p["x"], p["y"], ".", color="k" if p["role"] == "visitor" else "tab:purple", ms=5)
    ax.set_xlim(-42, 42); ax.set_ylim(-27, 27); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"Ground truth at sim time {sim_t:.1f} s ({len(people)} people)")
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)


class GroundTruth(Node):
    def __init__(self, args):
        super().__init__("aware_ground_truth")
        self.args = args
        self.truth, self.venue = load_world()
        self.tracks = [Track(a) for a in self.truth["actors"]]
        self.zones = self.venue["zones"]
        self.loc = self.venue["venue"]["location"]
        self.sim_t = None
        self.last_snap = -1e9
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.pub = self.create_publisher(String, "/aware/ground_truth", 10)
        self.pub_mp = self.create_publisher(PointStamped, "/aware/ground_truth/missing_person", 10)
        self.create_timer(1.0 / args.rate, self.tick)          # wall-clock timer
        self.create_timer(5.0, self.check_clock)
        self.get_logger().info(f"{len(self.tracks)} actors loaded "
                               f"(scenario: {self.truth['scenario']})")

    def on_clock(self, msg):
        self.sim_t = msg.clock.sec + msg.clock.nanosec * 1e-9

    def check_clock(self):
        if self.sim_t is None:
            self.get_logger().warn("no /clock yet: is aware_bridge running (it bridges /clock)?")

    def zone_of(self, x, y):
        for z in self.zones:
            if point_in_polygon(x, y, z["polygon"]):
                return z["id"]
        return None

    def tick(self):
        if self.sim_t is None:
            return
        people, counts = [], {z["id"]: 0 for z in self.zones}
        for tr in self.tracks:
            x, y = tr.position(self.sim_t)
            zone = self.zone_of(x, y)
            if zone:
                counts[zone] += 1
            lat, lon = enu_to_geo(x, y, self.loc)
            people.append(dict(name=tr.name, role=tr.role, outfit=tr.outfit, looks=tr.looks,
                               x=round(x, 2), y=round(y, 2),
                               lat=round(lat, 7), lon=round(lon, 7), zone=zone))
            if tr.role == "missing_person":
                pt = PointStamped()
                pt.header.stamp.sec = int(self.sim_t)
                pt.header.stamp.nanosec = int((self.sim_t % 1) * 1e9)
                pt.header.frame_id = "world"
                pt.point.x, pt.point.y = x, y
                self.pub_mp.publish(pt)
        self.pub.publish(String(data=json.dumps(dict(
            sim_time=round(self.sim_t, 2), scenario=self.truth["scenario"],
            zone_counts=counts, people=people))))
        if self.args.snapshot and self.sim_t - self.last_snap >= 5.0:
            self.last_snap = self.sim_t
            out = PKG_DIR / "docs" / "gt_snapshot.png"
            out.parent.mkdir(exist_ok=True)
            snapshot(people, self.venue, self.sim_t, out)


def main():
    ap = argparse.ArgumentParser(description="Publish simulator ground truth")
    ap.add_argument("--rate", type=float, default=2.0, help="messages per second")
    ap.add_argument("--snapshot", action="store_true", help="draw docs/gt_snapshot.png every 5 s")
    args = ap.parse_args()
    rclpy.init()
    node = GroundTruth(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
