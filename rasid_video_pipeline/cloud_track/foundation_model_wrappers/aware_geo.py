"""
aware_geo.py: WHERE on the ground is a detected person?

Used for remembering rejected people by PLACE instead of by track ID
(in the live simulation, unconfirmed people get a new temporary ID
every frame, so an ID-based rejection is forgotten immediately).

How a box becomes a ground position:

    drone GPS + altitude + compass heading     (patrol -> /aware/drone/state)
  + camera lens (focal length, image centre)   (/aware/camera/camera_info)
  + camera tilt, 45 degrees down               (aware_sim/config/drone.yaml)
  + the person's FEET = bottom-centre of their box
  -> follow that pixel's viewing ray down to the (flat) floor
  -> latitude / longitude of the person

Assumptions (fine for the simulated venue, and for a hovering drone):
  - the floor is flat and at the take-off height (relative altitude 0)
  - the camera faces the drone's nose and is tilted pitch_deg downward
  - roll/pitch telemetry describes the body-fixed camera attitude; missing
    roll/pitch falls back to a level drone

No ROS here: plain math, easy to test.
"""
import math
import time

import numpy as np

CAMERA_PITCH_DEG = 45.0          # must match pitch_deg in aware_sim/config/drone.yaml
CAMERA_STABILIZED = False       # camera is fixed to base_link, without a gimbal
METERS_PER_DEG_LAT = 111_320.0


def pixel_to_ground(u, v, pose, pitch_deg=CAMERA_PITCH_DEG):
    """Ground (lat, lon) seen at image pixel (u, v), or None.

    pose: dict with
        lat, lon, alt_rel_m, heading_deg    (drone)
        roll_deg, pitch_deg                 (optional; nose-up pitch positive)
        fx, fy, cx, cy                      (camera lens, in pixels)
    Returns None if something is missing or the pixel looks above the horizon.
    """
    needed = ("lat", "lon", "alt_rel_m", "heading_deg", "fx", "fy", "cx", "cy")
    if pose is None or any(pose.get(k) is None for k in needed):
        return None
    h = float(pose["alt_rel_m"])
    if h <= 0.5:                                   # on the ground: no useful view
        return None

    # Viewing ray in camera terms: x = right, y = down (image), z = forward (lens)
    xc = (u - pose["cx"]) / pose["fx"]
    yc = (v - pose["cy"]) / pose["fy"]
    zc = 1.0

    # Camera tilted down by pitch: express the ray as forward / right / down
    p = math.radians(pitch_deg)
    forward = zc * math.cos(p) - yc * math.sin(p)
    right = xc
    down = zc * math.sin(p) + yc * math.cos(p)
    # Body forward/right/down -> north/east/down: Rz(heading) Ry(pitch) Rx(roll).
    # Positive roll lowers the right wing; positive pitch raises the nose.
    roll = 0.0 if CAMERA_STABILIZED else math.radians(float(pose.get("roll_deg") or 0.0))
    pitch = 0.0 if CAMERA_STABILIZED else math.radians(float(pose.get("pitch_deg") or 0.0))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rolled_right = cr * right - sr * down
    rolled_down = sr * right + cr * down
    tilted_forward = cp * forward + sp * rolled_down
    down = -sp * forward + cp * rolled_down
    yaw = math.radians(pose["heading_deg"])
    north = math.cos(yaw) * tilted_forward - math.sin(yaw) * rolled_right
    east = math.sin(yaw) * tilted_forward + math.cos(yaw) * rolled_right
    if down <= 1e-6:
        return None                                # looking at or above the horizon
    t = h / down                                   # stretch the world ray to the floor
    north *= t
    east *= t

    lat = pose["lat"] + north / METERS_PER_DEG_LAT
    lon = pose["lon"] + east / (METERS_PER_DEG_LAT * math.cos(math.radians(pose["lat"])))
    return lat, lon


def box_to_ground(box, pose, pitch_deg=CAMERA_PITCH_DEG):
    """Ground (lat, lon) of a person's FEET (bottom-centre of the box), or None."""
    x1, y1, x2, y2 = [float(v) for v in np.asarray(box).reshape(4)]
    return pixel_to_ground((x1 + x2) / 2.0, y2, pose, pitch_deg)


def distance_m(a, b):
    """Distance in metres between two (lat, lon) points (fine for short distances)."""
    dn = (b[0] - a[0]) * METERS_PER_DEG_LAT
    de = (b[1] - a[1]) * METERS_PER_DEG_LAT * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot(dn, de)


class RejectedPlaces:
    """Remembers where the operator rejected someone, and what they looked like.

    A new detection is skipped (not sent to GPT, not offered again) for ttl_s
    seconds only if BOTH:
      - it stands within radius_m of a rejected place, and
      - it looks like the rejected person (same_look(signature_a, signature_b))
    The look check protects the real missing person: someone standing right
    next to a rejected person, but dressed differently, is still checked.
    The time limit matters too: people walk around.
    """

    def __init__(self, radius_m=4.0, ttl_s=180.0, same_look=None):
        self.radius_m = radius_m
        self.ttl_s = ttl_s
        self.same_look = same_look                 # function(sig_a, sig_b) -> bool
        self._places = []                          # [(lat, lon, time_rejected, signature)]

    def add(self, latlon, signature=None, now=None):
        now = time.monotonic() if now is None else now
        self._places.append((latlon[0], latlon[1], now, signature))

    def _prune(self, now):
        self._places = [p for p in self._places if now - p[2] < self.ttl_s]

    def is_near(self, latlon, now=None):
        """True if latlon is within radius_m of any remembered place."""
        if latlon is None:
            return False
        now = time.monotonic() if now is None else now
        self._prune(now)
        return any(distance_m(latlon, (p[0], p[1])) <= self.radius_m for p in self._places)

    def is_rejected(self, latlon, signature=None, now=None):
        if latlon is None:
            return False
        now = time.monotonic() if now is None else now
        self._prune(now)
        for lat, lon, _t, sig in self._places:
            if distance_m(latlon, (lat, lon)) > self.radius_m:
                continue
            if self.same_look is None or sig is None or signature is None:
                return True                        # no look information: place only
            if self.same_look(sig, signature):
                return True
        return False

    def clear(self):
        self._places = []

    def __len__(self):
        return len(self._places)