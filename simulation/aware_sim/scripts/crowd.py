"""
crowd.py: turns a scenario from crowd.yaml into actor waypoint lists.

Every actor is a list of waypoints (time, x, y, z, yaw). Gazebo moves the
actor between them in straight lines and plays the walk animation.

Key ideas
  - Routes are random but SEEDED: the same seed gives the same crowd,
    so a test can be repeated exactly (like replaying a recorded match).
  - Every straight leg is checked against obstacles (booths, stage) so
    nobody walks through a wall; bad legs are simply re-drawn.
"""
import math
import random

# ---------------------------------------------------------------- geometry
def point_in_polygon(x, y, poly):
    """Ray casting: shoot a ray to the right and count edge crossings.
    Odd = inside, even = outside (like counting doors you pass through)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def random_point_in(poly, rng, margin=0.5):
    """Rejection sampling: throw darts at the bounding box until one lands inside."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    for _ in range(1000):
        x = rng.uniform(min(xs) + margin, max(xs) - margin)
        y = rng.uniform(min(ys) + margin, max(ys) - margin)
        if point_in_polygon(x, y, poly):
            return x, y
    raise RuntimeError("could not sample a point inside polygon")


class Obstacles:
    """Footprints of solid structures, as rectangles in their own local frame."""

    def __init__(self, venue_cfg, clearance=0.4):
        self.rects = []   # (cx, cy, yaw, half_x, half_y)
        for b in venue_cfg["booths"]:
            x, y, yaw = b["pose"]
            d, w = b["size"]
            # the whole booth footprint is off-limits: visitors stay in front
            self.rects.append((x, y, yaw, d / 2 + clearance, w / 2 + clearance))
        s = venue_cfg["stage"]
        self.rects.append((s["pose"][0], s["pose"][1], s["pose"][2],
                           s["size"][0] / 2 + clearance, s["size"][1] / 2 + clearance))
        fx, fy = venue_cfg["venue"]["floor_size"]
        self.fx, self.fy = fx / 2 - 0.5, fy / 2 - 0.5

    def blocked(self, x, y):
        if abs(x) > self.fx or abs(y) > self.fy:
            return True                      # off the Expo floor
        for cx, cy, yaw, hx, hy in self.rects:
            # rotate the point into the rectangle's local frame
            dx, dy = x - cx, y - cy
            lx = dx * math.cos(-yaw) - dy * math.sin(-yaw)
            ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
            if abs(lx) <= hx and abs(ly) <= hy:
                return True
        return False

    def leg_clear(self, a, b, step=0.25):
        """Walk along the straight leg in small steps and test each point."""
        dist = math.dist(a, b)
        n = max(1, int(dist / step))
        return all(not self.blocked(a[0] + (b[0] - a[0]) * i / n,
                                    a[1] + (b[1] - a[1]) * i / n)
                   for i in range(n + 1))


# ---------------------------------------------------------------- trajectories
class Trajectory:
    """Builds waypoints while keeping track of the clock."""

    def __init__(self, x, y, yaw, z, turn_time):
        self.t = 0.0
        self.x, self.y, self.yaw, self.z = x, y, yaw, z
        self.turn_time = turn_time
        self.points = [(0.0, x, y, z, yaw)]

    def _add(self):
        self.points.append((round(self.t, 2), self.x, self.y, self.z, self.yaw))

    def walk_to(self, x, y, speed):
        dist = math.dist((self.x, self.y), (x, y))
        if dist < 0.05:
            return
        new_yaw = math.atan2(y - self.y, x - self.x)
        # 1) turn on the spot to face the next target
        self.t += self.turn_time
        self.yaw = new_yaw
        self._add()
        # 2) walk there in a straight line: time = distance / speed
        self.t += dist / speed
        self.x, self.y = x, y
        self._add()

    def wait(self, seconds, face_yaw=None):
        if seconds <= 0:
            return
        if face_yaw is not None:
            self.t += self.turn_time
            self.yaw = face_yaw
            self._add()
        self.t += seconds
        self._add()


# ---------------------------------------------------------------- scenario builder
def build_actors(venue_cfg, crowd_cfg, scenario_name):
    sc = crowd_cfg["scenarios"][scenario_name]
    d = crowd_cfg["defaults"]
    act = crowd_cfg["actor"]
    hip = act["hip_height"]
    stand_hip = act.get("stand_hip_height", hip)
    # skin = costume (Fuel URL; generate_world.py swaps it for an outfit variant)
    # anim_file = choreography; None = take the animation from the skin file itself
    WALK = dict(skin=act["skin"], anim="walk", anim_file=None, interpolate_x="true")
    STAND = dict(skin=act["stand_skin"], anim="talk_b", anim_file=None,
                 interpolate_x="false")
    floor_top = 0.02
    rng = random.Random(sc.get("seed", 0))
    obst = Obstacles(venue_cfg)
    zones = {z["id"]: z for z in venue_cfg["zones"]}
    booths = {b["id"]: b for b in venue_cfg["booths"]}
    actors = []

    fx, fy = venue_cfg["venue"]["floor_size"]
    whole_floor = [[-fx / 2, -fy / 2], [fx / 2, -fy / 2], [fx / 2, fy / 2], [-fx / 2, fy / 2]]

    def polygon_of(zone_id):
        # "floor" is a pseudo-zone: anywhere on the Expo floor (open-area wandering)
        return whole_floor if zone_id == "floor" else zones[zone_id]["polygon"]

    def zone_type(zone_id):
        return "area" if zone_id == "floor" else zones[zone_id]["type"]

    def free_point(zone_id):
        for _ in range(200):
            p = random_point_in(polygon_of(zone_id), rng, margin=1.0)
            if not obst.blocked(*p):
                return p
        raise RuntimeError(f"no free space in zone {zone_id}")

    def facing_booth(zone_id, x, y):
        if zone_id == "floor":
            return None
        b = booths.get(zones[zone_id].get("booth"))
        if not b:
            return None
        return math.atan2(b["pose"][1] - y, b["pose"][0] - x)

    # ---- visitors: loop through several destinations
    dest_ids = list(d["destinations"].keys())
    dest_w = list(d["destinations"].values())
    for i in range(sc.get("visitors", 0)):
        speed = rng.uniform(*d["adult_speed"])
        for _attempt in range(200):
            n_stops = rng.randint(*d["stops"])
            stops = []
            for _ in range(n_stops):
                zid = rng.choices(dest_ids, dest_w)[0]
                # avoid choosing the same zone twice in a row
                while stops and zid == stops[-1][0]:
                    zid = rng.choices(dest_ids, dest_w)[0]
                stops.append((zid, free_point(zid)))
            pts = [p for _, p in stops] + [stops[0][1]]      # close the loop
            if all(obst.leg_clear(a, b) for a, b in zip(pts, pts[1:])):
                break
        else:
            raise RuntimeError(f"visitor {i}: could not find an obstacle-free route")

        x0, y0 = stops[0][1]
        tr = Trajectory(x0, y0, 0.0, hip + floor_top, d["turn_time"])
        for k, (zid, (x, y)) in enumerate(stops):
            if k > 0:
                tr.walk_to(x, y, speed)
            ztype = zone_type(zid)
            tr.wait(rng.uniform(*d["dwell"][ztype]), facing_booth(zid, x, y))
        tr.walk_to(x0, y0, speed)                            # back to start
        actors.append(dict(name=f"visitor_{i:02d}", role="visitor", scale=1.0,
                           loop=True, points=tr.points, **WALK))

    # ---- standers: audience (face the stage) and hotspot crowds (face the booth)
    stage_xy = venue_cfg["stage"]["pose"][:2]
    standers = [("audience", "plaza", sc.get("audience", 0))]
    standers += [("hotspot", h["zone"], h["people"]) for h in sc.get("hotspots", [])]
    count = {}
    for role, zid, n in standers:
        for _ in range(n):
            x, y = free_point(zid)
            if role == "audience":
                yaw = math.atan2(stage_xy[1] - y, stage_xy[0] - x)
            else:
                yaw = facing_booth(zid, x, y) or rng.uniform(-math.pi, math.pi)
            k = count.get(role, 0)
            count[role] = k + 1
            actors.append(dict(name=f"{role}_{k:02d}", role=role, scale=1.0, loop=True,
                               points=[(0.0, x, y, stand_hip + floor_top, yaw),
                                       (3600.0, x, y, stand_hip + floor_top, yaw)],
                               **STAND))

    # ---- the missing person: follows a fixed path, then stays put.
    # NOTE: loop stays TRUE. An actor with <loop>false</loop> froze the whole
    # simulation in our tests (the old "lost_child"). Instead, the final wait is
    # one hour, so the path would only restart long after any test has ended.
    mp = sc.get("missing_person")
    if mp:
        path = mp["path"]
        for a, b in zip(path, path[1:]):
            if not obst.leg_clear(a, b):
                raise RuntimeError(f"missing_person path leg {a} -> {b} crosses an obstacle")
        speed = rng.uniform(*d["missing_speed"])
        tr = Trajectory(path[0][0], path[0][1], 0.0, hip + floor_top, d["turn_time"])
        tr.wait(mp.get("start_delay", 20))                   # with the group first
        for x, y in path[1:]:
            tr.walk_to(x, y, speed)
        tr.wait(3600)                                        # then stands alone
        actors.append(dict(name="missing_person", role="missing_person", scale=1.0,
                           loop=True, points=tr.points, **WALK))
    return actors
