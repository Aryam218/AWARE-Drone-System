#!/usr/bin/env python3
"""
patrol_plan.py: compute the lawnmower patrol from zones.yaml + patrol.yaml.
Pure geometry, no drone needed, so the plan can be checked on paper first.

    python3 patrol_plan.py              # print the plan
    python3 patrol_plan.py --preview    # + docs/patrol_plan.png (coverage map)

The idea, step by step:
  1. Camera footprint: a camera looking straight down from height h sees a
     rectangle on the ground:  width = 2 h tan(hfov/2),  length = 2 h tan(vfov/2)
  2. Swath: the strip of ground covered while flying a straight line. We use
     the SHORTER side of the footprint, so the plan stays valid whichever way
     the drone happens to face (conservative choice).
  3. Spacing: neighbouring strips overlap by `overlap` so nobody falls in a gap.
  4. Strips run along the long side of the floor; flown back-and-forth
     (boustrophedon = "as the ox ploughs").
  5. Strip ends stop half a footprint before the floor edge: the camera
     already sees that far ahead, so flying further only wastes time.
"""
import argparse
import math
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent.parent
EARTH_R = 6378137.0          # WGS84 equatorial radius, meters


def load():
    venue = yaml.safe_load((PKG_DIR / "config" / "zones.yaml").read_text())
    patrol = yaml.safe_load((PKG_DIR / "config" / "patrol.yaml").read_text())
    return venue, patrol


def footprint(h, cam):
    """Ground rectangle seen from height h (meters): (across, along)."""
    hfov = cam["hfov_rad"]
    vfov = 2 * math.atan(math.tan(hfov / 2) * cam["height"] / cam["width"])
    return 2 * h * math.tan(hfov / 2), 2 * h * math.tan(vfov / 2)


def plan(venue, patrol):
    fx, fy = venue["venue"]["floor_size"]
    h = patrol["altitude_m"]
    across, along = footprint(h, patrol["camera"])
    swath = min(across, along)
    spacing = swath * (1 - patrol["overlap"])

    # Strips along the LONGER floor side (fewer turns). Here: along x.
    long_x = fx >= fy
    length, width = (fx, fy) if long_x else (fy, fx)
    n = 1 if width <= swath else math.ceil((width - swath) / spacing) + 1
    if n == 1:
        offsets = [0.0]
    else:
        first, last = -width / 2 + swath / 2, width / 2 - swath / 2
        offsets = [first + i * (last - first) / (n - 1) for i in range(n)]

    # Start with the strip nearest the launch pad, and the end nearest to it
    pad_x, pad_y = venue["launch_pad"]["pose"][:2]
    pad_off = pad_y if long_x else pad_x
    pad_along = pad_x if long_x else pad_y
    if abs(offsets[-1] - pad_off) < abs(offsets[0] - pad_off):
        offsets.reverse()
    # Strip ends: the camera already sees along/2 ahead and behind, so the drone
    # can turn that much before the floor edge and still cover it.
    half = max(0.0, length / 2 - along / 2)
    start = -half if pad_along <= 0 else half

    waypoints = []
    for i, off in enumerate(offsets):
        a, b = (start, -start) if i % 2 == 0 else (-start, start)
        for s in (a, b):
            waypoints.append((s, off) if long_x else (off, s))

    gsd = h / (patrol["camera"]["width"] / 2 / math.tan(patrol["camera"]["hfov_rad"] / 2))
    actual = abs(offsets[1] - offsets[0]) if len(offsets) > 1 else 0.0
    return dict(altitude=h, across=across, along=along, swath=swath, spacing=spacing,
                actual_spacing=actual,
                strips=len(offsets), waypoints=waypoints, gsd_m=gsd,
                pattern_length=sum(math.dist(p, q) for p, q in zip(waypoints, waypoints[1:])))


def enu_to_geo(x, y, location):
    """Gazebo world (x East, y North, meters) -> latitude/longitude (degrees).
    Flat-Earth approximation: exact enough over a few hundred meters."""
    lat0 = location["latitude"]
    lat = lat0 + math.degrees(y / EARTH_R)
    lon = location["longitude"] + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def coverage(venue, p, step=0.5):
    """% of the floor that falls inside at least one strip's swath."""
    fx, fy = venue["venue"]["floor_size"]
    wps = p["waypoints"]
    strips = list(zip(wps[0::2], wps[1::2]))
    covered = total = 0
    y = -fy / 2
    while y <= fy / 2:
        x = -fx / 2
        while x <= fx / 2:
            total += 1
            for (x1, y1), (x2, y2) in strips:
                if y1 == y2:   # strip along x
                    if (min(x1, x2) - p["along"] / 2 <= x <= max(x1, x2) + p["along"] / 2
                            and abs(y - y1) <= p["swath"] / 2):
                        covered += 1
                        break
                else:          # strip along y
                    if (min(y1, y2) - p["along"] / 2 <= y <= max(y1, y2) + p["along"] / 2
                            and abs(x - x1) <= p["swath"] / 2):
                        covered += 1
                        break
            x += step
        y += step
    return 100.0 * covered / total


def preview(venue, patrol, p, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fx, fy = venue["venue"]["floor_size"]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.add_patch(Rectangle((-fx / 2, -fy / 2), fx, fy, fc="#e6e6e0", ec="k"))
    for b in venue["booths"]:
        x, y, _ = b["pose"]
        d, w = b["size"]
        ax.add_patch(Rectangle((x - d / 2, y - w / 2), d, w, fc=b["color"], ec="k"))
    wps = p["waypoints"]
    for (x1, y1), (x2, y2) in zip(wps[0::2], wps[1::2]):   # swath of each strip
        if y1 == y2:
            ax.add_patch(Rectangle((min(x1, x2) - p["along"] / 2, y1 - p["swath"] / 2),
                                   abs(x2 - x1) + p["along"], p["swath"],
                                   fc="tab:blue", alpha=0.15, ec="tab:blue", ls="--"))
        else:
            ax.add_patch(Rectangle((x1 - p["swath"] / 2, min(y1, y2) - p["along"] / 2),
                                   p["swath"], abs(y2 - y1) + p["along"],
                                   fc="tab:blue", alpha=0.15, ec="tab:blue", ls="--"))
    pad = venue["launch_pad"]["pose"][:2]
    route = [tuple(pad)] + wps + [tuple(pad)]
    ax.plot(*zip(*route), "-o", color="tab:red", lw=2, ms=5)
    for i, (x, y) in enumerate(wps):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(6, 6), color="tab:red")
    ax.plot(*pad, "s", color="k", ms=12)
    ax.text(pad[0] + 2, pad[1], "launch pad", va="center")
    ax.set_xlim(-60, 60); ax.set_ylim(-40, 35); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"Patrol at {p['altitude']} m: {p['strips']} strips, swath {p['swath']:.1f} m, "
                 f"spacing {p['spacing']:.1f} m, coverage {coverage(venue, p):.0f} %")
    fig.tight_layout(); fig.savefig(out, dpi=100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    venue, patrol = load()
    p = plan(venue, patrol)
    print(f"Altitude {p['altitude']} m -> footprint {p['across']:.1f} x {p['along']:.1f} m, "
          f"1 pixel = {p['gsd_m'] * 100:.1f} cm")
    print(f"Swath {p['swath']:.1f} m, max spacing {p['spacing']:.1f} m -> {p['strips']} strips, "
          f"spread evenly {p['actual_spacing']:.1f} m apart")
    print(f"Pattern length {p['pattern_length']:.0f} m "
          f"(~{p['pattern_length'] / patrol['speed_m_s']:.0f} s per loop at {patrol['speed_m_s']} m/s)")
    print(f"Floor coverage: {coverage(venue, p):.0f} %")
    for i, (x, y) in enumerate(p["waypoints"], 1):
        lat, lon = enu_to_geo(x, y, venue["venue"]["location"])
        print(f"  wp{i}: x={x:6.1f} y={y:6.1f}  ->  lat {lat:.7f}, lon {lon:.7f}")
    if args.preview:
        out = PKG_DIR / "docs" / "patrol_plan.png"
        out.parent.mkdir(exist_ok=True)
        print("Wrote", preview(venue, patrol, p, out))


if __name__ == "__main__":
    main()
