#!/usr/bin/env python3
"""
generate_world.py: build the AWARE Gazebo world

    zones.yaml (the stage) + crowd.yaml (the cast) + aware_expo.sdf.jinja (the form)
                                   |
                         generate_world.py (fills in the form)
                                   |
             worlds/aware_expo.sdf         (what Gazebo / PX4 load)
             worlds/aware_expo.actors.yaml (ground truth: who is where, for testing)

Usage (from anywhere):
    python3 generate_world.py                          # scenario "normal"
    python3 generate_world.py --scenario missing_person  # pick a scenario
    python3 generate_world.py --scenario none          # empty venue, no people
    python3 generate_world.py --debug-zones            # + transparent zone overlays
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import yaml
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError as e:
    sys.exit(f"Missing library: {e.name}. Install with: sudo apt install python3-yaml python3-jinja2")

# Paths are relative to THIS file, so the script works from any directory.
PKG_DIR = Path(__file__).resolve().parent.parent
CONFIG = PKG_DIR / "config" / "zones.yaml"
CROWD = PKG_DIR / "config" / "crowd.yaml"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crowd import build_actors  # noqa: E402  (our own module, next to this file)
import outfits as O               # noqa: E402
TEMPLATE_DIR = PKG_DIR / "worlds"
TEMPLATE = "aware_expo.sdf.jinja"


# ---------------------------------------------------------------- helpers
def fmt(*values):
    """Numbers -> 'a b c' string for SDF, rounded to keep the file readable."""
    return " ".join(f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
                    for v in values)


def polygon_area(pts):
    """Shoelace formula: positive = counter-clockwise, negative = clockwise."""
    return 0.5 * sum(x1 * y2 - x2 * y1
                     for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))


# ---------------------------------------------------------------- booth geometry
def booth_parts(b):
    """
    Break one booth into boxes, in the booth's LOCAL frame:
    origin at the booth center on the floor, +x toward the open front.
    Returns a list of {name, pose, size, color} ready for the template.
    """
    d, w = b["size"]                 # depth (x), width (y)
    h = b["wall_height"]
    t = 0.1                          # wall thickness
    color = fmt(*b["color"])
    wall = fmt(0.92, 0.92, 0.9)      # off-white walls
    return [
        # colored floor pad so each booth is recognizable from above
        dict(name="pad",   pose=fmt(0, 0, 0.015, 0, 0, 0),              size=fmt(d, w, 0.03),   color=color),
        dict(name="back",  pose=fmt(-d/2 + t/2, 0, h/2, 0, 0, 0),       size=fmt(t, w, h),      color=wall),
        dict(name="left",  pose=fmt(0,  w/2 - t/2, h/2, 0, 0, 0),       size=fmt(d, t, h),      color=wall),
        dict(name="right", pose=fmt(0, -w/2 + t/2, h/2, 0, 0, 0),       size=fmt(d, t, h),      color=wall),
        # counter near the open front, leaves gaps at both sides to walk in
        dict(name="counter", pose=fmt(d/2 - 0.4, 0, 0.5, 0, 0, 0),      size=fmt(0.6, w*0.6, 1.0), color=color),
        # sign board on top of the back wall, in the booth color
        dict(name="sign",  pose=fmt(-d/2 + t/2, 0, h + 0.4, 0, 0, 0),   size=fmt(t, w, 0.8),    color=color),
    ]


# ---------------------------------------------------------------- validation
def validate(cfg):
    """Catch blueprint mistakes BEFORE Gazebo does (its errors are far less clear)."""
    errors = []
    fx, fy = cfg["venue"]["floor_size"]
    inside = lambda x, y: -fx/2 <= x <= fx/2 and -fy/2 <= y <= fy/2

    booth_ids = [b["id"] for b in cfg["booths"]]
    if len(booth_ids) != len(set(booth_ids)):
        errors.append("duplicate booth id")
    for b in cfg["booths"]:
        x, y, _ = b["pose"]
        if not inside(x, y):
            errors.append(f"{b['id']}: center ({x}, {y}) is outside the floor")

    lp = cfg.get("launch_pad")
    if lp:
        x, y, _ = lp["pose"]
        half = lp["size"] / 2
        if abs(x) - half < fx / 2 and abs(y) - half < fy / 2:
            errors.append(f"launch_pad at ({x}, {y}) overlaps the Expo floor: keep it outside the crowd area")
    else:
        errors.append("launch_pad missing")

    zone_ids = [z["id"] for z in cfg["zones"]]
    if len(zone_ids) != len(set(zone_ids)):
        errors.append("duplicate zone id")
    for z in cfg["zones"]:
        pts = z["polygon"]
        if len(pts) < 3:
            errors.append(f"zone {z['id']}: needs at least 3 points")
            continue
        for x, y in pts:
            if not inside(x, y):
                errors.append(f"zone {z['id']}: point ({x}, {y}) is outside the floor")
        if polygon_area(pts) <= 0:
            errors.append(f"zone {z['id']}: points must go counter-clockwise")
        if z["type"] == "booth_front" and z.get("booth") not in booth_ids:
            errors.append(f"zone {z['id']}: references unknown booth '{z.get('booth')}'")
        if z.get("capacity", 0) <= 0:
            errors.append(f"zone {z['id']}: capacity must be > 0")
    return errors


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Generate the AWARE Gazebo world")
    ap.add_argument("--debug-zones", action="store_true",
                    help="draw zones as transparent overlays (development only)")
    ap.add_argument("--scenario", default="normal",
                    help="crowd scenario from crowd.yaml, or 'none' for no people")
    ap.add_argument("--no-outfits", action="store_true",
                    help="use the original Fuel skins (everyone dressed the same)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every people count in the scenario (e.g. 2 = double crowd)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output file (default: worlds/<venue name>.sdf)")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())

    errors = validate(cfg)
    if errors:
        print("Blueprint errors in zones.yaml:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    # ---- the cast
    crowd = yaml.safe_load(CROWD.read_text())
    if args.scenario == "none":
        actors = []
    elif args.scenario in crowd["scenarios"]:
        sc = crowd["scenarios"][args.scenario]
        if args.scale != 1.0:     # scale the cast without editing the YAML
            for key in ("visitors", "audience"):
                if key in sc:
                    sc[key] = round(sc[key] * args.scale)
            for h in sc.get("hotspots", []):
                h["people"] = round(h["people"] * args.scale)
        try:
            actors = build_actors(cfg, crowd, args.scenario)
        except RuntimeError as e:
            sys.exit(f"Crowd error: {e}")
    else:
        sys.exit(f"Unknown scenario '{args.scenario}'. "
                 f"Choose from: {', '.join(crowd['scenarios'])}, none")
    # ---- the costumes: swap each actor's skin for its outfit variant
    if actors and not args.no_outfits:
        ocfg = yaml.safe_load((PKG_DIR / "config" / "outfits.yaml").read_text())
        oerr = O.validate(ocfg)
        if oerr:
            sys.exit("outfits.yaml errors:\n  - " + "\n  - ".join(oerr))
        O.assign(actors, ocfg, crowd["scenarios"][args.scenario].get("seed", 0))
        missing = set()
        for a in actors:
            path = O.variant_path(ocfg, O.base_name(a["skin"]), a["outfit"])
            if not path.exists():
                missing.add(path.name)
            a["skin"] = str(path)        # absolute path: Gazebo loads it directly
        if missing:
            sys.exit(f"{len(missing)} outfit skins are missing (e.g. {sorted(missing)[0]}).\n"
                     f"Build them first:  python3 scripts/make_outfits.py")
    for a in actors:
        a["anim_file"] = a["anim_file"] or a["skin"]   # walkers: animation lives in the skin
    for a in actors:   # numbers -> SDF strings
        a["waypoints"] = [dict(time=fmt(round(t, 2)), pose=fmt(round(x, 3), round(y, 3), round(z, 3), 0, 0, round(yaw, 4)))
                          for t, x, y, z, yaw in a["points"]]
        a["loop"] = "true" if a["loop"] else "false"

    # Prepare everything the template needs as ready-to-print strings/numbers
    booths = [dict(id=b["id"], x=b["pose"][0], y=b["pose"][1], yaw=b["pose"][2],
                   parts=booth_parts(b)) for b in cfg["booths"]]
    s = cfg["stage"]
    stage = dict(x=s["pose"][0], y=s["pose"][1], yaw=s["pose"][2],
                 sx=s["size"][0], sy=s["size"][1], h=s["size"][2])
    g = cfg["entrance_gate"]
    gate = dict(x=g["pose"][0], y=g["pose"][1], yaw=g["pose"][2],
                width=g["width"], height=g["height"])
    lp = cfg["launch_pad"]
    pad = dict(x=lp["pose"][0], y=lp["pose"][1], yaw=lp["pose"][2], size=lp["size"])
    colors = cfg["zone_debug_colors"]
    zones = [dict(id=z["id"], polygon=z["polygon"], color=fmt(*colors[z["type"]]))
             for z in cfg["zones"]]

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR),
                      undefined=StrictUndefined,     # typo in template -> clear error
                      trim_blocks=True, lstrip_blocks=True)
    sdf = env.get_template(TEMPLATE).render(
        venue=cfg["venue"], booths=booths, stage=stage, gate=gate, pad=pad,
        zones=zones, debug_zones=args.debug_zones,
        actors=actors, scenario=args.scenario)

    # Sanity check: is the result valid XML at all?
    try:
        ET.fromstring(sdf.encode())
    except ET.ParseError as e:
        sys.exit(f"Generated SDF is not valid XML: {e}")

    out = args.output or (TEMPLATE_DIR / f"{cfg['venue']['name']}.sdf")
    out.write_text(sdf)

    # Ground truth for later testing: roles and starting positions of every actor
    # Ground truth: the "director's script". Full waypoint list per actor
    # ([time_s, x, y] in the world frame), so ground_truth.py can compute where
    # everyone is at any simulation time, exactly as Gazebo moves them.
    truth = out.with_suffix(".actors.yaml")
    truth.write_text(yaml.safe_dump(dict(
        scenario=args.scenario,
        actors=[dict(name=a["name"], role=a["role"], loop=a["loop"] == "true",
                     outfit=a.get("outfit", "original"), looks=a.get("outfit_desc", ""),
                     start=[round(a["points"][0][1], 2), round(a["points"][0][2], 2)],
                     waypoints=[[round(t, 2), round(x, 3), round(y, 3)]
                                for t, x, y, _z, _yaw in a["points"]])
                for a in actors]), sort_keys=False, default_flow_style=None))

    roles = {}
    for a in actors:
        roles[a["role"]] = roles.get(a["role"], 0) + 1
    print(f"Wrote {out}")
    print(f"  scenario: {args.scenario}   actors: {len(actors)} {roles or ''}")
    mp = next((a for a in actors if a["role"] == "missing_person"), None)
    if mp and "outfit_desc" in mp:
        print(f"  missing person wears: {mp['outfit_desc']}")
    outfits_used = len({a.get("outfit") for a in actors if a.get("outfit")})
    if outfits_used:
        print(f"  outfits in use: {outfits_used}")
    print(f"  launch pad at ({pad['x']}, {pad['y']})  ->  PX4_GZ_MODEL_POSE=\"{pad['x']},{pad['y']},0.3,0,0,0\"")
    print(f"  booths: {len(booths)}   zones: {len(zones)}   "
          f"debug zones: {'ON' if args.debug_zones else 'off'}")
    if len(actors) > 300:
        print("  WARNING: more than 300 actors (above your tested 248); check Real Time Factor and camera Hz")


if __name__ == "__main__":
    main()
