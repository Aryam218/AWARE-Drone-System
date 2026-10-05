#!/usr/bin/env python3
"""
preview_scenario.py: draw the venue + every actor's route as a top-view PNG.
Check a scenario on paper BEFORE loading it into Gazebo.

    python3 preview_scenario.py --scenario missing_person
    -> docs/scenario_missing_person.png
"""
import argparse
import math
import sys
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crowd import build_actors  # noqa: E402

ROLE_STYLE = {"visitor": ("tab:blue", 0.35), "audience": ("tab:purple", 1),
              "hotspot": ("tab:red", 1), "missing_person": ("tab:orange", 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="normal")
    args = ap.parse_args()
    venue = yaml.safe_load((PKG_DIR / "config" / "zones.yaml").read_text())
    crowd = yaml.safe_load((PKG_DIR / "config" / "crowd.yaml").read_text())
    actors = build_actors(venue, crowd, args.scenario)

    fig, ax = plt.subplots(figsize=(11, 7.2))
    fx, fy = venue["venue"]["floor_size"]
    ax.add_patch(Rectangle((-fx / 2, -fy / 2), fx, fy, fc="#e6e6e0", ec="k"))
    for z in venue["zones"]:
        ax.add_patch(Polygon(z["polygon"], fc="none", ec="gray", ls="--"))
    for b in venue["booths"]:
        x, y, yaw = b["pose"]
        d, w = b["size"]
        r = Rectangle((-d / 2, -w / 2), d, w, fc=b["color"], ec="k")
        r.set_transform(Affine2D().rotate(yaw).translate(x, y) + ax.transData)
        ax.add_patch(r)
    s = venue["stage"]
    ax.add_patch(Rectangle((s["pose"][0] - s["size"][0] / 2, s["pose"][1] - s["size"][1] / 2),
                           s["size"][0], s["size"][1], fc="#40404d"))

    for a in actors:
        color, alpha = ROLE_STYLE[a["role"]]
        xs = [p[1] for p in a["points"]]
        ys = [p[2] for p in a["points"]]
        if a["role"] in ("audience", "hotspot"):
            ax.plot(xs[0], ys[0], "o", color=color, ms=4)
        else:
            lw = 2.5 if a["role"] == "missing_person" else 1
            ax.plot(xs, ys, "-", color=color, alpha=alpha, lw=lw)
            ax.plot(xs[0], ys[0], "o", color=color, ms=4)
            if a["role"] == "missing_person":
                ax.plot(xs[-1], ys[-1], "*", color=color, ms=16, mec="k")

    handles = [plt.Line2D([], [], color=c, lw=2, label=r) for r, (c, _) in ROLE_STYLE.items()
               if any(a["role"] == r for a in actors)]
    ax.legend(handles=handles, loc="upper right")
    ax.set_xlim(-42, 42); ax.set_ylim(-27, 27); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"Scenario '{args.scenario}': {len(actors)} actors (dots = start, star = missing person ends)")
    out = PKG_DIR / "docs" / f"scenario_{args.scenario}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=100)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
