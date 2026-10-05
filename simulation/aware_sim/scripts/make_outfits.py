#!/usr/bin/env python3
"""
make_outfits.py: build repainted copies of the actor skins.

For each base skin (walk.dae, talk_b.dae) and each outfit in outfits.yaml:
  1. read the original .dae from the Fuel cache (never modified)
  2. find the clothing "paint pots" (effects whose id contains e.g. "sweater")
  3. replace their diffuse + ambient color with the outfit's colors
  4. save as models/aware_people/meshes/<skin>__<outfit>.dae

Skeleton, animation, skin tone and eyes are copied untouched, so every
variant still walks / talks exactly like the original.

    python3 make_outfits.py            # build all variants
    python3 make_outfits.py --list     # just show what would be built
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outfits as O  # noqa: E402

NS = "http://www.collada.org/2005/11/COLLADASchema"
ET.register_namespace("", NS)          # keep COLLADA as the default namespace on save
Q = lambda tag: f"{{{NS}}}{tag}"       # 'color' -> '{namespace}color'


def classify(effect_id, keywords):
    eid = effect_id.lower()
    for part, words in keywords.items():
        if any(w in eid for w in words):
            return part
    return None                         # skin, eyes, ... -> leave alone


def repaint(src, dst, top_rgb, bottom_rgb, keywords):
    tree = ET.parse(src)
    changed = []
    for effect in tree.getroot().iter(Q("effect")):
        part = classify(effect.get("id", ""), keywords)
        if part is None:
            continue
        rgb = top_rgb if part == "top" else bottom_rgb
        # diffuse = the main color of the surface. ambient = how it looks in shadow:
        # only repaint it if the original used one (a black ambient stays black).
        for channel in ("diffuse", "ambient"):
            for node in effect.iter(Q(channel)):
                color = node.find(Q("color"))
                if color is None:
                    continue
                vals = color.text.split()
                if channel == "ambient" and all(float(v) == 0 for v in vals[:3]):
                    continue
                alpha = vals[3] if len(vals) == 4 else "1"
                color.text = f"{rgb[0]} {rgb[1]} {rgb[2]} {alpha}"
        changed.append(f"{effect.get('id')}->{part}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, xml_declaration=True, encoding="utf-8")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show the plan only")
    args = ap.parse_args()

    cfg = yaml.safe_load((O.PKG_DIR / "config" / "outfits.yaml").read_text())
    crowd = yaml.safe_load((O.PKG_DIR / "config" / "crowd.yaml").read_text())
    errors = O.validate(cfg)
    if errors:
        sys.exit("outfits.yaml errors:\n  - " + "\n  - ".join(errors))

    # base skins: walkers get every outfit (incl. the missing person); standers get crowd outfits
    act = crowd["actor"]
    bases = [(act["skin"], cfg["crowd"] + [cfg["missing_person"]]),
             (act["stand_skin"], cfg["crowd"])]

    total = 0
    for url, outfit_list in bases:
        src = O.cached_skin(url)
        name = O.base_name(url)
        if src is None:
            sys.exit(f"Base skin not in cache: {url}\n"
                     f"Download its model first, e.g.:  gz fuel download -u \"{url.split('/tip/')[0]}\"")
        print(f"\n{name}.dae  <- {src}")
        for o in outfit_list:
            dst = O.variant_path(cfg, name, o["name"])
            if args.list:
                print(f"   would write {dst.name}  ({o['top']} / {o['bottom']})")
                continue
            changed = repaint(src, dst, cfg["colors"][o["top"]], cfg["colors"][o["bottom"]],
                              cfg["effect_keywords"])
            if len(changed) < 2:
                print(f"   WARNING {dst.name}: only repainted {changed}; check effect_keywords")
            print(f"   {dst.name:32s} {o['top']:>9s} / {o['bottom']:<6s}  [{', '.join(changed)}]")
            total += 1
    if not args.list:
        print(f"\nBuilt {total} skin variants in {O.PKG_DIR / cfg['output_dir']}")


if __name__ == "__main__":
    main()
