#!/usr/bin/env python3
"""
inspect_skin.py: look inside an actor's COLLADA (.dae) file to see how its
clothes get their color, so we know HOW to make recolored variants.

A .dae file is XML. The parts that matter for appearance:
  <library_images>    -> texture pictures (png/jpg) painted onto the body
  <library_effects>   -> "paint recipes": a flat color OR a texture reference
  <library_materials> -> named materials that point to an effect
  <geometry>          -> mesh pieces (e.g. body, shirt, pants) using a material

Usage:
  python3 inspect_skin.py                  # finds the cached walk.dae automatically
  python3 inspect_skin.py path/to/file.dae
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def find_cached(name="walk.dae"):
    root = Path.home() / ".gz" / "fuel"
    hits = sorted(root.rglob(name))
    return hits


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        hits = find_cached()
        if not hits:
            sys.exit("walk.dae not found in ~/.gz/fuel. Run the world once, or pass a path.")
        print("Cached skins found:")
        for h in hits:
            print("  ", h)
        path = hits[0]
    print(f"\nInspecting: {path}\n")
    tree = ET.parse(path)
    r = tree.getroot()

    print("== IMAGES (textures) ==")
    imgs = r.findall(".//c:library_images/c:image", NS)
    for im in imgs:
        f = im.find(".//c:init_from", NS)
        print(f"  id={im.get('id')}  file={f.text if f is not None else '?'}")
    if not imgs:
        print("  (none: colors come from flat material colors)")

    print("\n== EFFECTS (paint recipes) ==")
    for ef in r.findall(".//c:library_effects/c:effect", NS):
        diffuse = ef.find(".//c:diffuse", NS)
        desc = "no diffuse"
        if diffuse is not None:
            col = diffuse.find("c:color", NS)
            tex = diffuse.find("c:texture", NS)
            if col is not None:
                desc = f"flat color RGBA = {col.text.strip()}"
            elif tex is not None:
                desc = f"texture -> {tex.get('texture')}"
        print(f"  id={ef.get('id')}: {desc}")

    print("\n== MATERIALS -> EFFECT ==")
    for m in r.findall(".//c:library_materials/c:material", NS):
        ie = m.find("c:instance_effect", NS)
        print(f"  {m.get('id')} ({m.get('name')}) -> {ie.get('url') if ie is not None else '?'}")

    print("\n== MESH PIECES -> MATERIAL ==")
    for g in r.findall(".//c:library_geometries/c:geometry", NS):
        mats = {p.get("material") for p in g.iter() if p.get("material")}
        print(f"  {g.get('id')} ({g.get('name')}): materials {sorted(mats)}")

    # a Fuel model keeps meshes/ and materials/ side by side: search the model folder
    model_dir = path.parent.parent if path.parent.name == "meshes" else path.parent
    print(f"\nTexture files in {model_dir}:")
    found = [f for f in sorted(model_dir.rglob("*"))
             if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tga")]
    for f in found:
        print("  ", f)
    if not found:
        print("   (none)")


if __name__ == "__main__":
    main()
