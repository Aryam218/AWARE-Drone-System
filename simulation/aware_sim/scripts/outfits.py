"""
outfits.py: shared helpers for the costume system.

  - where a base skin lives in the Fuel cache
  - the file name of a repainted variant
  - which outfit each actor wears (seeded, so reproducible)
"""
import random
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent
FUEL_CACHE = Path.home() / ".gz" / "fuel"


def cached_skin(url):
    """Fuel URL -> file in ~/.gz/fuel (the cache mirrors the URL like an address).
    https://fuel.gazebosim.org/1.0/<owner>/models/<model>/tip/files/meshes/<file>
    -> ~/.gz/fuel/fuel.gazebosim.org/<owner>/models/<model>/<version>/meshes/<file>"""
    parts = url.split("/")
    host = parts[2]
    owner = parts[4].lower()
    model = parts[6].lower()
    filename = parts[-1]
    model_dir = FUEL_CACHE / host / owner / "models" / model
    versions = sorted((d for d in model_dir.glob("*") if d.name.isdigit()),
                      key=lambda d: int(d.name))
    if not versions:
        return None
    path = versions[-1] / "meshes" / filename      # newest version = "tip"
    return path if path.exists() else None


def base_name(url):
    """'.../meshes/walk.dae' -> 'walk'"""
    return url.rsplit("/", 1)[-1].removesuffix(".dae")


def variant_path(cfg, base, outfit_name):
    return PKG_DIR / cfg["output_dir"] / f"{base}__{outfit_name}.dae"


def validate(cfg):
    errors = []
    colors = cfg["colors"]
    for o in cfg["crowd"] + [cfg["missing_person"]]:
        for part in ("top", "bottom"):
            if o[part] not in colors:
                errors.append(f"outfit {o['name']}: unknown color '{o[part]}'")
    reserved = cfg["missing_person"]["top"]
    for o in cfg["crowd"]:
        if reserved in (o["top"], o["bottom"]):
            errors.append(f"outfit {o['name']} uses '{reserved}', which is reserved for the missing person")
    names = [o["name"] for o in cfg["crowd"]] + [cfg["missing_person"]["name"]]
    if len(names) != len(set(names)):
        errors.append("duplicate outfit name")
    return errors


def assign(actors, cfg, seed):
    """Give every actor an outfit. Uses its OWN random generator, so changing
    costumes never changes anyone's route (and vice versa)."""
    rng = random.Random(seed + 1000)
    crowd = cfg["crowd"]
    weights = [o.get("weight", 1) for o in crowd]
    for a in actors:
        outfit = cfg["missing_person"] if a["role"] == "missing_person" else rng.choices(crowd, weights)[0]
        a["outfit"] = outfit["name"]
        a["outfit_desc"] = f"{outfit['top']} top, {outfit['bottom']} bottom"
