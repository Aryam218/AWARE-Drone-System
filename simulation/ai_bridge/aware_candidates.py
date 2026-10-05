"""
aware_candidates.py: choose WHICH detected people are sent to the VLM.

Problem it solves (seen in the live simulation):
  ByteTrack only confirms a person after matching their box in consecutive
  frames. Live, frames are processed 1-4 s apart while the drone moves, so
  boxes rarely overlap and almost nobody gets confirmed. Only static things
  (the drone's own legs, a whole-frame box) reached the VLM.

What it does instead (search mode):
  1. Start from ALL Grounding DINO detections in the frame.
  2. Drop impossible boxes: huge (whole-frame) boxes, and large boxes glued to
     the left/right image edge (the drone's legs in a tilted camera).
  3. Read the colors named in the description ("red top", "white jeans").
  4. Score each box by how much of its upper body is the top color (and lower
     body the bottom color): a CHEAP filter, milliseconds for 150 boxes.
  5. Send only the best few (default 3) to the EXPENSIVE VLM check.
  Keeps ByteTrack IDs where a box matches a confirmed track; other boxes get a
  temporary negative ID for this frame.

Plug-in point: detector_vlm_pipeline.py, right after ByteTrack's update().
"""
import re
from itertools import count

import cv2
import numpy as np
from loguru import logger

# ---------------------------------------------------------------- settings
MAX_CANDIDATES = 3          # VLM checks per frame (cost / time budget)
MIN_COLOR_SCORE = 0.04      # below this, a box doesn't look like the description
MAX_BOX_WIDTH = 0.60        # fraction of image width: wider = not a person
MAX_BOX_HEIGHT = 0.80
EDGE_PX = 4                 # "touching the image edge"
EDGE_MIN_AREA = 0.015       # large boxes on the side edges = drone legs
MATCH_IOU = 0.5             # box <-> ByteTrack track association

# ---------------------------------------------------------------- colors
# OpenCV HSV: H 0-179, S 0-255, V 0-255
def _mask(hsv, name):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if name == "red":
        return ((h <= 10) | (h >= 170)) & (s >= 100) & (v >= 60)
    if name == "orange":
        return (h >= 11) & (h <= 19) & (s >= 100) & (v >= 100)
    if name == "yellow":
        return (h >= 20) & (h <= 34) & (s >= 100) & (v >= 100)
    if name == "green":
        return (h >= 35) & (h <= 85) & (s >= 60) & (v >= 40)
    if name == "blue":
        return (h >= 90) & (h <= 130) & (s >= 60) & (v >= 40)
    if name == "purple":
        return (h >= 131) & (h <= 160) & (s >= 60) & (v >= 40)
    if name == "pink":
        return (h >= 145) & (h <= 169) & (s >= 50) & (v >= 120)
    if name == "brown":
        return (h >= 8) & (h <= 25) & (s >= 40) & (v >= 50) & (v <= 200)
    if name == "white":
        return (s <= 40) & (v >= 170)
    if name == "black":
        return v <= 50
    if name == "grey":
        return (s <= 40) & (v > 60) & (v < 170)
    raise ValueError(name)


COLOR_WORDS = {
    "red": "red", "maroon": "red", "crimson": "red", "orange": "orange",
    "yellow": "yellow", "gold": "yellow", "green": "green", "olive": "green",
    "blue": "blue", "navy": "blue", "sky": "blue", "purple": "purple",
    "violet": "purple", "pink": "pink", "brown": "brown", "beige": "brown",
    "khaki": "brown", "tan": "brown", "white": "white", "cream": "white",
    "black": "black", "grey": "grey", "gray": "grey", "silver": "grey",
}
_C = "|".join(COLOR_WORDS)
TOP_WORDS = r"top|shirt|t-shirt|tshirt|sweater|jacket|hoodie|blouse|dress|thobe|abaya|coat"
BOTTOM_WORDS = r"bottom|bottoms|trousers|pants|jeans|skirt|shorts|leggings"


def parse_colors(description):
    """'a person in a red top and white jeans' -> ('red', 'white')"""
    d = description.lower()
    top = re.search(rf"\b({_C})\b(?:\s+\w+)?\s+({TOP_WORDS})\b", d)
    bottom = re.search(rf"\b({_C})\b(?:\s+\w+)?\s+({BOTTOM_WORDS})\b", d)
    top_c = COLOR_WORDS[top.group(1)] if top else None
    bot_c = COLOR_WORDS[bottom.group(1)] if bottom else None
    if top_c is None and bot_c is None:            # e.g. "a person in red"
        first = re.search(rf"\b({_C})\b", d)
        top_c = COLOR_WORDS[first.group(1)] if first else None
    return top_c, bot_c


def color_score(hsv, box, top_c, bot_c):
    """Returns (top_fraction, bottom_fraction): how much of the upper body is
    the top color and how much of the lower body is the bottom color.
    Central 60 % of the width avoids background around the person."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 6:
        return 0.0, 0.0
    cx1, cx2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    top_f = bot_f = 0.0
    if top_c:
        region = hsv[int(y1 + 0.15 * h):int(y1 + 0.50 * h), cx1:cx2]
        if region.size:
            top_f = float(_mask(region, top_c).mean())
    if bot_c:
        region = hsv[int(y1 + 0.50 * h):int(y1 + 0.90 * h), cx1:cx2]
        if region.size:
            bot_f = float(_mask(region, bot_c).mean())
    return top_f, bot_f


# ---------------------------------------------------------------- helpers
_temp_ids = count(start=-1, step=-1)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _as_array(x):
    if hasattr(x, "detach"):                       # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def plausible(box, W, H):
    """False for boxes that cannot be a single person in our camera view."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w > MAX_BOX_WIDTH * W or h > MAX_BOX_HEIGHT * H:
        return False                               # whole-frame / huge box
    on_side = x1 <= EDGE_PX or x2 >= W - EDGE_PX
    if on_side and (w * h) > EDGE_MIN_AREA * W * H:
        return False                               # drone leg at the image side
    return True


# ---------------------------------------------------------------- main entry
def select_candidates(image, boxes, scores, tracked_people, description,
                      max_candidates=MAX_CANDIDATES):
    """Return a list shaped like ByteTrack's output:
    [{"bbox": [x1,y1,x2,y2], "confidence": float, "track_id": int}, ...]"""
    rgb = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    boxes = _as_array(boxes).reshape(-1, 4) if len(boxes) else np.empty((0, 4), np.float32)
    scores = _as_array(scores).reshape(-1) if len(scores) else np.empty((0,), np.float32)
    top_c, bot_c = parse_colors(description)

    tracks = [(np.asarray(p["bbox"], np.float32), int(p["track_id"])) for p in tracked_people]
    ranked, dropped = [], 0
    for box, conf in zip(boxes, scores):
        if not plausible(box, W, H):
            dropped += 1
            continue
        tid = None
        for tbox, t in tracks:                      # keep a confirmed ByteTrack ID
            if _iou(box, tbox) >= MATCH_IOU:
                tid = t
                break
        if top_c or bot_c:
            top_f, bot_f = color_score(hsv, box, top_c, bot_c)
            # the TOP color is the strongest clue: a box must show it to qualify
            key_f = top_f if top_c else bot_f
            if key_f < MIN_COLOR_SCORE:
                continue
            cs = top_f + 0.5 * bot_f
        else:
            cs = float(conf)                       # no color named: detector score
        ranked.append((cs, float(conf), box, tid))

    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    chosen = ranked[:max_candidates]

    logger.info(f"AWARE candidates: {len(boxes)} detections, {dropped} implausible, "
                f"colors top={top_c} bottom={bot_c}, "
                f"best color scores={[round(r[0], 3) for r in chosen]}")
    return [dict(bbox=[float(v) for v in box], confidence=conf,
                 track_id=tid if tid is not None else next(_temp_ids))
            for cs, conf, box, tid in chosen]
