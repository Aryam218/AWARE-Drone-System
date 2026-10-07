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
  3. Drop boxes belonging to track IDs the operator already rejected, and
     people standing where the operator rejected someone (exclude_box).
  4. Read the colors named in the description ("red top", "white jeans").
  5. Score each box by how much of its upper body is the top color (and lower
     body the bottom color): a CHEAP filter, milliseconds for 150 boxes.
  6. Return the best few candidates, skipping near-duplicate boxes of the
     same person. The pipeline then decides which of them actually need a
     VLM call (cache + per-frame VLM budget live in detector_vlm_pipeline.py).
  Keeps ByteTrack IDs where a box matches a confirmed track; other boxes get a
  temporary negative ID for this frame.

Plug-in point: detector_vlm_pipeline.py, right after ByteTrack's update().

verify_target() (re-detection while TRACKING):
  The small frame-to-frame tracker can slide off the person onto the
  background (e.g. a booth wall). Every few seconds, CloudTrack runs the
  person detector again and verify_target() checks that the tracked box is
  still on a detected person who still shows the described colours. If the
  person moved a little, it re-finds them nearby; if not, the target is
  reported missing.
"""
import re
from itertools import count

import cv2
import numpy as np
from loguru import logger

# ---------------------------------------------------------------- settings
MAX_CANDIDATES = 3          # default number of candidates returned per frame
MIN_COLOR_SCORE = 0.04      # below this, a box doesn't look like the description
MAX_BOX_WIDTH = 0.60        # fraction of image width: wider = not a person
MAX_BOX_HEIGHT = 0.80
EDGE_PX = 4                 # "touching the image edge"
EDGE_MIN_AREA = 0.015       # large boxes on the side edges = drone legs
MATCH_IOU = 0.5             # box <-> ByteTrack track association
DUPLICATE_IOU = 0.6         # two boxes overlapping this much = the same person
REDETECT_MATCH_IOU = 0.3    # tracked box <-> detected person: "still on them"
REDETECT_SEARCH_RADIUS = 2.5  # re-find the person within this many box diagonals

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


def _overlap(a, b):
    """Intersection divided by the SMALLER box's area. Catches a small box
    drawn inside a bigger box of the same person (IoU misses that)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


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
                      max_candidates=MAX_CANDIDATES, exclude_track_ids=None,
                      exclude_box=None):
    """Return a list shaped like ByteTrack's output:
    [{"bbox": [x1,y1,x2,y2], "confidence": float, "track_id": int}, ...]

    exclude_track_ids: ByteTrack IDs (e.g. operator-rejected people) that
    must never be returned, so they don't take up a candidate slot.
    exclude_box: optional function box -> True to skip that box (e.g. the
    person stands where the operator already rejected someone)."""
    exclude = set(exclude_track_ids or ())

    rgb = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    if boxes is None or len(boxes) == 0:
        boxes = np.empty((0, 4), np.float32)
        scores = np.empty((0,), np.float32)
    else:
        boxes = _as_array(boxes).reshape(-1, 4)
        if scores is None or len(scores) == 0:
            scores = np.ones((len(boxes),), np.float32)
        else:
            scores = _as_array(scores).reshape(-1)

    top_c, bot_c = parse_colors(description)

    tracks = [(np.asarray(p["bbox"], np.float32), int(p["track_id"])) for p in tracked_people]
    ranked, dropped, excluded = [], 0, 0
    for box, conf in zip(boxes, scores):
        if not plausible(box, W, H):
            dropped += 1
            continue
        tid = None
        for tbox, t in tracks:                      # keep a confirmed ByteTrack ID
            if _iou(box, tbox) >= MATCH_IOU:
                tid = t
                break
        if tid is not None and tid in exclude:      # operator already rejected
            excluded += 1
            continue
        if exclude_box is not None and exclude_box(box):   # rejected place
            excluded += 1
            continue
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
    # Grounding DINO often returns two almost identical boxes for one person:
    # keep only the best of overlapping boxes, so GPT isn't asked twice.
    chosen = []
    for r in ranked:
        if all(_overlap(r[2], c[2]) < DUPLICATE_IOU for c in chosen):
            chosen.append(r)
        if len(chosen) == max_candidates:
            break

    logger.info(f"AWARE candidates: {len(boxes)} detections, {dropped} implausible, "
                f"{excluded} rejected, colors top={top_c} bottom={bot_c}, "
                f"best color scores={[round(r[0], 3) for r in chosen]}")
    return [dict(bbox=[float(v) for v in box], confidence=conf,
                 track_id=tid if tid is not None else next(_temp_ids))
            for cs, conf, box, tid in chosen]


# ---------------------------------------------------------------- re-detection
def verify_target(image, boxes, scores, track_box, anchor_box, description,
                  anchor_distance=None):
    """Check the tracked target against a fresh person detection.

    image:       the current frame (PIL or RGB array)
    boxes:       person detections in this frame (x1, y1, x2, y2)
    track_box:   where the small tracker thinks the target is now
    anchor_box:  the target's last VERIFIED box (detector-confirmed)
    description: the missing-person description (for the colour check)
    anchor_distance: optional function box -> metres on the GROUND from the
                 target's last verified position, or None if too far / unknown.
                 Used instead of the pixel radius when the drone position is
                 known: the drone moving shifts the whole image, but not where
                 people stand.

    Returns (box, how):
      (box, "confirmed")   the tracked box is still on a matching person
      (box, "reacquired")  the tracker had slipped; the person was found nearby
      (None, reason)       no matching person here; ground search includes counts
    """
    rgb = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if boxes is None or len(boxes) == 0:
        return None, "missing: 0 people in view; 0 near last position with wrong colors"
    boxes = _as_array(boxes).reshape(-1, 4)
    people = [b for b in boxes if plausible(b, W, H)]
    top_c, bot_c = parse_colors(description or "")

    def colour(box):
        if not (top_c or bot_c):
            return 0.0
        top_f, bot_f = color_score(hsv, box, top_c, bot_c)
        return top_f + 0.5 * bot_f

    def looks_right(box):
        if not (top_c or bot_c):
            return True                              # no colour named: can't check
        top_f, bot_f = color_score(hsv, box, top_c, bot_c)
        return (top_f if top_c else bot_f) >= MIN_COLOR_SCORE

    # 1. Is the tracked box still on a detected person who looks right?
    if track_box is not None:
        track_box = np.asarray(track_box, np.float32).reshape(4)
        best, best_iou = None, 0.0
        for b in people:
            i = _iou(b, track_box)
            if i > best_iou:
                best, best_iou = b, i
        if best is not None and best_iou >= REDETECT_MATCH_IOU and looks_right(best):
            return best, "confirmed"

    # 2. The tracker slipped: look for the person near their last verified spot.
    if anchor_distance is not None:                 # by ground position
        nearby = []
        wrong_colors = 0
        for b in people:
            d = anchor_distance(b)
            if d is not None:
                if looks_right(b):
                    nearby.append((colour(b), -d, b))
                else:
                    wrong_colors += 1
        if nearby:
            nearby.sort(key=lambda r: (r[0], r[1]), reverse=True)
            return nearby[0][2], "reacquired"
        return None, (f"missing: {len(people)} people in view; "
                      f"{wrong_colors} near last position with wrong colors")

    ref = anchor_box if anchor_box is not None else track_box
    if ref is None:
        return None, "missing"
    ref = np.asarray(ref, np.float32).reshape(4)
    cx, cy = (ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2
    radius = REDETECT_SEARCH_RADIUS * float(np.hypot(ref[2] - ref[0], ref[3] - ref[1]))
    nearby = []
    for b in people:
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d = float(np.hypot(bx - cx, by - cy))
        if d <= radius and looks_right(b):
            nearby.append((colour(b), -d, b))
    if nearby:
        nearby.sort(key=lambda r: (r[0], r[1]), reverse=True)  # best colour, then closest
        return nearby[0][2], "reacquired"

    return None, "missing"


# ---------------------------------------------------------------- appearance
SAME_LOOK_MAX_DISTANCE = 0.45   # colour-histogram distance: below = same clothes


def appearance_signature(image, box):
    """A small colour fingerprint of a person's clothes (upper and lower body),
    used to tell whether two detections are probably the same person.
    Returns None if the box is too small."""
    rgb = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    x1, y1, x2, y2 = [int(v) for v in np.asarray(box).reshape(4)]
    H, W = hsv.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    w, h = x2 - x1, y2 - y1
    if w < 4 or h < 6:
        return None
    cx1, cx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    parts = []
    for top, bottom in ((0.15, 0.50), (0.50, 0.90)):          # upper body, lower body
        region = hsv[y1 + int(top * h):y1 + int(bottom * h), cx1:cx2]
        if region.size == 0:
            return None
        # hue x saturation x brightness (brightness separates white from grey)
        hist = cv2.calcHist([region], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        parts.append(hist.astype(np.float32))
    return parts


def same_look(sig_a, sig_b, max_distance=SAME_LOOK_MAX_DISTANCE):
    """True if two appearance signatures probably show the same clothes."""
    if sig_a is None or sig_b is None:
        return False
    d = max(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA) for a, b in zip(sig_a, sig_b))
    return d <= max_distance