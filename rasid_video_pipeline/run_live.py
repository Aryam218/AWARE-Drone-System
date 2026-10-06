#!/usr/bin/env python3
"""
run_live.py: run the AWARE search engine on the LIVE Gazebo camera, from a terminal.

The video window shows the camera LIVE (about 15 frames/s, like the YOLO test),
with the AI's latest result drawn on top. The AI itself runs in the background at
its own, slower pace (Grounding DINO + GPT), so the view never waits for it.

    camera ──► LIVE VIEW (this window, every frame)
       └─────► AI search (background, as fast as it can) ──► boxes + status

You play the operator: press c (confirm) or r (reject) in the window, or type
c / r + Enter in the terminal. q in the window quits.

    python3 run_live.py
    python3 run_live.py --description "a person wearing a red top and white trousers"
    python3 run_live.py --save live_run.mp4
"""
import argparse
import importlib
import sys
import threading
import time

import cv2

from ros_frame_source import RosFrameSource
from aware_status import publish_status         # tells the patrol to hold / resume

SEARCH_MODULE = "pipeline_runner"  # your search engine file (no .py)
WINDOW = "AWARE live search"

# ---- shared state between the AI thread and the live view --------------------
_pending = {"confirm": False, "reject": False}
_lock = threading.Lock()
_overlay = {"state": "Searching", "bbox": None, "label": "", "time": 0.0}

STATUS = {   # event name -> (status bar text, color BGR)
    "Searching": ("SEARCHING", (200, 200, 200)),
    "CandidateEvent": ("POSSIBLE MATCH - press c (confirm) / r (reject)", (0, 165, 255)),
    "TrackUpdateEvent": ("TRACKING", (0, 255, 0)),
    "ConfirmedEvent": ("CONFIRMED", (0, 255, 0)),
    "RejectedEvent": ("REJECTED - searching again", (0, 0, 255)),
    "LostEvent": ("TARGET LOST - searching again", (0, 0, 255)),
}


def _keyboard():
    for line in sys.stdin:
        key = line.strip().lower()
        if key in ("c", "r"):
            _pending["confirm" if key == "c" else "reject"] = True


def _take(name):
    """True once per key press (like a button that resets after being read)."""
    if _pending[name]:
        _pending[name] = False
        return True
    return False


def _on_ai_frame(annotated, event):
    """Called by the pipeline after each processed frame: remember the result."""
    name = type(event).__name__ if event is not None else "Searching"
    with _lock:
        if name == "Searching" and _overlay["state"] == "CandidateEvent":
            return                                 # keep the candidate until you decide
        _overlay["state"] = name
        _overlay["time"] = time.time()
        bbox = getattr(event, "bbox", None)
        _overlay["bbox"] = None if bbox is None else [int(v) for v in bbox]
        if name == "TrackUpdateEvent":
            _overlay["label"] = "confirmed target" if event.confirmed else "candidate"
        elif name in ("CandidateEvent", "ConfirmedEvent"):
            _overlay["label"] = f"ID {event.track_id}"


def _on_ai_candidate(event):
    """The pipeline waits for your c/r decision BEFORE its next on_frame call,
    so the candidate box must be set here, as soon as the candidate appears."""
    with _lock:
        _overlay.update(state="CandidateEvent", time=time.time(),
                        bbox=[int(v) for v in event.bbox], label=f"candidate ID {event.track_id}")


def _draw(frame):
    with _lock:
        st = dict(_overlay)
    text, color = STATUS.get(st["state"], (st["state"], (255, 255, 255)))
    if st["state"] == "TrackUpdateEvent" and st["label"] == "confirmed target":
        text = "TRACKING CONFIRMED TARGET"
    age = time.time() - st["time"]
    keep = st["state"] in ("CandidateEvent", "ConfirmedEvent")   # stay until the next event
    if st["bbox"] is not None and (keep or age < 3.0):              # hide stale tracking boxes
        x1, y1, x2, y2 = st["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, st["label"], (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if st["state"] != "Searching":
        cv2.putText(frame, f"AI result {age:.1f}s ago", (frame.shape[1] - 230, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="person")
    ap.add_argument("--description", default="a person wearing a red top and white trousers")
    ap.add_argument("--save", default=None, help="optional output video of the AI frames")
    args = ap.parse_args()

    search = importlib.import_module(SEARCH_MODULE)
    viewer = RosFrameSource()                      # the LIVE view's own camera feed
    publish_status("searching")                    # announce ourselves early (ROS discovery)
    threading.Thread(target=_keyboard, daemon=True).start()

    def ai_worker():
        search.run_on_video(
            "ros",
            category=args.category,
            description=args.description,
            on_candidate=lambda e: (_on_ai_candidate(e),
                                    publish_status("candidate", track_id=e.track_id),
                                    print(f"[CANDIDATE] ID {e.track_id} bbox {e.bbox.astype(int).tolist()}"
                                          f"\n            why: {e.justification}\n            -> c / r ?")),
            on_track_update=lambda e: None,
            on_lost=lambda e: (publish_status("lost", track_id=e.track_id),
                               print(f"[LOST] ID {e.track_id} (was confirmed: {e.was_confirmed})")),
            on_confirmed=lambda e: (publish_status("confirmed", track_id=e.track_id),
                                    print(f"[CONFIRMED] ID {e.track_id}")),
            on_rejected=lambda e: (publish_status("rejected", track_id=e.track_id),
                                   print(f"[REJECTED] ID {e.track_id}: will not be shown again")),
            should_confirm=lambda: _take("confirm"),
            should_reject=lambda: _take("reject"),
            output_video_path=args.save,
            on_frame=_on_ai_frame,
        )

    ai = threading.Thread(target=ai_worker, daemon=True)
    ai.start()

    print(f"Searching for: {args.description}")
    print("Live view: press c (confirm) / r (reject) / q (quit) in the window.\n")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    # The window must be serviced (waitKey) many times per second, even when no new
    # camera frame has arrived; otherwise Ubuntu reports "not responding".
    last, last_time, warned = None, time.time(), False
    try:
        while ai.is_alive():
            ok, frame = viewer.read(timeout=0.05)      # never block the window long
            if ok:
                last, last_time, warned = frame, time.time(), False
            elif time.time() - last_time > 10 and not warned:
                print("No camera frames for 10 s: is aware_bridge (and the simulation) running?")
                warned = True
            if last is not None:
                cv2.imshow(WINDOW, _draw(last.copy()))
            key = cv2.waitKey(15) & 0xFF
            if key == ord("c"):
                _pending["confirm"] = True
            elif key == ord("r"):
                _pending["reject"] = True
            elif key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopped.")
        viewer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
