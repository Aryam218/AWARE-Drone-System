#!/usr/bin/env python3
"""
run_live.py: run the AWARE search engine on the LIVE Gazebo camera, from a terminal.
(A stand-in for the dashboard while testing: a live video window shows the
 search/tracking boxes like the YOLO test, events are printed, and YOU play the
 operator: press c (confirm) or r (reject) in the window, or type c/r + Enter.)

Before running:
  - simulation + aware_bridge running (frames on /aware/camera/image)
  - ros_frame_source.py next to this file
  - SEARCH_MODULE below = the file name of your search engine (without .py)

    python3 run_live.py
    python3 run_live.py --description "a person wearing a red top and white trousers"
    python3 run_live.py --save live_run.mp4
    python3 run_live.py --no-window          # terminal only
"""
import argparse
import importlib
import sys
import threading

SEARCH_MODULE = "pipeline_runner"  # your search engine file (no .py)

# ---- operator input: typed in this terminal OR pressed in the video window ----
_pending = {"confirm": False, "reject": False}
_window = {"enabled": True}


def _keyboard():
    for line in sys.stdin:
        key = line.strip().lower()
        if key == "c":
            _pending["confirm"] = True
        elif key == "r":
            _pending["reject"] = True


def _poll_window():
    """Keep the video window alive and read its keys (c / r / q).
    Called often, including while the pipeline waits for your decision."""
    if not _window["enabled"]:
        return
    import cv2
    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        _pending["confirm"] = True
    elif key == ord("r"):
        _pending["reject"] = True
    elif key == ord("q"):
        raise KeyboardInterrupt


def _take(name):
    """True once per key press (like a button that resets after being read)."""
    _poll_window()
    if _pending[name]:
        _pending[name] = False
        return True
    return False


def _show(annotated, event):
    """Show the pipeline's annotated frame with a status bar (like the YOLO test)."""
    if not _window["enabled"]:
        return
    import cv2
    name = type(event).__name__ if event is not None else "Searching"
    status = {
        "Searching": ("SEARCHING", (200, 200, 200)),
        "CandidateEvent": ("POSSIBLE MATCH - press c (confirm) / r (reject)", (0, 165, 255)),
        "TrackUpdateEvent": ("TRACKING" + (" CONFIRMED" if getattr(event, "confirmed", False)
                                           else " candidate"), (0, 255, 0)),
        "ConfirmedEvent": ("CONFIRMED", (0, 255, 0)),
        "RejectedEvent": ("REJECTED - searching again", (0, 0, 255)),
        "LostEvent": ("TARGET LOST - searching again", (0, 0, 255)),
    }.get(name, (name, (255, 255, 255)))
    frame = annotated.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, status[0], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status[1], 2)
    cv2.imshow("AWARE live search", frame)
    _poll_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="person")
    ap.add_argument("--description", default="a person wearing a red top and white trousers")
    ap.add_argument("--save", default=None, help="optional output video, e.g. live_run.mp4")
    ap.add_argument("--no-window", action="store_true", help="terminal only, no video window")
    args = ap.parse_args()
    _window["enabled"] = not args.no_window
    if _window["enabled"]:
        import cv2
        cv2.namedWindow("AWARE live search", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AWARE live search", 960, 720)

    search = importlib.import_module(SEARCH_MODULE)
    threading.Thread(target=_keyboard, daemon=True).start()

    print(f"Searching for: {args.description}")
    print("When a possible match appears: press c / r in the video window "
          "(or type c / r + Enter here). q in the window quits.\n")

    try:
        search.run_on_video(
            "ros",                                   # live camera instead of a video file
            category=args.category,
            description=args.description,
            on_candidate=lambda e: print(f"[CANDIDATE] ID {e.track_id} bbox {e.bbox.astype(int).tolist()}"
                                         f"\n            why: {e.justification}\n            -> c / r ?"),
            on_track_update=lambda e: print(f"[TRACK] ID {e.track_id} "
                                            f"{'CONFIRMED' if e.confirmed else 'candidate'}"),
            on_lost=lambda e: print(f"[LOST] ID {e.track_id} (was confirmed: {e.was_confirmed})"),
            on_confirmed=lambda e: print(f"[CONFIRMED] ID {e.track_id}"),
            on_rejected=lambda e: print(f"[REJECTED] ID {e.track_id}: will not be shown again"),
            should_confirm=lambda: _take("confirm"),
            should_reject=lambda: _take("reject"),
            output_video_path=args.save,
            on_frame=_show,                          # live video window (needs the on_frame change)
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    else:
        print("Search ended: no more frames (is the simulation + aware_bridge still running?)")
    finally:
        if _window["enabled"]:
            import cv2
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
