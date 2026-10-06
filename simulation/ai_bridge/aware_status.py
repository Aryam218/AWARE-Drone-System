"""
aware_status.py: tell the rest of AWARE what the search is doing.

Publishes on ROS 2 topic /aware/search_state (std_msgs/String, JSON):
    {"state": "candidate" | "confirmed" | "rejected" | "lost" | "searching",
     "track_id": ..., "time": ...}

The patrol script listens to it: the drone HOLDS POSITION while a candidate
waits for the operator and after confirmation, and resumes the patrol when the
candidate is rejected or the target is lost.

Usage (anywhere in the AI code):
    from aware_status import publish_status
    publish_status("confirmed", track_id=7)
"""
import json
import threading
import time

_lock = threading.Lock()
_pub = None


def _publisher():
    global _pub
    with _lock:
        if _pub is None:
            import rclpy
            from std_msgs.msg import String
            if not rclpy.ok():
                if threading.current_thread() is threading.main_thread():
                    rclpy.init()
                else:
                    from rclpy.signals import SignalHandlerOptions
                    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            node = rclpy.create_node("aware_search_status")
            _pub = (node, node.create_publisher(String, "/aware/search_state", 10), String)
        return _pub


def publish_status(state, **info):
    """Never raises: a status message must not crash the search."""
    try:
        _node, pub, String = _publisher()
        msg = dict(state=state, time=round(time.time(), 2), **info)
        pub.publish(String(data=json.dumps(msg, default=str)))
    except Exception as exc:                      # noqa: BLE001
        print(f"[aware_status] could not publish '{state}': {exc}")
