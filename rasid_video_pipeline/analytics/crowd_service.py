"""Backend-owned live crowd worker with search-priority inference scheduling."""
import threading
import time
import cv2
from PIL import Image
from analytics.moving_crowd import MovingCrowdCounter


def protocol_analytics(result):
    zones = {}
    for zone in result['zones']:
        fraction = zone['coverage_fraction']
        seen = fraction is not None and fraction > 0
        full = fraction is not None and fraction >= 0.8
        occupancy = zone['occupancy_percent'] if full else None
        zones[zone['id']] = dict(id=zone['id'], type=zone['type'], display_name=zone['display_name'],
            booth=zone['booth'], count=zone['visible_count'] if seen else None,
            capacity=zone['capacity'], coverage_fraction=fraction,
            coverage='full' if full else 'partial' if seen else 'unseen',
            occupancy_percent=occupancy, crowded=occupancy >= 80 if occupancy is not None else None,
            last_seen_sim_time=zone['last_seen'])
    return dict(type='crowd_analytics', sim_time=result['timestamp'],
                total_people_in_view=result['total_people'], zones=zones)


class CrowdService:
    def __init__(self, detector, publish, report_error=None, idle_interval_s=10, extra_interval_s=15):
        self.counter = MovingCrowdCounter(detector=detector)
        self.detector = self.counter.detector
        self.publish = publish
        self.report_error = report_error or (lambda exc: print('[AWARE CROWD] '+str(exc), flush=True))
        self.idle_interval_s = idle_interval_s
        self.extra_interval_s = extra_interval_s
        self.search_active = threading.Event()
        self.stopping = threading.Event()
        # Handoff ensures the idle writer finishes before the search writer changes last_seen.
        self._counter_lock = threading.RLock()
        self._last_extra_wall = float('-inf')
        self._last_idle_wall = float('-inf')
        self._thread = None
        self._source = self._pose = None
        self.updates = 0

    def start(self):
        from ros_frame_source import RosFrameSource
        from drone_state import DroneStateReader
        self._source = RosFrameSource(timeout_s=1)
        self._pose = DroneStateReader()
        self._thread = threading.Thread(target=self._idle_loop, name='aware_crowd', daemon=True)
        self._thread.start()

    def set_search_active(self, active):
        if active:
            self.search_active.set()
        else:
            self.search_active.clear()

    def _emit(self, frame, boxes, scores, pose, timestamp, source, started):
        with self._counter_lock:
            height, width = frame.shape[:2]
            result = self.counter.count_detections(boxes, scores, pose, width, height, timestamp)
            annotated = frame.copy()
            for person in result['people']:
                x1,y1,x2,y2 = map(int, person['box'])
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            if width > 640:
                annotated = cv2.resize(annotated, (640, round(height*640/width)), interpolation=cv2.INTER_AREA)
            self.publish(protocol_analytics(result), annotated)
            self.updates += 1
            print(f'[AWARE CROWD] source={source} sim_time={timestamp} people={result["total_people"]} processing_s={time.perf_counter()-started:.3f} inference_calls={self.detector.inference_calls}', flush=True)

    def process_idle_frame(self, frame, pose, timestamp, wall_now=None):
        now = time.monotonic() if wall_now is None else wall_now
        if self.search_active.is_set() or self.stopping.is_set() or now-self._last_idle_wall < self.idle_interval_s:
            return False
        started = time.perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raw = self.detector.try_background_inference(image, lambda: not self.search_active.is_set() and not self.stopping.is_set())
        if raw is None:
            return False
        self._last_idle_wall = now
        # Search may have started during a non-preemptible CPU pass. Give it the
        # detector immediately; a stale idle frame must not overwrite its results.
        if self.search_active.is_set() or self.stopping.is_set():
            return False
        self._emit(frame, raw[2], raw[3], pose, timestamp, 'idle', started)
        return True

    def on_search_frame(self, frame, pose, timestamp, controller):
        if self.stopping.is_set():
            return
        started = time.perf_counter()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        raw = self.detector.cached_detections(image)
        if raw is not None:
            self._emit(frame, raw[0], raw[1], pose, timestamp, 'search_reuse', started)
            return
        now = time.monotonic()
        if now-self._last_extra_wall < self.extra_interval_s:
            return
        pipeline = controller.pipeline
        if not pipeline.tracker_initialized:
            return  # The next search frame needs DINO; never delay it with crowd-only work.
        if pipeline.tracker_initialized:
            # Forced/due re-detection always wins. Reserve enough time for a
            # full measured detector pass before the next scheduled check.
            if timestamp is None or pipeline._force_redetect or pipeline._last_redetect_time is None:
                return
            remaining = pipeline.redetect_interval_s - (timestamp-pipeline._last_redetect_time)
            if remaining <= self.detector.inference_seconds + 0.5:
                return
        # Search is the sole inference owner while active; this is called only
        # after it processed the frame and acknowledged operator decisions.
        self._last_extra_wall = now
        _, _, boxes, scores = self.detector.run_inference(image, prompt='person', mark_results=False, print_results=False)
        self._emit(frame, boxes, scores, pose, timestamp, 'search_extra', started)

    def _idle_loop(self):
        while not self.stopping.is_set():
            if self.search_active.is_set():
                self.stopping.wait(.2)
                continue
            ok, frame = self._source.read(timeout=.5)
            if not ok:
                continue
            pose = self._pose.frame_pose()
            timestamp = self._source.last_stamp
            try:
                self.process_idle_frame(frame, pose, timestamp)
            except Exception as exc:
                self.report_error(exc)
                self.stopping.wait(2)

    def close(self):
        self.stopping.set()
        if self._thread is not None:
            self._thread.join(30)
        if self._source is not None:
            self._source.release()
        if self._pose is not None:
            self._pose.release()
