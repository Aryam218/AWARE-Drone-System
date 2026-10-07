"""One inference owner and bounded, exact-image cache for search/crowd consumers."""
import copy
import hashlib
import threading
import time
from collections import OrderedDict
import numpy as np

class SharedDetector:
    def __init__(self, detector):
        self.detector = detector
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self.inference_calls = 0
        self.cache_hits = 0
        self.inference_seconds = 10.0

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def _key(self, image, prompt):
        return (image.mode, image.size, hashlib.sha256(image.tobytes()).digest(),
                prompt.strip().rstrip('.'), getattr(self.detector, 'box_threshold', None),
                getattr(self.detector, 'text_threshold', None))

    def cached_detections(self, image, prompt='person'):
        """Read raw detections without ever starting or waiting for inference."""
        key = self._key(image, prompt)
        if not self._lock.acquire(blocking=False):
            return None
        try:
            item = self._cache.get(key)
            return None if item is None else (item[1].copy(), item[2].copy())
        finally:
            self._lock.release()

    def try_background_inference(self, image, can_run, prompt='person'):
        """Low-priority work never queues ahead of search and rechecks under lock."""
        if not can_run() or not self._lock.acquire(blocking=False):
            return None
        try:
            if not can_run():
                return None
            return self.run_inference(image, prompt=prompt, mark_results=False, print_results=False)
        finally:
            self._lock.release()

    def run_inference(self, image, prompt, mark_results=False, **kwargs):
        # Marked images/masks are not shared; consumers request raw person boxes.
        with self._lock:
            if mark_results:
                self.inference_calls += 1
                return self.detector.run_inference(image, prompt=prompt, mark_results=True, **kwargs)
            key = self._key(image, prompt)
            if key in self._cache:
                self.cache_hits += 1
                masks, boxes, scores = self._cache[key]
                self._cache.move_to_end(key)
            else:
                started = time.perf_counter()
                _, masks, boxes, scores = self.detector.run_inference(
                    image, prompt=prompt, mark_results=False, **kwargs)
                elapsed = time.perf_counter() - started
                self.inference_seconds = max(elapsed, self.inference_seconds * 0.9)
                self.inference_calls += 1
                boxes = np.asarray(boxes if boxes is not None else [], dtype=np.float32).reshape(-1, 4)
                scores = np.asarray(scores if scores is not None else np.ones(len(boxes)), dtype=np.float32)
                self._cache[key] = (copy.deepcopy(masks), boxes.copy(), scores.copy())
                while len(self._cache) > 2:
                    self._cache.popitem(last=False)
            # Search modifies boxes when expanding candidate crops. Never share mutable arrays.
            return image, copy.deepcopy(masks), boxes.copy(), scores.copy()

def shared_detector(detector):
    if isinstance(detector, SharedDetector):
        return detector
    # Constructor calls are made by the owning pipeline before worker threads start.
    existing = vars(detector).get('_aware_shared_detector')
    if existing is None:
        existing = SharedDetector(detector)
        detector._aware_shared_detector = existing
    return existing
