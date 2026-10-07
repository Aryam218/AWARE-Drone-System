#!/usr/bin/env python3
"""Headless bounded live test. Ground truth is used only by the audit logger.

Run from rasid_video_pipeline with ai_venv active and ROS sourced:
    python3 tools/auto_live_test.py --time-limit 480 --log-dir ~/aware_test_logs/run
The automatic operator rejects candidate one, then confirms candidate two.
The within-3-m label is a position-based scoring proxy, not visual identity proof.
"""
import argparse
from bisect import bisect_left
from collections import deque
import json
import math
import re
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))


class Audit:
    def __init__(self, directory, location):
        self.directory = directory
        self.location = location
        self.lock = threading.RLock()
        self.truth = deque(maxlen=20000)
        self.sim_time = None
        self.records = []
        self.events = (directory / 'events.jsonl').open('w', buffering=1)
        self.truth_file = (directory / 'ground_truth.jsonl').open('w', buffering=1)

    def write(self, kind, timestamp=None, **data):
        with self.lock:
            row = dict(event=kind, sim_time=timestamp, wall_time=time.time(), **data)
            self.records.append(row)
            self.events.write(json.dumps(row, default=str) + '\n')
            print('AUDIT ' + json.dumps(row, default=str), flush=True)
        return row

    def add_truth(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        row = dict(sim_time=stamp, x_east_m=msg.point.x, y_north_m=msg.point.y)
        with self.lock:
            self.truth.append(row)
            self.truth_file.write(json.dumps(row) + '\n')

    def score(self, ground, timestamp):
        if ground is None or timestamp is None:
            return dict(distance_to_missing_m=None, within_3m=None, scoring_unavailable='position or timestamp missing')
        with self.lock:
            samples = list(self.truth)
        if not samples:
            return dict(distance_to_missing_m=None, within_3m=None, scoring_unavailable='no ground truth')
        i = bisect_left([s['sim_time'] for s in samples], timestamp)
        # Interpolate only across nearby samples; never score against stale truth.
        if 0 < i < len(samples) and samples[i]['sim_time'] > samples[i-1]['sim_time']:
            a, b = samples[i-1], samples[i]
            gap = max(timestamp - a['sim_time'], b['sim_time'] - timestamp)
            f = (timestamp - a['sim_time']) / (b['sim_time'] - a['sim_time'])
            x = a['x_east_m'] + f * (b['x_east_m'] - a['x_east_m'])
            y = a['y_north_m'] + f * (b['y_north_m'] - a['y_north_m'])
        else:
            sample = samples[min(i, len(samples)-1)]
            gap = abs(sample['sim_time'] - timestamp)
            x, y = sample['x_east_m'], sample['y_north_m']
        if gap > 2.0:
            return dict(distance_to_missing_m=None, within_3m=None, scoring_unavailable='truth more than 2 sim seconds away', truth_gap_s=gap)
        lat0, lon0 = self.location['latitude'], self.location['longitude']
        north = math.radians(ground[0] - lat0) * 6378137.0
        east = math.radians(ground[1] - lon0) * 6378137.0 * math.cos(math.radians(lat0))
        d = math.hypot(east - x, north - y)
        return dict(distance_to_missing_m=round(d, 3), within_3m=d <= 3.0,
                    estimated_xy_m=[east, north], true_xy_m=[x, y], truth_gap_s=gap)

    def scored(self, kind, timestamp, ground, **data):
        return self.write(kind, timestamp, ground=ground, **self.score(ground, timestamp), **data)

    def finish(self):
        # Re-score after the run so samples received during slow inference are included.
        with self.lock:
            rows = []
            for row in self.records:
                if 'ground' in row:
                    row = dict(row, **self.score(row['ground'], row.get('position_sim_time', row['sim_time'])))
                rows.append(row)
            (self.directory / 'scored_events.jsonl').write_text(''.join(json.dumps(r, default=str) + '\n' for r in rows))
            self.events.close()
            self.truth_file.close()


def instrument_gpt_http(audit):
    """Observe HTTP attempts and API-reported usage without recording request content."""
    from types import SimpleNamespace
    from cloud_track.foundation_model_wrappers import gpt_four_wrapper as gpt
    original_requests = gpt.requests
    counter = 0
    counter_lock = threading.Lock()

    def post(*args, **kwargs):
        nonlocal counter
        with counter_lock:
            counter += 1
            call_id = counter
        audit.write('GPTHTTPRequest', audit.sim_time, call_id=call_id,
                    model=kwargs.get('json', {}).get('model'))
        try:
            response = original_requests.post(*args, **kwargs)
        except Exception as exc:
            audit.write('GPTHTTPResult', audit.sim_time, call_id=call_id,
                        error_type=type(exc).__name__, usage=None)
            raise
        try:
            data = response.json()
            usage = data.get('usage')
            choices = data.get('choices') or []
            answer = choices[0].get('message', {}).get('content', '') if choices else ''
            decision_match = re.search(r'Decision:\s*(MATCH|NO MATCH|NO_MATCH|UNCERTAIN)', answer, re.IGNORECASE)
            decision = decision_match.group(1).upper() if decision_match else None
            justification_match = re.search(r'Justification:\s*(.*)', answer, re.DOTALL)
            justification = justification_match.group(1).strip() if justification_match else None
            error = data.get('error') or {}
            error_code = error.get('code') if isinstance(error, dict) else None
        except (ValueError, AttributeError):
            usage, error_code, decision, justification = None, None, None, None
        audit.write('GPTHTTPResult', audit.sim_time, call_id=call_id,
                    http_status=response.status_code, usage=usage, error_code=error_code,
                    decision=decision, justification=justification)
        return response

    # Isolate this instrumentation to the GPT wrapper; other requests users are untouched.
    gpt.requests = SimpleNamespace(post=post, exceptions=original_requests.exceptions)


class Operator:
    def __init__(self, audit, delay):
        self.audit, self.delay = audit, delay
        self.lock = threading.Lock()
        self.count = 0
        self.active = None
        self.action_sent = False
        self.confirmed = False

    def candidate(self, event):
        with self.lock:
            self.count += 1
            self.active = (self.count, event.track_id, time.monotonic())
            self.action_sent = False

    def take(self, action):
        with self.lock:
            if self.active is None or self.action_sent or self.confirmed:
                return False
            number, track_id, started = self.active
            desired = 'reject' if number == 1 else 'confirm'
            elapsed = time.monotonic() - started
            if desired != action or elapsed < self.delay:
                return False
            self.action_sent = True
        self.audit.write('OperatorAction', self.audit.sim_time, action=action,
                         candidate_number=number, track_id=track_id, delay_wall_s=elapsed)
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--time-limit', type=float, default=480.0, help='wall seconds, including model loading')
    ap.add_argument('--decision-delay', type=float, default=3.0)
    ap.add_argument('--description', default='a person wearing a red top and white trousers')
    ap.add_argument('--category', default='person')
    ap.add_argument('--log-dir', type=Path, default=Path.home() / 'aware_test_logs' / time.strftime('%Y%m%d_%H%M%S'))
    args = ap.parse_args()
    if args.time_limit <= 0 or args.decision_delay < 0:
        ap.error('time-limit must be positive and decision-delay nonnegative')
    args.log_dir.mkdir(parents=True, exist_ok=True)
    import yaml
    import rclpy
    from geometry_msgs.msg import PointStamped
    from rosgraph_msgs.msg import Clock
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from aware_status import publish_status
    import cv2
    import pipeline_runner as runner

    venue = yaml.safe_load((AI_DIR.parent / 'simulation/aware_sim/config/zones.yaml').read_text())
    audit = Audit(args.log_dir, venue['venue']['location'])
    instrument_gpt_http(audit)
    operator = Operator(audit, args.decision_delay)
    stop = threading.Event()
    failed = threading.Event()
    streams = []
    holder = {}
    deadline = time.monotonic() + args.time_limit
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    node = rclpy.create_node('aware_auto_test_scoring')
    node.create_subscription(PointStamped, '/aware/ground_truth/missing_person', audit.add_truth, 10)
    def clock(msg):
        audit.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
    node.create_subscription(Clock, '/clock', clock, 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    def spin():
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()

    original_controller = runner.SearchController
    class ObservedController(original_controller):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            holder['controller'] = self
            places = self.pipeline.backend.rejected_places
            original_add = places.add
            original_rejected = places.is_rejected
            def add(ground, signature=None, now=None):
                original_add(ground, signature, now)
                audit.scored('RejectedPlace', getattr(self, '_audit_frame_time', None), ground,
                             signature_present=signature is not None, remembered_places=len(places))
            def is_rejected(ground, signature=None, now=None):
                answer = original_rejected(ground, signature, now)
                if answer:
                    audit.scored('RejectedPlaceSkip', getattr(self, '_audit_frame_time', None), ground)
                return answer
            places.add = add
            places.is_rejected = is_rejected
        def process_frame(self, *a, **kw):
            self._audit_frame_time = kw.get('timestamp')
            return super().process_frame(*a, **kw)
    runner.SearchController = ObservedController
    original_stream = runner.open_stream
    def open_stream(*a, **kw):
        stream = original_stream(*a, **kw)
        streams.append(stream)
        class BoundedStream:
            def __getattr__(self, name):
                return getattr(stream, name)
            def __iter__(self):
                for frame in stream:
                    if stop.is_set() or time.monotonic() >= deadline:
                        break
                    yield frame
        return BoundedStream()
    runner.open_stream = open_stream

    def event_log(event):
        name = type(event).__name__
        pipeline = holder['controller'].pipeline
        target = pipeline.verified_target()
        ground = target['ground'] if target else None
        position_time = pipeline._anchor_time
        data = dict(track_id=event.track_id, frame_index=event.frame_index,
                    confirmed=getattr(event, 'confirmed', None),
                    was_confirmed=getattr(event, 'was_confirmed', None),
                    justification=getattr(event, 'justification', None),
                    candidate_number=operator.count,
                    position_sim_time=position_time)
        box = getattr(event, 'bbox', None)
        if box is not None:
            data['bbox'] = [float(v) for v in box]
            data['box_ground'] = pipeline.backend.ground_position(box)
            data['box_score'] = audit.score(data['box_ground'], event.timestamp)
        if name in ('CandidateEvent', 'ConfirmedEvent'):
            filename = f'{name}_{event.frame_index}_{event.track_id}.jpg'
            cv2.imwrite(str(args.log_dir / filename), event.frame)
            data['image'] = filename
        # Score the same VERIFIED position sent to the patrol, at its sighting time.
        audit.write(name, event.timestamp, ground=ground,
                    **audit.score(ground, position_time), **data)
    def candidate(event):
        operator.candidate(event)
        event_log(event)
        publish_status('candidate', track_id=event.track_id)
    def confirmed(event):
        operator.confirmed = True
        event_log(event)
        publish_status('confirmed', track_id=event.track_id)
    def rejected(event):
        event_log(event)
        publish_status('rejected', track_id=event.track_id)
        operator.active = None
    def lost(event):
        event_log(event)
        publish_status('lost', track_id=event.track_id)
        operator.active = None
    def on_frame(frame, event):
        if event is None:
            controller = holder['controller']
            audit.write('Searching', getattr(controller, '_audit_frame_time', None))
    def work():
        try:
            runner.run_on_video('ros', category=args.category, description=args.description,
                on_candidate=candidate, on_track_update=event_log, on_lost=lost,
                on_confirmed=confirmed, on_rejected=rejected, on_frame=on_frame,
                should_reject=lambda: operator.take('reject'),
                should_confirm=lambda: operator.take('confirm'))
        except BaseException as exc:
            failed.set()
            audit.write('Error', audit.sim_time, message=str(exc), traceback=traceback.format_exc())
        finally:
            stop.set()
    publish_status('searching')
    audit.write('TestStart', audit.sim_time, time_limit_wall_s=args.time_limit,
                description=args.description, truth_used_only_for_scoring=True)
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while not stop.wait(0.2):
        if time.monotonic() >= deadline:
            audit.write('TimeLimit', audit.sim_time)
            stop.set()
    # Request stream closure without disturbing an in-progress model/API call.
    for stream in streams:
        stream._open = False
        with stream._lock:
            stream._lock.notify_all()
    worker.join(timeout=3.0)
    spin_thread.join(timeout=1.0)
    audit.write('TestEnd', audit.sim_time, candidates=operator.count,
                confirmed=operator.confirmed, worker_still_running=worker.is_alive(), failed=failed.is_set())
    audit.finish()
    if worker.is_alive():
        # Hard deadline: terminate daemon inference and any pending GPT requests.
        import os
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if failed.is_set() else 0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 1 if failed.is_set() else 0


if __name__ == '__main__':
    raise SystemExit(main())
