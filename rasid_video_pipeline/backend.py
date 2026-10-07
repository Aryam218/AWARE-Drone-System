"""Missing-person and live crowd dashboard with one shared detector."""
from __future__ import annotations
import asyncio
import base64
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import uuid
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from aware_status import publish_status
from pipeline_runner import draw_bbox, run_on_video

app = FastAPI(title="AWARE — Missing Person Search Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

class StartRequest(BaseModel):
    video_path: str
    description: str
    category: str = "person"

class ConfirmRequest(BaseModel):
    decision: str
    candidate_id: str | None = None

class Hub:
    def __init__(self):
        self.clients = []
        self.loop = None
    async def connect(self, ws):
        await ws.accept()
        self.clients.append(ws)
    def disconnect(self, ws):
        if ws in self.clients:
            self.clients.remove(ws)
    def broadcast_threadsafe(self, payload):
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)
    async def _broadcast(self, payload):
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self.disconnect(ws)

HUB = Hub()
_lock = threading.RLock()
_reject_requested = threading.Event()
_confirm_requested = threading.Event()
_awaiting_decision = threading.Event()
_run_thread = None
_crowd = None
_crowd_latest = None
_snapshot_latest = None
_shutdown = threading.Event()
_candidate = None
_location = None
_quota = None
_pending_decision = False
_status = dict(type="search_status", sim_time=None, state="idle", message="Ready to search.", request_id=None)

def status(state, message, timestamp=None, request_id=None, **extra):
    global _status
    with _lock:
        _status = dict(type="search_status", sim_time=timestamp, state=state, message=message, request_id=request_id, gpt_verification_off=_quota is not None, **extra)
        HUB.broadcast_threadsafe(_status)

def error(message, code="command_rejected", request_id=None, recoverable=True, timestamp=None):
    return dict(type="error", sim_time=timestamp, code=code, message=message, request_id=request_id, recoverable=recoverable)

def _encode_full(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        raise ValueError("Could not encode camera image.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

def _encode_crop(frame, bbox):
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = map(int, bbox)
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0:
        raise ValueError("Candidate crop is outside the camera image.")
    return _encode_full(crop)

def _worker(video_path, category, description, output_path):
    position = {}
    last_preview = 0.0
    def on_position(value):
        position.clear()
        position.update(value)
    def stamp(event):
        return event.timestamp if video_path == "ros" else None
    def identity(event):
        return dict(candidate_id=_candidate["candidate_id"] if _candidate else None,
                    track_id=event.track_id, frame_index=event.frame_index, sim_time=stamp(event))
    def location(event):
        global _location
        _location = dict(type="person_location", **identity(event), confirmed=True,
                         location=position.get("location"), drone=position.get("drone"),
                         position_sim_time=position.get("position_sim_time"))
        HUB.broadcast_threadsafe(_location)
    def preview(event):
        nonlocal last_preview
        last_preview = time.monotonic()
        HUB.broadcast_threadsafe(dict(type="tracking_update", **identity(event),
            confirmed=bool(getattr(event,"confirmed",False)),
            image=_encode_full(draw_bbox(event.frame, event.bbox, label="tracked person"))))
    def on_candidate(event):
        global _candidate, _location
        with _lock:
            if _awaiting_decision.is_set():
                return
            _location = None
            _candidate = dict(type="candidate", candidate_id=uuid.uuid4().hex,
                track_id=event.track_id, frame_index=event.frame_index, sim_time=stamp(event),
                image=_encode_crop(event.frame,event.bbox),
                frame=_encode_full(draw_bbox(event.frame,event.bbox,label="possible match")),
                justification=event.justification)
            _awaiting_decision.set()
            publish_status("candidate", track_id=event.track_id)
            HUB.broadcast_threadsafe(_candidate)
            status("awaiting_decision", "Review this candidate and Confirm or Reject.",stamp(event))
            preview(event)
    def on_track_update(event):
        with _lock:
            if _candidate is None or time.monotonic()-last_preview < 1.0:
                return
            preview(event)
            if event.confirmed:
                location(event)
    def on_confirmed(event):
        global _pending_decision
        with _lock:
            _awaiting_decision.clear()
            _confirm_requested.clear()
            _pending_decision = False
            publish_status("confirmed",track_id=event.track_id)
            status("confirmed","Person confirmed. Tracking continues.",stamp(event))
            location(event)
            # ConfirmedEvent has no confirmed attribute; its acknowledgement precedes this preview.
            payload = dict(type="tracking_update", **identity(event), confirmed=True,
                image=_encode_full(draw_bbox(event.frame,event.bbox,label="confirmed target")))
            HUB.broadcast_threadsafe(payload)
    def clear(event, kind):
        global _candidate, _location, _pending_decision
        with _lock:
            payload = dict(type=kind, **identity(event), status="searching")
            if kind == "lost":
                payload["was_confirmed"] = event.was_confirmed
            publish_status(kind,track_id=event.track_id)
            _awaiting_decision.clear()
            _reject_requested.clear()
            _confirm_requested.clear()
            _pending_decision = False
            _candidate = _location = None
            HUB.broadcast_threadsafe(payload)
            status("searching","Searching for another matching person.",stamp(event))
    def on_verification_error(event):
        global _quota
        with _lock:
            if _quota is None:
                _quota = dict(type="gpt_verification_off",sim_time=stamp(event),code=event.code,message=event.message)
                HUB.broadcast_threadsafe(_quota)
    def crowd_frame(frame, pose, timestamp, controller):
        try:
            _crowd.on_search_frame(frame, pose, timestamp, controller)
        except Exception as exc:
            print(f"[AWARE] crowd update failed: {exc}", flush=True)
            HUB.broadcast_threadsafe(error(str(exc), "crowd_failed", timestamp=timestamp))
    try:
        run_on_video(video_path=video_path,category=category,description=description,
            on_candidate=on_candidate,on_track_update=on_track_update,on_lost=lambda e:clear(e,"lost"),
            output_video_path=output_path,should_reject=_reject_requested.is_set,
            should_confirm=_confirm_requested.is_set,on_confirmed=on_confirmed,
            on_rejected=lambda e:clear(e,"rejected"),on_verification_error=on_verification_error,
            on_target_position=on_position,
            detector=_crowd.detector if _crowd is not None else None,
            on_processed_frame=crowd_frame if _crowd is not None and video_path == "ros" else None,
            should_stop=_shutdown.is_set)
        status("finished","Search source finished.",output_video_path=output_path)
    except Exception as exc:
        print(f"[AWARE] dashboard search failed: {exc}",flush=True)
        HUB.broadcast_threadsafe(error(str(exc),"search_failed",recoverable=False))
        status("finished","Search failed.")
    finally:
        global _candidate, _location, _pending_decision
        if _crowd is not None:
            _crowd.set_search_active(False)
        with _lock:
            _reject_requested.clear()
            _confirm_requested.clear()
            _awaiting_decision.clear()
            _pending_decision = False
            _candidate = _location = None

def start_search(req, request_id=None):
    global _run_thread, _candidate, _location, _quota, _pending_decision
    with _lock:
        if _run_thread is not None and _run_thread.is_alive():
            return {"error":"A search is already running. Wait for it to finish."}
        video_path, description = req.video_path.strip(), req.description.strip()
        if not video_path or not description:
            return {"error":"video_path and a missing-person description are required."}
        _candidate = _location = _quota = None
        _pending_decision = False
        _reject_requested.clear(); _confirm_requested.clear(); _awaiting_decision.clear()
        stem = "live_"+datetime.now().strftime("%Y%m%d_%H%M%S") if video_path == "ros" else Path(video_path).stem
        output_path = str(OUTPUT_DIR / f"annotated_{stem}.mp4")
        _run_thread = threading.Thread(target=_worker,args=(video_path,req.category.strip() or "person",description,output_path),daemon=True)
        publish_status("searching")
        status("searching","Searching the drone camera for matching people.",request_id=request_id)
        if _crowd is not None:
            _crowd.set_search_active(True)
        _run_thread.start()
        return dict(ok=True,status="searching",output_video_path=output_path)

def decide(req, request_id=None):
    global _pending_decision
    with _lock:
        decision = req.decision.strip().lower()
        if decision not in {"confirm","reject"}:
            return {"error":"decision must be either 'confirm' or 'reject'."}
        if not _awaiting_decision.is_set() or _candidate is None:
            return {"error":"There is no candidate awaiting confirmation."}
        if req.candidate_id is not None and req.candidate_id != _candidate["candidate_id"]:
            return {"error":"This candidate is no longer awaiting a decision."}
        if _pending_decision:
            return {"error":"A decision is already pending."}
        _pending_decision = True
        # Enqueue acknowledgement before exposing the decision to the worker.
        state = "confirming" if decision == "confirm" else "resuming_search"
        status(state,"Waiting for the pipeline to apply your decision.",request_id=request_id)
        (_confirm_requested if decision == "confirm" else _reject_requested).set()
        return dict(ok=True,decision=decision,status="confirming_target" if decision == "confirm" else "resuming_search")

def publish_crowd(analytics, image):
    global _crowd_latest, _snapshot_latest
    snapshot = dict(type="crowd_snapshot", sim_time=analytics["sim_time"], image=_encode_full(image))
    with _lock:
        _crowd_latest, _snapshot_latest = analytics, snapshot
        HUB.broadcast_threadsafe(analytics)
        HUB.broadcast_threadsafe(snapshot)


def create_crowd_service():
    from analytics.crowd_service import CrowdService
    from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import get_detector
    # Exactly one weight load for this server process, before any search starts.
    detector = get_detector("sam_lq")
    return CrowdService(detector, publish_crowd,
        report_error=lambda exc: HUB.broadcast_threadsafe(error(str(exc), "crowd_failed")))


@app.on_event("startup")
async def _capture_loop():
    global _crowd
    HUB.loop = asyncio.get_running_loop()
    _shutdown.clear()
    _crowd = await asyncio.to_thread(create_crowd_service)
    if _crowd is not None:
        await asyncio.to_thread(_crowd.start)


@app.on_event("shutdown")
async def _stop_workers():
    global _crowd
    _shutdown.set()
    if _crowd is not None:
        _crowd.stopping.set()
    if _run_thread is not None and _run_thread.is_alive():
        await asyncio.to_thread(_run_thread.join, 30)
    if _crowd is not None:
        await asyncio.to_thread(_crowd.close)
        _crowd = None

@app.post("/start")
async def start(req: StartRequest):
    return start_search(req)

@app.post("/confirm")
async def confirm(req: ConfirmRequest):
    return decide(req)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await HUB.connect(ws)
    try:
        with _lock:
            replay = [_status, _candidate if _awaiting_decision.is_set() else None, _location, _quota, _crowd_latest, _snapshot_latest]
        for payload in replay:
            if payload is not None:
                await ws.send_json(payload)
        while True:
            request_id = None
            try:
                command = json.loads(await ws.receive_text())
                if not isinstance(command,dict):
                    raise ValueError("Command must be a JSON object.")
                request_id = command.get("request_id")
                if request_id is not None and not isinstance(request_id,str):
                    raise ValueError("request_id must be a string.")
                kind = command.get("type")
                if kind == "start_person_search":
                    result = start_search(StartRequest(video_path=command.get("video_path","ros"),description=command.get("description"),category=command.get("category","person")),request_id)
                elif kind in {"confirm_candidate","reject_candidate"}:
                    result = decide(ConfirmRequest(decision="confirm" if kind=="confirm_candidate" else "reject",candidate_id=command.get("candidate_id")),request_id)
                else:
                    await ws.send_json(error("Unknown WebSocket command.","invalid_command",request_id))
                    continue
                if "error" in result:
                    await ws.send_json(error(result["error"],request_id=request_id))
            except (ValueError,ValidationError) as exc:
                await ws.send_json(error(str(exc),"invalid_request",request_id if isinstance(request_id,str) else None))
    except WebSocketDisconnect:
        pass
    finally:
        HUB.disconnect(ws)
