"""
AWARE Dashboard Backend

Runs the blocking CloudTrack / VLM video-search pipeline in a background
thread and broadcasts search events to the dashboard through WebSocket.

Current search flow:

    SEARCHING
        ↓
    Grounding DINO detects people
        ↓
    VLM verifies description
        ↓
    MATCH
        ↓
    POSSIBLE CANDIDATE + TRACKING
        ↓
       User
      /    \
 CONFIRM   REJECT
    ↓         ↓
CONFIRMED   reset tracker
    ↓         ↓
continue    SEARCHING
tracking

Later, the video-file source can be replaced with a Gazebo / ROS2 camera
stream without changing the dashboard protocol or SearchController logic.

Crowd analytics is intentionally kept separate from this search backend.
"""

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import List

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline_runner import (
    CandidateEvent,
    ConfirmedEvent,
    LostEvent,
    RejectedEvent,
    TrackUpdateEvent,
    draw_bbox,
    run_on_video,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AWARE — Missing Person Search Dashboard"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# REQUEST MODELS
# ============================================================


class StartRequest(BaseModel):
    video_path: str
    description: str
    category: str = "person"


class ConfirmRequest(BaseModel):
    # Expected:
    #     "confirm"
    #     "reject"
    decision: str


# ============================================================
# WEBSOCKET HUB
# ============================================================


class Hub:
    """
    Small WebSocket broadcaster.

    The search pipeline runs in a worker thread, while FastAPI owns
    the asyncio event loop.

    broadcast_threadsafe() safely hands messages from the worker
    thread back to FastAPI's event loop.
    """

    def __init__(self) -> None:
        self.clients: List[WebSocket] = []
        self.loop = None

    async def connect(
        self,
        ws: WebSocket,
    ) -> None:

        await ws.accept()

        self.clients.append(ws)

    def disconnect(
        self,
        ws: WebSocket,
    ) -> None:

        if ws in self.clients:
            self.clients.remove(ws)

    def broadcast_threadsafe(
        self,
        payload: dict,
    ) -> None:

        if self.loop is None:
            return

        import asyncio

        asyncio.run_coroutine_threadsafe(
            self._broadcast(payload),
            self.loop,
        )

    async def _broadcast(
        self,
        payload: dict,
    ) -> None:

        dead = []

        for ws in self.clients:

            try:
                await ws.send_text(
                    json.dumps(payload)
                )

            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)


HUB = Hub()


# ============================================================
# SHARED SEARCH STATE
# ============================================================

# User clicked:
#     Reject — Keep Searching
_reject_requested = threading.Event()

# User clicked:
#     Confirm Person
_confirm_requested = threading.Event()

# True while a possible candidate is waiting for the user's decision.
_awaiting_decision = threading.Event()

# Only one search worker is allowed at a time.
_run_thread: threading.Thread | None = None


# ============================================================
# IMAGE ENCODING
# ============================================================


def _encode_crop(
    frame,
    bbox,
) -> str:
    """
    Crop the candidate from the full frame and encode it as JPEG/base64
    for the dashboard candidate card.
    """

    x1, y1, x2, y2 = (
        int(v) for v in bbox
    )

    height, width = frame.shape[:2]

    # Clamp coordinates to image boundaries.
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))

    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return ""

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return ""

    ok, buf = cv2.imencode(
        ".jpg",
        crop,
    )

    if not ok:
        return ""

    return base64.b64encode(
        buf.tobytes()
    ).decode("ascii")


def _encode_full(
    frame,
) -> str:
    """
    Encode a full preview frame for live tracking display.
    """

    ok, buf = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            60,
        ],
    )

    if not ok:
        return ""

    return base64.b64encode(
        buf.tobytes()
    ).decode("ascii")


# ============================================================
# SEARCH WORKER
# ============================================================


def _worker(
    video_path: str,
    category: str,
    description: str,
    output_path: str,
) -> None:
    """
    Background search worker.

    run_on_video() blocks while inference is running, therefore this
    function executes inside a background thread.
    """

    # --------------------------------------------------------
    # POSSIBLE CANDIDATE
    # --------------------------------------------------------

    def on_candidate(
        event: CandidateEvent,
    ) -> None:

        # If a candidate is already waiting for a decision,
        # do not replace the dashboard candidate card.
        if _awaiting_decision.is_set():
            return

        _awaiting_decision.set()

        HUB.broadcast_threadsafe(
            {
                "type": "candidate_found",
                "crop_jpeg_b64": _encode_crop(
                    event.frame,
                    event.bbox,
                ),
                "justification": event.justification,
                "frame_index": event.frame_index,
            }
        )

    # --------------------------------------------------------
    # TRACKING UPDATE
    # --------------------------------------------------------

    def on_track_update(
        event: TrackUpdateEvent,
    ) -> None:

        # Do not flood the WebSocket.
        # Send one preview every 10 frames.
        if event.frame_index % 10 != 0:
            return

        if event.confirmed:
            label = "confirmed target"
        else:
            label = "candidate tracking"

        preview = draw_bbox(
            event.frame,
            event.bbox,
            label=label,
        )

        HUB.broadcast_threadsafe(
            {
                "type": "tracking_update",
                "frame_jpeg_b64": _encode_full(
                    preview
                ),
                "score": event.score,
                "frame_index": event.frame_index,
                "confirmed": event.confirmed,
            }
        )

    # --------------------------------------------------------
    # USER CONFIRMED CANDIDATE
    # --------------------------------------------------------

    def on_confirmed(
        event: ConfirmedEvent,
    ) -> None:

        # Candidate no longer needs a yes/no decision.
        _awaiting_decision.clear()
        _confirm_requested.clear()
        
        preview = draw_bbox(
            event.frame,
            event.bbox,
            label="confirmed target",
        )

        HUB.broadcast_threadsafe(
            {
                "type": "target_confirmed",
                "frame_jpeg_b64": _encode_full(
                    preview
                ),
                "crop_jpeg_b64": _encode_crop(
                    event.frame,
                    event.bbox,
                ),
                "frame_index": event.frame_index,

                # Location is intentionally NOT invented here.
                #
                # Later the Gazebo / ROS2 layer will provide the
                # actual drone / target position.
                "location": None,
            }
        )

    # --------------------------------------------------------
    # USER REJECTED CANDIDATE
    # --------------------------------------------------------

    def on_rejected(
        event: RejectedEvent,
    ) -> None:

        _awaiting_decision.clear()

        # The request has now been consumed.
        _reject_requested.clear()

        HUB.broadcast_threadsafe(
            {
                "type": "candidate_rejected",
                "frame_index": event.frame_index,
                "status": "searching",
            }
        )

    # --------------------------------------------------------
    # TRACK LOST
    # --------------------------------------------------------

    def on_lost(
        event: LostEvent,
    ) -> None:

        # A lost candidate must not leave the UI stuck in
        # "waiting for confirmation".
        _awaiting_decision.clear()

        _confirm_requested.clear()
        _reject_requested.clear()

        HUB.broadcast_threadsafe(
            {
                "type": "track_lost",
                "frame_index": event.frame_index,
                "was_confirmed": event.was_confirmed,
                "status": "searching",
            }
        )

    # --------------------------------------------------------
    # RUN SEARCH
    # --------------------------------------------------------

    try:

        run_on_video(
            video_path=video_path,
            category=category,
            description=description,

            on_candidate=on_candidate,
            on_track_update=on_track_update,
            on_lost=on_lost,

            output_video_path=output_path,

            # Dashboard actions
            should_reject=lambda: _reject_requested.is_set(),
            should_confirm=lambda: _confirm_requested.is_set(),

            # New AWARE callbacks
            on_confirmed=on_confirmed,
            on_rejected=on_rejected,
        )

        HUB.broadcast_threadsafe(
            {
                "type": "finished",
                "output_video_path": output_path,
            }
        )

    except Exception as exc:

        HUB.broadcast_threadsafe(
            {
                "type": "error",
                "message": str(exc),
            }
        )

    finally:

        # Always clean up state when a search finishes or crashes.
        _reject_requested.clear()
        _confirm_requested.clear()
        _awaiting_decision.clear()


# ============================================================
# FASTAPI STARTUP
# ============================================================


@app.on_event("startup")
async def _capture_loop() -> None:

    import asyncio

    HUB.loop = asyncio.get_event_loop()


# ============================================================
# START SEARCH
# ============================================================


@app.post("/start")
async def start(
    req: StartRequest,
) -> dict:

    global _run_thread

    # --------------------------------------------------------
    # Prevent two simultaneous searches
    # --------------------------------------------------------

    if (
        _run_thread is not None
        and _run_thread.is_alive()
    ):

        return {
            "error":
                "A search is already running. "
                "Wait for it to finish."
        }

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    video_path = req.video_path.strip()
    description = req.description.strip()

    if not video_path:
        return {
            "error": "video_path is required."
        }

    if not description:
        return {
            "error":
                "A missing-person description is required."
        }

    # AWARE currently searches for people.
    # Keep category configurable for compatibility,
    # but default to person.
    category = (
        req.category.strip()
        if req.category
        else "person"
    )

    # --------------------------------------------------------
    # Clear previous search state
    # --------------------------------------------------------

    _reject_requested.clear()
    _confirm_requested.clear()
    _awaiting_decision.clear()

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    output_path = str(
        OUTPUT_DIR
        / f"annotated_{Path(video_path).stem}.mp4"
    )

    # --------------------------------------------------------
    # Start worker
    # --------------------------------------------------------

    _run_thread = threading.Thread(
        target=_worker,
        args=(
            video_path,
            category,
            description,
            output_path,
        ),
        daemon=True,
    )

    _run_thread.start()

    return {
        "ok": True,
        "status": "searching",
        "output_video_path": output_path,
    }


# ============================================================
# CONFIRM / REJECT CANDIDATE
# ============================================================


@app.post("/confirm")
async def confirm(
    req: ConfirmRequest,
) -> dict:
    """
    Receive the user's decision for the current possible match.

    Accepted:

        {"decision": "confirm"}

        {"decision": "reject"}
    """

    decision = req.decision.strip().lower()

    if decision not in {
        "confirm",
        "reject",
    }:

        return {
            "error":
                "decision must be either "
                "'confirm' or 'reject'."
        }

    # No candidate currently waiting.
    if not _awaiting_decision.is_set():

        return {
            "error":
                "There is no candidate awaiting confirmation."
        }

    # --------------------------------------------------------
    # REJECT
    # --------------------------------------------------------

    if decision == "reject":

        # Prevent a stale confirm request.
        _confirm_requested.clear()

        _reject_requested.set()

        # Keep _awaiting_decision set until SearchController
        # actually consumes the rejection and fires RejectedEvent.

        return {
            "ok": True,
            "decision": "reject",
            "status": "resuming_search",
        }

    # --------------------------------------------------------
    # CONFIRM
    # --------------------------------------------------------

    _reject_requested.clear()
    _confirm_requested.set()

    # Keep _awaiting_decision set until ConfirmedEvent arrives.
    # That prevents another candidate card from replacing this one
    # during the hand-off.

    return {
        "ok": True,
        "decision": "confirm",
        "status": "confirming_target",
    }


# ============================================================
# WEBSOCKET
# ============================================================


@app.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
) -> None:

    await HUB.connect(ws)

    try:

        while True:
            # The frontend may send heartbeat messages.
            # We do not currently need to inspect their contents.
            await ws.receive_text()

    except WebSocketDisconnect:

        HUB.disconnect(ws)

    except Exception:

        HUB.disconnect(ws)