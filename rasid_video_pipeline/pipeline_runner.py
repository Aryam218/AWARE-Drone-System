"""
The AWARE search engine.

Architecture:
- detector + VLM backend
- ByteTrack person IDs
- local frontend tracker
- callbacks for dashboard / flight controller

AWARE search flow:
- candidate detection
- persistent person ID
- candidate confirmation
- candidate rejection
- confirmed-person tracking

Crowd analytics remains separate from this module.
"""
from __future__ import annotations


import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

from cloud_track.foundation_model_wrappers.detector_vlm_pipeline import (
    get_vlm_pipeline,
)
from cloud_track.pipeline.cloud_track import CloudTrack
from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.utils.video_streamer import VideoStreamer


load_dotenv()
from ros_frame_source import RosVideoStreamer


# ============================================================
# EVENTS
# ============================================================
#
# Event timestamps are the frame's SIMULATION time (seconds)
# for the live Gazebo camera, and wall-clock time for
# recorded videos.
# ============================================================


@dataclass
class VerificationErrorEvent:
    """Permanent verification failure, reported once to interested UIs."""
    message: str
    frame_index: int
    timestamp: float
    code: str = "insufficient_quota"


@dataclass
class CandidateEvent:
    """
    Fired when detector + VLM finds a possible matching person.

    This is still only a POSSIBLE MATCH until the operator
    explicitly confirms the candidate.
    """

    frame: np.ndarray
    bbox: np.ndarray
    justification: Optional[str]
    track_id: Optional[int]
    frame_index: int
    timestamp: float


@dataclass
class TrackUpdateEvent:
    """
    Fired while the selected candidate/target is being tracked.

    confirmed=False: candidate still waiting for the operator's
    decision (the box keeps following the person meanwhile).
    """

    frame: np.ndarray
    bbox: np.ndarray
    score: Optional[float]
    track_id: Optional[int]
    frame_index: int
    timestamp: float
    confirmed: bool = False


@dataclass
class LostEvent:
    """
    Fired when AWARE loses the currently tracked candidate/target.
    """

    frame_index: int
    timestamp: float
    track_id: Optional[int] = None
    was_confirmed: bool = False


@dataclass
class ConfirmedEvent:
    """
    Fired once when the operator confirms that the current
    candidate is the missing person.

    Tracking continues after confirmation.
    """

    frame: np.ndarray
    bbox: np.ndarray
    track_id: Optional[int]
    frame_index: int
    timestamp: float


@dataclass
class RejectedEvent:
    """
    Fired when the operator rejects the current candidate.

    The candidate's ByteTrack ID is remembered so the same
    person is not repeatedly presented during the same search.
    """

    frame_index: int
    timestamp: float
    track_id: Optional[int] = None


# ============================================================
# SEARCH CONTROLLER
# ============================================================


class SearchController:
    """
    Coordinates CloudTrack with the dashboard.

    Flow:

        SEARCHING
            ↓
        Grounding DINO
            ↓
        ByteTrack IDs
            ↓
        VLM
            ↓
        MATCH
            ↓
        CANDIDATE (tracked while the operator decides)
           ↙     ↘
       REJECT   CONFIRM
          ↓        ↓
     remember ID  CONFIRMED
          ↓        ↓
      SEARCHING  TRACKING

    ByteTrack handles persistent person identity.

    The local OpenCV tracker follows the currently selected
    candidate efficiently after the VLM match.
    """

    def __init__(
        self,
        category: str,
        description: str,
        openai_api_key: Optional[str] = None,
        detector=None,
    ):
        self.category = category
        self.description = description

        # --------------------------------------------------------
        # DETECTOR + VLM + BYTETRACK BACKEND
        # --------------------------------------------------------

        backend = get_vlm_pipeline(
            vl_model_name="gpt-4o-mini",
            system_description=(
                "You are assisting the AWARE autonomous "
                "missing-person search system at a crowded venue. "
                f"{description}"
            ),
            simulate_time_delay=False,
            detector_name="sam_lq",
            detector=detector,
            openai_api_key=(
                openai_api_key
                or os.environ.get("OPENAI_API_KEY")
            ),
        )

        # --------------------------------------------------------
        # LOCAL TARGET TRACKER
        # --------------------------------------------------------

        frontend_tracker = OpenCVWrapper(
            tracker_type="nano",
            reinit_threshold=0.5,
        )

        self.pipeline = CloudTrack(
            backend=backend,
            frontend_tracker=frontend_tracker,
        )

        # --------------------------------------------------------
        # STATE
        # --------------------------------------------------------

        self._was_tracking = False

        self._parent_rejected_pending = False
        self._parent_confirmed_pending = False

        self._candidate_confirmed = False

        self._last_candidate_frame = None
        self._last_candidate_bbox = None
        self._last_candidate_track_id = None

        # A candidate that was LOST a moment ago. On a slow
        # computer the operator often presses "reject" just
        # after the box disappeared: that should still count.
        self._recently_lost = None
        self._late_reject_pending = False
        self.late_reject_window_s = 20.0


    # ========================================================
    # USER ACTIONS
    # ========================================================

    def reject_current_candidate(self) -> None:
        """
        Request rejection of the currently displayed candidate.

        The actual state mutation happens safely inside
        process_frame().
        """

        if self.pipeline.tracker_initialized:
            self._parent_rejected_pending = True

        elif (
            self._recently_lost is not None
            and time.monotonic() - self._recently_lost["time"]
            <= self.late_reject_window_s
        ):
            # Reject pressed just after the candidate was lost.
            self._late_reject_pending = True

    def confirm_current_candidate(self) -> None:
        """
        Confirm the current candidate.

        Confirmation does not stop tracking.
        """

        if self.pipeline.tracker_initialized:
            self._parent_confirmed_pending = True

    # ========================================================
    # FRAME PROCESSING
    # ========================================================

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: Optional[float] = None,
    ):
        """
        Process one frame and return one event.

        timestamp:
            Frame time in seconds (simulation time for the live
            camera). None = use wall-clock time.
        """

        now = (
            timestamp
            if timestamp is not None
            else time.time()
        )

        # --------------------------------------------------------
        # REJECTION
        # --------------------------------------------------------

        if self._parent_rejected_pending:

            # Save the ID before CloudTrack clears its
            # currently selected target.
            rejected_track_id = (
                self.pipeline.current_track_id
            )

            # IMPORTANT:
            # Do not simply reset().
            #
            # reject_current_target() stores the ByteTrack ID
            # in rejected-ID memory and then resumes SEARCHING.
            self.pipeline.reject_current_target()

            self._parent_rejected_pending = False
            self._parent_confirmed_pending = False

            self._was_tracking = False
            self._candidate_confirmed = False

            self._last_candidate_frame = None
            self._last_candidate_bbox = None
            self._last_candidate_track_id = None

            return RejectedEvent(
                frame_index=frame_index,
                timestamp=now,
                track_id=rejected_track_id,
            )

        # --------------------------------------------------------
        # LATE REJECTION (candidate was lost a moment ago)
        # --------------------------------------------------------

        if self._late_reject_pending:

            self._late_reject_pending = False

            lost = self._recently_lost
            self._recently_lost = None

            backend = self.pipeline.backend

            if lost is not None:

                if hasattr(backend, "reject_place"):
                    backend.reject_place(
                        lost["box"],
                        lost["pose"],
                        lost["frame"],
                        latlon=lost.get("ground"),
                    )

                if (
                    lost["track_id"] is not None
                    and hasattr(backend, "reject_track_id")
                ):
                    backend.reject_track_id(lost["track_id"])

                return RejectedEvent(
                    frame_index=frame_index,
                    timestamp=now,
                    track_id=lost["track_id"],
                )

        # --------------------------------------------------------
        # CONFIRMATION
        # --------------------------------------------------------

        if self._parent_confirmed_pending:

            self._parent_confirmed_pending = False
            self._candidate_confirmed = True

            confirmed_track_id = (
                self.pipeline.current_track_id
            )

            self._last_candidate_track_id = (
                confirmed_track_id
            )

            if (
                self._last_candidate_frame is not None
                and self._last_candidate_bbox is not None
            ):
                # Acknowledge immediately; the next frame performs the expensive check.
                self.pipeline.request_redetection()
                return ConfirmedEvent(
                    frame=self._last_candidate_frame.copy(),
                    bbox=self._last_candidate_bbox.copy(),
                    track_id=confirmed_track_id,
                    frame_index=frame_index,
                    timestamp=now,
                )

        # --------------------------------------------------------
        # CLOUDTRACK
        # --------------------------------------------------------

        was_tracking = (
            self.pipeline.tracker_initialized
        )

        # Save current ID before forward(), because if the
        # local tracker is lost, CloudTrack.reset() clears it.
        previous_track_id = (
            self.pipeline.current_track_id
        )

        # Same for the target's last VERIFIED position (for a
        # late rejection).
        previous_target = (
            self.pipeline.verified_target()
            if hasattr(self.pipeline, "verified_target")
            else None
        )

        if previous_target is not None:
            previous_target = dict(previous_target)
            previous_target["track_id"] = previous_track_id

        bbox, justification, score = (
            self.pipeline.forward(
                frame,
                category=self.category,
                description=self.description,
                timestamp=timestamp,
            )
        )

        is_tracking_now = (
            self.pipeline.tracker_initialized
        )

        current_track_id = (
            self.pipeline.current_track_id
        )

        # --------------------------------------------------------
        # LOST TRACK
        # --------------------------------------------------------

        if was_tracking and not is_tracking_now:

            was_confirmed = (
                self._candidate_confirmed
            )

            lost_track_id = (
                previous_track_id
            )

            self._was_tracking = False
            self._candidate_confirmed = False

            self._last_candidate_frame = None
            self._last_candidate_bbox = None
            self._last_candidate_track_id = None

            # Only an unconfirmed candidate can be rejected late.
            if not was_confirmed and previous_target is not None:
                previous_target["time"] = time.monotonic()
                self._recently_lost = previous_target
            else:
                self._recently_lost = None

            return LostEvent(
                frame_index=frame_index,
                timestamp=now,
                track_id=lost_track_id,
                was_confirmed=was_confirmed,
            )

        # --------------------------------------------------------
        # STILL SEARCHING
        # --------------------------------------------------------

        if bbox is None:

            self._was_tracking = False

            return None

        bbox = np.asarray(
            bbox
        )

        # --------------------------------------------------------
        # NEW POSSIBLE MATCH
        # --------------------------------------------------------

        if (
            not was_tracking
            and is_tracking_now
        ):

            self._was_tracking = True
            self._candidate_confirmed = False

            self._last_candidate_frame = (
                frame.copy()
            )

            self._last_candidate_bbox = (
                bbox.copy()
            )

            self._last_candidate_track_id = (
                current_track_id
            )

            return CandidateEvent(
                frame=frame,
                bbox=bbox,
                justification=justification,
                track_id=current_track_id,
                frame_index=frame_index,
                timestamp=now,
            )

        # --------------------------------------------------------
        # CONTINUOUS TRACKING
        # --------------------------------------------------------

        if is_tracking_now:

            self._last_candidate_frame = (
                frame.copy()
            )

            self._last_candidate_bbox = (
                bbox.copy()
            )

            self._last_candidate_track_id = (
                current_track_id
            )

            return TrackUpdateEvent(
                frame=frame,
                bbox=bbox,
                score=score,
                track_id=current_track_id,
                frame_index=frame_index,
                timestamp=now,
                confirmed=self._candidate_confirmed,
            )

        return None


# ============================================================
# DRAWING
# ============================================================


def draw_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    label: str = "",
) -> np.ndarray:

    x1, y1, x2, y2 = (
        int(v)
        for v in bbox
    )

    out = frame.copy()

    cv2.rectangle(
        out,
        (x1, y1),
        (x2, y2),
        (0, 165, 255),
        2,
    )

    if label:

        cv2.putText(
            out,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    return out


# ============================================================
# VIDEO RUNNER
# ============================================================

# ============================================================
# STREAM SOURCE
# ============================================================


def open_stream(video_path: str):
    """'ros' = live Gazebo camera; anything else = a video file, as before."""
    if video_path == "ros":
        return RosVideoStreamer("/aware/camera/image")
    return VideoStreamer(video_path)

def run_on_video(
    video_path: str,
    category: str,
    description: str,
    on_candidate: Callable[[CandidateEvent], None],
    on_track_update: Callable[[TrackUpdateEvent], None],
    on_lost: Callable[[LostEvent], None],
    output_video_path: Optional[str] = None,
    should_reject: Optional[Callable[[], bool]] = None,
    should_confirm: Optional[Callable[[], bool]] = None,
    on_confirmed: Optional[
        Callable[[ConfirmedEvent], None]
    ] = None,
    on_rejected: Optional[
        Callable[[RejectedEvent], None]
    ] = None,
    on_frame: Optional[
        Callable[[np.ndarray, object], None]
    ] = None,
    on_verification_error: Optional[Callable[[VerificationErrorEvent], None]] = None,
    on_target_position: Optional[Callable[[dict], None]] = None,
    on_crowd_analysis: Optional[Callable[[dict], None]] = None,
    detector=None,
    on_processed_frame: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Run AWARE on a test video or the live Gazebo camera
    (video_path="ros").

    The frame loop never pauses for the operator: while a
    candidate waits for confirm / reject, the local tracker
    keeps following them (TrackUpdateEvent with
    confirmed=False). The decision is picked up at the start
    of the next frame.
    """

    controller = SearchController(
        category=category,
        description=description,
        detector=detector,
    )

    crowd_counter = None
    if on_crowd_analysis is not None:
        from analytics.moving_crowd import MovingCrowdCounter
        crowd_counter = MovingCrowdCounter(detector=controller.pipeline.backend.detector)

    stream = open_stream(
        video_path
    )

    # --------------------------------------------------------
    # DRONE POSITION (live simulation only)
    # --------------------------------------------------------
    #
    # Read with every frame, so a rejected person can be
    # remembered as a PLACE on the ground (see aware_geo.py).
    # --------------------------------------------------------

    drone_reader = None
    target_pub = None

    if video_path == "ros":

        try:
            from drone_state import DroneStateReader, TargetPublisher

            drone_reader = DroneStateReader()

            # Tells the patrol where the candidate stands, so the
            # drone can fly to look at them and follow them.
            target_pub = TargetPublisher()

        except Exception as exc:
            print(
                "[AWARE] drone position unavailable, rejected "
                f"places will not be remembered: {exc}"
            )

    def publish_target(event) -> None:
        """Send the candidate's ground position to the patrol."""

        if target_pub is None or event is None:
            return

        try:

            if isinstance(event, (LostEvent, RejectedEvent)):
                target_pub.publish("none", track_id=event.track_id)
                return

            if isinstance(event, CandidateEvent):
                state = "candidate"
            elif isinstance(event, ConfirmedEvent):
                state = "confirmed"
            elif isinstance(event, TrackUpdateEvent):
                state = "confirmed" if event.confirmed else "candidate"
            else:
                return

            # The last VERIFIED ground position, not the small
            # tracker's box: while the drone flies and turns, that
            # box can stay on the wrong spot of the image, which
            # would send the drone after a phantom.
            target = controller.pipeline.verified_target()
            ground = target["ground"] if target is not None else None

            if ground is not None:
                target_pub.publish(state, ground, track_id=event.track_id)

        except Exception as exc:
            print(f"[AWARE] could not publish the target position: {exc}")

    writer = None
    verification_error_reported = False

    # --------------------------------------------------------
    # OPTIONAL OUTPUT VIDEO
    # --------------------------------------------------------

    if output_video_path:

        first_frame = next(
            iter(stream)
        )

        h, w = (
            first_frame.shape[:2]
        )

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            20.0,
            (w, h),
        )

    if video_path != "ros":
        stream = VideoStreamer(video_path)

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for frame_index, frame in enumerate(
        stream
    ):

        if should_stop is not None and should_stop():
            break

        # Simulation time of this frame (live camera only;
        # recorded videos have no last_stamp -> None).
        frame_time = getattr(
            stream,
            "last_stamp",
            None,
        )

        # Drone pose for THIS frame (read right after the frame
        # arrived, so both describe the same moment).
        if drone_reader is not None:
            controller.pipeline.backend.frame_pose = (
                drone_reader.frame_pose()
            )

        # ----------------------------------------------------
        # DASHBOARD ACTIONS
        # ----------------------------------------------------
        #
        # Checked every frame. The controller applies the
        # decision inside process_frame() below.
        # ----------------------------------------------------

        if (
            should_reject
            and should_reject()
        ):
            controller.reject_current_candidate()

        if (
            should_confirm
            and should_confirm()
        ):
            controller.confirm_current_candidate()

        # ----------------------------------------------------
        # PROCESS FRAME
        # ----------------------------------------------------

        event = (
            controller.process_frame(
                frame,
                frame_index,
                timestamp=frame_time,
            )
        )

        verification_error = getattr(controller.pipeline.backend, "verification_error", None)
        if verification_error is not None and not verification_error_reported:
            verification_error_reported = True
            if on_verification_error is not None:
                on_verification_error(VerificationErrorEvent(
                    message=verification_error,
                    frame_index=frame_index,
                    timestamp=frame_time if frame_time is not None else time.time(),
                ))

        # Expose the existing verified anchor before the dashboard event callbacks.
        # Ground truth never enters this observer; current drone pose is separate.
        if on_target_position is not None and isinstance(event, (CandidateEvent, ConfirmedEvent, TrackUpdateEvent)):
            target = controller.pipeline.verified_target()
            ground = target.get("ground") if target is not None else None
            pose = getattr(controller.pipeline.backend, "frame_pose", None)
            on_target_position({
                "location": {"lat": float(ground[0]), "lon": float(ground[1])} if ground is not None else None,
                "position_sim_time": getattr(controller.pipeline, "_anchor_time", None) if target is not None and frame_time is not None else None,
                "drone": {key: pose.get(key) for key in ("lat", "lon", "alt_rel_m", "heading_deg", "roll_deg", "pitch_deg")} if pose else None,
            })

        annotated = frame

        # ----------------------------------------------------
        # POSSIBLE MATCH
        # ----------------------------------------------------
        #
        # No waiting here: the next frames keep tracking the
        # candidate until the operator confirms or rejects.
        # ----------------------------------------------------

        if isinstance(
            event,
            CandidateEvent,
        ):

            label = "possible match"

            if event.track_id is not None:
                label += (
                    f" ID {event.track_id}"
                )

            annotated = draw_bbox(
                frame,
                event.bbox,
                label=label,
            )

            on_candidate(
                event
            )

        # ----------------------------------------------------
        # TRACKING
        # ----------------------------------------------------

        elif isinstance(
            event,
            TrackUpdateEvent,
        ):

            if event.confirmed:

                label = (
                    "confirmed target"
                )

            else:

                label = (
                    "candidate tracking"
                )

            if event.track_id is not None:

                label += (
                    f" ID {event.track_id}"
                )

            annotated = draw_bbox(
                frame,
                event.bbox,
                label=label,
            )

            on_track_update(
                event
            )

        # ----------------------------------------------------
        # CONFIRMED
        # ----------------------------------------------------

        elif isinstance(
            event,
            ConfirmedEvent,
        ):

            label = (
                "confirmed target"
            )

            if event.track_id is not None:

                label += (
                    f" ID {event.track_id}"
                )

            annotated = draw_bbox(
                event.frame,
                event.bbox,
                label=label,
            )

            if on_confirmed is not None:

                on_confirmed(
                    event
                )

        # ----------------------------------------------------
        # REJECTED
        # ----------------------------------------------------

        elif isinstance(
            event,
            RejectedEvent,
        ):

            if on_rejected is not None:

                on_rejected(
                    event
                )

        # ----------------------------------------------------
        # LOST
        # ----------------------------------------------------

        elif isinstance(
            event,
            LostEvent,
        ):

            on_lost(
                event
            )


        # ----------------------------------------------------
        # TARGET POSITION -> PATROL
        # ----------------------------------------------------

        publish_target(event)

        if on_processed_frame is not None:
            # Search acknowledgements/target steering precede low-priority crowd work.
            on_processed_frame(frame, getattr(controller.pipeline.backend, "frame_pose", None),
                               frame_time, controller)

        if crowd_counter is not None:
            # Search's raw full-frame person inference is reused by the shared
            # detector cache. Tracking-only frames need one crowd inference.
            on_crowd_analysis(crowd_counter.analyze_frame(
                frame, pose=getattr(controller.pipeline.backend, "frame_pose", None),
                timestamp=frame_time))


        # ----------------------------------------------------
        # LIVE VIEW
        # ----------------------------------------------------

        if on_frame is not None:
            on_frame(annotated, event)

        # ----------------------------------------------------
        # OPTIONAL VIDEO OUTPUT
        # ----------------------------------------------------

        if writer is not None:

            writer.write(
                annotated
            )

    if writer is not None:

        writer.release()

    if hasattr(stream, "release"):
        stream.release()

    if drone_reader is not None:
        drone_reader.release()

    if target_pub is not None:
        target_pub.release()