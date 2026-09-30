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


# ============================================================
# EVENTS
# ============================================================


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
        CANDIDATE
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
    ):
        """
        Process one frame and return one event.
        """

        now = time.time()

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

        bbox, justification, score = (
            self.pipeline.forward(
                frame,
                category=self.category,
                description=self.description,
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
) -> None:
    """
    Run AWARE on a test video.

    Later, VideoStreamer will be replaced by the Gazebo /
    ROS2 camera stream while SearchController remains the same.
    """

    controller = SearchController(
        category=category,
        description=description,
    )

    stream = VideoStreamer(
        video_path
    )

    writer = None

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

        stream = VideoStreamer(
            video_path
        )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    for frame_index, frame in enumerate(
        stream
    ):

        # ----------------------------------------------------
        # DASHBOARD ACTIONS
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
            )
        )

        annotated = frame

        # ----------------------------------------------------
        # POSSIBLE MATCH
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

            # ------------------------------------------------
            # WAIT FOR OPERATOR DECISION
            # ------------------------------------------------

            while True:

                # --------------------------------------------
                # REJECT
                # --------------------------------------------

                if (
                    should_reject
                    and should_reject()
                ):

                    controller.reject_current_candidate()

                    rejected_event = (
                        controller.process_frame(
                            frame,
                            frame_index,
                        )
                    )

                    if isinstance(
                        rejected_event,
                        RejectedEvent,
                    ):

                        if on_rejected is not None:
                            on_rejected(
                                rejected_event
                            )

                    break

                # --------------------------------------------
                # CONFIRM
                # --------------------------------------------

                if (
                    should_confirm
                    and should_confirm()
                ):

                    controller.confirm_current_candidate()

                    confirmed_event = (
                        controller.process_frame(
                            frame,
                            frame_index,
                        )
                    )

                    if isinstance(
                        confirmed_event,
                        ConfirmedEvent,
                    ):

                        if on_confirmed is not None:
                            on_confirmed(
                                confirmed_event
                            )

                    break

                time.sleep(
                    0.05
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
        # OPTIONAL VIDEO OUTPUT
        # ----------------------------------------------------

        if writer is not None:

            writer.write(
                annotated
            )

    if writer is not None:

        writer.release()