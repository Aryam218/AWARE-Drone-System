import time

import cv2
import numpy as np
import PIL
import PIL.Image
from loguru import logger

from cloud_track.tracker_wrapper import OpenCVWrapper
from cloud_track.utils import get_best_box
from cloud_track.utils.flow_control import PerformanceTimer


class CloudTrack:
    """
    AWARE missing-person search and target-tracking pipeline.

    Flow:

        SEARCHING
            ↓
        Grounding DINO
            ↓
        ByteTrack person IDs
            ↓
        VLM verification
            ↓
        MATCH
            ↓
        candidate track ID selected
            ↓
        local target tracker
            ↓
        TRACKING

    Important:

    ByteTrack:
        Maintains persistent IDs for all detected people.

    Local OpenCV tracker:
        Follows the currently selected missing-person candidate
        efficiently between frames.

    Re-detection while TRACKING:
        The local tracker follows pixels, not people, so it can
        slide onto the background (e.g. a booth wall) when the
        person walks away. Every redetect_interval_s seconds (and
        whenever the local tracker fails), the person detector runs
        again (no VLM call) to check the box is still on a matching
        person:

            still on them      -> snap the tracker onto the detection
            moved a little     -> re-find them nearby, re-anchor
            not found          -> count a miss; after
                                  max_redetect_misses AND lost_timeout_s unseen -> LOST

    The higher-level SearchController handles:
        - operator confirmation
        - operator rejection
        - confirmed-person state
    """

    def __init__(
        self,
        backend,
        frontend_tracker: OpenCVWrapper,
    ):
        self.backend = backend
        self.frontend_tracker = frontend_tracker
        self.fm_timer = PerformanceTimer()

        # --------------------------------------------------
        # LOCAL TARGET TRACKER STATE
        # --------------------------------------------------

        self.tracker_initialized = False
        self.box = None

        self.current_justification = None
        self.state = "SEARCHING"

        # --------------------------------------------------
        # BYTETRACK PERSON ID
        # --------------------------------------------------
        #
        # This is the persistent ByteTrack ID of the person
        # currently selected as the candidate/target.
        #
        # Example:
        #
        #     Person ID 7 -> VLM MATCH
        #
        # AWARE keeps that ID so Confirm / Reject can refer
        # to the same person.
        # --------------------------------------------------

        self.current_track_id = None

        # --------------------------------------------------
        # RE-DETECTION WHILE TRACKING
        # --------------------------------------------------

        # Seconds between detector checks while tracking
        # (frame time: simulation time for the live camera).
        self.redetect_interval_s = 3.0

        # Both consecutive failed checks and time unseen are required for LOST.
        self.max_redetect_misses = 2
        self.lost_timeout_s = 30.0

        self._last_redetect_time = None
        self._redetect_misses = 0

        # Last box confirmed by the detector (not just the
        # local tracker), its ground position and frame time.
        self._anchor_box = None
        self._anchor_ground = None
        self._anchor_time = None
        self._anchor_pose = None
        self._anchor_frame = None

        # Re-finding by ground position: the person may have
        # walked this far: base + walking speed x time since
        # they were last verified (capped).
        self.reacquire_base_m = 4.0
        self.reacquire_walk_speed_m_s = 1.5
        self.reacquire_max_m = 15.0

        # Drone/camera pose of the frame self.box comes from
        # (live simulation only), so a rejection can be
        # remembered as a PLACE on the ground.
        self.box_frame_pose = None

        # The frame (RGB) self.box comes from, so a rejection
        # can also remember what the person looked like.
        self.box_frame = None

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):
        """
        Reset ONLY the active target lock.

        This is used when:

            - tracking is lost
            - a candidate is rejected
            - AWARE needs to resume searching

        IMPORTANT:

        This does NOT reset the detector-side ByteTrack
        history and does NOT clear rejected IDs.

        That allows AWARE to remember people across the same
        search session.
        """

        logger.info(
            "AWARE target tracker reset -> SEARCHING"
        )

        self.tracker_initialized = False
        self.box = None
        self.current_justification = None
        self.current_track_id = None

        self._last_redetect_time = None
        self._redetect_misses = 0
        self._anchor_box = None
        self._anchor_ground = None
        self._anchor_time = None
        self._anchor_pose = None
        self._anchor_frame = None
        self.box_frame_pose = None
        self.box_frame = None

        self.state = "SEARCHING"

    def reset_search_session(self):
        """
        Start a completely new missing-person search.

        Unlike reset(), this clears:

            - active target tracking
            - ByteTrack history
            - rejected person IDs

        Use this when the user starts a NEW report/search,
        not when rejecting one candidate.
        """

        self.reset()

        if hasattr(
            self.backend,
            "reset_person_tracker",
        ):
            self.backend.reset_person_tracker()

        logger.info(
            "AWARE complete search session reset."
        )

    # ======================================================
    # REJECT CURRENT PERSON
    # ======================================================

    def reject_current_target(self):
        """
        Reject the currently selected candidate.

        The current ByteTrack ID is added to the detector
        pipeline's rejected-ID memory.

        Therefore, if the same person appears again with the
        same ByteTrack ID, AWARE will not offer them again.

        After recording the rejection, the local target lock
        is reset and AWARE resumes SEARCHING.
        """

        rejected_id = self.current_track_id

        # Remember WHERE the person stands (works even when
        # the track ID is only temporary). Uses the last
        # VERIFIED position: the small tracker's box can be
        # stuck on the wrong spot while the drone moves.
        target = self.verified_target()

        if (
            target is not None
            and hasattr(
                self.backend,
                "reject_place",
            )
        ):
            self.backend.reject_place(
                target["box"],
                target["pose"],
                target["frame"],
                latlon=target["ground"],
            )

        if rejected_id is not None:

            if hasattr(
                self.backend,
                "reject_track_id",
            ):
                self.backend.reject_track_id(
                    rejected_id
                )

            logger.info(
                "AWARE operator rejected "
                f"person track ID {rejected_id}."
            )

        else:

            logger.warning(
                "AWARE rejection requested but "
                "there is no active ByteTrack ID."
            )

        self.reset()

        return rejected_id

    # ======================================================
    # FRAME PROCESSING
    # ======================================================

    def forward(
        self,
        frame: np.ndarray,
        category: str,
        description: str = None,
        timestamp: float = None,
    ):
        """
        Process one frame.

        SEARCHING:
            Grounding DINO
                ↓
            ByteTrack IDs
                ↓
            VLM verification

        If VLM returns MATCH:
            select candidate box + corresponding ByteTrack ID
                ↓
            initialize local tracker
                ↓
            TRACKING

        TRACKING:
            follow the selected candidate efficiently.

        If tracking fails:
            reset active target
                ↓
            return to SEARCHING

        timestamp:
            Optional frame time in seconds (simulation time for
            the live Gazebo camera). Passed to the detector/VLM
            backend for its answer cache. None = wall-clock time.

        Returns:
            bbox:
                Current target bounding box or None.

            justification:
                VLM explanation for why the candidate matched.

            score:
                Local tracking quality score.

        The selected ByteTrack ID is available through:

            self.current_track_id
        """

        # OpenCV frame = BGR
        # detector/VLM pipeline expects RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        score = None

        # --------------------------------------------------
        # SEARCHING
        # --------------------------------------------------

        if not self.tracker_initialized:

            self.state = "SEARCHING"

            bbox, justification = (
                self.add_keyframe(
                    frame_rgb,
                    category,
                    description,
                    timestamp=timestamp,
                )
            )

            self.current_justification = (
                justification
            )

            if bbox is not None:

                self.state = "TRACKING"

                logger.info(
                    "AWARE candidate MATCH found "
                    "-> TRACKING "
                    f"(ByteTrack ID: "
                    f"{self.current_track_id})"
                )

            else:

                logger.debug(
                    "AWARE: no matching person "
                    "in this frame. "
                    "Continuing search."
                )

        # --------------------------------------------------
        # TRACKING
        # --------------------------------------------------

        else:

            self.state = "TRACKING"

            bbox, success, score = (
                self.track_frame(
                    frame_rgb
                )
            )

            # ----------------------------------------------
            # RE-DETECTION: check the box is still on the
            # person (periodically, or right away if the
            # local tracker just failed).
            # ----------------------------------------------

            now = self._clock(timestamp)

            tracker_failed = (
                not success
                or bbox is None
            )

            if (
                self._can_redetect()
                and (
                    tracker_failed
                    or self._redetect_due(now)
                )
            ):

                bbox, success = self._redetect(
                    frame_rgb,
                    category,
                    description,
                    None if tracker_failed else bbox,
                    now,
                )

            if not success or bbox is None:

                lost_track_id = (
                    self.current_track_id
                )

                logger.warning(
                    "AWARE lost candidate "
                    f"track ID {lost_track_id} "
                    "-> returning to SEARCHING"
                )

                self.reset()

                bbox = None
                score = None

            else:

                self.box = np.asarray(
                    bbox
                )

                self.box_frame_pose = getattr(
                    self.backend,
                    "frame_pose",
                    None,
                )
                self.box_frame = frame_rgb

        # --------------------------------------------------
        # NORMALIZE BBOX
        # --------------------------------------------------

        if bbox is not None:

            bbox = np.asarray(
                bbox
            )

        # Keep the current API compatible with
        # pipeline_runner.py.
        return (
            bbox,
            self.current_justification,
            score,
        )

    # ======================================================
    # SEARCH KEYFRAME
    # ======================================================

    def add_keyframe(
        self,
        frame: np.ndarray,
        category: str,
        description: str,
        timestamp: float = None,
    ):
        """
        Search one frame for matching people.

        detector_vlm_pipeline.py now performs:

            Grounding DINO
                ↓
            ByteTrack
                ↓
            stable person IDs
                ↓
            VLM
                ↓
            MATCH / NO_MATCH / UNCERTAIN

        Only MATCH detections reach this function.

        backend.last_match_track_ids contains IDs aligned
        with boxes_filt.

        Example:

            boxes_filt:
                [box A, box B]

            last_match_track_ids:
                [4, 11]

        If get_best_box chooses box B, then the selected
        missing-person candidate has ByteTrack ID 11.
        """

        image_pil = PIL.Image.fromarray(
            frame
        )

        # Only pass the timestamp when there is one, so
        # backends without a timestamp argument still work.
        extra_args = {}

        if timestamp is not None:
            extra_args["timestamp"] = timestamp

        # --------------------------------------------------
        # DETECTOR + BYTETRACK + VLM
        # --------------------------------------------------

        with self.fm_timer:

            (
                image_pil,
                masks,
                boxes_filt,
                scores,
                justifications,
            ) = self.backend.run_inference(
                image_pil,
                category,
                description,
                **extra_args,
            )

        # --------------------------------------------------
        # NO MATCH
        # --------------------------------------------------

        if (
            boxes_filt is None
            or len(boxes_filt) == 0
        ):

            logger.debug(
                "AWARE: detector/VLM "
                "returned no MATCH."
            )

            return None, None

        # --------------------------------------------------
        # GET MATCH TRACK IDS
        # --------------------------------------------------

        match_track_ids = []

        if hasattr(
            self.backend,
            "get_last_match_track_ids",
        ):

            match_track_ids = (
                self.backend
                .get_last_match_track_ids()
            )

        elif hasattr(
            self.backend,
            "last_match_track_ids",
        ):

            match_track_ids = list(
                self.backend
                .last_match_track_ids
            )

        # Safety:
        # boxes and IDs should have identical ordering.
        if (
            len(match_track_ids) > 0
            and len(match_track_ids)
            != len(boxes_filt)
        ):

            logger.warning(
                "AWARE ByteTrack ID alignment "
                "mismatch: "
                f"{len(boxes_filt)} boxes vs "
                f"{len(match_track_ids)} IDs."
            )

        # --------------------------------------------------
        # SELECT BEST MATCH
        # --------------------------------------------------

        selected_box, idx = get_best_box(
            boxes_filt,
            scores,
        )

        # get_best_box may return NumPy / Tensor-like index
        selected_index = int(idx)

        selected_box = np.asarray(
            selected_box
        ).squeeze()

        # --------------------------------------------------
        # VALIDATE BBOX
        # --------------------------------------------------

        if selected_box.size != 4:

            logger.warning(
                "AWARE received invalid bbox: "
                f"{selected_box}"
            )

            return None, None

        x1, y1, x2, y2 = (
            selected_box
        )

        if x2 <= x1 or y2 <= y1:

            logger.warning(
                "AWARE received degenerate bbox: "
                f"{selected_box}"
            )

            return None, None

        # --------------------------------------------------
        # SELECT CORRESPONDING BYTETRACK ID
        # --------------------------------------------------

        selected_track_id = None

        if (
            len(match_track_ids)
            > selected_index
        ):

            selected_track_id = int(
                match_track_ids[
                    selected_index
                ]
            )

        else:

            logger.warning(
                "AWARE selected candidate "
                "has no corresponding "
                "ByteTrack ID."
            )

        # --------------------------------------------------
        # INITIALIZE LOCAL TARGET TRACKER
        # --------------------------------------------------

        try:

            self.frontend_tracker.init(
                frame,
                selected_box,
            )

        except Exception as exc:

            logger.exception(
                "AWARE could not initialize "
                f"local tracker: {exc}"
            )

            self.reset()

            return None, None

        # --------------------------------------------------
        # STORE ACTIVE TARGET
        # --------------------------------------------------

        self.tracker_initialized = True

        self.box = selected_box

        self.box_frame_pose = getattr(
            self.backend,
            "frame_pose",
            None,
        )
        self.box_frame = frame

        self.current_track_id = (
            selected_track_id
        )

        # The detector + VLM just verified this box.
        self._anchor_box = np.asarray(
            selected_box,
            dtype=np.float32,
        )
        self._redetect_misses = 0
        self._last_redetect_time = self._clock(
            timestamp
        )
        self._set_anchor_ground(
            self._anchor_box,
            self._last_redetect_time,
            frame,
        )

        self.state = "TRACKING"

        # --------------------------------------------------
        # VLM JUSTIFICATION
        # --------------------------------------------------

        justification = None

        if (
            justifications is not None
            and len(justifications)
            > selected_index
        ):

            justification = (
                justifications[
                    selected_index
                ]
            )

        logger.info(
            "AWARE MATCH accepted by VLM. "
            f"ByteTrack ID: "
            f"{self.current_track_id} | "
            f"Tracker initialized at "
            f"{selected_box.tolist()}"
        )

        return (
            selected_box,
            justification,
        )

    # ======================================================
    # RE-DETECTION HELPERS
    # ======================================================

    @staticmethod
    def _clock(timestamp):
        """Frame time if known (simulation time), else wall clock."""

        return (
            float(timestamp)
            if timestamp is not None
            else time.monotonic()
        )

    def _set_anchor_ground(self, box, now, frame=None) -> None:
        """Remember where on the ground the verified target stands,
        plus the frame and drone pose it was verified in."""

        self._anchor_ground = None
        self._anchor_time = now
        self._anchor_frame = frame
        self._anchor_pose = getattr(
            self.backend,
            "frame_pose",
            None,
        )

        if box is not None and hasattr(
            self.backend,
            "ground_position",
        ):
            self._anchor_ground = self.backend.ground_position(box)

    def verified_target(self):
        """
        The target as last VERIFIED by the detector (not the
        small tracker's latest guess), or None:

            {"box", "pose", "frame", "ground" (lat, lon) or None}

        Used for rejections and for telling the drone where to
        look: while the drone moves, the small tracker's box can
        stay on the wrong spot of the image.
        """

        if self._anchor_box is None:
            return None

        return {
            "box": self._anchor_box,
            "pose": self._anchor_pose,
            "frame": self._anchor_frame,
            "ground": self._anchor_ground,
        }

    def _can_redetect(self) -> bool:

        return hasattr(
            self.backend,
            "redetect_target",
        )

    def _redetect_due(self, now) -> bool:

        if self._last_redetect_time is None:
            return True

        elapsed = now - self._last_redetect_time

        # elapsed < 0: the clock went backwards
        # (e.g. the simulation was restarted).
        return (
            elapsed < 0
            or elapsed >= self.redetect_interval_s
        )

    def _redetect(
        self,
        frame_rgb: np.ndarray,
        category: str,
        description: str,
        track_box,
        now,
    ):
        """
        Run the person detector and check the target.

        track_box:
            where the local tracker thinks the target is,
            or None if the local tracker just failed.

        Returns (bbox, success).
        """

        self._last_redetect_time = now

        ref_box = (
            track_box
            if track_box is not None
            else self.box
        )

        # Ground-position search, if the target's ground
        # position is known (live simulation).
        ground_args = {}

        if self._anchor_ground is not None:

            elapsed = max(
                0.0,
                now - (self._anchor_time if self._anchor_time is not None else now),
            )

            ground_args = {
                "anchor_ground": self._anchor_ground,
                "max_ground_distance_m": min(
                    self.reacquire_max_m,
                    self.reacquire_base_m
                    + self.reacquire_walk_speed_m_s * elapsed,
                ),
            }

        try:

            with self.fm_timer:

                new_box, how = (
                    self.backend.redetect_target(
                        PIL.Image.fromarray(frame_rgb),
                        category,
                        description,
                        ref_box,
                        self._anchor_box,
                        **ground_args,
                    )
                )

        except Exception as exc:

            # Detector problem: don't drop the target
            # because of it, keep the tracker's answer.
            logger.warning(
                "AWARE re-detection failed: "
                f"{exc}"
            )

            return ref_box, ref_box is not None

        # --------------------------------------------------
        # PERSON FOUND: re-anchor the local tracker on them
        # --------------------------------------------------

        if new_box is not None:

            new_box = np.asarray(
                new_box,
                dtype=np.float32,
            ).reshape(4)

            try:

                self.frontend_tracker.init(
                    frame_rgb,
                    new_box,
                )

            except Exception as exc:

                logger.warning(
                    "AWARE could not re-anchor "
                    f"the tracker: {exc}"
                )

                # The detector still verified this sighting even if the local
                # tracker could not be initialized; retry tracking next frame.

            self._anchor_box = new_box
            self._redetect_misses = 0
            self._set_anchor_ground(new_box, now, frame_rgb)

            if how == "reacquired":

                logger.warning(
                    "AWARE re-detection: tracker had "
                    "slipped off the target; re-found "
                    "the person nearby at "
                    f"{new_box.astype(int).tolist()}"
                )

            else:

                logger.info(
                    "AWARE re-detection: target "
                    "still on a matching person."
                )

            return new_box, True

        # --------------------------------------------------
        # PERSON NOT FOUND
        # --------------------------------------------------

        self._redetect_misses += 1

        logger.warning(
            "AWARE re-detection: no matching person "
            "at the tracked box "
            f"(miss {self._redetect_misses}/"
            f"{self.max_redetect_misses}): {how}."
        )

        unseen_s = max(0.0, now - self._anchor_time) if self._anchor_time is not None else 0.0
        if (
            self._redetect_misses >= self.max_redetect_misses
            and unseen_s >= self.lost_timeout_s
        ):
            return None, False

        # Keep the last box through temporary misses, including tracker failures.
        return ref_box, ref_box is not None

    # ======================================================
    # LOCAL TARGET TRACKING
    # ======================================================

    def track_frame(
        self,
        frame: np.ndarray,
    ):
        """
        Track the currently selected candidate.

        ByteTrack identifies people globally.

        This local tracker follows the selected candidate
        efficiently once that person has passed the VLM.

        If this local tracker fails, forward() returns AWARE
        to SEARCHING.
        """

        try:

            success, bbox, score = (
                self.frontend_tracker.update(
                    frame
                )
            )

        except Exception as exc:

            logger.exception(
                "AWARE tracker update failed: "
                f"{exc}"
            )

            return (
                None,
                False,
                None,
            )

        # --------------------------------------------------
        # TRACK LOST
        # --------------------------------------------------

        if (
            not success
            or bbox is None
        ):

            return (
                None,
                False,
                score,
            )

        bbox = np.asarray(
            bbox
        )

        # --------------------------------------------------
        # INVALID TRACKER OUTPUT
        # --------------------------------------------------

        if bbox.size != 4:

            logger.warning(
                "AWARE tracker returned "
                f"invalid bbox: {bbox}"
            )

            return (
                None,
                False,
                score,
            )

        return (
            bbox,
            True,
            score,
        )