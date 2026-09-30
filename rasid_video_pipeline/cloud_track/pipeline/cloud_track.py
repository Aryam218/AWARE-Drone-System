import cv2
import numpy as np
import PIL
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

        self.current_track_id = (
            selected_track_id
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