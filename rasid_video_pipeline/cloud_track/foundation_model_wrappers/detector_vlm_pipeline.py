from pathlib import Path
import io
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from cloud_track.foundation_model_wrappers.aware_candidates import (
    appearance_signature,
    same_look,
    select_candidates,
    verify_target,
)
from cloud_track.foundation_model_wrappers.aware_geo import (
    RejectedPlaces,
    box_to_ground,
    distance_m,
)
from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase
from cloud_track.tracker_wrapper.bytetrack_wrapper import ByteTrackWrapper

from .gpt_four_wrapper import (
    GPTFourWrapper, VlmInsufficientQuotaError, NO_CREDITS_MESSAGE,
)
from .grounding_dino_huggingface_wrapper import (
    GroundingDinoHuggingfaceWrapper,
)

try:
    from .grounding_dino_wrapper import GroundingDinoWrapper
except ImportError:
    GroundingDinoWrapper = None

from .llava_wrapper import LlavaWrapper
from .paligemma_wrapper import PaligemmaWrapper


# ============================================================
# FIXED RESPONSES (same format as a real VLM reply)
# ============================================================

RESPONSE_REJECTED = """
Decision: NO_MATCH
Justification: This tracked person was previously rejected by the operator.
"""

RESPONSE_INVALID_BOX = """
Decision: NO_MATCH
Justification: Skipped invalid detection box.
"""

RESPONSE_INVALID_BOX_OVERSCAN = """
Decision: NO_MATCH
Justification: Skipped invalid detection box after overscan.
"""

RESPONSE_DETECTOR_ONLY = """
Decision: MATCH
Justification: Detector-only mode; no VLM verification available.
"""

RESPONSE_DEFERRED = """
Decision: UNCERTAIN
Justification: Not checked this frame (VLM call budget reached).
"""

RESPONSE_TIMEOUT = """
Decision: UNCERTAIN
Justification: VLM call timed out.
"""

RESPONSE_VERIFICATION_OFF = f"""
Decision: UNCERTAIN
Justification: {NO_CREDITS_MESSAGE}
"""

RESPONSE_FAILED = """
Decision: UNCERTAIN
Justification: VLM call failed.
"""


# ============================================================
# DETECTOR + VLM + BYTETRACK PIPELINE
# ============================================================


class DetectorVlmPipeline(WrapperBase):
    """
    AWARE detector + VLM verification pipeline.

    Flow:

        Frame
          ↓
        Grounding DINO
          ↓
        Person detections
          ↓
        ByteTrack
          ↓
        select_candidates (cheap color / plausibility filter)
          ↓
        VLM cache lookup (confirmed ByteTrack IDs only)
          ↓
        VLM verification (parallel, budgeted, with timeout)
          ↓
        MATCH / NO_MATCH / UNCERTAIN

    Only MATCH detections are returned to CloudTrack.

    TRACK IDS:
        Positive IDs come from ByteTrack and persist across
        frames. Negative IDs are temporary (one frame only),
        assigned by select_candidates to boxes ByteTrack has
        not confirmed yet. Only positive IDs are cached or
        can be rejected.

    VLM CACHE:
        Each confirmed ID's VLM decision is cached and expires
        on a time-based schedule depending on the decision:

            UNCERTAIN -> short TTL, also re-verified early if
                         the person's box grows (closer view)
            NO_MATCH  -> long TTL (moderate, to survive ID swaps)
            MATCH     -> long TTL, periodically re-verified

        Time comes from the optional `timestamp` argument of
        run_inference() (e.g. ROS image header / Gazebo sim
        time). Without it, wall-clock time is used.
    """

    def __init__(
        self,
        vlm: WrapperBase,
        detector: WrapperBase,
        enable_overscan=True,
        overscan_value=50,
    ):
        self.vlm = vlm
        from analytics.shared_detector import shared_detector
        self.detector = shared_detector(detector)

        self.enable_overscan = enable_overscan
        self.overscan_value = overscan_value

        self.debug_enable_gpt = True
        self.verification_error = None

        # ----------------------------------------------------
        # SHARED PERSON TRACKER
        # ----------------------------------------------------

        self.person_tracker = ByteTrackWrapper(
            track_activation_threshold=0.15,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
            frame_rate=30,
        )

        # ----------------------------------------------------
        # TRACK-ID STATE
        # ----------------------------------------------------

        # Every track ID in the most recently processed frame.
        self.last_track_ids = []

        # Track IDs of MATCH detections, same order as the
        # filtered boxes returned from run_inference().
        self.last_match_track_ids = []

        # Confirmed ByteTrack IDs rejected by the operator.
        self.rejected_track_ids = set()

        # ----------------------------------------------------
        # CANDIDATE / VLM BUDGET
        # ----------------------------------------------------

        # How many candidates select_candidates may return.
        # Cached candidates don't cost a VLM call, so this
        # can be larger than the VLM budget.
        self.max_candidates = 6

        # Maximum number of real VLM calls per frame.
        self.max_vlm_calls_per_frame = 3

        # Max seconds to wait for all VLM calls of one frame.
        self.vlm_timeout_s = 20.0

        # VLM calls of one frame run in parallel.
        self._vlm_executor = ThreadPoolExecutor(
            max_workers=self.max_vlm_calls_per_frame * 2,
            thread_name_prefix="aware_vlm",
        )

        # ----------------------------------------------------
        # VLM DECISION CACHE (confirmed ByteTrack IDs only)
        # ----------------------------------------------------
        #
        # track_id -> {
        #     "response": raw VLM text,
        #     "decision": MATCH / NO_MATCH / UNCERTAIN,
        #     "time": time when the VLM was queried,
        #     "box_area": box area at query time,
        #     "last_seen": last time this ID was tracked,
        # }
        # ----------------------------------------------------

        self.vlm_cache = {}

        # How many seconds each decision stays valid.
        self.cache_ttl_s = {
            "MATCH": 30.0,
            "NO_MATCH": 45.0,
            "UNCERTAIN": 5.0,
        }

        # Re-query an UNCERTAIN person early if their box
        # grows by this factor (drone got closer).
        self.uncertain_area_growth = 1.5

        # Drop entries for IDs not seen for this many seconds.
        self.cache_forget_after_s = 60.0

        # Current frame time and counter.
        self._now = 0.0
        self.frame_index = 0

        # ----------------------------------------------------
        # REJECTED PLACES (live simulation)
        # ----------------------------------------------------
        #
        # frame_pose: drone GPS + heading + camera lens for the
        # frame being processed. Set by pipeline_runner for the
        # live camera (None for video files).
        #
        # When the operator rejects someone, the place where
        # they stand is remembered, and people standing there
        # are skipped for a while. This works even though
        # unconfirmed people get a new temporary ID per frame.
        # ----------------------------------------------------

        self.frame_pose = None
        # Skipped only if near a rejected place AND dressed
        # like the rejected person (protects the real target
        # standing next to someone who was rejected).
        self.rejected_places = RejectedPlaces(
            radius_m=4.0,
            ttl_s=180.0,
            same_look=same_look,
        )

    # ========================================================
    # BYTE TRACK STATE
    # ========================================================

    def reject_track_id(self, track_id: int) -> None:
        """
        Remember a person that the operator explicitly rejected.

        Only confirmed (positive) ByteTrack IDs can be
        remembered. Temporary negative IDs exist for a single
        frame and cannot be rejected persistently.
        """

        if track_id is None:
            return

        track_id = int(track_id)

        if track_id < 0:

            logger.warning(
                f"AWARE: track ID {track_id} is temporary "
                "(not confirmed by ByteTrack); rejection "
                "cannot be remembered across frames."
            )

            return

        self.rejected_track_ids.add(track_id)
        self.vlm_cache.pop(track_id, None)

        logger.info(
            f"AWARE: track ID {track_id} marked as rejected."
        )

    def clear_rejected_track_ids(self) -> None:
        """
        Clear the rejected-person memory.
        """

        self.rejected_track_ids.clear()

    def reset_person_tracker(self) -> None:
        """
        Reset ByteTrack and all ID-related state.

        Use only when starting a completely new search session.
        """

        self.person_tracker.reset()

        self.last_track_ids = []
        self.last_match_track_ids = []
        self.rejected_track_ids.clear()

        self.vlm_cache.clear()
        self.frame_index = 0
        self.rejected_places.clear()

        logger.info(
            "AWARE: ByteTrack person-tracking state "
            "and VLM cache reset."
        )

    # ========================================================
    # REJECTED PLACES
    # ========================================================

    def ground_position(self, box, pose=None):
        """(lat, lon) of the person's feet, or None if unknown."""

        return box_to_ground(
            box,
            pose if pose is not None else self.frame_pose,
        )

    def reject_place(self, box, pose=None, image=None, latlon=None) -> None:
        """
        Remember WHERE the operator rejected someone, and what
        they looked like.

        box:   the rejected person's box in the image
        pose:  the frame pose that box belongs to (falls back
               to the current frame pose)
        image: the frame that box belongs to (for the look)
        latlon: the person's ground position if already known
        """

        if box is None:
            return

        if latlon is None:
            latlon = self.ground_position(box, pose)

        if latlon is None:

            logger.warning(
                "AWARE: rejected person's ground position "
                "unknown (no drone position / camera info); "
                "only the track ID is remembered."
            )

            return

        signature = (
            appearance_signature(image, box)
            if image is not None
            else None
        )

        self.rejected_places.add(
            latlon,
            signature,
        )

        logger.info(
            "AWARE: rejected place remembered at "
            f"lat {latlon[0]:.7f}, lon {latlon[1]:.7f} "
            f"(radius {self.rejected_places.radius_m:.0f} m, "
            f"{self.rejected_places.ttl_s:.0f} s). "
            f"{len(self.rejected_places)} place(s) remembered."
        )

    def _box_at_rejected_place(self, box, image) -> bool:
        """
        True if this person stands where someone was rejected
        AND is dressed like them.
        """

        latlon = self.ground_position(box)

        # Cheap test first: most people are nowhere near a
        # rejected place.
        if not self.rejected_places.is_near(latlon):
            return False

        return self.rejected_places.is_rejected(
            latlon,
            appearance_signature(image, box),
        )

    def get_last_match_track_ids(self):
        """
        Return the track IDs corresponding to the latest
        MATCH boxes returned by run_inference().
        """

        return list(self.last_match_track_ids)

    def shutdown(self) -> None:
        """
        Stop the VLM worker threads (call on program exit).
        """

        self._vlm_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    # ========================================================
    # VLM CACHE
    # ========================================================

    @staticmethod
    def _is_cacheable(track_id: int) -> bool:
        """
        Only confirmed ByteTrack IDs are cached. Negative IDs
        are temporary and change every frame.
        """

        return track_id >= 0

    @staticmethod
    def _extract_decision(reply) -> str:
        """
        Quiet decision parse (no logging) used for caching.
        """

        if not isinstance(reply, str):
            return "UNCERTAIN"

        for line in reply.splitlines():

            line = line.strip()

            if line.lower().startswith("decision:"):

                value = (
                    line.split(":", 1)[1]
                    .strip()
                    .upper()
                )

                if value in (
                    "MATCH",
                    "NO_MATCH",
                    "UNCERTAIN",
                ):
                    return value

                return "UNCERTAIN"

        return "UNCERTAIN"

    def _get_cached_response(
        self,
        track_id: int,
        box_area: float,
    ):
        """
        Return a cached VLM response for this track ID,
        or None if a fresh VLM query is needed.
        """

        entry = self.vlm_cache.get(track_id)

        if entry is None:
            return None

        age = self._now - entry["time"]
        ttl = self.cache_ttl_s.get(entry["decision"], 0.0)

        if age >= ttl:

            logger.debug(
                f"AWARE cache: ID {track_id} expired "
                f"({entry['decision']}, age {age:.1f}s)."
            )

            return None

        if (
            entry["decision"] == "UNCERTAIN"
            and entry["box_area"] > 0
            and box_area
            >= entry["box_area"] * self.uncertain_area_growth
        ):

            logger.debug(
                f"AWARE cache: ID {track_id} box grew "
                f"({entry['box_area']:.0f} -> {box_area:.0f}), "
                "re-verifying."
            )

            return None

        response = entry["response"]

        if isinstance(response, str):

            response = response.replace(
                "Justification:",
                "Justification: [cached]",
                1,
            )

        return response

    def _store_in_cache(
        self,
        track_id: int,
        response: str,
        box_area: float,
    ) -> None:

        self.vlm_cache[track_id] = {
            "response": response,
            "decision": self._extract_decision(response),
            "time": self._now,
            "box_area": box_area,
            "last_seen": self._now,
        }

    def _update_and_prune_cache(
        self,
        active_track_ids,
    ) -> None:

        for tid in active_track_ids:

            if tid in self.vlm_cache:
                self.vlm_cache[tid]["last_seen"] = self._now

        stale = [
            tid
            for tid, entry in self.vlm_cache.items()
            if self._now - entry["last_seen"]
            > self.cache_forget_after_s
        ]

        for tid in stale:
            del self.vlm_cache[tid]

        if stale:

            logger.debug(
                f"AWARE cache: pruned stale IDs {stale}."
            )

    def clear_vlm_cache(self) -> None:

        self.vlm_cache.clear()

    # ========================================================
    # RE-DETECTION WHILE TRACKING (no VLM call)
    # ========================================================

    def redetect_target(
        self,
        image: Image,
        category: str,
        description: str,
        track_box,
        anchor_box,
        anchor_ground=None,
        max_ground_distance_m=None,
    ):
        """
        Run ONLY the person detector (Grounding DINO, no GPT) and
        check that the tracked target is still on a person who
        matches the description's colours.

        anchor_ground / max_ground_distance_m: the target's last
        verified ground position and how far they may have walked
        since. When given (and the drone position is known), the
        person is searched for by GROUND position, which does not
        shift when the drone moves or turns.

        Called by CloudTrack every few seconds while TRACKING, so
        the small tracker cannot drift onto the background
        unnoticed.

        Returns (box, how): see aware_candidates.verify_target.
        """

        if "//" in category:

            parts = category.split("//", 1)

            category = parts[0]

            if not description:
                description = parts[1]

        (
            _image_pil,
            _masks,
            boxes,
            scores,
        ) = self.detector.run_inference(
            image,
            prompt=category,
            mark_results=False,
        )

        anchor_distance = None

        if (
            anchor_ground is not None
            and max_ground_distance_m is not None
            and self.frame_pose is not None
        ):

            def anchor_distance(box):

                ground = self.ground_position(box)

                if ground is None:
                    return None

                d = distance_m(ground, anchor_ground)

                return d if d <= max_ground_distance_m else None

        return verify_target(
            image,
            boxes,
            scores,
            track_box,
            anchor_box,
            description,
            anchor_distance=anchor_distance,
        )

    # ========================================================
    # VLM RESPONSE PARSING
    # ========================================================

    def parse_vlm_response(self, reply):
        """
        Parse the AWARE VLM response.

        Expected format:

            Decision: MATCH/NO_MATCH/UNCERTAIN
            Justification: explanation
        """

        if not isinstance(reply, str):

            logger.warning(
                "VLM response is not a string. "
                "Treating response as UNCERTAIN."
            )

            return (
                "UNCERTAIN",
                "Invalid VLM response format.",
            )

        decision = "UNCERTAIN"
        justification = "No justification provided."

        for line in reply.splitlines():

            line = line.strip()

            if line.lower().startswith("decision:"):

                value = (
                    line.split(":", 1)[1]
                    .strip()
                    .upper()
                )

                if value in [
                    "MATCH",
                    "NO_MATCH",
                    "UNCERTAIN",
                ]:
                    decision = value

                else:

                    logger.warning(
                        f"Unknown VLM decision '{value}'. "
                        "Treating as UNCERTAIN."
                    )

                    decision = "UNCERTAIN"

                break

        for line in reply.splitlines():

            line = line.strip()

            if line.lower().startswith("justification:"):

                justification = (
                    line.split(":", 1)[1]
                    .strip()
                )

                break

        if decision == "MATCH":

            logger.info(
                "AWARE VLM decision: MATCH | "
                f"{justification}"
            )

        elif decision == "NO_MATCH":

            logger.info(
                "AWARE VLM decision: NO_MATCH | "
                f"{justification}"
            )

        else:

            logger.info(
                "AWARE VLM decision: UNCERTAIN | "
                f"{justification}"
            )

        return decision, justification

    def parse_vlm_response_list(
        self,
        vlm_responses: list[str],
    ):

        decisions = []
        justifications = []

        for response in vlm_responses:

            decision, justification = (
                self.parse_vlm_response(
                    response
                )
            )

            decisions.append(decision)
            justifications.append(
                justification
            )

        return decisions, justifications

    # ========================================================
    # DETECTOR + BYTETRACK + VLM
    # ========================================================

    def run_inference_inner(
        self,
        cathegory: str,
        verbal_description: str,
        image: Image,
    ):
        """
        Detect people, assign ByteTrack IDs, pick candidates,
        then verify them using the cache or the VLM.

        Returns:
            vlm_responses
            image_pil
            masks
            tracked_boxes
            tracked_scores
            track_ids
        """

        self.frame_index += 1

        # If there is no VLM, pass the full description
        # directly to the detector.
        if self.vlm is None:
            cathegory = verbal_description

        # ----------------------------------------------------
        # 1. GROUNDING DINO DETECTION
        # ----------------------------------------------------

        (
            image_pil,
            masks,
            boxes_filt,
            scores,
        ) = self.detector.run_inference(
            image,
            prompt=cathegory,
            mark_results=False,
        )

        # ----------------------------------------------------
        # 2. BYTE TRACK
        # ----------------------------------------------------

        tracked_people = (
            self.person_tracker.update(
                boxes_filt,
                scores,
            )
        )

        # ----------------------------------------------------
        # 3. CANDIDATE SELECTION (see aware_candidates.py)
        # ----------------------------------------------------

        tracked_people = select_candidates(
            image,
            boxes_filt,
            scores,
            tracked_people,
            verbal_description,
            max_candidates=self.max_candidates,
            exclude_track_ids=self.rejected_track_ids,
            exclude_box=(
                (lambda box: self._box_at_rejected_place(box, image))
                if len(self.rejected_places) > 0
                else None
            ),
        )

        if len(tracked_people) == 0:

            self.last_track_ids = []
            self._update_and_prune_cache([])

            return (
                [],
                image_pil,
                None,
                np.empty(
                    (0, 4),
                    dtype=np.float32,
                ),
                [],
                [],
            )

        # ----------------------------------------------------
        # Aligned arrays: tracked_boxes[i], tracked_scores[i],
        # track_ids[i] always refer to the same person.
        # ----------------------------------------------------

        tracked_boxes = []
        tracked_scores = []
        track_ids = []

        for person in tracked_people:

            tracked_boxes.append(
                np.asarray(
                    person["bbox"],
                    dtype=np.float32,
                )
            )

            confidence = person.get("confidence")

            if confidence is None:
                confidence = 1.0

            tracked_scores.append(float(confidence))

            track_ids.append(int(person["track_id"]))

        tracked_boxes = np.asarray(
            tracked_boxes,
            dtype=np.float32,
        )

        self.last_track_ids = list(track_ids)

        self._update_and_prune_cache(
            [t for t in track_ids if self._is_cacheable(t)]
        )

        prompt = verbal_description

        # One slot per candidate, filled below.
        vlm_responses = [None] * len(track_ids)

        # Candidates that need a real VLM call:
        # (index, track_id, cropped_image, box_area)
        pending = []

        num_cache_hits = 0

        # ----------------------------------------------------
        # 4. CACHE / VALIDITY PASS
        # ----------------------------------------------------

        for index, row in enumerate(tracked_boxes):

            track_id = int(track_ids[index])

            row = [int(x) for x in row]

            x1, y1, x2, y2 = row

            # Safety net: select_candidates already excludes
            # rejected IDs.
            if track_id in self.rejected_track_ids:

                vlm_responses[index] = RESPONSE_REJECTED
                continue

            if x2 <= x1 or y2 <= y1:

                logger.warning(
                    f"Skipping degenerate box {row} "
                    f"for track ID {track_id}."
                )

                vlm_responses[index] = RESPONSE_INVALID_BOX
                continue

            # Area BEFORE overscan, so growth comparisons
            # reflect the person's real size.
            box_area = float((x2 - x1) * (y2 - y1))

            if self.verification_error is not None:
                vlm_responses[index] = RESPONSE_VERIFICATION_OFF
                continue

            if self._is_cacheable(track_id):

                cached = self._get_cached_response(
                    track_id,
                    box_area,
                )

                if cached is not None:

                    logger.info(
                        "AWARE: using cached VLM result "
                        f"for track ID {track_id}."
                    )

                    vlm_responses[index] = cached
                    num_cache_hits += 1
                    continue

            if self.enable_overscan:

                x1 = max(0, x1 - self.overscan_value)
                y1 = max(0, y1 - self.overscan_value)
                x2 = min(image.width, x2 + self.overscan_value)
                y2 = min(image.height, y2 + self.overscan_value)

            if x2 <= x1 or y2 <= y1:

                logger.warning(
                    "Skipping degenerate box after "
                    "overscan clamp for "
                    f"track ID {track_id}: "
                    f"({x1}, {y1}, {x2}, {y2})."
                )

                vlm_responses[index] = (
                    RESPONSE_INVALID_BOX_OVERSCAN
                )
                continue

            if self.vlm is None:

                vlm_responses[index] = RESPONSE_DETECTOR_ONLY
                continue

            if not self.debug_enable_gpt:

                raise NotImplementedError(
                    "GPTFourWrapper is disabled "
                    "to save tokens during debug."
                )

            cropped_image = image.crop((x1, y1, x2, y2))

            pending.append(
                (index, track_id, cropped_image, box_area)
            )

        # ----------------------------------------------------
        # 5. VLM BUDGET
        # ----------------------------------------------------
        #
        # Candidates are already ranked best-first by
        # select_candidates, so the best uncached ones get
        # the VLM calls. The rest are marked UNCERTAIN for
        # this frame (not cached, so they are retried later).
        # ----------------------------------------------------

        to_call = pending[: self.max_vlm_calls_per_frame]
        deferred = pending[self.max_vlm_calls_per_frame:]

        for index, track_id, _, _ in deferred:

            logger.info(
                f"AWARE: deferring track ID {track_id} "
                "(VLM budget reached this frame)."
            )

            vlm_responses[index] = RESPONSE_DEFERRED

        # ----------------------------------------------------
        # 6. PARALLEL VLM CALLS WITH TIMEOUT
        # ----------------------------------------------------

        futures = []

        for index, track_id, cropped_image, box_area in to_call:

            ground = self.ground_position(
                tracked_boxes[index]
            )

            ground_text = (
                f" (ground ~ lat {ground[0]:.7f}, "
                f"lon {ground[1]:.7f})"
                if ground is not None
                else ""
            )

            logger.info(
                f"AWARE: detected {cathegory} with track ID "
                f"{track_id} at "
                f"{[int(v) for v in tracked_boxes[index]]}"
                f"{ground_text}. "
                "Running VLM verification."
            )

            future = self._vlm_executor.submit(
                self.vlm.run_inference,
                prompt,
                cropped_image,
            )

            futures.append(
                (index, track_id, box_area, future)
            )

        num_vlm_calls = 0

        # One shared deadline for all calls in this frame
        # (real time, not sim time).
        deadline = time.monotonic() + self.vlm_timeout_s

        for index, track_id, box_area, future in futures:

            if self.verification_error is not None:
                future.cancel()
                vlm_responses[index] = RESPONSE_VERIFICATION_OFF
                continue

            remaining = max(0.0, deadline - time.monotonic())

            try:

                vlm_response = future.result(
                    timeout=remaining
                )

                num_vlm_calls += 1

                if self._is_cacheable(track_id):

                    self._store_in_cache(
                        track_id,
                        vlm_response,
                        box_area,
                    )

            except VlmInsufficientQuotaError:

                if self.verification_error is None:
                    self.verification_error = NO_CREDITS_MESSAGE
                    logger.error(NO_CREDITS_MESSAGE)
                    self.clear_vlm_cache()
                # Cancel queued calls; requests already in flight cannot be recalled.
                for _, _, _, pending_future in futures:
                    pending_future.cancel()
                vlm_response = RESPONSE_VERIFICATION_OFF

            except FutureTimeoutError:

                future.cancel()

                logger.warning(
                    f"AWARE: VLM call for track ID "
                    f"{track_id} timed out after "
                    f"{self.vlm_timeout_s:.0f}s."
                )

                vlm_response = RESPONSE_TIMEOUT

            except Exception as e:

                logger.warning(
                    "AWARE: VLM call failed for "
                    f"track ID {track_id}: {e}"
                )

                vlm_response = RESPONSE_FAILED

            logger.info(
                "AWARE raw VLM response for "
                f"track ID {track_id}: "
                f"{vlm_response}"
            )

            vlm_responses[index] = vlm_response

        if self.verification_error is not None:
            # Do not emit a new MATCH from cached or concurrent results on this frame.
            vlm_responses = [RESPONSE_VERIFICATION_OFF] * len(track_ids)

        logger.info(
            f"AWARE frame {self.frame_index}: "
            f"{len(track_ids)} candidate(s), "
            f"{num_vlm_calls} VLM call(s), "
            f"{num_cache_hits} cache hit(s), "
            f"{len(deferred)} deferred."
        )

        return (
            vlm_responses,
            image_pil,
            masks,
            tracked_boxes,
            tracked_scores,
            track_ids,
        )

    # ========================================================
    # MAIN INFERENCE
    # ========================================================

    def run_inference(
        self,
        image: Image,
        category: str,
        description: str = None,
        mark_results=False,
        filter_results=True,
        timestamp: float = None,
    ):
        """
        Run the complete AWARE candidate-search pipeline.

        timestamp:
            Optional frame time in seconds (e.g. ROS image
            header stamp / Gazebo sim time, or video frame
            time). If None, wall-clock time is used. Only used
            for cache expiry.

        Only MATCH candidates are returned to CloudTrack.
        With filter_results=True this still returns the same
        five values as before. Track IDs of the returned boxes
        are in self.last_match_track_ids.
        """

        self._now = (
            float(timestamp)
            if timestamp is not None
            else time.monotonic()
        )

        if not description:

            description = (
                f"Find a {category} matching "
                "the user's description."
            )

        if "//" in category:

            parts = category.split("//", 1)

            category = parts[0]
            description = parts[1]

        (
            vlm_responses,
            image_pil,
            masks,
            boxes_filt,
            scores,
            track_ids,
        ) = self.run_inference_inner(
            category,
            description,
            image,
        )

        # ----------------------------------------------------
        # NOTHING TRACKED IN THIS FRAME
        # ----------------------------------------------------

        if len(track_ids) == 0:

            self.last_match_track_ids = []

            if filter_results:

                return (
                    image_pil,
                    None,
                    None,
                    None,
                    None,
                )

            return (
                image_pil,
                None,
                boxes_filt,
                scores,
                [],
                [],
                [],
            )

        decisions, justifications = (
            self.parse_vlm_response_list(
                vlm_responses
            )
        )

        # ----------------------------------------------------
        # VISUALIZATION LABELS
        # ----------------------------------------------------

        labels = []

        for i, decision in enumerate(decisions):

            track_id = track_ids[i]

            if decision == "MATCH":
                tag = "MATCH"
            elif decision == "UNCERTAIN":
                tag = "UNCERTAIN"
            else:
                tag = "NO MATCH"

            labels.append(
                f"ID {track_id} | "
                f"{category} ({tag}) | "
                f"{justifications[i]}"
            )

        if mark_results:

            image_pil = self.visualize(
                image_pil,
                torch.as_tensor(boxes_filt),
                labels=labels,
            )

        # ====================================================
        # FILTER RESULTS
        # ====================================================

        if filter_results:

            keep_idx = [
                i
                for i, decision in enumerate(decisions)
                if decision == "MATCH"
            ]

            if len(keep_idx) == 0:

                self.last_match_track_ids = []

                logger.info(
                    "AWARE: No MATCH candidate "
                    "in this frame. Continuing search."
                )

                return (
                    image_pil,
                    None,
                    None,
                    None,
                    None,
                )

            boxes_filt = torch.as_tensor(
                boxes_filt,
                dtype=torch.float32,
            )[keep_idx]

            scores = [scores[i] for i in keep_idx]

            justifications = [
                justifications[i] for i in keep_idx
            ]

            match_track_ids = [
                int(track_ids[i]) for i in keep_idx
            ]

            self.last_match_track_ids = match_track_ids

            masks = None

            logger.info(
                f"AWARE: {len(keep_idx)} matching "
                "candidate(s) found. "
                f"Track IDs: {match_track_ids}"
            )

            # Keep existing 5-value return signature so
            # cloud_track.py keeps working.
            return (
                image_pil,
                masks,
                boxes_filt,
                scores,
                justifications,
            )

        # ====================================================
        # DEBUG / UNFILTERED RETURN
        # ====================================================

        return (
            image_pil,
            masks,
            boxes_filt,
            scores,
            decisions,
            labels,
            justifications,
        )


# ============================================================
# AWARE VLM SYSTEM PROMPT
# ============================================================


def system_prompt_from_description(
    description: str,
):
    """
    Generate the AWARE system prompt used to verify whether
    a detected person matches the user-provided description.
    """

    prompt = f"""
You are the visual verification module of AWARE, an autonomous
missing-person search system.

Your task is to determine whether the detected PERSON in the provided
image matches the missing-person description given by the user.

Missing-person description:
{description}

Evaluate ONLY visible evidence in the image.

Rules:

1. Compare the detected person with ALL observable attributes in the
   missing-person description, such as clothing, hair, glasses, age
   appearance, gender presentation, or other visual characteristics.

2. Do NOT assume an attribute is present if it cannot be clearly seen.

3. Return MATCH only when the visible characteristics provide strong
   evidence that this person matches the description.

4. Return NO_MATCH when one or more clearly visible characteristics
   contradict the description.

5. Return UNCERTAIN when important characteristics cannot be verified,
   for example because:
   - the person is partially occluded,
   - the image is blurry,
   - the viewing angle hides important features,
   - the person is too far away,
   - or there is not enough visible evidence.

6. A partial resemblance is NOT enough for MATCH.

7. Do not identify the person by name or infer identity.
   Only compare visible characteristics with the supplied description.

8. If the candidate is UNCERTAIN, do not treat them as the target.
   The search should continue until stronger visual evidence is available.

Return EXACTLY this format:

Decision: MATCH/NO_MATCH/UNCERTAIN
Justification: <brief explanation based only on visible evidence>
"""

    return prompt


# ============================================================
# LOCAL IMAGE TEST
# ============================================================


if __name__ == "__main__":

    output_folder = Path(
        "~/Downloads/flextrack_gpt_output"
    ).expanduser()

    output_folder.mkdir(exist_ok=True)

    # Retained from the original project structure.
    # Normal AWARE video testing is performed through
    # pipeline_runner.py.

    opject_cathegory = "person"

    description = (
        "You are looking for a person with a gray shirt, "
        "who is missing after being injured."
    )

    search_and_rescue_desctiption = (
        "You are on a search and rescue mission. "
        f"{description} "
        "Determine whether the detected person matches "
        "the description."
    )

    image_folder = Path(__file__).parent / "images"

    answers = []

    # Local debugging placeholder.


# ============================================================
# DETECTOR FACTORY
# ============================================================


def get_detector(
    detector_model,
):

    box_threshold = 0.5

    detector_model = detector_model.lower()

    if "sam" in detector_model:

        if "hq" in detector_model:

            use_sam_hq = True

            if GroundingDinoWrapper is None:

                raise ImportError(
                    "GroundingDinoWrapper is unavailable."
                )

            detector = GroundingDinoWrapper(
                box_threshold=box_threshold,
                use_sam_hq=use_sam_hq,
            )

        elif "lq" in detector_model:

            # Lower thresholds are useful here because
            # Grounding DINO proposes people and the VLM
            # performs the final visual verification.

            box_threshold = 0.1
            text_threshold = 0.05

            detector = GroundingDinoHuggingfaceWrapper(
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

        else:

            raise ValueError(
                f"Unknown SAM model {detector_model}."
            )

    elif "glee" in detector_model:

        split = detector_model.split("_")

        if len(split) == 1:

            raise ValueError(
                f"Unknown GLEE model {detector_model}: "
                "Please specify model name like "
                "GLEE_[lite/plus/pro]."
            )

        else:

            from .glee_wrapper import GLEEWrapper

            detector = GLEEWrapper(
                model_name=split[1],
                box_threshold=box_threshold,
            )

    else:

        return None

    return detector


# ============================================================
# VLM FACTORY
# ============================================================


def get_vlm(
    vl_model,
    system_description,
    simulate_time_delay,
):

    if "gpt" in vl_model:

        system_prompt = system_prompt_from_description(
            system_description
        )

        vlm = GPTFourWrapper(
            enable_caching=False,
            simulate_time_delay=simulate_time_delay,
            model=vl_model,
            system_prompt=system_prompt,
            cache_file_name="sard_single_shot_cache.json",
            image_detail="low",
        )

    elif "paligemma" in vl_model:

        vlm = PaligemmaWrapper(
            system_prompt=system_description
        )

    elif "llava" in vl_model:

        system_prompt = system_description

        vlm = LlavaWrapper(
            system_prompt=system_prompt,
            model_name=vl_model,
        )

    else:

        vlm = None

    return vlm


# ============================================================
# COMPLETE PIPELINE FACTORY
# ============================================================


def get_vlm_pipeline(
    vl_model_name: str,
    system_description: str,
    simulate_time_delay: bool,
    detector_name: str,
    openai_api_key: str = None,
    detector=None,
):
    """
    Create the detector + ByteTrack + VLM pipeline used by AWARE.
    """

    vlm = get_vlm(
        vl_model_name,
        system_description,
        simulate_time_delay,
    )

    if detector is None:
        detector = get_detector(detector_name)

    if detector is None:

        raise ValueError(
            f"Unknown model {detector_name} - "
            "cannot run without detector."
        )

    if vlm is None:

        logger.warning(
            f"No VLM or unknown name "
            f"{vl_model_name} specified. "
            "Running detector-only fallback."
        )

    model = DetectorVlmPipeline(
        vlm,
        detector,
        overscan_value=20,
    )

    return model