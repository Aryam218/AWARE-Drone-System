from pathlib import Path
import io

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from cloud_track.foundation_model_wrappers.aware_candidates import select_candidates
from cloud_track.foundation_model_wrappers.wrapper_base import WrapperBase
from cloud_track.tracker_wrapper.bytetrack_wrapper import ByteTrackWrapper

from .gpt_four_wrapper import GPTFourWrapper
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
        Persistent person IDs
          ↓
        VLM verification
          ↓
        MATCH / NO_MATCH / UNCERTAIN

    Only MATCH detections are returned to CloudTrack.

    ByteTrack IDs are kept internally so the higher-level
    AWARE search controller can later use them for:

        - candidate confirmation
        - candidate rejection
        - avoiding repeated rejected candidates
        - consistent person tracking
    """

    def __init__(
        self,
        vlm: WrapperBase,
        detector: WrapperBase,
        enable_overscan=True,
        overscan_value=50,
    ):
        self.vlm = vlm
        self.detector = detector

        self.enable_overscan = enable_overscan
        self.overscan_value = overscan_value

        self.debug_enable_gpt = True

        # ----------------------------------------------------
        # SHARED PERSON TRACKER
        # ----------------------------------------------------
        #
        # Grounding DINO detects people independently in each
        # frame. ByteTrack connects those detections across
        # frames and assigns persistent IDs.
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

        # Every track ID returned by ByteTrack in the
        # most recently processed frame.
        self.last_track_ids = []

        # Track IDs corresponding ONLY to detections that
        # passed VLM verification as MATCH.
        #
        # This list is kept in exactly the same order as the
        # filtered boxes returned from run_inference().
        self.last_match_track_ids = []

        # IDs explicitly rejected by the operator.
        #
        # pipeline_runner.py will use reject_track_id()
        # in the next integration step.
        self.rejected_track_ids = set()

    # ========================================================
    # BYTE TRACK STATE
    # ========================================================

    def reject_track_id(self, track_id: int) -> None:
        """
        Remember a person that the operator explicitly rejected.

        Once rejected, the same ByteTrack ID will not be sent
        to the VLM again during the same search session.
        """

        if track_id is None:
            return

        track_id = int(track_id)

        self.rejected_track_ids.add(track_id)

        logger.info(
            f"AWARE: track ID {track_id} marked as rejected."
        )

    def clear_rejected_track_ids(self) -> None:
        """
        Clear the rejected-person memory.

        Useful when beginning a completely new missing-person
        search.
        """

        self.rejected_track_ids.clear()

    def reset_person_tracker(self) -> None:
        """
        Reset ByteTrack and all ID-related state.

        This should be used only when starting a completely
        new search session, not when rejecting one candidate.
        """

        self.person_tracker.reset()

        self.last_track_ids = []
        self.last_match_track_ids = []
        self.rejected_track_ids.clear()

        logger.info(
            "AWARE: ByteTrack person-tracking state reset."
        )

    def get_last_match_track_ids(self):
        """
        Return the ByteTrack IDs corresponding to the latest
        MATCH boxes returned by run_inference().
        """

        return list(self.last_match_track_ids)

    # ========================================================
    # VLM RESPONSE PARSING
    # ========================================================

    def parse_vlm_response(self, reply):
        """
        Parse the AWARE VLM response.

        Expected format:

            Decision: MATCH/NO_MATCH/UNCERTAIN
            Justification: explanation

        Returns:
            decision:
                MATCH, NO_MATCH, or UNCERTAIN

            justification:
                Explanation returned by the VLM.
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

        # ----------------------------------------------------
        # Parse decision
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Parse justification
        # ----------------------------------------------------

        for line in reply.splitlines():

            line = line.strip()

            if line.lower().startswith("justification:"):

                justification = (
                    line.split(":", 1)[1]
                    .strip()
                )

                break

        # ----------------------------------------------------
        # Debug logging
        # ----------------------------------------------------

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
        """
        Parse multiple AWARE VLM responses.

        Returns:
            decisions:
                MATCH / NO_MATCH / UNCERTAIN

            justifications:
                Explanation for every result.
        """

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
        Detect people, assign ByteTrack IDs, then verify each
        tracked person using the VLM.

        Grounding DINO:
            detects possible people.

        ByteTrack:
            gives each detected person a persistent track ID.

        VLM:
            evaluates the tracked person against the user's
            missing-person description.

        Returns:
            vlm_responses
            image_pil
            masks
            tracked_boxes
            tracked_scores
            track_ids
        """

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
        
        # AWARE: choose WHICH people the VLM checks (see aware_candidates.py)
        tracked_people = select_candidates(
            image,
            boxes_filt,
            scores,
            tracked_people,
            verbal_description,
        )

        # Nothing currently tracked/detected.
        if len(tracked_people) == 0:

            self.last_track_ids = []

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
        # Convert ByteTrack results into aligned arrays.
        #
        # Everything below uses these arrays, so:
        #
        # tracked_boxes[i]
        # tracked_scores[i]
        # track_ids[i]
        #
        # always refer to the same person.
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

            confidence = person.get(
                "confidence"
            )

            if confidence is None:
                confidence = 1.0

            tracked_scores.append(
                float(confidence)
            )

            track_ids.append(
                int(person["track_id"])
            )

        tracked_boxes = np.asarray(
            tracked_boxes,
            dtype=np.float32,
        )

        self.last_track_ids = list(
            track_ids
        )

        prompt = verbal_description

        vlm_responses = []

        # ----------------------------------------------------
        # 3. VERIFY EACH TRACKED PERSON
        # ----------------------------------------------------

        for index, row in enumerate(
            tracked_boxes
        ):

            track_id = int(
                track_ids[index]
            )

            row = [
                int(x)
                for x in row
            ]

            x1, y1, x2, y2 = row

            # ------------------------------------------------
            # REJECTED PERSON
            # ------------------------------------------------
            #
            # If the user previously rejected this specific
            # ByteTrack ID, do not waste another VLM call and
            # do not offer the same tracked person again.
            # ------------------------------------------------

            if track_id in self.rejected_track_ids:

                logger.info(
                    "AWARE: skipping previously rejected "
                    f"track ID {track_id}."
                )

                vlm_responses.append(
                    """
Decision: NO_MATCH
Justification: This tracked person was previously rejected by the operator.
"""
                )

                continue

            # ------------------------------------------------
            # INVALID DETECTION BOX
            # ------------------------------------------------

            if x2 <= x1 or y2 <= y1:

                logger.warning(
                    f"Skipping degenerate box {row} "
                    f"for track ID {track_id}."
                )

                vlm_responses.append(
                    """
Decision: NO_MATCH
Justification: Skipped invalid detection box.
"""
                )

                continue

            # ------------------------------------------------
            # ADD CONTEXT AROUND PERSON
            # ------------------------------------------------

            if self.enable_overscan:

                x1 = max(
                    0,
                    x1 - self.overscan_value,
                )

                y1 = max(
                    0,
                    y1 - self.overscan_value,
                )

                x2 = min(
                    image.width,
                    x2 + self.overscan_value,
                )

                y2 = min(
                    image.height,
                    y2 + self.overscan_value,
                )

            # ------------------------------------------------
            # VALIDATE AGAIN AFTER OVERSCAN
            # ------------------------------------------------

            if x2 <= x1 or y2 <= y1:

                logger.warning(
                    "Skipping degenerate box after "
                    "overscan clamp for "
                    f"track ID {track_id}: "
                    f"({x1}, {y1}, {x2}, {y2})."
                )

                vlm_responses.append(
                    """
Decision: NO_MATCH
Justification: Skipped invalid detection box after overscan.
"""
                )

                continue

            # ------------------------------------------------
            # CROP TRACKED PERSON
            # ------------------------------------------------

            cropped_image = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            logger.info(
                f"AWARE: detected {cathegory} "
                f"with track ID {track_id} "
                f"at {row}. "
                "Running VLM verification."
            )

            if not self.debug_enable_gpt:

                raise NotImplementedError(
                    "GPTFourWrapper is disabled "
                    "to save tokens during debug."
                )

            logger.info(
                "AWARE verification prompt for "
                f"track ID {track_id}: "
                f"{prompt}"
            )

            # ------------------------------------------------
            # VLM VERIFICATION
            # ------------------------------------------------

            if self.vlm is not None:

                vlm_response = (
                    self.vlm.run_inference(
                        prompt,
                        cropped_image,
                    )
                )

            else:

                # Detector-only fallback.
                vlm_response = """
Decision: MATCH
Justification: Detector-only mode; no VLM verification available.
"""

            logger.info(
                "AWARE raw VLM response for "
                f"track ID {track_id}: "
                f"{vlm_response}"
            )

            vlm_responses.append(
                vlm_response
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
    ):
        """
        Run the complete AWARE candidate-search pipeline.

        Flow:

            Frame
              ↓
            Grounding DINO
              ↓
            Candidate people
              ↓
            ByteTrack IDs
              ↓
            VLM verification
              ↓
            MATCH / NO_MATCH / UNCERTAIN

        Only MATCH candidates are returned to CloudTrack.

        NO_MATCH:
            Ignore candidate and continue searching.

        UNCERTAIN:
            Do not initialize target tracking.

        MATCH:
            Candidate may initialize CloudTrack.

        IMPORTANT:
            To remain compatible with the current CloudTrack
            implementation, this function still returns the
            same five values as before when filter_results=True.

            ByteTrack IDs corresponding to the returned boxes
            are stored in:

                self.last_match_track_ids

            The next integration step will connect those IDs
            to CloudTrack and pipeline_runner.
        """

        # ----------------------------------------------------
        # DEFAULT DESCRIPTION
        # ----------------------------------------------------

        if not description:

            description = (
                f"Find a {category} matching "
                "the user's description."
            )

        # ----------------------------------------------------
        # CLOUDTRACK CATEGORY//DESCRIPTION COMPATIBILITY
        # ----------------------------------------------------

        if "//" in category:

            parts = category.split(
                "//",
                1,
            )

            category = parts[0]
            description = parts[1]

        # ----------------------------------------------------
        # DETECTOR + BYTETRACK + VLM
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RAW VLM RESPONSE -> DECISION
        # ----------------------------------------------------

        decisions, justifications = (
            self.parse_vlm_response_list(
                vlm_responses
            )
        )

        # ----------------------------------------------------
        # VISUALIZATION LABELS
        # ----------------------------------------------------

        labels = []

        for i, decision in enumerate(
            decisions
        ):

            track_id = track_ids[i]

            if decision == "MATCH":

                labels.append(
                    f"ID {track_id} | "
                    f"{category} (MATCH) | "
                    f"{justifications[i]}"
                )

            elif decision == "UNCERTAIN":

                labels.append(
                    f"ID {track_id} | "
                    f"{category} (UNCERTAIN) | "
                    f"{justifications[i]}"
                )

            else:

                labels.append(
                    f"ID {track_id} | "
                    f"{category} (NO MATCH) | "
                    f"{justifications[i]}"
                )

        # ----------------------------------------------------
        # OPTIONAL VISUALIZATION
        # ----------------------------------------------------

        if mark_results:

            image_pil = self.visualize(
                image_pil,
                torch.as_tensor(
                    boxes_filt
                ),
                labels=labels,
            )

        # ====================================================
        # FILTER RESULTS
        # ====================================================

        if filter_results:

            # Only VLM MATCH is allowed to become candidate.
            keep_idx = [
                i
                for i, decision
                in enumerate(decisions)
                if decision == "MATCH"
            ]

            # ------------------------------------------------
            # NO MATCH
            # ------------------------------------------------

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

            # ------------------------------------------------
            # KEEP MATCHES
            # ------------------------------------------------

            boxes_filt = torch.as_tensor(
                boxes_filt,
                dtype=torch.float32,
            )[keep_idx]

            scores = [
                scores[i]
                for i in keep_idx
            ]

            justifications = [
                justifications[i]
                for i in keep_idx
            ]

            match_track_ids = [
                int(track_ids[i])
                for i in keep_idx
            ]

            # This ordering corresponds exactly to boxes_filt.
            self.last_match_track_ids = (
                match_track_ids
            )

            masks = None

            logger.info(
                f"AWARE: {len(keep_idx)} matching "
                "candidate(s) found. "
                f"Track IDs: {match_track_ids}"
            )

            # IMPORTANT:
            # Keep existing 5-value return signature so
            # cloud_track.py keeps working right now.
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
        #
        # Keep the old return shape for compatibility.
        # Track IDs can be read through self.last_track_ids.
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

    output_folder.mkdir(
        exist_ok=True
    )

    # This section is retained from the original
    # project structure.
    #
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

    image_folder = (
        Path(__file__).parent / "images"
    )

    answers = []

    # Local debugging placeholder.


# ============================================================
# DETECTOR FACTORY
# ============================================================


def get_detector(
    detector_model,
):

    box_threshold = 0.5

    detector_model = (
        detector_model.lower()
    )

    # --------------------------------------------------------
    # GROUNDING DINO / SAM
    # --------------------------------------------------------

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

            detector = (
                GroundingDinoHuggingfaceWrapper(
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            )

        else:

            raise ValueError(
                f"Unknown SAM model "
                f"{detector_model}."
            )

    # --------------------------------------------------------
    # GLEE
    # --------------------------------------------------------

    elif "glee" in detector_model:

        split = detector_model.split("_")

        if len(split) == 1:

            raise ValueError(
                f"Unknown GLEE model "
                f"{detector_model}: "
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

        system_prompt = (
            system_prompt_from_description(
                system_description
            )
        )

        vlm = GPTFourWrapper(
            enable_caching=False,
            simulate_time_delay=(
                simulate_time_delay
            ),
            model=vl_model,
            system_prompt=system_prompt,
            cache_file_name=(
                "sard_single_shot_cache.json"
            ),
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
):
    """
    Create the detector + ByteTrack + VLM pipeline used by AWARE.

    Detector:
        Grounding DINO finds candidate persons.

    ByteTrack:
        assigns persistent IDs to detected persons.

    VLM:
        evaluates each tracked person against the
        missing-person description.

    CloudTrack:
        receives only MATCH candidates.
    """

    vlm = get_vlm(
        vl_model_name,
        system_description,
        simulate_time_delay,
    )

    detector = get_detector(
        detector_name
    )

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
