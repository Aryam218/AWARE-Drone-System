"""
Wrapper around the OpenCV implementation of the tracker classes.
"""

import cv2
import numpy as np
import torch
from loguru import logger

from .cv_models import (
    dasiamrpn_tracker_factory,
    goturn_tracker_factory,
    nano_tracker_factory,
    vit_tracker_factory,
)


class OpenCVWrapper:
    def __init__(
        self, tracker_type: str = "nano", reinit_threshold: float = 0.5
    ):
        self.tracker_type = tracker_type

        OPENCV_OBJECT_TRACKERS = {
            # Disabled since not supported in newer OpenCV versions.
            # "csrt": cv2.TrackerCSRT_create,
            "daSiamRpn": dasiamrpn_tracker_factory,
            "goturn": goturn_tracker_factory,
            # "kcf": cv2.TrackerKCF_create,
            # "mil": cv2.TrackerMIL_create,
            "nano": nano_tracker_factory,
            "vit": vit_tracker_factory,
        }

        if self.tracker_type not in OPENCV_OBJECT_TRACKERS:
            raise ValueError(
                f"Tracker type {self.tracker_type} not supported."
            )

        self.tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]()
        self.reinit_threshold = reinit_threshold

    def init(self, image: np.ndarray, bbox):
        """
        Initialize the tracker with the first frame and bounding box.

        image:
            NumPy array containing the frame.

        bbox:
            Bounding box in xyxy format:
            (x1, y1, x2, y2)

            Supports:
            - torch.Tensor
            - numpy.ndarray
            - list / tuple
        """

        # ----------------------------------------------------------
        # Normalize bbox to NumPy.
        # The original CloudTrack implementation expected a
        # torch.Tensor, but AWARE may pass a NumPy array.
        # ----------------------------------------------------------
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()
        else:
            bbox = np.asarray(bbox)

        # Flatten in case bbox has shape such as (1, 4).
        bbox = bbox.squeeze()

        if bbox.size != 4:
            raise ValueError(
                f"Expected bbox with 4 values (x1, y1, x2, y2), "
                f"but received shape {bbox.shape}."
            )

        # Convert coordinates to float first.
        x1, y1, x2, y2 = [float(v) for v in bbox]

        # ----------------------------------------------------------
        # Clamp coordinates to image boundaries.
        # ----------------------------------------------------------
        height, width = image.shape[:2]

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box after clamping: "
                f"({x1}, {y1}, {x2}, {y2})"
            )

        # ----------------------------------------------------------
        # OpenCV trackers expect:
        # (xmin, ymin, width, height)
        # ----------------------------------------------------------
        tracker_bbox = (
            int(round(x1)),
            int(round(y1)),
            int(round(x2 - x1)),
            int(round(y2 - y1)),
        )

        logger.debug(
            f"Re-Initializing tracker with bbox: {tracker_bbox}"
        )

        self.tracker.init(image, tracker_bbox)

    def update(self, image):
        """
        Update the tracker with the next frame.

        Returns:
            success:
                Whether tracking is still valid.

            new_box:
                Bounding box in xyxy format.

            score:
                Tracker confidence score when supported.
        """

        tracking_success, new_box = self.tracker.update(image)

        # Convert (xmin, ymin, width, height) -> xyxy.
        new_box = (
            int(new_box[0]),
            int(new_box[1]),
            int(new_box[0] + new_box[2]),
            int(new_box[1] + new_box[3]),
        )

        tracking_quality, score = self.__tracking_quality()

        success = tracking_success and tracking_quality

        return success, new_box, score

    def __tracking_quality(self):
        """
        Check whether tracking quality is still sufficient.

        Returns:
            success:
                True if tracking quality is acceptable.

            score:
                Tracking confidence score.
        """

        supported_trackers = ["nano", "daSiamRpn", "vit"]

        if self.tracker_type not in supported_trackers:
            logger.warning(
                "Tracker type does not support tracking score calculation."
            )

            # Keep compatibility with update(), which expects
            # both success and score.
            return True, None

        score = self.tracker.getTrackingScore()
        success = score > self.reinit_threshold

        if not success:
            logger.info(
                f"Tracking failed with score {score}. "
                "Reinitializing tracker."
            )

        return success, score

    def shutdown(self):
        """
        Shutdown the tracker.
        """
        pass

    def get_Timings(self):
        """
        Get tracker timings.
        """
        pass


if __name__ == "__main__":
    tracker = OpenCVWrapper()
    print("OpenCVWrapper initialized.")