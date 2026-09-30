import numpy as np
import supervision as sv


class ByteTrackWrapper:
    """
    Shared ByteTrack wrapper for AWARE.

    Used by:
    - missing-person search
    - crowd analytics

    Input:
        detection boxes in xyxy format
        confidence scores

    Output:
        stable person track IDs across frames
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.15,
        lost_track_buffer: int = 60,
        minimum_matching_threshold: float = 0.7,
        frame_rate: int = 30,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(
        self,
        boxes,
        scores=None,
    ):
        """
        Update ByteTrack using detections from the current frame.

        Args:
            boxes:
                Person bounding boxes in xyxy format.

            scores:
                Detection confidence scores.

        Returns:
            list of dictionaries:

            [
                {
                    "track_id": 1,
                    "bbox": np.array([x1, y1, x2, y2]),
                    "confidence": 0.91,
                },
                ...
            ]
        """

        # ------------------------------------------------------
        # NO DETECTIONS
        # ------------------------------------------------------

        if boxes is None or len(boxes) == 0:

            empty_detections = sv.Detections(
                xyxy=np.empty(
                    (0, 4),
                    dtype=np.float32,
                ),
                confidence=np.empty(
                    (0,),
                    dtype=np.float32,
                ),
            )

            self.tracker.update_with_detections(
                empty_detections
            )

            return []

        # ------------------------------------------------------
        # NORMALIZE BOXES
        # ------------------------------------------------------

        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()

        boxes = np.asarray(
            boxes,
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # NORMALIZE SCORES
        # ------------------------------------------------------

        if scores is None:

            scores = np.ones(
                len(boxes),
                dtype=np.float32,
            )

        else:

            if hasattr(scores, "cpu"):
                scores = scores.cpu().numpy()

            scores = np.asarray(
                scores,
                dtype=np.float32,
            )

        # ------------------------------------------------------
        # CREATE SUPERVISION DETECTIONS
        # ------------------------------------------------------

        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
        )

        # ------------------------------------------------------
        # BYTE TRACK
        # ------------------------------------------------------

        tracked = self.tracker.update_with_detections(
            detections
        )

        results = []

        # ------------------------------------------------------
        # RETURN TRACK IDS
        # ------------------------------------------------------

        if (
            tracked.tracker_id is None
            or len(tracked) == 0
        ):
            return results

        for i in range(len(tracked)):

            track_id = int(
                tracked.tracker_id[i]
            )

            bbox = np.asarray(
                tracked.xyxy[i],
                dtype=np.float32,
            )

            confidence = None

            if tracked.confidence is not None:
                confidence = float(
                    tracked.confidence[i]
                )

            results.append(
                {
                    "track_id": track_id,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

        return results

    def reset(self):
        """
        Reset ByteTrack and remove all stored track history.
        """

        self.tracker.reset()