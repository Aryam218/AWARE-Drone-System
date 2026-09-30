import time

import cv2
import numpy as np
from PIL import Image

from cloud_track.foundation_model_wrappers.grounding_dino_huggingface_wrapper import (
    GroundingDinoHuggingfaceWrapper,
)
from cloud_track.tracker_wrapper.bytetrack_wrapper import (
    ByteTrackWrapper,
)


class CrowdAnalytics:
    """
    AWARE Crowd Analytics

    Pipeline:
        Camera / Gazebo / Video
                ↓
        Resize for inference
                ↓
        Grounding DINO
                ↓
        Person detections
                ↓
        Scale boxes to original frame
                ↓
        NMS
                ↓
        ByteTrack
                ↓
        Track IDs
                ↓
        AWARE identity continuity
                ↓
        Zone / Booth analytics

    Important design:

    1. CURRENT PEOPLE:
       Count ONLY tracks returned by ByteTrack in the current
       analytics frame.

       Old/stale tracks are NOT counted as current occupants.

    2. TRACK MEMORY:
       Old tracks are kept briefly only for:
           - ID-switch recovery
           - dwell continuity
           - footfall continuity

    3. FOOTFALL:
       Uses an AWARE logical person ID instead of blindly using
       every raw ByteTrack ID.

       If ByteTrack changes an ID for a person who reappears
       almost immediately in nearly the same location, the new
       ByteTrack ID inherits the old logical person ID.

    This prevents:
        3 real people -> temporary 4 people
    and reduces:
        one person -> multiple false footfall visitors
    """

    def __init__(
        self,
        zones=None,
        zone_capacities=None,
        crowd_alert_threshold=0.80,
        stale_track_seconds=2.0,
        inference_width=1280,
        tracker_frame_rate=5,
        id_switch_max_seconds=1.0,
        id_switch_max_distance_ratio=0.08,
    ):
        # ====================================================
        # PERSON DETECTOR
        # ====================================================

        self.detector = GroundingDinoHuggingfaceWrapper(
            box_threshold=0.1,
            text_threshold=0.05,
        )

        # ====================================================
        # INFERENCE SIZE
        # ====================================================

        self.inference_width = inference_width

        # ====================================================
        # NMS
        # ====================================================

        self.nms_threshold = 0.4

        # ====================================================
        # BYTE TRACK
        # ====================================================

        self.tracker = ByteTrackWrapper(
            track_activation_threshold=0.15,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
            frame_rate=tracker_frame_rate,
        )

        # ====================================================
        # ZONES
        # ====================================================
        #
        # Temporary test zones:
        # left / middle / right thirds.
        #
        # For Gazebo:
        # replace these with the actual booth/zone regions.
        #
        # Normalized coordinates:
        # (x1, y1, x2, y2)
        # ====================================================

        if zones is None:
            zones = {
                "Zone A": (
                    0.00,
                    0.00,
                    0.33,
                    1.00,
                ),
                "Zone B": (
                    0.33,
                    0.00,
                    0.66,
                    1.00,
                ),
                "Zone C": (
                    0.66,
                    0.00,
                    1.00,
                    1.00,
                ),
            }

        self.zones = zones

        # ====================================================
        # CAPACITY
        # ====================================================

        if zone_capacities is None:
            zone_capacities = {
                zone_name: 10
                for zone_name in self.zones
            }

        self.zone_capacities = zone_capacities

        self.crowd_alert_threshold = (
            crowd_alert_threshold
        )

        # ====================================================
        # TRACK MEMORY
        # ====================================================
        #
        # Key:
        #   raw ByteTrack ID
        #
        # Value:
        #   {
        #       logical_id,
        #       bbox,
        #       center,
        #       current_zone,
        #       zone_entered_at,
        #       last_seen
        #   }
        #
        # IMPORTANT:
        # stale tracks are MEMORY ONLY.
        # They are NOT counted as current people.
        # ====================================================

        self.track_state = {}

        self.stale_track_seconds = (
            float(stale_track_seconds)
        )

        # ====================================================
        # LOGICAL PERSON IDs
        # ====================================================
        #
        # ByteTrack IDs can occasionally switch:
        #
        #   ID 3 -> ID 5
        #
        # AWARE keeps a logical identity:
        #
        #   ByteTrack 3 -> Person 3
        #   ByteTrack 5 -> Person 3
        #
        # if the switch happens quickly and spatially nearby.
        # ====================================================

        self.next_logical_id = 1

        self.id_switch_max_seconds = float(
            id_switch_max_seconds
        )

        self.id_switch_max_distance_ratio = float(
            id_switch_max_distance_ratio
        )

        # ====================================================
        # UNIQUE FOOTFALL
        # ====================================================

        self.zone_unique_visitors = {
            zone_name: set()
            for zone_name in self.zones
        }

        # ====================================================
        # DWELL
        # ====================================================
        #
        # Dwell is stored by logical person ID,
        # NOT raw ByteTrack ID.
        # ====================================================

        self.zone_dwell_seconds = {
            zone_name: {}
            for zone_name in self.zones
        }

        # ====================================================
        # CURRENT OCCUPANTS
        # ====================================================
        #
        # These contain ONLY people visible in the current
        # ByteTrack output.
        # ====================================================

        self.current_zone_ids = {
            zone_name: set()
            for zone_name in self.zones
        }

        self.current_people_ids = set()

        # ====================================================
        # LATEST OUTPUT
        # ====================================================

        self.latest_analytics = (
            self._empty_analytics()
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """
        Reset tracking and all analytics.
        """

        self.tracker.reset()

        self.track_state.clear()

        self.next_logical_id = 1

        self.zone_unique_visitors = {
            zone_name: set()
            for zone_name in self.zones
        }

        self.zone_dwell_seconds = {
            zone_name: {}
            for zone_name in self.zones
        }

        self.current_zone_ids = {
            zone_name: set()
            for zone_name in self.zones
        }

        self.current_people_ids = set()

        self.latest_analytics = (
            self._empty_analytics()
        )

    # ========================================================
    # EMPTY ANALYTICS
    # ========================================================

    def _empty_analytics(self):
        return {
            "total_people": 0,

            "zones": {
                zone_name: {
                    "current_count": 0,

                    "capacity": (
                        self.zone_capacities.get(
                            zone_name,
                            0,
                        )
                    ),

                    "occupancy_ratio": 0.0,
                    "occupancy_percent": 0.0,

                    "crowded": False,

                    "unique_footfall": 0,

                    "average_dwell_seconds": 0.0,
                }

                for zone_name in self.zones
            },

            "footfall_ranking": [],
        }

    # ========================================================
    # RESIZE FOR INFERENCE
    # ========================================================

    def _prepare_inference_frame(
        self,
        frame,
    ):
        """
        Grounding DINO runs on a smaller copy.

        Bounding boxes are later converted back to original
        camera coordinates.
        """

        original_height, original_width = (
            frame.shape[:2]
        )

        if (
            self.inference_width is None
            or self.inference_width <= 0
            or original_width <= self.inference_width
        ):
            return (
                frame,
                1.0,
                1.0,
            )

        resize_ratio = (
            self.inference_width
            / float(original_width)
        )

        inference_width = int(
            original_width
            * resize_ratio
        )

        inference_height = int(
            original_height
            * resize_ratio
        )

        inference_frame = cv2.resize(
            frame,
            (
                inference_width,
                inference_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        scale_x = (
            original_width
            / float(inference_width)
        )

        scale_y = (
            original_height
            / float(inference_height)
        )

        return (
            inference_frame,
            scale_x,
            scale_y,
        )

    # ========================================================
    # SCALE BOXES TO ORIGINAL FRAME
    # ========================================================

    def _scale_boxes_to_original(
        self,
        boxes,
        scale_x,
        scale_y,
    ):
        if boxes is None or len(boxes) == 0:
            return np.empty(
                (0, 4),
                dtype=np.float32,
            )

        if hasattr(boxes, "cpu"):
            boxes = (
                boxes
                .cpu()
                .numpy()
            )
        else:
            boxes = np.asarray(
                boxes
            )

        boxes = np.asarray(
            boxes,
            dtype=np.float32,
        ).copy()

        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x

        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

        return boxes

    # ========================================================
    # NMS
    # ========================================================

    def _apply_nms(
        self,
        boxes,
        scores,
    ):
        """
        Remove overlapping duplicate Grounding DINO boxes
        before ByteTrack.
        """

        if boxes is None or len(boxes) == 0:
            return (
                np.empty(
                    (0, 4),
                    dtype=np.float32,
                ),
                [],
            )

        boxes = np.asarray(
            boxes,
            dtype=np.float32,
        )

        if scores is None:
            scores = [
                1.0
            ] * len(boxes)

        else:
            scores = [
                float(score)
                for score in scores
            ]

        nms_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box

            width = max(
                0.0,
                x2 - x1,
            )

            height = max(
                0.0,
                y2 - y1,
            )

            nms_boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(width),
                    float(height),
                ]
            )

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores,
            score_threshold=0.0,
            nms_threshold=self.nms_threshold,
        )

        if (
            indices is None
            or len(indices) == 0
        ):
            return (
                np.empty(
                    (0, 4),
                    dtype=np.float32,
                ),
                [],
            )

        indices = np.asarray(
            indices
        ).reshape(-1)

        filtered_boxes = boxes[
            indices
        ]

        filtered_scores = [
            scores[int(i)]
            for i in indices
        ]

        return (
            filtered_boxes,
            filtered_scores,
        )

    # ========================================================
    # GEOMETRY HELPERS
    # ========================================================

    def _bbox_center(
        self,
        bbox,
    ):
        """
        Return bounding-box center.
        """

        x1, y1, x2, y2 = [
            float(v)
            for v in bbox
        ]

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    def _center_distance_ratio(
        self,
        center_a,
        center_b,
        frame_width,
        frame_height,
    ):
        """
        Distance normalized by frame diagonal.

        This makes the threshold resolution-independent,
        which is useful when moving from test video to
        Gazebo camera frames.
        """

        dx = (
            float(center_a[0])
            - float(center_b[0])
        )

        dy = (
            float(center_a[1])
            - float(center_b[1])
        )

        distance = (
            dx * dx
            + dy * dy
        ) ** 0.5

        frame_diagonal = (
            frame_width * frame_width
            + frame_height * frame_height
        ) ** 0.5

        if frame_diagonal <= 0:
            return float("inf")

        return (
            distance
            / frame_diagonal
        )

    # ========================================================
    # ZONE
    # ========================================================

    def _get_zone_for_bbox(
        self,
        bbox,
        frame_width,
        frame_height,
    ):
        """
        Use bottom-center because it approximates the person's
        ground position better than bbox center.
        """

        x1, y1, x2, y2 = [
            float(v)
            for v in bbox
        ]

        person_x = (
            (x1 + x2)
            / 2.0
        )

        person_y = y2

        normalized_x = (
            person_x
            / frame_width
        )

        normalized_y = (
            person_y
            / frame_height
        )

        for (
            zone_name,
            (
                zx1,
                zy1,
                zx2,
                zy2,
            ),
        ) in self.zones.items():

            if (
                zx1
                <= normalized_x
                < zx2
                and
                zy1
                <= normalized_y
                <= zy2
            ):
                return zone_name

        return None

    # ========================================================
    # LOGICAL ID
    # ========================================================

    def _new_logical_id(self):
        logical_id = (
            self.next_logical_id
        )

        self.next_logical_id += 1

        return logical_id

    def _find_id_switch_candidate(
        self,
        bbox,
        current_zone,
        visible_raw_ids,
        already_reused_raw_ids,
        frame_width,
        frame_height,
        now,
    ):
        """
        Try to determine whether a NEW ByteTrack ID is actually
        a recently-lost person whose raw ID changed.

        Requirements:
            1. old raw ID is not currently visible
            2. old track disappeared very recently
            3. spatial position is close
            4. zone is compatible

        Returns:
            old_raw_id or None
        """

        new_center = self._bbox_center(
            bbox
        )

        best_raw_id = None
        best_distance = float("inf")

        for (
            old_raw_id,
            state,
        ) in self.track_state.items():

            # Old ID is still visible -> cannot be the same
            # physical person as this new track.
            if old_raw_id in visible_raw_ids:
                continue

            # Do not reuse the same old identity twice
            # in one analytics frame.
            if old_raw_id in already_reused_raw_ids:
                continue

            elapsed = (
                now
                - state["last_seen"]
            )

            if (
                elapsed < 0.0
                or elapsed
                > self.id_switch_max_seconds
            ):
                continue

            old_zone = state.get(
                "current_zone"
            )

            # If both have a valid zone and they are different,
            # reject the match.
            #
            # This avoids merging two different people from
            # distant booth regions.
            if (
                old_zone is not None
                and current_zone is not None
                and old_zone != current_zone
            ):
                continue

            old_center = state.get(
                "center"
            )

            if old_center is None:
                continue

            distance_ratio = (
                self._center_distance_ratio(
                    new_center,
                    old_center,
                    frame_width,
                    frame_height,
                )
            )

            if (
                distance_ratio
                <= self.id_switch_max_distance_ratio
                and distance_ratio
                < best_distance
            ):
                best_distance = (
                    distance_ratio
                )

                best_raw_id = (
                    old_raw_id
                )

        return best_raw_id

    # ========================================================
    # DWELL
    # ========================================================

    def _close_zone_visit(
        self,
        logical_id,
        zone_name,
        entered_at,
        now,
    ):
        if (
            zone_name is None
            or entered_at is None
        ):
            return

        duration = max(
            0.0,
            now - entered_at,
        )

        previous = (
            self.zone_dwell_seconds[
                zone_name
            ].get(
                logical_id,
                0.0,
            )
        )

        self.zone_dwell_seconds[
            zone_name
        ][logical_id] = (
            previous
            + duration
        )

    # ========================================================
    # TRACK STATE
    # ========================================================

    def _update_track_state(
        self,
        tracks,
        frame_width,
        frame_height,
        now,
    ):
        """
        Core AWARE tracking logic.

        Current occupancy:
            ONLY tracks visible now.

        Memory:
            stale tracks remain temporarily for ID continuity,
            but they are NOT added to current occupancy.
        """

        # ----------------------------------------------------
        # RESET CURRENT FRAME OCCUPANCY
        # ----------------------------------------------------

        self.current_zone_ids = {
            zone_name: set()
            for zone_name in self.zones
        }

        self.current_people_ids = set()

        # ----------------------------------------------------
        # RAW IDs VISIBLE IN THIS FRAME
        # ----------------------------------------------------

        visible_raw_ids = {
            int(track["track_id"])
            for track in tracks
        }

        # Old raw IDs already used to repair a switch during
        # this same frame.
        already_reused_raw_ids = set()

        # ----------------------------------------------------
        # PROCESS CURRENT BYTETRACK OUTPUT
        # ----------------------------------------------------

        for track in tracks:

            raw_track_id = int(
                track["track_id"]
            )

            bbox = np.asarray(
                track["bbox"],
                dtype=np.float32,
            )

            center = self._bbox_center(
                bbox
            )

            current_zone = (
                self._get_zone_for_bbox(
                    bbox,
                    frame_width,
                    frame_height,
                )
            )

            # =================================================
            # EXISTING RAW ID
            # =================================================

            if (
                raw_track_id
                in self.track_state
            ):
                state = (
                    self.track_state[
                        raw_track_id
                    ]
                )

            # =================================================
            # NEW RAW ID
            # =================================================

            else:
                # ByteTrack may have switched the ID.
                old_raw_id = (
                    self._find_id_switch_candidate(
                        bbox=bbox,
                        current_zone=current_zone,
                        visible_raw_ids=visible_raw_ids,
                        already_reused_raw_ids=(
                            already_reused_raw_ids
                        ),
                        frame_width=frame_width,
                        frame_height=frame_height,
                        now=now,
                    )
                )

                # ---------------------------------------------
                # ID SWITCH RECOVERY
                # ---------------------------------------------

                if old_raw_id is not None:

                    old_state = (
                        self.track_state[
                            old_raw_id
                        ]
                    )

                    already_reused_raw_ids.add(
                        old_raw_id
                    )

                    # Move the identity state from old
                    # ByteTrack ID to new ByteTrack ID.
                    state = {
                        "logical_id": (
                            old_state[
                                "logical_id"
                            ]
                        ),

                        "bbox": bbox.copy(),

                        "center": center,

                        "current_zone": (
                            old_state[
                                "current_zone"
                            ]
                        ),

                        "zone_entered_at": (
                            old_state[
                                "zone_entered_at"
                            ]
                        ),

                        "last_seen": now,
                    }

                    # Remove old raw-ID mapping immediately.
                    #
                    # This is important:
                    # we do NOT want old ID + new ID to coexist
                    # as two people.
                    del self.track_state[
                        old_raw_id
                    ]

                    self.track_state[
                        raw_track_id
                    ] = state

                # ---------------------------------------------
                # GENUINELY NEW PERSON
                # ---------------------------------------------

                else:
                    logical_id = (
                        self._new_logical_id()
                    )

                    state = {
                        "logical_id": (
                            logical_id
                        ),

                        "bbox": bbox.copy(),

                        "center": center,

                        "current_zone": (
                            current_zone
                        ),

                        "zone_entered_at": (
                            now
                            if current_zone
                            is not None
                            else None
                        ),

                        "last_seen": now,
                    }

                    self.track_state[
                        raw_track_id
                    ] = state

                    if current_zone is not None:
                        self.zone_unique_visitors[
                            current_zone
                        ].add(
                            logical_id
                        )

            # =================================================
            # UPDATE ZONE TRANSITION
            # =================================================

            logical_id = int(
                state["logical_id"]
            )

            previous_zone = (
                state[
                    "current_zone"
                ]
            )

            if (
                current_zone
                != previous_zone
            ):
                # Close previous visit.
                if previous_zone is not None:
                    self._close_zone_visit(
                        logical_id,
                        previous_zone,
                        state[
                            "zone_entered_at"
                        ],
                        now,
                    )

                # Start new visit.
                state[
                    "current_zone"
                ] = current_zone

                state[
                    "zone_entered_at"
                ] = (
                    now
                    if current_zone
                    is not None
                    else None
                )

                if current_zone is not None:
                    self.zone_unique_visitors[
                        current_zone
                    ].add(
                        logical_id
                    )

            # =================================================
            # UPDATE TRACK STATE
            # =================================================

            state["bbox"] = (
                bbox.copy()
            )

            state["center"] = center

            state["last_seen"] = now

            # =================================================
            # CURRENT PEOPLE
            # =================================================
            #
            # ONLY currently visible ByteTrack tracks reach here.
            #
            # Therefore stale tracks can never inflate current
            # occupancy.
            # =================================================

            self.current_people_ids.add(
                logical_id
            )

            if current_zone is not None:
                self.current_zone_ids[
                    current_zone
                ].add(
                    logical_id
                )

        # ----------------------------------------------------
        # REMOVE EXPIRED MEMORY
        # ----------------------------------------------------
        #
        # Missing tracks are NOT counted as current people.
        #
        # We retain them only long enough to:
        #   - recover short ByteTrack ID switches
        #   - preserve dwell continuity
        # ----------------------------------------------------

        expired_raw_ids = []

        for (
            raw_track_id,
            state,
        ) in list(
            self.track_state.items()
        ):

            if raw_track_id in visible_raw_ids:
                continue

            elapsed = (
                now
                - state["last_seen"]
            )

            if (
                elapsed
                < self.stale_track_seconds
            ):
                continue

            previous_zone = (
                state[
                    "current_zone"
                ]
            )

            if previous_zone is not None:
                self._close_zone_visit(
                    state[
                        "logical_id"
                    ],
                    previous_zone,
                    state[
                        "zone_entered_at"
                    ],
                    state[
                        "last_seen"
                    ],
                )

            expired_raw_ids.append(
                raw_track_id
            )

        for raw_track_id in expired_raw_ids:
            self.track_state.pop(
                raw_track_id,
                None,
            )

    # ========================================================
    # ANALYTICS
    # ========================================================

    def _build_analytics(
        self,
        tracks,
        now,
    ):
        zones_output = {}

        for zone_name in self.zones:

            # =================================================
            # CURRENT COUNT
            # =================================================
            #
            # Current count comes ONLY from current visible
            # logical IDs.
            # =================================================

            current_ids = (
                self.current_zone_ids[
                    zone_name
                ]
            )

            current_count = len(
                current_ids
            )

            capacity = int(
                self.zone_capacities.get(
                    zone_name,
                    0,
                )
            )

            if capacity > 0:
                occupancy_ratio = (
                    current_count
                    / capacity
                )
            else:
                occupancy_ratio = 0.0

            crowded = (
                occupancy_ratio
                >= self.crowd_alert_threshold
            )

            # =================================================
            # DWELL
            # =================================================

            dwell_by_person = {
                int(logical_id): float(value)

                for (
                    logical_id,
                    value,
                ) in self.zone_dwell_seconds[
                    zone_name
                ].items()
            }

            # Add ongoing visits.
            for (
                raw_track_id,
                state,
            ) in self.track_state.items():

                if (
                    state[
                        "current_zone"
                    ]
                    != zone_name
                ):
                    continue

                entered_at = (
                    state[
                        "zone_entered_at"
                    ]
                )

                if entered_at is None:
                    continue

                logical_id = int(
                    state[
                        "logical_id"
                    ]
                )

                active_dwell = max(
                    0.0,
                    now - entered_at,
                )

                completed_dwell = (
                    self.zone_dwell_seconds[
                        zone_name
                    ].get(
                        logical_id,
                        0.0,
                    )
                )

                dwell_by_person[
                    logical_id
                ] = (
                    float(completed_dwell)
                    + float(active_dwell)
                )

            if dwell_by_person:
                average_dwell = (
                    sum(
                        dwell_by_person.values()
                    )
                    / len(
                        dwell_by_person
                    )
                )
            else:
                average_dwell = 0.0

            zones_output[
                zone_name
            ] = {
                "current_count": (
                    current_count
                ),

                "capacity": capacity,

                "occupancy_ratio": (
                    round(
                        occupancy_ratio,
                        3,
                    )
                ),

                "occupancy_percent": (
                    round(
                        occupancy_ratio
                        * 100.0,
                        1,
                    )
                ),

                "crowded": crowded,

                "unique_footfall": len(
                    self.zone_unique_visitors[
                        zone_name
                    ]
                ),

                "average_dwell_seconds": (
                    round(
                        average_dwell,
                        2,
                    )
                ),
            }

        # ====================================================
        # FOOTFALL RANKING
        # ====================================================

        ranking = sorted(
            [
                {
                    "zone": zone_name,

                    "unique_footfall": len(
                        self.zone_unique_visitors[
                            zone_name
                        ]
                    ),
                }

                for zone_name
                in self.zones
            ],

            key=lambda item: (
                item[
                    "unique_footfall"
                ]
            ),

            reverse=True,
        )

        # ====================================================
        # TOTAL PEOPLE
        # ====================================================
        #
        # CRITICAL FIX:
        #
        # DO NOT use stale track_state here.
        #
        # This is the number of unique logical people represented
        # by ByteTrack in the CURRENT analytics frame.
        # ====================================================

        total_people = len(
            self.current_people_ids
        )

        return {
            "total_people": (
                total_people
            ),

            "zones": (
                zones_output
            ),

            "footfall_ranking": (
                ranking
            ),
        }

    # ========================================================
    # DRAW ZONES
    # ========================================================

    def _draw_zones(
        self,
        frame,
    ):
        height, width = (
            frame.shape[:2]
        )

        for (
            zone_name,
            (
                zx1,
                zy1,
                zx2,
                zy2,
            ),
        ) in self.zones.items():

            x1 = int(
                zx1 * width
            )

            y1 = int(
                zy1 * height
            )

            x2 = int(
                zx2 * width
            )

            y2 = int(
                zy2 * height
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )

            current_count = len(
                self.current_zone_ids[
                    zone_name
                ]
            )

            capacity = (
                self.zone_capacities.get(
                    zone_name,
                    0,
                )
            )

            label = (
                f"{zone_name}: "
                f"{current_count}/{capacity}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 10,
                    y1 + 30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

    # ========================================================
    # DRAW TRACKS
    # ========================================================

    def _draw_tracks(
        self,
        frame,
        tracks,
    ):
        for track in tracks:

            raw_track_id = int(
                track[
                    "track_id"
                ]
            )

            bbox = np.asarray(
                track[
                    "bbox"
                ]
            )

            x1, y1, x2, y2 = [
                int(v)
                for v in bbox
            ]

            confidence = (
                track.get(
                    "confidence"
                )
            )

            state = (
                self.track_state.get(
                    raw_track_id,
                    {},
                )
            )

            logical_id = (
                state.get(
                    "logical_id",
                    raw_track_id,
                )
            )

            current_zone = (
                state.get(
                    "current_zone"
                )
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2,
            )

            # Show both IDs while debugging.
            #
            # P = logical AWARE person
            # BT = raw ByteTrack ID

            label = (
                f"P{logical_id} "
                f"(BT{raw_track_id})"
            )

            if confidence is not None:
                label += (
                    f" {float(confidence):.2f}"
                )

            if current_zone is not None:
                label += (
                    f" | {current_zone}"
                )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
            )

    # ========================================================
    # MAIN ANALYSIS
    # ========================================================

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float = None,
    ):
        """
        Analyze one analytics frame.

        Works with:
            - recorded video
            - Gazebo camera frames
            - live camera frames

        Returns:
            annotated_frame
            people_count
            boxes
            scores
        """

        if timestamp is None:
            now = time.monotonic()
        else:
            now = float(
                timestamp
            )

        # ====================================================
        # 1. SMALLER FRAME FOR DINO
        # ====================================================

        (
            inference_frame,
            scale_x,
            scale_y,
        ) = self._prepare_inference_frame(
            frame
        )

        # ====================================================
        # 2. BGR -> RGB
        # ====================================================

        frame_rgb = cv2.cvtColor(
            inference_frame,
            cv2.COLOR_BGR2RGB,
        )

        image_pil = Image.fromarray(
            frame_rgb
        )

        # ====================================================
        # 3. GROUNDING DINO
        # ====================================================

        (
            _,
            _,
            boxes,
            scores,
        ) = self.detector.run_inference(
            image_pil,
            prompt="person",
            mark_results=False,
        )

        raw_dino_count = (
            0
            if boxes is None
            else len(boxes)
        )

        # ====================================================
        # 4. SCALE TO ORIGINAL FRAME
        # ====================================================

        if (
            boxes is None
            or len(boxes) == 0
        ):
            boxes = np.empty(
                (0, 4),
                dtype=np.float32,
            )

            scores = []

        else:
            boxes = (
                self._scale_boxes_to_original(
                    boxes,
                    scale_x,
                    scale_y,
                )
            )

            # =================================================
            # 5. NMS
            # =================================================

            (
                boxes,
                scores,
            ) = self._apply_nms(
                boxes,
                scores,
            )

        after_nms_count = len(
            boxes
        )

        # ====================================================
        # 6. BYTE TRACK
        # ====================================================

        tracks = self.tracker.update(
            boxes,
            scores,
        )

        # ====================================================
        # 7. ANALYTICS STATE
        # ====================================================

        height, width = (
            frame.shape[:2]
        )

        self._update_track_state(
            tracks,
            width,
            height,
            now,
        )

        # ====================================================
        # 8. BUILD OUTPUT
        # ====================================================

        self.latest_analytics = (
            self._build_analytics(
                tracks,
                now,
            )
        )

        people_count = (
            self.latest_analytics[
                "total_people"
            ]
        )

        # ====================================================
        # DEBUG
        # ====================================================

        raw_ids = [
            int(
                track[
                    "track_id"
                ]
            )
            for track in tracks
        ]

        logical_ids = [
            self.track_state.get(
                raw_id,
                {},
            ).get(
                "logical_id"
            )
            for raw_id in raw_ids
        ]

        print(
            "\n[AWARE DEBUG] "
            f"DINO raw: {raw_dino_count} | "
            f"After NMS: {after_nms_count} | "
            f"ByteTrack: {len(tracks)} | "
            f"BT IDs: {raw_ids} | "
            f"AWARE IDs: {logical_ids} | "
            f"Analytics people: {people_count}"
        )

        # ====================================================
        # 9. DRAW
        # ====================================================

        annotated_frame = (
            frame.copy()
        )

        self._draw_zones(
            annotated_frame
        )

        self._draw_tracks(
            annotated_frame,
            tracks,
        )

        # ====================================================
        # 10. TOTAL PEOPLE
        # ====================================================

        cv2.putText(
            annotated_frame,
            f"People: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            2,
        )

        return (
            annotated_frame,
            people_count,
            boxes,
            scores,
        )

    # ========================================================
    # PUBLIC ANALYTICS GETTER
    # ========================================================

    def get_analytics(self):
        return self.latest_analytics