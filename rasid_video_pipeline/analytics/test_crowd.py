import cv2

from analytics.crowd_analytics import CrowdAnalytics


VIDEO_PATH = "7415436-uhd_3840_2160_25fps.mp4"

# ============================================================
# ANALYTICS PROCESSING RATE
# ============================================================
#
# The source video may be 25 or 30 FPS.
#
# AWARE does NOT need to run the heavy Grounding DINO detector
# on every camera frame.
#
# Instead, the crowd analytics pipeline operates at its own
# processing rate.
#
# Example:
# Camera/video: 25 FPS
# Analytics:     5 FPS
#
# ByteTrack receives every frame that the analytics pipeline
# actually processes.
# ============================================================

TARGET_ANALYTICS_FPS = 5.0


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

if video_fps <= 0:
    video_fps = 25.0


# Number of source frames between analytics updates.
#
# Example:
# video = 25 FPS
# target analytics = 5 FPS
#
# 25 / 5 = 5
#
# Therefore:
# process frames:
# 0, 5, 10, 15, 20, ...
#
# NOT:
# 0, 30, 60, 90, ...
# like the old test.

process_every_n_frames = max(
    1,
    int(
        round(
            video_fps
            / TARGET_ANALYTICS_FPS
        )
    ),
)

actual_analytics_fps = (
    video_fps
    / process_every_n_frames
)


print(
    f"Video FPS: "
    f"{video_fps:.2f}"
)

print(
    f"Target analytics FPS: "
    f"{TARGET_ANALYTICS_FPS:.2f}"
)

print(
    f"Actual analytics FPS: "
    f"{actual_analytics_fps:.2f}"
)

print(
    f"Processing every "
    f"{process_every_n_frames} "
    f"source frames."
)

print(
    "AWARE Crowd Analytics started."
)

print(
    "Press Q to stop."
)


# ============================================================
# INITIALIZE AWARE CROWD ANALYTICS
# ============================================================
#
# tracker_frame_rate matches the rate at which ByteTrack
# actually receives detections.
#
# Grounding DINO internally receives a resized frame
# (1280 px wide by default), not the original 4K frame.
# ============================================================

analytics = CrowdAnalytics(
    inference_width=1280,
    tracker_frame_rate=int(
        round(
            actual_analytics_fps
        )
    ),
)


# ============================================================
# MAIN LOOP
# ============================================================

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ========================================================
    # PROCESS AT ANALYTICS RATE
    # ========================================================
    #
    # The camera/video can produce more frames than the
    # analytics system needs.
    #
    # Frames that are not selected here are simply camera
    # frames that do not enter the heavy AI pipeline.
    #
    # Grounding DINO therefore does NOT run at the full
    # source-video FPS.
    # ========================================================

    if (
        frame_number
        % process_every_n_frames
        != 0
    ):
        frame_number += 1
        continue


    # ========================================================
    # VIDEO TIMESTAMP
    # ========================================================
    #
    # Use the real video timeline for dwell time.
    #
    # This means:
    #
    # If Grounding DINO takes 2 seconds to process,
    # we do NOT incorrectly add those 2 processing seconds
    # to the person's dwell time.
    # ========================================================

    timestamp_seconds = (
        cap.get(
            cv2.CAP_PROP_POS_MSEC
        )
        / 1000.0
    )


    # ========================================================
    # AWARE CROWD ANALYTICS
    # ========================================================
    #
    # Inside analyze_frame():
    #
    # Original frame
    #       ↓
    # Resize inference copy
    #       ↓
    # Grounding DINO
    #       ↓
    # Scale boxes back
    #       ↓
    # NMS
    #       ↓
    # ByteTrack
    #       ↓
    # Stable IDs
    #       ↓
    # Zone / Booth assignment
    #       ↓
    # Occupancy
    # Footfall
    # Dwell time
    # Crowd alerts
    # ========================================================

    (
        annotated_frame,
        people_count,
        boxes,
        scores,
    ) = analytics.analyze_frame(
        frame,
        timestamp=timestamp_seconds,
    )


    # ========================================================
    # GET STRUCTURED ANALYTICS
    # ========================================================

    stats = (
        analytics.get_analytics()
    )


    # ========================================================
    # PRINT LIVE ANALYTICS
    # ========================================================

    print(
        "\r"
        f"Video frame: {frame_number} | "
        f"People: "
        f"{stats['total_people']} | "
        + " | ".join(
            (
                f"{zone_name}: "
                f"{zone_data['current_count']} people, "
                f"{zone_data['occupancy_percent']:.1f}% occupancy, "
                f"{zone_data['unique_footfall']} footfall"
            )
            for (
                zone_name,
                zone_data,
            )
            in stats[
                "zones"
            ].items()
        ),
        end="",
        flush=True,
    )


    # ========================================================
    # RESIZE FOR DISPLAY ONLY
    # ========================================================
    #
    # The output is drawn using original-frame coordinates.
    #
    # We shrink it here only so the 4K video fits on screen.
    # ========================================================

    preview = cv2.resize(
        annotated_frame,
        None,
        fx=0.35,
        fy=0.35,
        interpolation=cv2.INTER_AREA,
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AWARE Crowd Analytics",
        preview,
    )


    # Q = stop
    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):
        break


    frame_number += 1


# ============================================================
# FINAL ANALYTICS
# ============================================================

print(
    "\n\nAWARE Crowd Analytics finished."
)

final_stats = (
    analytics.get_analytics()
)

print(
    "\nFinal analytics:"
)


for (
    zone_name,
    zone_data,
) in final_stats[
    "zones"
].items():

    print(
        f"\n{zone_name}"
        f"\n  Current people: "
        f"{zone_data['current_count']}"
        f"\n  Capacity: "
        f"{zone_data['capacity']}"
        f"\n  Occupancy: "
        f"{zone_data['occupancy_percent']:.1f}%"
        f"\n  Unique footfall: "
        f"{zone_data['unique_footfall']}"
        f"\n  Average dwell: "
        f"{zone_data['average_dwell_seconds']:.2f} sec"
        f"\n  Crowded: "
        f"{zone_data['crowded']}"
    )


print(
    "\nFootfall ranking:"
)


for rank, item in enumerate(
    final_stats[
        "footfall_ranking"
    ],
    start=1,
):

    print(
        f"  {rank}. "
        f"{item['zone']} -> "
        f"{item['unique_footfall']} visitors"
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()