from pathlib import Path

import cv2

from services.vision_service.app.detection.detector import YOLODetector
from services.vision_service.app.tracking.tracker import ByteTrackTracker
from services.vision_service.app.tracking.track_manager import TrackManager


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "detectorfinal.pt"
)

VIDEO_PATH = (
    BASE_DIR
    / "videos"
    / "input"
    / "Grabación 2026-08-26 142508.mp4"
)


# ==========================================
# CONFIGURATION
# ==========================================

CONF_THRESHOLD = 0.70
IOU_THRESHOLD = 0.60
IMGSZ = 640


# ==========================================
# INITIALIZE
# ==========================================

print("Initializing detector...")

detector = YOLODetector(
    model_path=MODEL_PATH,
    confidence=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    image_size=IMGSZ,
)


tracker = ByteTrackTracker(
    model=detector.model,
    confidence=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    image_size=IMGSZ,
)


track_manager = TrackManager(
    max_missing_frames=30
)


# ==========================================
# OPEN VIDEO
# ==========================================

print("\nOpening video...")

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ==========================================
# PROCESS VIDEO
# ==========================================

print("\nStarting tracking state test...\n")


for frame_number in range(1, 101):

    success, frame = cap.read()

    if not success:
        break

    # --------------------------------------
    # YOLO + ByteTrack
    # --------------------------------------

    detections = tracker.update(
        frame
    )

    # --------------------------------------
    # Update TrackManager
    # --------------------------------------

    active_tracks = track_manager.update(
        detections
    )

    # --------------------------------------
    # Print every 10 frames
    # --------------------------------------

    if frame_number % 10 == 0:

        print(
            f"\nFrame {frame_number}"
        )

        print(
            f"Active tracks: "
            f"{len(active_tracks)}"
        )

        for track in active_tracks:

            center_x, center_y = (
                track.center
            )

            dx, dy = (
                track.movement
            )

            print(
                f"  "
                f"ID={track.track_id:<3} "
                f"class={track.class_name:<12} "
                f"center=({center_x:.1f}, "
                f"{center_y:.1f}) "
                f"movement=({dx:.1f}, "
                f"{dy:.1f})"
            )


# ==========================================
# CLEANUP
# ==========================================

cap.release()

print("\nTrack state test finished successfully.")

print(
    f"Total active tracks: "
    f"{len(track_manager.get_active_tracks())}"
)