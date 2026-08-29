from pathlib import Path

import cv2

from services.vision_service.app.detection.detector import YOLODetector
from services.vision_service.app.tracking.tracker import ByteTrackTracker
from services.vision_service.app.tracking.track_manager import TrackManager
from services.vision_service.app.tracking.motion import MotionAnalyzer


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

print("Initializing motion analysis...")

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

motion_analyzer = MotionAnalyzer()


# ==========================================
# OPEN VIDEO
# ==========================================

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

print("\nProcessing first 100 frames...\n")


for frame_number in range(1, 101):

    success, frame = cap.read()

    if not success:
        break

    # --------------------------------------
    # Detection + tracking
    # --------------------------------------

    detections = tracker.update(
        frame
    )

    # --------------------------------------
    # Track manager
    # --------------------------------------

    tracks = track_manager.update(
        detections
    )

    # --------------------------------------
    # Motion analysis
    # --------------------------------------

    if frame_number % 10 == 0:

        print(
            f"\nFrame {frame_number}"
        )

        print(
            f"Active tracks: {len(tracks)}"
        )

        for track in tracks:

            motion = motion_analyzer.analyze(
                track_id=track.track_id,
                movement=track.movement,
            )

            print(
                f"  "
                f"ID={motion.track_id:<3} "
                f"class={track.class_name:<12} "
                f"dx={motion.dx:>6.2f} "
                f"dy={motion.dy:>6.2f} "
                f"speed={motion.speed:>6.2f} "
                f"direction={motion.direction:>7.2f}° "
                f"moving={motion.moving}"
            )


# ==========================================
# CLEANUP
# ==========================================

cap.release()

print(
    "\nMotion analysis test finished successfully."
)