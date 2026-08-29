from pathlib import Path

import cv2

from services.vision_service.app.detection.detector import YOLODetector
from services.vision_service.app.tracking.tracker import ByteTrackTracker
from services.vision_service.app.tracking.track_manager import TrackManager
from services.vision_service.app.events.event_engine import EventEngine


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

print("Initializing vision pipeline...")

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

event_engine = EventEngine()


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

print("\nStarting event engine test...\n")


for frame_number in range(1, 101):

    success, frame = cap.read()

    if not success:
        break

    # YOLO + ByteTrack
    detections = tracker.update(
        frame
    )

    # Track state
    tracks = track_manager.update(
        detections
    )

    # Event engine
    events = event_engine.process(
        tracks
    )

    # --------------------------------------
    # Print generated events
    # --------------------------------------

    for event in events:

        print(
            f"[EVENT] "
            f"type={event.event_type} "
            f"track_ids={event.track_ids} "
            f"confidence={event.confidence:.2f} "
            f"data={event.data}"
        )


# ==========================================
# CLEANUP
# ==========================================

cap.release()

print(
    "\nEvent engine test finished successfully."
)
