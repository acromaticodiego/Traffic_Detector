from pathlib import Path

import cv2

from services.vision_service.app.detection.detector import YOLODetector
from services.vision_service.app.tracking.tracker import ByteTrackTracker


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "detectorfinal.pt"

VIDEO_PATH = (
    BASE_DIR
    / "videos"
    / "input"
    / "Grabación 2026-08-26 142508.mp4"
)


CONF_THRESHOLD = 0.70
IOU_THRESHOLD = 0.60
IMGSZ = 640


print("Initializing YOLO detector...")

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


print("\nOpening video...")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


print("\nStarting tracking test...\n")


for frame_number in range(1, 101):

    success, frame = cap.read()

    if not success:
        break

    detections = tracker.update(frame)

    print(
        f"\nFrame {frame_number}"
    )

    for detection in detections:

        print(
            f"  "
            f"{detection.class_name:<15} "
            f"confidence={detection.confidence:.2f} "
            f"track_id={detection.track_id}"
        )


cap.release()

print("\nTracking test finished successfully.")