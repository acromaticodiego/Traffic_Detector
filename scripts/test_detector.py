from pathlib import Path

import cv2

from services.vision_service.app.detection.detector import YOLODetector


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "detectorfinal.pt"

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
# LOAD DETECTOR
# ==========================================

print("Initializing detector...\n")

detector = YOLODetector(
    model_path=MODEL_PATH,
    confidence=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    image_size=IMGSZ,
)


# ==========================================
# OPEN VIDEO
# ==========================================

print("\nOpening video...")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ==========================================
# READ FIRST FRAME
# ==========================================

success, frame = cap.read()

if not success:
    cap.release()
    raise RuntimeError("Could not read first frame.")


print("First frame loaded successfully.")


# ==========================================
# RUN DETECTION
# ==========================================

print("\nRunning detection...\n")

detections = detector.predict(frame)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("=" * 60)
print(f"Detections found: {len(detections)}")
print("=" * 60)

for index, detection in enumerate(detections, start=1):

    print(f"\nDetection #{index}")

    print(f"  Class ID:   {detection.class_id}")
    print(f"  Class name: {detection.class_name}")
    print(f"  Confidence: {detection.confidence:.4f}")

    print("  Bounding box:")

    print(f"    x1: {detection.bbox.x1:.2f}")
    print(f"    y1: {detection.bbox.y1:.2f}")
    print(f"    x2: {detection.bbox.x2:.2f}")
    print(f"    y2: {detection.bbox.y2:.2f}")

    print(f"  Track ID:   {detection.track_id}")


# ==========================================
# CLEANUP
# ==========================================

cap.release()

print("\nDetection test finished successfully.")