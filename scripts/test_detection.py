from pathlib import Path

import cv2
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "detectorfinal.pt"

INPUT_VIDEO = (
    BASE_DIR
    / "videos"
    / "input"
    / "Grabación 2026-08-26 142508.mp4"
)

OUTPUT_VIDEO = (
    BASE_DIR
    / "videos"
    / "output"
    / "tracking_result.mp4"
)


# ==========================================
# YOLO CONFIGURATION
# ==========================================

CONF_THRESHOLD = 0.70

IOU_THRESHOLD = 0.60

IMGSZ = 640


# ==========================================
# TRACKING CONFIGURATION
# ==========================================

TRACKER = "bytetrack.yaml"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print(f"Classes: {model.names}")

print("\nYOLO configuration:")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"IoU threshold: {IOU_THRESHOLD}")
print(f"Image size: {IMGSZ}")
print(f"Tracker: {TRACKER}")


# ==========================================
# OPEN VIDEO
# ==========================================

print("\nOpening video:")
print(INPUT_VIDEO)

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {INPUT_VIDEO}"
    )


# ==========================================
# VIDEO INFORMATION
# ==========================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("\nVideo information:")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")


# ==========================================
# OUTPUT VIDEO
# ==========================================

OUTPUT_VIDEO.parent.mkdir(
    parents=True,
    exist_ok=True
)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        f"Could not create output video: {OUTPUT_VIDEO}"
    )


# ==========================================
# PROCESS VIDEO
# ==========================================

frame_number = 0

print("\nStarting tracking...\n")


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # ======================================
    # YOLO + TRACKING
    # ======================================

    results = model.track(
        frame,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        persist=True,
        tracker=TRACKER,
        verbose=False
    )

    # ======================================
    # DRAW DETECTIONS + TRACKING IDs
    # ======================================

    annotated_frame = results[0].plot()

    # ======================================
    # WRITE FRAME
    # ======================================

    writer.write(annotated_frame)

    # ======================================
    # PROGRESS
    # ======================================

    if frame_number % 30 == 0:

        progress = (
            frame_number / total_frames
        ) * 100

        print(
            f"Progress: "
            f"{progress:.1f}% "
            f"({frame_number}/{total_frames})"
        )


# ==========================================
# CLEANUP
# ==========================================

cap.release()
writer.release()

print("\nTracking finished.")

print(
    f"Output video:\n{OUTPUT_VIDEO}"
)