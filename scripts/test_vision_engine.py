from pathlib import Path

import cv2

from services.vision_service.app.detection.detector import YOLODetector

from services.vision_service.app.tracking.tracker import (
    ByteTrackTracker,
)

from services.vision_service.app.tracking.track_manager import (
    TrackManager,
)

from services.vision_service.app.motion.motion_analyzer import (
    MotionAnalyzer,
)

from services.vision_service.app.events.event_engine import (
    EventEngine,
)

from services.vision_service.app.incidents.incident_engine import (
    IncidentEngine,
)

from services.vision_service.app.incidents.evidence import (
    IncidentEvidence,
)

from services.vision_service.app.vision_engine import (
    VisionEngine,
)


# ============================================================
# CONFIGURATION
# ============================================================

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

EVIDENCE_DIR = (
    BASE_DIR
    / "outputs"
    / "incidents"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "videos"
)

OUTPUT_VIDEO = (
    OUTPUT_DIR
    / "vision_result.mp4"
)


# ============================================================
# CLASS COLORS
# ============================================================
#
# OpenCV uses BGR:
#
# Blue      = (255, 0, 0)
# Green     = (0, 255, 0)
# Red       = (0, 0, 255)
# Cyan      = (255, 255, 0)
# Yellow    = (0, 255, 255)
# Magenta   = (255, 0, 255)
# Orange    = (0, 165, 255)
#
# ============================================================

CLASS_COLORS = {

    "car": (255, 0, 0),

    "motorcycle": (0, 165, 255),

    "truck": (255, 0, 255),

    "bus": (0, 255, 255),

    "pedestrian": (0, 0, 255),

    "ciclist": (255, 255, 0),

}


DEFAULT_COLOR = (0, 255, 0)


# ============================================================
# DRAW TRACKS
# ============================================================

def draw_tracks(
    frame,
    tracks,
):
    """
    Draw bounding boxes and tracking IDs.

    Each object class receives its own color.
    """

    for track in tracks:

        x1 = int(track.x1)
        y1 = int(track.y1)
        x2 = int(track.x2)
        y2 = int(track.y2)

        # ----------------------------------------------------
        # Color according to class
        # ----------------------------------------------------

        color = CLASS_COLORS.get(
            track.class_name,
            DEFAULT_COLOR,
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{track.class_name} "
            f"ID:{track.track_id}"
        )

        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
        )

        label_y = max(
            y1 - 10,
            text_height + 10,
        )

        cv2.rectangle(
            frame,
            (
                x1,
                label_y - text_height - baseline - 5,
            ),
            (
                x1 + text_width + 5,
                label_y + 3,
            ),
            color,
            -1,
        )

        # ----------------------------------------------------
        # Label text
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return frame


# ============================================================
# DRAW INCIDENTS
# ============================================================

def draw_incidents(
    frame,
    incidents,
):
    """
    Draw incident bounding boxes.
    """

    INCIDENT_COLOR = (0, 0, 255)

    for incident in incidents:

        bbox = incident.bbox

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        # ----------------------------------------------------
        # Incident bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            INCIDENT_COLOR,
            3,
        )

        # ----------------------------------------------------
        # Incident label
        # ----------------------------------------------------

        label = (
            f"{incident.incident_type} "
            f"{incident.confidence:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            INCIDENT_COLOR,
            2,
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISION ENGINE FULL VIDEO TEST")
    print("=" * 60)

    # ========================================================
    # DETECTOR
    # ========================================================

    print("\nCreating detector...")

    detector = YOLODetector(
        model_path=MODEL_PATH,

        # Minimum detection confidence (matches the API default)
        confidence=0.60,

        # IoU threshold
        iou=0.60,

        # YOLO input image size
        image_size=640,
    )

    # ========================================================
    # TRACKER
    # ========================================================

    print("\nCreating tracker...")

    tracker = ByteTrackTracker(
        model=detector.model,
        confidence=detector.confidence,
        iou=detector.iou,
        image_size=detector.image_size,
        device=detector.device,
    )

    # ========================================================
    # TRACK MANAGER
    # ========================================================

    print("\nCreating track manager...")

    track_manager = TrackManager()

    # ========================================================
    # MOTION ANALYZER
    # ========================================================

    print("\nCreating motion analyzer...")

    motion_analyzer = MotionAnalyzer()

    # ========================================================
    # EVENT ENGINE
    # ========================================================

    print("\nCreating event engine...")

    event_engine = EventEngine()

    # ========================================================
    # INCIDENT ENGINE
    # ========================================================

    print("\nCreating incident engine...")

    incident_engine = IncidentEngine()

    # ========================================================
    # EVIDENCE
    # ========================================================

    print("\nCreating incident evidence manager...")

    evidence = IncidentEvidence(
        output_dir=EVIDENCE_DIR
    )

    print(
        f"Evidence directory: {EVIDENCE_DIR}"
    )

    # ========================================================
    # VISION ENGINE
    # ========================================================

    print("\nCreating vision engine...")

    vision_engine = VisionEngine(
        detector=detector,
        tracker=tracker,
        track_manager=track_manager,
        event_engine=event_engine,
        motion_analyzer=motion_analyzer,
        incident_engine=incident_engine,
    )

    print(
        "\nVision engine initialized successfully."
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    print("\nOpening video...")

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video:"
        )

        print(
            VIDEO_PATH
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Video FPS: {fps}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Resolution: {width}x{height}"
    )

    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        print(
            "ERROR: Could not create output video."
        )

        cap.release()

        return

    print(
        f"Output video: {OUTPUT_VIDEO}"
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    frame_id = 0

    total_events = 0

    total_incidents = 0

    evidence_saved = 0

    incident_records = []

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_id += 1

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        if fps > 0:

            timestamp_seconds = (
                frame_id / fps
            )

        else:

            timestamp_seconds = 0.0

        # ----------------------------------------------------
        # Vision pipeline
        #
        # IMPORTANT:
        # YOLO receives the COMPLETE FRAME here.
        # It is NOT processing object crops individually.
        # ----------------------------------------------------

        result = vision_engine.process_frame(
            frame=frame,
            frame_id=frame_id,
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_events += len(
            result.events
        )

        total_incidents += len(
            result.incidents
        )

        # ----------------------------------------------------
        # Create annotated frame
        # ----------------------------------------------------

        annotated_frame = frame.copy()

        # ----------------------------------------------------
        # Draw tracked objects
        # ----------------------------------------------------

        annotated_frame = draw_tracks(
            annotated_frame,
            result.tracks,
        )

        # ----------------------------------------------------
        # Draw incidents
        # ----------------------------------------------------

        annotated_frame = draw_incidents(
            annotated_frame,
            result.incidents,
        )

        # ----------------------------------------------------
        # Save incident evidence
        # ----------------------------------------------------

        for incident in result.incidents:

            print(
                f"\n[INCIDENT DETECTED] "
                f"Frame {frame_id} | "
                f"{incident.incident_type} | "
                f"Confidence: "
                f"{incident.confidence:.2f}"
            )

            try:

                evidence_path = evidence.save(
                    frame=annotated_frame,
                    incident=incident,
                    tracks=result.tracks,
                    frame_id=frame_id,
                )

                evidence_saved += 1

                print(
                    f"Evidence saved: "
                    f"{evidence_path}"
                )

            except Exception as error:

                print(
                    "ERROR saving evidence:"
                )

                print(error)

            # ------------------------------------------------
            # Store incident
            # ------------------------------------------------

            incident_record = {

                "frame_id":
                    frame_id,

                "timestamp_seconds":
                    round(
                        timestamp_seconds,
                        2,
                    ),

                "incident_type":
                    incident.incident_type,

                "track_ids":
                    incident.track_ids,

                "confidence":
                    round(
                        incident.confidence,
                        2,
                    ),

                "data":
                    incident.data,
            }

            incident_records.append(
                incident_record
            )

        # ----------------------------------------------------
        # Write annotated frame
        # ----------------------------------------------------

        writer.write(
            annotated_frame
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if frame_id % 50 == 0:

            progress = (
                frame_id
                / total_frames
                * 100
            )

            print(
                f"\rProcessing: "
                f"{frame_id}/{total_frames} "
                f"({progress:.1f}%)",
                end="",
                flush=True,
            )

    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()

    writer.release()

    print("\n")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("=" * 60)
    print("VISION ENGINE TEST FINISHED")
    print("=" * 60)

    print(
        f"Frames processed: {frame_id}"
    )

    print(
        f"Total events: {total_events}"
    )

    print(
        f"Total incidents: {total_incidents}"
    )

    print(
        f"Evidence saved: {evidence_saved}"
    )

    print(
        f"Total persistent tracks: "
        f"{len(track_manager.tracks)}"
    )

    print(
        f"Output video: {OUTPUT_VIDEO}"
    )

    # ========================================================
    # INCIDENT SUMMARY
    # ========================================================

    print("\n")

    print("=" * 60)
    print("INCIDENT SUMMARY")
    print("=" * 60)

    if not incident_records:

        print(
            "\nNo incidents detected."
        )

    else:

        for index, incident in enumerate(
            incident_records,
            start=1,
        ):

            print(
                f"\nIncident #{index}"
            )

            print(
                f"  Frame: "
                f"{incident['frame_id']}"
            )

            print(
                f"  Timestamp: "
                f"{incident['timestamp_seconds']:.2f}s"
            )

            print(
                f"  Type: "
                f"{incident['incident_type']}"
            )

            print(
                f"  Track IDs: "
                f"{incident['track_ids']}"
            )

            print(
                f"  Confidence: "
                f"{incident['confidence']:.2f}"
            )

            print(
                f"  Data: "
                f"{incident['data']}"
            )

    # ========================================================
    # EVIDENCE LOCATION
    # ========================================================

    print("\n")

    print("=" * 60)
    print("EVIDENCE")
    print("=" * 60)

    print(
        "\nEvidence directory:"
    )

    print(
        EVIDENCE_DIR
    )

    print(
        f"\nEvidence files generated: "
        f"{evidence_saved}"
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")

    print("=" * 60)
    print("FULL VIDEO TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

