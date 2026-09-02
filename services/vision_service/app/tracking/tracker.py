from ultralytics import YOLO

from ..detection.schemas import BoundingBox, Detection


class ByteTrackTracker:
    """
    YOLO + ByteTrack tracker.

    The YOLO model performs detection and tracking
    in a single inference call per frame.
    """

    def __init__(
        self,
        model: YOLO,
        confidence: float = 0.70,
        iou: float = 0.60,
        image_size: int = 640,
        tracker: str = "bytetrack.yaml",
        device=0,
    ):

        self.model = model

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.tracker = tracker
        self.device = device

        print(
            "ByteTrack configured."
        )

        print(
            f"Tracker confidence: "
            f"{self.confidence}"
        )

        print(
            f"Tracker IoU: "
            f"{self.iou}"
        )

        print(
            f"Tracker image size: "
            f"{self.image_size}"
        )

        print(
            f"Tracker device: "
            f"{self.device}"
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        frame,
        persist: bool = True,
    ) -> list[Detection]:
        """
        Run YOLO detection + ByteTrack on the
        complete image.

        Only one model inference is performed
        per frame.

        persist=True keeps the track IDs between
        frames. Passing persist=False on the first
        frame of a new run resets the internal
        ByteTrack state (used when the same YOLO
        model object is reused across sessions).
        """

        results = self.model.track(

            source=frame,

            # YOLO input size
            imgsz=self.image_size,

            # Detection confidence
            conf=self.confidence,

            # NMS IoU
            iou=self.iou,

            # Keep track IDs between frames
            persist=persist,

            # ByteTrack
            tracker=self.tracker,

            # GPU
            device=self.device,

            verbose=False,
        )

        result = results[0]

        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for box in boxes:

            # ================================================
            # CLASS
            # ================================================

            class_id = int(
                box.cls[0].item()
            )

            # ================================================
            # CONFIDENCE
            # ================================================

            confidence = float(
                box.conf[0].item()
            )

            # ================================================
            # BOUNDING BOX
            # ================================================

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            # ================================================
            # TRACK ID
            # ================================================

            track_id = None

            if box.id is not None:

                track_id = int(
                    box.id[0].item()
                )

            # ================================================
            # DETECTION
            # ================================================

            detection = Detection(

                class_id=class_id,

                class_name=self.model.names[
                    class_id
                ],

                confidence=confidence,

                bbox=BoundingBox(

                    x1=float(x1),
                    y1=float(y1),

                    x2=float(x2),
                    y2=float(y2),
                ),

                track_id=track_id,
            )

            detections.append(
                detection
            )

        return detections

