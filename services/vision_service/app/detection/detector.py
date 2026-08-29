from pathlib import Path

import torch
from ultralytics import YOLO

from .schemas import BoundingBox, Detection


class YOLODetector:

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.70,
        iou: float = 0.60,
        image_size: int = 640,
    ):

        self.model_path = Path(model_path)

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        # ====================================================
        # LOAD MODEL
        # ====================================================

        print("Loading YOLO model...")

        self.model = YOLO(
            str(self.model_path)
        )

        print(
            "YOLO model loaded successfully."
        )

        print(
            f"Classes: {self.model.names}"
        )

        # ====================================================
        # DEVICE
        # ====================================================

        if torch.cuda.is_available():

            self.device = 0

            print(
                "CUDA available."
            )

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            print(
                "Inference device: GPU"
            )

        else:

            self.device = "cpu"

            print(
                "CUDA not available."
            )

            print(
                "Inference device: CPU"
            )

        # ====================================================
        # CONFIGURATION
        # ====================================================

        print(
            f"Confidence threshold: "
            f"{self.confidence}"
        )

        print(
            f"IoU threshold: "
            f"{self.iou}"
        )

        print(
            f"Image size: "
            f"{self.image_size}"
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        frame,
    ) -> list[Detection]:

        # ====================================================
        # ONE YOLO INFERENCE
        # ====================================================
        #
        # YOLO recibe el FRAME COMPLETO.
        #
        # No se procesa objeto por objeto.
        # ====================================================

        results = self.model.predict(

            source=frame,

            imgsz=self.image_size,

            conf=self.confidence,

            iou=self.iou,

            device=self.device,

            verbose=False,
        )

        result = results[0]

        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        # ====================================================
        # PARSE YOLO RESULTS
        # ====================================================

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

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
            )

            detections.append(
                detection
            )

        return detections

