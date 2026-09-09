"""
Seguimiento ByteTrack con estado propio por sesión.

Antes esto era `model.track(persist=True)`. Ultralytics guarda ahí el estado
del tracker colgado del modelo (`model.predictor.trackers`), que es un objeto
compartido por todo el proceso: con una cámara funcionaba, con dos las dos
sesiones escriben en el mismo tracker y los IDs se mezclan. Peor aún, el
`persist=False` del primer frame de una sesión le borraba los tracks a la
cámara que ya estaba corriendo.

Ahora la detección sigue siendo compartida —un solo modelo cargado en la
GPU— pero la asociación de tracks ocurre en un `BYTETracker` propio de cada
instancia de esta clase, así que dos cámaras concurrentes no se ven entre sí.

Esto usa clases internas de ultralytics (`BYTETracker`, `BaseTrack`), no su
API pública. La versión está fijada en STACK.md justamente por eso: al
subirla hay que releer este archivo.
"""

from __future__ import annotations

import threading

from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

from ..detection.schemas import BoundingBox, Detection

# El modelo YOLO es uno solo para todo el proceso y `predict()` reutiliza su
# predictor interno, que no es reentrante. Con una sola GPU las inferencias de
# varias cámaras se serializan de todas formas, así que el lock no cuesta
# rendimiento: lo que evita es que dos hilos se pisen el predictor.
_MODEL_LOCK = threading.Lock()

_COUNTER_LOCK = threading.Lock()


def _load_tracker_config(name: str) -> IterableSimpleNamespace:
    """Los umbrales de ByteTrack, leídos del YAML que trae ultralytics."""

    config = IterableSimpleNamespace(**YAML.load(check_yaml(name)))

    if config.tracker_type != "bytetrack":
        raise ValueError(
            f"'{name}' configura un tracker '{config.tracker_type}'; "
            f"esta clase solo implementa bytetrack."
        )

    return config


def _new_bytetrack(config: IterableSimpleNamespace) -> BYTETracker:
    """
    Un BYTETracker nuevo que NO le reinicia los IDs a los que ya existen.

    `BYTETracker.__init__` llama a `reset_id()`, que pone a cero
    `BaseTrack._count`: un contador de CLASE, común a todo el proceso. Sin
    esta salvaguarda, abrir una segunda cámara haría que la primera empezara a
    repartir IDs que ya tiene asignados a tracks vivos, y sus tracks se
    fusionarían entre sí. Preservando el contador, los IDs siguen creciendo y
    además quedan únicos entre cámaras, que ayuda al leer los logs.
    """

    with _COUNTER_LOCK:
        preserved = BaseTrack._count
        tracker = BYTETracker(config)
        BaseTrack._count = preserved

    return tracker


class ByteTrackTracker:
    """
    YOLO + ByteTrack tracker.

    One YOLO inference per frame; the association runs
    afterwards on this instance's own tracker state.
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

        self._config = _load_tracker_config(tracker)
        self._tracker = _new_bytetrack(self._config)

        # Todo lo que vio YOLO en el último frame, con track o sin él. Lo usa
        # la anonimización de la evidencia: un vehículo recién entrado al
        # frame todavía no tiene track, pero su placa se lee igual.
        self.last_detections: list[Detection] = []

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
    ) -> list[Detection]:
        """
        Run YOLO detection on the complete image and
        associate the boxes with this session's tracks.

        Only one model inference is performed
        per frame.
        """

        boxes = self._detect(frame)

        if boxes is None:
            self.last_detections = []
            return []

        self.last_detections = self._raw_detections(boxes)

        # Se llama en TODOS los frames, también en los que no traen ninguna
        # detección: el tracker cuenta frames para envejecer los tracks
        # perdidos, y saltárselos alargaría artificialmente las oclusiones.
        tracked = self._tracker.update(boxes, frame)

        detections: list[Detection] = []

        # Cada fila es [x1, y1, x2, y2, track_id, score, class_id, índice de
        # la detección de la que salió]; el índice aquí no hace falta.
        for x1, y1, x2, y2, track_id, confidence, class_id, _index in tracked:

            class_id = int(class_id)

            detection = Detection(

                class_id=class_id,

                class_name=self.model.names[
                    class_id
                ],

                confidence=float(confidence),

                bbox=BoundingBox(

                    x1=float(x1),
                    y1=float(y1),

                    x2=float(x2),
                    y2=float(y2),
                ),

                track_id=int(track_id),
            )

            detections.append(
                detection
            )

        return detections

    def _raw_detections(self, boxes) -> list[Detection]:
        """Las cajas del detector antes de asociarlas, sin track_id."""

        detections: list[Detection] = []

        for corners, confidence, class_id in zip(
            boxes.xyxy, boxes.conf, boxes.cls
        ):
            x1, y1, x2, y2 = corners
            class_id = int(class_id)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.model.names[class_id],
                    confidence=float(confidence),
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                )
            )

        return detections

    # ========================================================
    # DETECT
    # ========================================================

    def _detect(self, frame):
        """
        Las cajas crudas de YOLO, ya en CPU.

        El paso a numpy va dentro del lock a propósito: lo que devuelve
        ultralytics son tensores en la GPU, y soltarlo antes dejaría que la
        siguiente inferencia toque memoria que todavía estamos leyendo.
        """

        with _MODEL_LOCK:

            results = self.model.predict(

                source=frame,

                imgsz=self.image_size,

                conf=self.confidence,

                iou=self.iou,

                device=self.device,

                verbose=False,
            )

            boxes = results[0].boxes

            if boxes is None:
                return None

            return boxes.cpu().numpy()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Olvidar los tracks de esta sesión sin tocar los de las demás.

        Se construye uno nuevo en vez de llamar a `reset()` del tracker porque
        ese método también reinicia el contador global de IDs.
        """

        self._tracker = _new_bytetrack(self._config)
