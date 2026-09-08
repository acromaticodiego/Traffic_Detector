"""
Constructores de objetos de dominio para los tests.

Los tests trabajan sobre la lógica pura del servicio (geometría de la
calzada, movimiento, incidentes, registro de cámaras): no abren video ni
cargan el modelo, así que no hace falta torch, ultralytics ni opencv.
"""

from services.vision_service.app.detection.schemas import BoundingBox, Detection
from services.vision_service.app.motion.motion_analyzer import MotionAnalysis
from services.vision_service.app.tracking.track_state import TrackState


def make_track(
    track_id: int = 1,
    class_name: str = "car",
    box: tuple[float, float, float, float] = (100.0, 100.0, 140.0, 140.0),
    confidence: float = 0.9,
) -> TrackState:
    x1, y1, x2, y2 = box

    return TrackState(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def make_detection(
    track_id: int | None = 1,
    class_name: str = "car",
    box: tuple[float, float, float, float] = (100.0, 100.0, 140.0, 140.0),
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(*box),
        track_id=track_id,
    )


def make_motion(
    track_id: int = 1,
    speed: float = 0.0,
    dx: float = 0.0,
    dy: float = 0.0,
    acceleration: float | None = 0.0,
    abrupt_change: bool = False,
) -> MotionAnalysis:
    return MotionAnalysis(
        track_id=track_id,
        dx=dx,
        dy=dy,
        speed=speed,
        direction=0.0,
        moving=speed >= 0.5,
        acceleration=acceleration,
        abrupt_change=abrupt_change,
    )
