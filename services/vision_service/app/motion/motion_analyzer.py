import math
from dataclasses import dataclass
from typing import Optional

from ..tracking.track_state import TrackState


@dataclass
class MotionAnalysis:
    """
    Resultado del análisis de movimiento de un objeto.
    """

    track_id: int

    dx: float
    dy: float

    speed: float

    direction: float

    moving: bool

    acceleration: Optional[float] = None

    abrupt_change: bool = False


class MotionAnalyzer:
    """
    Analiza el movimiento de los objetos trackeados.

    Se utiliza suavizado para reducir el ruido producido
    por pequeñas variaciones del tracker.
    """

    # ==========================================
    # CONFIGURATION
    # ==========================================

    MOVEMENT_THRESHOLD = 0.50

    # Cambio real de velocidad necesario para
    # considerar un movimiento abrupto.
    ABRUPT_SPEED_CHANGE = 4.0

    # Factor de suavizado.
    # 0.0 = sin movimiento nuevo
    # 1.0 = sin suavizado
    SPEED_SMOOTHING = 0.35

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(
        self,
        movement_threshold: float = MOVEMENT_THRESHOLD,
        abrupt_speed_change: float = ABRUPT_SPEED_CHANGE,
    ):

        self.movement_threshold = movement_threshold

        self.abrupt_speed_change = (
            abrupt_speed_change
        )

        # Velocidad suavizada anterior
        self.previous_speeds: dict[int, float] = {}

    # ==========================================
    # SPEED
    # ==========================================

    def _calculate_speed(
        self,
        dx: float,
        dy: float,
    ) -> float:

        return math.sqrt(
            dx * dx + dy * dy
        )

    # ==========================================
    # DIRECTION
    # ==========================================

    def _calculate_direction(
        self,
        dx: float,
        dy: float,
    ) -> float:

        if dx == 0 and dy == 0:
            return 0.0

        return math.degrees(
            math.atan2(dy, dx)
        )

    # ==========================================
    # SMOOTH SPEED
    # ==========================================

    def _smooth_speed(
        self,
        track_id: int,
        raw_speed: float,
    ) -> tuple[float, float]:

        previous_speed = self.previous_speeds.get(
            track_id
        )

        # Primer frame del track.
        if previous_speed is None:

            self.previous_speeds[
                track_id
            ] = raw_speed

            return raw_speed, 0.0

        smoothed_speed = (
            self.SPEED_SMOOTHING * raw_speed
            +
            (1.0 - self.SPEED_SMOOTHING)
            * previous_speed
        )

        acceleration = (
            smoothed_speed
            - previous_speed
        )

        self.previous_speeds[
            track_id
        ] = smoothed_speed

        return (
            smoothed_speed,
            acceleration,
        )

    # ==========================================
    # ANALYZE
    # ==========================================

    def analyze(
        self,
        track: TrackState,
    ) -> MotionAnalysis:

        dx, dy = track.movement

        raw_speed = self._calculate_speed(
            dx,
            dy,
        )

        speed, acceleration = (
            self._smooth_speed(
                track.track_id,
                raw_speed,
            )
        )

        direction = self._calculate_direction(
            dx,
            dy,
        )

        moving = (
            speed >= self.movement_threshold
        )

        abrupt_change = (
            abs(acceleration)
            >= self.abrupt_speed_change
        )

        return MotionAnalysis(
            track_id=track.track_id,

            dx=dx,

            dy=dy,

            speed=speed,

            direction=direction,

            moving=moving,

            acceleration=acceleration,

            abrupt_change=abrupt_change,
        )

    # ==========================================
    # ANALYZE MANY
    # ==========================================

    def analyze_all(
        self,
        tracks: list[TrackState],
    ) -> list[MotionAnalysis]:

        results = []

        for track in tracks:

            results.append(
                self.analyze(track)
            )

        return results

    # ==========================================
    # REMOVE TRACK
    # ==========================================

    def remove_track(
        self,
        track_id: int,
    ):

        self.previous_speeds.pop(
            track_id,
            None,
        )

    # ==========================================
    # RESET
    # ==========================================

    def reset(self):

        self.previous_speeds.clear()