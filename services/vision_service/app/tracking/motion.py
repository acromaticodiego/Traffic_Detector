import math
from dataclasses import dataclass


@dataclass
class MotionState:
    """
    Represents the movement state of a tracked object.
    """

    track_id: int

    dx: float
    dy: float

    speed: float

    direction: float

    moving: bool


class MotionAnalyzer:
    """
    Calculates movement information from tracked objects.
    """

    # Minimum displacement considered as movement.
    MOVEMENT_THRESHOLD = 0.5

    def analyze(
        self,
        track_id: int,
        movement: tuple[float, float],
    ) -> MotionState:
        """
        Analyze the movement vector of a track.
        """

        dx = float(movement[0])
        dy = float(movement[1])

        # --------------------------------------
        # Speed
        # --------------------------------------

        speed = math.sqrt(
            dx * dx +
            dy * dy
        )

        # --------------------------------------
        # Direction
        # --------------------------------------

        direction = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        # --------------------------------------
        # Movement state
        # --------------------------------------

        moving = (
            speed
            >= self.MOVEMENT_THRESHOLD
        )

        return MotionState(
            track_id=track_id,
            dx=dx,
            dy=dy,
            speed=speed,
            direction=direction,
            moving=moving,
        )