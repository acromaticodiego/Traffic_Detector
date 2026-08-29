from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Tuple


@dataclass
class TrackState:
    """
    Maintains the temporal state of a tracked object.
    """

    track_id: int
    class_id: int
    class_name: str

    confidence: float

    # Current bounding box
    x1: float
    y1: float
    x2: float
    y2: float

    # Position history
    max_history: int = 30

    positions: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=30)
    )

    first_seen: datetime = field(
        default_factory=datetime.now
    )

    last_seen: datetime = field(
        default_factory=datetime.now
    )

    def __post_init__(self):
        """
        Add the initial center position.
        """

        self.positions.append(
            self.center
        )

    @property
    def center(self) -> Tuple[float, float]:
        """
        Calculate the center point of the bounding box.
        """

        center_x = (
            self.x1 + self.x2
        ) / 2

        center_y = (
            self.y1 + self.y2
        ) / 2

        return center_x, center_y

    @property
    def previous_position(self):
        """
        Return the previous position if available.
        """

        if len(self.positions) < 2:
            return None

        return self.positions[-2]

    def update(
        self,
        confidence: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ):
        """
        Update the tracked object's current state.
        """

        self.confidence = confidence

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.last_seen = datetime.now()

        self.positions.append(
            self.center
        )

    @property
    def movement(self):
        """
        Calculate the displacement between
        the previous and current position.
        """

        previous = self.previous_position

        if previous is None:
            return 0.0, 0.0

        current = self.center

        dx = current[0] - previous[0]
        dy = current[1] - previous[1]

        return dx, dy