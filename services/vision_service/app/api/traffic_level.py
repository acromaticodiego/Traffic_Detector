"""
Traffic level estimation.

Counts the vehicles currently tracked in the scene and
turns a time-smoothed count into a coarse level:

    "bajo"  -> few / no vehicles
    "medio" -> moderate
    "alto"  -> congested

Thresholds are scene-dependent and configurable
(VISION_TRAFFIC_MEDIUM / VISION_TRAFFIC_HIGH).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tracking.track_state import TrackState
from .config import settings

VEHICLE_CLASSES = {"car", "motorcycle", "truck", "bus"}


@dataclass
class TrafficLevel:
    level: str          # "bajo" | "medio" | "alto"
    vehicles: int       # raw vehicle count this frame
    people: int         # raw pedestrian / cyclist count this frame
    score: float        # smoothed vehicle count


class TrafficLevelEstimator:

    def __init__(
        self,
        medium: float | None = None,
        high: float | None = None,
        smoothing: float | None = None,
    ):
        self.medium = medium if medium is not None else settings.traffic_medium
        self.high = high if high is not None else settings.traffic_high
        self.smoothing = (
            smoothing if smoothing is not None else settings.traffic_smoothing
        )
        self._score: float | None = None

    def update(self, tracks: list[TrackState]) -> TrafficLevel:
        vehicles = sum(
            1 for t in tracks if t.class_name in VEHICLE_CLASSES
        )
        people = sum(
            1 for t in tracks if t.class_name not in VEHICLE_CLASSES
        )

        if self._score is None:
            self._score = float(vehicles)
        else:
            self._score = (
                self.smoothing * vehicles
                + (1.0 - self.smoothing) * self._score
            )

        if self._score >= self.high:
            level = "alto"
        elif self._score >= self.medium:
            level = "medio"
        else:
            level = "bajo"

        return TrafficLevel(
            level=level,
            vehicles=vehicles,
            people=people,
            score=round(self._score, 2),
        )

    def reset(self) -> None:
        self._score = None
