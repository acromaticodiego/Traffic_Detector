from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None