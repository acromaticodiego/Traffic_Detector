from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class IncidentCandidate:
    """
    Represents a possible incident detected
    by the incident engine.
    """

    incident_type: str

    track_ids: list[int]

    confidence: float

    data: dict[str, Any] = field(
        default_factory=dict
    )

    # Bounding box that contains all objects
    # involved in the incident.
    bbox: Optional[dict[str, float]] = None

    # Stable id for the incident. Overlapping detections of the
    # same real event share this id, so clients upsert instead of
    # accumulating duplicates.
    incident_id: Optional[str] = None