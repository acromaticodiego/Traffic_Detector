from pathlib import Path

import cv2

from ..tracking.track_state import TrackState
from .schemas import IncidentCandidate


class IncidentEvidence:
    """
    Generates visual evidence for detected incidents.

    For each incident, it saves:

    - original.jpg
    - annotated.jpg

    The annotated image contains the bounding
    boxes of the vehicles involved in the incident.
    """

    def __init__(
        self,
        output_dir: Path,
    ):
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        frame,
        incident: IncidentCandidate,
        tracks: list[TrackState],
        frame_id: int,
    ) -> Path:

        # ==================================================
        # INCIDENT DIRECTORY
        # ==================================================

        incident_dir = (
            self.output_dir
            / f"incident_{frame_id}"
        )

        incident_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==================================================
        # SAVE ORIGINAL FRAME
        # ==================================================

        original_path = (
            incident_dir
            / "original.jpg"
        )

        cv2.imwrite(
            str(original_path),
            frame,
        )

        # ==================================================
        # CREATE ANNOTATED FRAME
        # ==================================================

        annotated = frame.copy()

        incident_track_ids = set(
            incident.track_ids
        )

        # ==================================================
        # DRAW INVOLVED TRACKS
        # ==================================================

        for track in tracks:

            if track.track_id not in incident_track_ids:
                continue

            x1 = int(track.x1)
            y1 = int(track.y1)
            x2 = int(track.x2)
            y2 = int(track.y2)

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3,
            )

            # ----------------------------------------------
            # Label
            # ----------------------------------------------

            label = (
                f"{track.class_name} "
                f"ID:{track.track_id}"
            )

            cv2.putText(
                annotated,
                label,
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # ----------------------------------------------
            # Center point
            # ----------------------------------------------

            center_x = int(
                track.center[0]
            )

            center_y = int(
                track.center[1]
            )

            cv2.circle(
                annotated,
                (
                    center_x,
                    center_y,
                ),
                5,
                (0, 0, 255),
                -1,
            )

        # ==================================================
        # INCIDENT INFORMATION
        # ==================================================

        title = (
            f"INCIDENT: "
            f"{incident.incident_type}"
        )

        confidence_text = (
            f"Confidence: "
            f"{incident.confidence:.2f}"
        )

        frame_text = (
            f"Frame: {frame_id}"
        )

        # ==================================================
        # DRAW TITLE
        # ==================================================

        cv2.putText(
            annotated,
            title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )

        # ==================================================
        # DRAW CONFIDENCE
        # ==================================================

        cv2.putText(
            annotated,
            confidence_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # ==================================================
        # DRAW FRAME
        # ==================================================

        cv2.putText(
            annotated,
            frame_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # ==================================================
        # SAVE ANNOTATED FRAME
        # ==================================================

        annotated_path = (
            incident_dir
            / "annotated.jpg"
        )

        cv2.imwrite(
            str(annotated_path),
            annotated,
        )

        return annotated_path