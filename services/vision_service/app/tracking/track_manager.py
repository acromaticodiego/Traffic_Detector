from typing import Dict

from ..detection.schemas import Detection
from .track_state import TrackState


class TrackManager:
    """
    Maintains the temporal state of all active tracks.
    """

    def __init__(
        self,
        max_missing_frames: int = 30,
    ):
        self.tracks: Dict[int, TrackState] = {}

        self.max_missing_frames = max_missing_frames

        self.missing_frames: Dict[int, int] = {}

    def update(
        self,
        detections: list[Detection],
    ) -> list[TrackState]:
        """
        Update the state of all tracked objects.

        Only detections with a valid track_id
        are converted into TrackState objects.
        """

        current_track_ids = set()

        active_tracks = []

        for detection in detections:

            if detection.track_id is None:
                continue

            track_id = detection.track_id

            current_track_ids.add(track_id)

            # ----------------------------------
            # Existing track
            # ----------------------------------

            if track_id in self.tracks:

                track = self.tracks[track_id]

                track.update(
                    confidence=detection.confidence,
                    x1=detection.bbox.x1,
                    y1=detection.bbox.y1,
                    x2=detection.bbox.x2,
                    y2=detection.bbox.y2,
                )

            # ----------------------------------
            # New track
            # ----------------------------------

            else:

                track = TrackState(
                    track_id=track_id,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    x1=detection.bbox.x1,
                    y1=detection.bbox.y1,
                    x2=detection.bbox.x2,
                    y2=detection.bbox.y2,
                )

                self.tracks[track_id] = track

            # Object is currently visible
            self.missing_frames[track_id] = 0

            active_tracks.append(track)

        # --------------------------------------
        # Update missing tracks
        # --------------------------------------

        existing_track_ids = list(
            self.tracks.keys()
        )

        for track_id in existing_track_ids:

            if track_id in current_track_ids:
                continue

            self.missing_frames[track_id] += 1

            # Remove old tracks
            if (
                self.missing_frames[track_id]
                > self.max_missing_frames
            ):

                del self.tracks[track_id]

                del self.missing_frames[track_id]

        return active_tracks

    def get_track(
        self,
        track_id: int,
    ) -> TrackState | None:

        return self.tracks.get(track_id)

    def get_active_tracks(self) -> list[TrackState]:

        return list(
            self.tracks.values()
        )

    def clear(self):

        self.tracks.clear()

        self.missing_frames.clear()