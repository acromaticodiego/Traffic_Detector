import math
from datetime import datetime

from ..tracking.track_state import TrackState
from .schemas import Event


class EventEngine:
    """
    Engine responsible for generating events
    from tracked objects.
    """

    # ==========================================
    # CONFIGURATION
    # ==========================================

    PROXIMITY_THRESHOLD = 80.0

    # ==========================================
    # INITIALIZATION
    # ==========================================

    def __init__(self):

        # Tracks that have already generated
        # a vehicle_detected event.
        self.known_tracks: set[int] = set()

        # Pairs of tracks that have already
        # generated a proximity event.
        self.proximity_pairs: set[
            tuple[int, int]
        ] = set()

    # ==========================================
    # DISTANCE
    # ==========================================

    def _distance(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> float:
        """
        Calculate Euclidean distance between
        the centers of two tracked objects.

        The distance is expressed in pixels.
        """

        dx = (
            track_a.center[0]
            - track_b.center[0]
        )

        dy = (
            track_a.center[1]
            - track_b.center[1]
        )

        return math.sqrt(
            dx * dx + dy * dy
        )

    # ==========================================
    # PROCESS
    # ==========================================

    def process(
        self,
        tracks: list[TrackState],
    ) -> list[Event]:
        """
        Process active tracks and generate events.
        """

        events: list[Event] = []

        # ======================================
        # VEHICLE DETECTED
        # ======================================

        for track in tracks:

            if (
                track.track_id
                not in self.known_tracks
            ):

                event = Event(
                    event_type="vehicle_detected",
                    timestamp=datetime.now(),
                    confidence=track.confidence,
                    track_ids=[
                        track.track_id
                    ],
                    data={
                        "class_id":
                            track.class_id,
                        "class_name":
                            track.class_name,
                    },
                )

                events.append(event)

                self.known_tracks.add(
                    track.track_id
                )

        # ======================================
        # VEHICLE PROXIMITY
        # ======================================

        for i in range(len(tracks)):

            for j in range(i + 1, len(tracks)):

                track_a = tracks[i]
                track_b = tracks[j]

                distance = self._distance(
                    track_a,
                    track_b
                )

                # Objects are close enough
                if (
                    distance
                    <= self.PROXIMITY_THRESHOLD
                ):

                    # Normalize pair order.
                    pair = tuple(
                        sorted(
                            [
                                track_a.track_id,
                                track_b.track_id,
                            ]
                        )
                    )

                    # Only generate the event once.
                    if (
                        pair
                        not in self.proximity_pairs
                    ):

                        confidence = min(
                            track_a.confidence,
                            track_b.confidence,
                        )

                        event = Event(
                            event_type=(
                                "vehicle_proximity"
                            ),
                            timestamp=datetime.now(),
                            confidence=confidence,
                            track_ids=[
                                track_a.track_id,
                                track_b.track_id,
                            ],
                            data={
                                "distance_px":
                                    round(
                                        distance,
                                        2
                                    ),
                                "class_a":
                                    track_a.class_name,
                                "class_b":
                                    track_b.class_name,
                            },
                        )

                        events.append(event)

                        self.proximity_pairs.add(
                            pair
                        )

        return events

    # ==========================================
    # RESET
    # ==========================================

    def reset(self):
        """
        Reset the internal state of the
        event engine.
        """

        self.known_tracks.clear()

        self.proximity_pairs.clear()