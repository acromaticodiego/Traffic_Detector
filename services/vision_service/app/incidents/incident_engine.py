from typing import Optional
import math

from ..tracking.track_state import TrackState
from ..motion.motion_analyzer import MotionAnalysis

from .schemas import IncidentCandidate


class IncidentEngine:

    # ==========================================
    # CONFIGURATION
    # ==========================================

    # Distancia máxima entre centros.
    MAX_CENTER_DISTANCE = 45.0

    # Distancia extremadamente cercana.
    VERY_CLOSE_DISTANCE = 25.0

    # Separación máxima entre bounding boxes.
    MAX_BBOX_GAP = 10.0

    # Cambio mínimo de velocidad.
    MIN_SPEED_CHANGE = 4.0

    # Velocidad mínima para considerar interacción.
    MIN_SPEED = 1.0

    # Número de frames consecutivos necesarios.
    REQUIRED_FRAMES = 3

    # Confianza mínima.
    MIN_COLLISION_CONFIDENCE = 0.70

    VEHICLE_CLASSES = {
        "car",
        "motorcycle",
        "truck",
        "bus",
    }

    def __init__(self):

        # Pares que ya generaron incidente.
        self.collision_pairs: set[
            tuple[int, int]
        ] = set()

        # Cantidad de frames consecutivos
        # donde el par cumple las condiciones.
        self.candidate_frames: dict[
            tuple[int, int],
            int
        ] = {}

    # ==========================================
    # CENTER DISTANCE
    # ==========================================

    def _calculate_distance(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> float:

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
    # BBOX GAP
    # ==========================================

    def _calculate_bbox_gap(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> float:
        """
        Calcula la distancia mínima entre
        dos bounding boxes.

        Si se están tocando o solapando,
        devuelve 0.
        """

        horizontal_gap = max(
            track_a.x1 - track_b.x2,
            track_b.x1 - track_a.x2,
            0.0,
        )

        vertical_gap = max(
            track_a.y1 - track_b.y2,
            track_b.y1 - track_a.y2,
            0.0,
        )

        return math.sqrt(
            horizontal_gap ** 2
            +
            vertical_gap ** 2
        )

    # ==========================================
    # BBOX IOU
    # ==========================================

    def _calculate_iou(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> float:

        x1 = max(
            track_a.x1,
            track_b.x1,
        )

        y1 = max(
            track_a.y1,
            track_b.y1,
        )

        x2 = min(
            track_a.x2,
            track_b.x2,
        )

        y2 = min(
            track_a.y2,
            track_b.y2,
        )

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            intersection_width
            *
            intersection_height
        )

        if intersection <= 0:
            return 0.0

        area_a = (
            track_a.x2 - track_a.x1
        ) * (
            track_a.y2 - track_a.y1
        )

        area_b = (
            track_b.x2 - track_b.x1
        ) * (
            track_b.y2 - track_b.y1
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # ==========================================
    # APPROACHING
    # ==========================================

    def _are_approaching(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> bool:
        """
        Determina si los objetos se están
        moviendo aproximadamente uno hacia otro.
        """

        dx = (
            track_b.center[0]
            - track_a.center[0]
        )

        dy = (
            track_b.center[1]
            - track_a.center[1]
        )

        distance = math.sqrt(
            dx * dx + dy * dy
        )

        if distance <= 0:
            return True

        direction_x = dx / distance
        direction_y = dy / distance

        velocity_a_x = track_a.movement[0]
        velocity_a_y = track_a.movement[1]

        velocity_b_x = track_b.movement[0]
        velocity_b_y = track_b.movement[1]

        # A debe moverse hacia B.
        projection_a = (
            velocity_a_x * direction_x
            +
            velocity_a_y * direction_y
        )

        # B debe moverse hacia A.
        projection_b = (
            velocity_b_x * direction_x
            +
            velocity_b_y * direction_y
        )

        return (
            projection_a > 0
            and
            projection_b < 0
        )

    # ==========================================
    # INCIDENT BBOX
    # ==========================================

    def _calculate_incident_bbox(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> dict[str, float]:

        return {
            "x1": float(
                min(
                    track_a.x1,
                    track_b.x1,
                )
            ),

            "y1": float(
                min(
                    track_a.y1,
                    track_b.y1,
                )
            ),

            "x2": float(
                max(
                    track_a.x2,
                    track_b.x2,
                )
            ),

            "y2": float(
                max(
                    track_a.y2,
                    track_b.y2,
                )
            ),
        }

    # ==========================================
    # CONFIDENCE
    # ==========================================

    def _calculate_collision_confidence(
        self,
        distance: float,
        bbox_gap: float,
        iou: float,
        motion_a: MotionAnalysis,
        motion_b: MotionAnalysis,
        approaching: bool,
    ) -> float:

        confidence = 0.0

        # --------------------------------------
        # DISTANCE
        # --------------------------------------

        if distance <= self.VERY_CLOSE_DISTANCE:

            confidence += 0.35

        elif distance <= self.MAX_CENTER_DISTANCE:

            confidence += 0.20

        # --------------------------------------
        # BBOX
        # --------------------------------------

        if iou > 0.05:

            confidence += 0.30

        elif bbox_gap <= 5:

            confidence += 0.25

        elif bbox_gap <= self.MAX_BBOX_GAP:

            confidence += 0.15

        # --------------------------------------
        # APPROACHING
        # --------------------------------------

        if approaching:

            confidence += 0.20

        # --------------------------------------
        # ABRUPT MOVEMENT
        # --------------------------------------

        if motion_a.abrupt_change:

            confidence += 0.10

        if motion_b.abrupt_change:

            confidence += 0.10

        # --------------------------------------
        # SPEED CHANGE
        # --------------------------------------

        if (
            abs(motion_a.acceleration or 0)
            >= self.MIN_SPEED_CHANGE
        ):

            confidence += 0.10

        if (
            abs(motion_b.acceleration or 0)
            >= self.MIN_SPEED_CHANGE
        ):

            confidence += 0.10

        return min(
            confidence,
            1.0,
        )

    # ==========================================
    # COLLISION DETECTION
    # ==========================================

    def _detect_collision(
        self,
        track_a: TrackState,
        track_b: TrackState,
        motion_a: MotionAnalysis,
        motion_b: MotionAnalysis,
    ) -> Optional[IncidentCandidate]:

        # --------------------------------------
        # ONLY VEHICLES
        # --------------------------------------

        if (
            track_a.class_name
            not in self.VEHICLE_CLASSES
        ):

            return None

        if (
            track_b.class_name
            not in self.VEHICLE_CLASSES
        ):

            return None

        # --------------------------------------
        # PAIR
        # --------------------------------------

        pair = tuple(
            sorted(
                [
                    track_a.track_id,
                    track_b.track_id,
                ]
            )
        )

        # Ya reportado.
        if pair in self.collision_pairs:

            return None

        # --------------------------------------
        # DISTANCE
        # --------------------------------------

        distance = self._calculate_distance(
            track_a,
            track_b,
        )

        if distance > self.MAX_CENTER_DISTANCE:

            self.candidate_frames.pop(
                pair,
                None,
            )

            return None

        # --------------------------------------
        # BBOX GAP
        # --------------------------------------

        bbox_gap = self._calculate_bbox_gap(
            track_a,
            track_b,
        )

        iou = self._calculate_iou(
            track_a,
            track_b,
        )

        # --------------------------------------
        # OBJECTS MUST BE VERY CLOSE
        # --------------------------------------

        if (
            bbox_gap
            > self.MAX_BBOX_GAP
            and
            iou <= 0.0
        ):

            self.candidate_frames.pop(
                pair,
                None,
            )

            return None

        # --------------------------------------
        # SPEED
        # --------------------------------------

        speed_a = motion_a.speed
        speed_b = motion_b.speed

        if (
            speed_a < self.MIN_SPEED
            and
            speed_b < self.MIN_SPEED
        ):

            self.candidate_frames.pop(
                pair,
                None,
            )

            return None

        # --------------------------------------
        # APPROACHING
        # --------------------------------------

        approaching = self._are_approaching(
            track_a,
            track_b,
        )

        # Si no se están aproximando,
        # solamente permitimos el caso de
        # solapamiento real.
        if (
            not approaching
            and
            iou <= 0.05
        ):

            self.candidate_frames.pop(
                pair,
                None,
            )

            return None

        # --------------------------------------
        # CONFIDENCE
        # --------------------------------------

        confidence = (
            self._calculate_collision_confidence(
                distance=distance,
                bbox_gap=bbox_gap,
                iou=iou,
                motion_a=motion_a,
                motion_b=motion_b,
                approaching=approaching,
            )
        )

        if (
            confidence
            < self.MIN_COLLISION_CONFIDENCE
        ):

            self.candidate_frames.pop(
                pair,
                None,
            )

            return None

        # --------------------------------------
        # TEMPORAL VALIDATION
        # --------------------------------------

        current_count = (
            self.candidate_frames.get(
                pair,
                0,
            )
        )

        current_count += 1

        self.candidate_frames[
            pair
        ] = current_count

        # Todavía no tenemos suficiente
        # evidencia temporal.
        if (
            current_count
            < self.REQUIRED_FRAMES
        ):

            return None

        # --------------------------------------
        # REGISTER INCIDENT
        # --------------------------------------

        self.collision_pairs.add(
            pair
        )

        self.candidate_frames.pop(
            pair,
            None,
        )

        # --------------------------------------
        # INCIDENT
        # --------------------------------------

        incident_bbox = (
            self._calculate_incident_bbox(
                track_a,
                track_b,
            )
        )

        return IncidentCandidate(
            incident_type="possible_collision",

            track_ids=[
                track_a.track_id,
                track_b.track_id,
            ],

            confidence=confidence,

            bbox=incident_bbox,

            data={
                "distance_px": round(
                    distance,
                    2,
                ),

                "bbox_gap_px": round(
                    bbox_gap,
                    2,
                ),

                "iou": round(
                    iou,
                    3,
                ),

                "approaching":
                    approaching,

                "class_a":
                    track_a.class_name,

                "class_b":
                    track_b.class_name,

                "speed_a":
                    round(
                        speed_a,
                        2,
                    ),

                "speed_b":
                    round(
                        speed_b,
                        2,
                    ),

                "acceleration_a":
                    round(
                        motion_a.acceleration or 0,
                        2,
                    ),

                "acceleration_b":
                    round(
                        motion_b.acceleration or 0,
                        2,
                    ),

                "abrupt_change_a":
                    motion_a.abrupt_change,

                "abrupt_change_b":
                    motion_b.abrupt_change,
            },
        )

    # ==========================================
    # PROCESS
    # ==========================================

    def process(
        self,
        tracks: list[TrackState],
        motion_analysis: list[MotionAnalysis],
    ) -> list[IncidentCandidate]:

        incidents: list[
            IncidentCandidate
        ] = []

        motion_by_id = {
            analysis.track_id: analysis
            for analysis in motion_analysis
        }

        active_pairs: set[
            tuple[int, int]
        ] = set()

        for i in range(
            len(tracks)
        ):

            for j in range(
                i + 1,
                len(tracks),
            ):

                track_a = tracks[i]

                track_b = tracks[j]

                pair = tuple(
                    sorted(
                        [
                            track_a.track_id,
                            track_b.track_id,
                        ]
                    )
                )

                active_pairs.add(pair)

                motion_a = motion_by_id.get(
                    track_a.track_id
                )

                motion_b = motion_by_id.get(
                    track_b.track_id
                )

                if (
                    motion_a is None
                    or motion_b is None
                ):

                    continue

                incident = (
                    self._detect_collision(
                        track_a=track_a,
                        track_b=track_b,
                        motion_a=motion_a,
                        motion_b=motion_b,
                    )
                )

                if incident is not None:

                    incidents.append(
                        incident
                    )

        # Limpa candidatos que deixaram
        # de estar ativos.
        inactive_pairs = (
            set(
                self.candidate_frames.keys()
            )
            - active_pairs
        )

        for pair in inactive_pairs:

            self.candidate_frames.pop(
                pair,
                None,
            )

        return incidents

    # ==========================================
    # RESET
    # ==========================================

    def reset(self):

        self.collision_pairs.clear()

        self.candidate_frames.clear()