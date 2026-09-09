from typing import Optional
import math

from ..geometry.homography import CalibrationError, GroundPlane, ground_point
from ..tracking.track_state import TrackState
from ..motion.motion_analyzer import MotionAnalysis

from .schemas import IncidentCandidate


class IncidentEngine:
    """
    Turns tracked objects + motion analysis into incident
    candidates.

    Detects:
      - possible_collision  : two vehicles in contact together
                              with a crash-like sudden deceleration.
      - vehiculo_detenido   : a vehicle that was moving and is now
                              stopped (abruptly, or for a long time).

    Geometry thresholds are expressed relative to the vehicle
    bounding-box size, so the engine is not tied to one camera
    resolution.
    """

    # ==========================================
    # CONFIGURATION - collision (geometry)
    # ==========================================

    # Overlap (IoU) that already counts as contact.
    CONTACT_IOU = 0.10

    # Gap between boxes counts as "in contact" when it is
    # below this fraction of the average vehicle size.
    GAP_RATIO = 0.30

    # ==========================================
    # CONFIGURATION - collision (dynamics)
    # ==========================================

    # A |acceleration| at/above this is a crash-like change.
    HARD_DECEL = 3.0

    # How many processed frames a hard decel / abrupt change
    # stays "recent" for the collision logic.
    ABRUPT_RECENT_FRAMES = 12

    # At least one vehicle must be moving now, or have been
    # moving recently, for a contact to be a collision.
    MIN_INVOLVED_SPEED = 1.0

    # ==========================================
    # CONFIGURATION - collision (decision)
    # ==========================================

    REQUIRED_FRAMES = 2

    MIN_COLLISION_CONFIDENCE = 0.60

    # ==========================================
    # CONFIGURATION - incident clustering
    # ==========================================

    # Overlapping collision detections within this many processed
    # frames are merged into one incident (same real event).
    MERGE_WINDOW = 60

    # Two incident boxes belong together when their gap is below
    # this fraction of their average size (or they overlap).
    MERGE_NEAR_RATIO = 0.6

    # Confidence at/above which an incident is "confirmed" rather
    # than "pending review". Clients use it for severity display.
    ALERT_CONFIDENCE = 0.80

    # ==========================================
    # CONFIGURATION - metric conflict (needs calibration)
    # ==========================================
    #
    # Con la cámara calibrada el motor deja de razonar en píxeles. Y el dato
    # que importa NO es la distancia: medidos sobre el asfalto, los pares que
    # este motor venía marcando estaban a metros o menos unos de otros, que
    # es lo normal en tráfico urbano denso. Estar cerca no distingue nada.
    #
    # Lo que distingue un conflicto es la DINÁMICA: a qué velocidad se cierra
    # la pareja y en cuántos segundos se tocarían si nadie reaccionara. Es el
    # TTC (time-to-collision) de la literatura de seguridad vial, donde por
    # debajo de ~1,5 s se considera un conflicto serio.

    # Segundos de TTC por debajo de los cuales la pareja está en conflicto.
    TTC_CONFLICT_S = 1.5

    # Velocidad de cierre mínima (m/s) para tomarse en serio un TTC. Dos
    # vehículos parados en un trancón tienen TTC infinito y no molestan, pero
    # el ruido de medición puede fabricar cierres diminutos.
    MIN_CLOSING_SPEED = 0.5

    # Separación (m) por debajo de la cual dos vehículos están tan juntos que
    # cualquier cierre importa. Dos carros lado a lado ocupan ~4 m de ancho.
    CONTACT_DISTANCE_M = 1.0

    # ==========================================
    # CONFIGURATION - aftermath confirmation
    # ==========================================
    #
    # Two boxes touching in the image says almost nothing: the camera
    # projects a 3D scene onto a plane, so a bus in a far lane and a car in
    # a near one can overlap perfectly while being metres apart. That single
    # fact is behind most of this engine's false positives, and no threshold
    # on pixel geometry can fix it.
    #
    # What a real crash does leave behind is unambiguous and observable: it
    # IMMOBILISES at least one of the vehicles. So contact alone is now only
    # a suspicion, and the incident is confirmed or dropped by what happens
    # in the seconds after it.
    #
    # This mirrors how deployed incident-detection systems work: they report
    # the consequence (a stopped vehicle, a queue forming) rather than the
    # instant of impact, because the consequence is what a camera can
    # actually see.

    # Processed frames to wait for one of the vehicles to stop before
    # concluding that they simply drove on.
    AFTERMATH_WINDOW = 90

    # Consecutive stopped frames that count as "immobilised".
    AFTERMATH_STOP_FRAMES = 20

    # The stop has to START around the incident. A vehicle already parked
    # before the contact is not evidence of anything, and this is exactly
    # the case a naive check would confirm.
    AFTERMATH_STOP_TOLERANCE = 8

    # Ceiling on the confidence of an incident whose aftermath is still
    # unknown. Below ALERT_CONFIDENCE on purpose: nothing reaches
    # "confirmed" on geometry alone any more.
    UNCONFIRMED_CAP = 0.75

    # Added once a vehicle is confirmed immobilised.
    AFTERMATH_BONUS = 0.15

    # Applied when the window closes and everyone kept driving. Not zero:
    # the detection still happened and a reviewer may want to see it, but it
    # drops to the bottom of the queue.
    KEPT_MOVING_FACTOR = 0.5

    # ==========================================
    # CONFIGURATION - stopped vehicle
    # ==========================================

    # Speed below this = "stopped".
    STOP_SPEED = 0.6

    # Speed above this = "moving" (used to arm / clear).
    MOVING_SPEED = 2.5

    # Processed frames stopped before an ABRUPT stop is reported.
    STOPPED_FRAMES = 25

    # Processed frames stopped before a gentle/long stop is
    # reported (avoids firing on normal red-light stops).
    STOPPED_FRAMES_LONG = 150

    VEHICLE_CLASSES = {
        "car",
        "motorcycle",
        "truck",
        "bus",
    }

    # ==========================================
    # CONFIGURATION - olvido
    # ==========================================
    #
    # Tres estructuras tienen que sobrevivir a que un track desaparezca unos
    # frames: si un vehículo se pierde tras un bus y vuelve, no puede
    # reportarse otra vez ni dejar de constar que ya se movió. Por eso NO se
    # limpian junto con las demás.
    #
    # Pero "sobrevivir a una oclusión" no es "guardarse para siempre". Sin
    # vencimiento crecen mientras el proceso viva, y ahora las sesiones no
    # terminan nunca: con la fuente en bucle una cámara puede llevar semanas
    # abierta, y los track_id ya no se reinician entre cámaras.
    #
    # El plazo es holgado a propósito. ByteTrack descarta un track perdido a
    # los 30 frames y los id nunca se reutilizan, así que pasado ese punto un
    # id no puede volver: 300 frames son diez veces ese margen.
    FORGET_AFTER_FRAMES = 300

    # Cada cuántos frames se pasa el olvido. No hace falta cada frame —el
    # plazo se mide en cientos— y recorrer las parejas de colisión en cada
    # uno sería trabajo repetido dentro del bucle de visión.
    FORGET_EVERY_FRAMES = 60

    # ==========================================
    # INIT
    # ==========================================

    def __init__(
        self,
        ground_plane: Optional["GroundPlane"] = None,
        fps: float = 25.0,
    ):
        """
        `ground_plane` es la cámara calibrada contra el asfalto. Con ella el
        motor razona en metros y segundos; sin ella se comporta exactamente
        como antes, en píxeles. Una cámara sin calibrar no puede quedarse sin
        detección solo porque nadie haya marcado cuatro puntos todavía.
        """

        self._plane = ground_plane
        self._fps = max(fps, 1.0)

        # Processed-frame counter (this engine is called once per frame).
        self._frame = 0

        # Posición en metros de cada track, y la del frame anterior. Es lo
        # que permite calcular a qué velocidad se cierra una pareja, que es
        # el dato que de verdad separa un conflicto del tráfico normal:
        # en una vía urbana dos vehículos a dos metros son la norma, no una
        # anomalía, así que la distancia por sí sola no discrimina nada.
        self._world: dict[int, tuple[float, float]] = {}
        self._world_prev: dict[int, tuple[float, float]] = {}

        # Pairs that already produced a collision hit (won't re-fire).
        self.collision_pairs: set[tuple[int, int]] = set()

        # Consecutive frames a pair meets the collision conditions.
        self.candidate_frames: dict[tuple[int, int], int] = {}

        # Active collision incident clusters (merged detections).
        self._clusters: list[dict] = []

        # Frames since the last abrupt change / hard decel per track.
        self._abrupt_recency: dict[int, int] = {}

        # Highest smoothed speed ever seen per track.
        self._speed_peak: dict[int, float] = {}

        # Consecutive "stopped" frames per track.
        self._still_frames: dict[int, int] = {}

        # Was the stop that started the current still streak abrupt?
        self._stop_was_abrupt: dict[int, bool] = {}

        # Tracks already reported as stopped (cleared when moving again).
        self._stopped_reported: set[int] = set()

        # Último frame en que se vio cada track. Es lo que permite olvidar lo
        # que ya no puede volver, sin perder lo que solo está tapado.
        self._last_seen: dict[int, int] = {}

    # ==========================================
    # GEOMETRY HELPERS
    # ==========================================

    def _calculate_distance(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> float:

        dx = track_a.center[0] - track_b.center[0]
        dy = track_a.center[1] - track_b.center[1]

        return math.sqrt(dx * dx + dy * dy)

    def _vehicle_size(self, track: TrackState) -> float:
        """Average of box width and height (a scale reference)."""

        width = track.x2 - track.x1
        height = track.y2 - track.y1

        return (abs(width) + abs(height)) / 2.0

    def _calculate_bbox_gap(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> float:
        """
        Minimum distance between the two boxes.
        0 if they touch or overlap.
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

        return math.sqrt(horizontal_gap ** 2 + vertical_gap ** 2)

    def _calculate_iou(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> float:

        x1 = max(track_a.x1, track_b.x1)
        y1 = max(track_a.y1, track_b.y1)
        x2 = min(track_a.x2, track_b.x2)
        y2 = min(track_a.y2, track_b.y2)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        intersection = inter_w * inter_h

        if intersection <= 0:
            return 0.0

        area_a = (track_a.x2 - track_a.x1) * (track_a.y2 - track_a.y1)
        area_b = (track_b.x2 - track_b.x1) * (track_b.y2 - track_b.y1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _are_approaching(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> bool:
        """
        Whether the two objects are moving roughly toward
        each other.
        """

        dx = track_b.center[0] - track_a.center[0]
        dy = track_b.center[1] - track_a.center[1]

        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= 0:
            return True

        direction_x = dx / distance
        direction_y = dy / distance

        va_x, va_y = track_a.movement
        vb_x, vb_y = track_b.movement

        projection_a = va_x * direction_x + va_y * direction_y
        projection_b = vb_x * direction_x + vb_y * direction_y

        return projection_a > 0 and projection_b < 0

    def _calculate_incident_bbox(
        self,
        *tracks: TrackState,
    ) -> dict[str, float]:

        return {
            "x1": float(min(t.x1 for t in tracks)),
            "y1": float(min(t.y1 for t in tracks)),
            "x2": float(max(t.x2 for t in tracks)),
            "y2": float(max(t.y2 for t in tracks)),
        }

    # ==========================================
    # PER-TRACK STATE
    # ==========================================

    def _recent_abrupt(self, track_id: int) -> bool:
        return (
            self._abrupt_recency.get(track_id, 999)
            <= self.ABRUPT_RECENT_FRAMES
        )

    def _ever_moved(self, track_id: int) -> bool:
        return self._speed_peak.get(track_id, 0.0) >= self.MOVING_SPEED

    def _update_track_state(
        self,
        tracks: list[TrackState],
        motion_by_id: dict[int, MotionAnalysis],
    ) -> None:

        active_ids = set()

        for track in tracks:

            active_ids.add(track.track_id)
            self._last_seen[track.track_id] = self._frame

            motion = motion_by_id.get(track.track_id)

            if motion is None:
                continue

            track_id = track.track_id
            speed = motion.speed

            # --- abrupt / hard-decel recency ---
            hard = (
                motion.abrupt_change
                or abs(motion.acceleration or 0.0) >= self.HARD_DECEL
            )

            if hard:
                self._abrupt_recency[track_id] = 0
            else:
                self._abrupt_recency[track_id] = (
                    self._abrupt_recency.get(track_id, 999) + 1
                )

            # --- speed peak ---
            self._speed_peak[track_id] = max(
                self._speed_peak.get(track_id, 0.0),
                speed,
            )

            # --- stopped streak ---
            if speed <= self.STOP_SPEED:

                previous = self._still_frames.get(track_id, 0)

                if previous == 0:
                    # just stopped: was it an abrupt stop?
                    self._stop_was_abrupt[track_id] = self._recent_abrupt(
                        track_id
                    )

                self._still_frames[track_id] = previous + 1

            else:

                self._still_frames[track_id] = 0

                if speed >= self.MOVING_SPEED:
                    # moving again -> allow a future report
                    self._stopped_reported.discard(track_id)

        # --- light GC for tracks that disappeared ---
        gone = [
            track_id
            for track_id in list(self._abrupt_recency)
            if track_id not in active_ids
        ]

        for track_id in gone:
            self._abrupt_recency.pop(track_id, None)
            self._still_frames.pop(track_id, None)
            self._stop_was_abrupt.pop(track_id, None)
            # _speed_peak y _stopped_reported NO se tocan aquí: tienen que
            # sobrevivir a una oclusión. Vencen aparte, mucho más tarde.

        self._forget_stale()

    def _forget_stale(self) -> None:
        """
        Soltar lo que se guarda de tracks que ya no pueden volver.

        `_speed_peak`, `_stopped_reported` y `collision_pairs` se conservan a
        propósito cuando un track desaparece, porque puede estar tapado unos
        frames. El problema nunca fue conservarlos, era que no vencían: en una
        cámara que lleva semanas abierta eso es una fuga lenta, una entrada por
        cada vehículo que ha pasado por la vía.

        Una pareja de colisión se olvida en cuanto se olvida cualquiera de sus
        dos vehículos: sin uno de los dos, esa pareja no se puede repetir.
        """

        if self._frame % self.FORGET_EVERY_FRAMES:
            return

        horizon = self._frame - self.FORGET_AFTER_FRAMES

        stale = {
            track_id
            for track_id, seen in self._last_seen.items()
            if seen < horizon
        }

        if not stale:
            return

        for track_id in stale:
            self._last_seen.pop(track_id, None)
            self._speed_peak.pop(track_id, None)
            self._stopped_reported.discard(track_id)

        self.collision_pairs = {
            pair
            for pair in self.collision_pairs
            if pair[0] not in stale and pair[1] not in stale
        }

    # ==========================================
    # STOPPED VEHICLE
    # ==========================================

    def _detect_stopped(
        self,
        tracks: list[TrackState],
        motion_by_id: dict[int, MotionAnalysis],
    ) -> list[IncidentCandidate]:

        incidents: list[IncidentCandidate] = []

        for track in tracks:

            if track.class_name not in self.VEHICLE_CLASSES:
                continue

            track_id = track.track_id

            if track_id in self._stopped_reported:
                continue

            still = self._still_frames.get(track_id, 0)

            if still < self.STOPPED_FRAMES:
                continue

            if not self._ever_moved(track_id):
                continue

            # Esta parada ya está contada: ES el desenlace del choque que ya
            # se reportó, no un incidente aparte. Sin esto, un choque que
            # inmoviliza a dos vehículos sale como TRES incidentes —la
            # colisión y una parada por cabeza— y quien abre la bandeja ve un
            # sistema que no sabe contar. La información no se pierde: el
            # incidente de colisión lleva a los dos implicados y su desenlace.
            if self._explained_by_collision(track_id):
                # Se marca como reportada aunque no se emita nada. Si no, al
                # caducar el cluster unos segundos después el vehículo seguiría
                # quieto y la parada saldría entonces: el mismo duplicado, solo
                # que más tarde y más difícil de relacionar.
                self._stopped_reported.add(track_id)
                continue

            abrupt_stop = self._stop_was_abrupt.get(track_id, False)

            if not abrupt_stop and still < self.STOPPED_FRAMES_LONG:
                continue

            self._stopped_reported.add(track_id)

            confidence = 0.85 if abrupt_stop else 0.65

            incidents.append(
                IncidentCandidate(
                    incident_type="vehiculo_detenido",
                    incident_id=f"stop-{track_id}",
                    track_ids=[track_id],
                    confidence=confidence,
                    bbox=self._calculate_incident_bbox(track),
                    data={
                        "class": track.class_name,
                        "still_frames": still,
                        "abrupt_stop": abrupt_stop,
                        "peak_speed": round(
                            self._speed_peak.get(track_id, 0.0), 2
                        ),
                    },
                )
            )

        return incidents

    # ==========================================
    # COLLISION CONFIDENCE
    # ==========================================

    def _collision_confidence(
        self,
        *,
        center_distance: float,
        bbox_gap: float,
        iou: float,
        ref_size: float,
        recent_crash: bool,
        approaching: bool,
        motion_a: MotionAnalysis,
        motion_b: MotionAnalysis,
    ) -> float:

        confidence = 0.0

        # --- contact strength ---
        if iou >= self.CONTACT_IOU:
            confidence += 0.40
        elif bbox_gap <= 1.0:
            confidence += 0.35
        elif bbox_gap <= 0.15 * ref_size:
            confidence += 0.25
        else:
            confidence += 0.15

        # --- crash-like sudden change ---
        if recent_crash:
            confidence += 0.35

        # --- were closing in before contact ---
        if approaching:
            confidence += 0.15

        # --- instantaneous abrupt change ---
        if motion_a.abrupt_change or motion_b.abrupt_change:
            confidence += 0.10

        # --- centers really close relative to size ---
        if center_distance <= 0.6 * ref_size:
            confidence += 0.10

        return min(confidence, 1.0)

    # ==========================================
    # COLLISION HIT (pair, before clustering)
    # ==========================================

    def _collision_hit(
        self,
        track_a: TrackState,
        track_b: TrackState,
        motion_a: MotionAnalysis,
        motion_b: MotionAnalysis,
    ) -> Optional[dict]:

        if track_a.class_name not in self.VEHICLE_CLASSES:
            return None

        if track_b.class_name not in self.VEHICLE_CLASSES:
            return None

        pair = tuple(sorted([track_a.track_id, track_b.track_id]))

        if pair in self.collision_pairs:
            return None

        # --------------------------------------
        # GEOMETRY - contact relative to size
        # --------------------------------------

        ref_size = (
            self._vehicle_size(track_a) + self._vehicle_size(track_b)
        ) / 2.0

        if ref_size <= 0:
            self.candidate_frames.pop(pair, None)
            return None

        bbox_gap = self._calculate_bbox_gap(track_a, track_b)
        iou = self._calculate_iou(track_a, track_b)

        in_contact = (
            iou >= self.CONTACT_IOU
            or bbox_gap <= self.GAP_RATIO * ref_size
        )

        if not in_contact:
            self.candidate_frames.pop(pair, None)
            return None

        # --------------------------------------
        # DYNAMICS
        # --------------------------------------

        speed_a = motion_a.speed
        speed_b = motion_b.speed

        involved = (
            speed_a >= self.MIN_INVOLVED_SPEED
            or speed_b >= self.MIN_INVOLVED_SPEED
            or self._ever_moved(track_a.track_id)
            or self._ever_moved(track_b.track_id)
        )

        if not involved:
            self.candidate_frames.pop(pair, None)
            return None

        recent_crash = (
            self._recent_abrupt(track_a.track_id)
            or self._recent_abrupt(track_b.track_id)
        )

        approaching = self._are_approaching(track_a, track_b)

        # A plain "cars parked next to each other" case: in contact,
        # not approaching, no recent sudden change -> not a collision.
        if not recent_crash and not approaching and iou < self.CONTACT_IOU:
            self.candidate_frames.pop(pair, None)
            return None

        # --------------------------------------
        # CONFIDENCE
        # --------------------------------------

        center_distance = self._calculate_distance(track_a, track_b)

        confidence = self._collision_confidence(
            center_distance=center_distance,
            bbox_gap=bbox_gap,
            iou=iou,
            ref_size=ref_size,
            recent_crash=recent_crash,
            approaching=approaching,
            motion_a=motion_a,
            motion_b=motion_b,
        )

        if confidence < self.MIN_COLLISION_CONFIDENCE:
            self.candidate_frames.pop(pair, None)
            return None

        # --------------------------------------
        # TEMPORAL VALIDATION
        # --------------------------------------

        count = self.candidate_frames.get(pair, 0) + 1
        self.candidate_frames[pair] = count

        if count < self.REQUIRED_FRAMES:
            return None

        # --------------------------------------
        # HIT (merged into a cluster downstream)
        # --------------------------------------

        self.candidate_frames.pop(pair, None)

        # --------------------------------------
        # CONFLICTO EN UNIDADES FISICAS
        # --------------------------------------

        metrics = self._conflict_metrics(track_a, track_b)

        if metrics is not None:
            confidence = self._apply_conflict(confidence, metrics)

        return {
            "pair": pair,
            "track_ids": [track_a.track_id, track_b.track_id],
            "confidence": confidence,
            "bbox": self._calculate_incident_bbox(track_a, track_b),
            "data": {
                "distance_px": round(center_distance, 2),
                "bbox_gap_px": round(bbox_gap, 2),
                "iou": round(iou, 3),
                "ref_size_px": round(ref_size, 1),
                "recent_crash": recent_crash,
                "approaching": approaching,
                "class_a": track_a.class_name,
                "class_b": track_b.class_name,
                "speed_a": round(speed_a, 2),
                "speed_b": round(speed_b, 2),
                "acceleration_a": round(motion_a.acceleration or 0, 2),
                "acceleration_b": round(motion_b.acceleration or 0, 2),
                **(metrics or {}),
            },
        }

    # ==========================================
    # COLLISION CLUSTERING
    # ==========================================

    @staticmethod
    def _bbox_size(bbox: dict) -> float:
        w = bbox["x2"] - bbox["x1"]
        h = bbox["y2"] - bbox["y1"]
        return (abs(w) + abs(h)) / 2.0

    @staticmethod
    def _bbox_gap(a: dict, b: dict) -> float:
        hg = max(a["x1"] - b["x2"], b["x1"] - a["x2"], 0.0)
        vg = max(a["y1"] - b["y2"], b["y1"] - a["y2"], 0.0)
        return math.sqrt(hg * hg + vg * vg)

    @staticmethod
    def _bbox_union(a: dict, b: dict) -> dict:
        return {
            "x1": float(min(a["x1"], b["x1"])),
            "y1": float(min(a["y1"], b["y1"])),
            "x2": float(max(a["x2"], b["x2"])),
            "y2": float(max(a["y2"], b["y2"])),
        }

    def _match_cluster(self, bbox: dict) -> Optional[dict]:
        """
        Find an active cluster whose box is close to `bbox`
        and that is still within the merge time window.
        """

        for cluster in self._clusters:

            if self._frame - cluster["last_frame"] > self.MERGE_WINDOW:
                continue

            ref = (
                self._bbox_size(cluster["bbox"]) + self._bbox_size(bbox)
            ) / 2.0

            gap = self._bbox_gap(cluster["bbox"], bbox)

            if gap <= self.MERGE_NEAR_RATIO * ref:
                return cluster

        return None

    def _cluster_incident(self, cluster: dict) -> IncidentCandidate:

        confidence = self._effective_confidence(cluster)

        track_ids = sorted(cluster["track_ids"])

        return IncidentCandidate(
            incident_type="possible_collision",
            incident_id=cluster["id"],
            track_ids=track_ids,
            confidence=confidence,
            bbox=dict(cluster["bbox"]),
            data={
                **cluster["data"],
                "involved": track_ids,
                "first_frame": cluster["first_frame"],
                "last_frame": cluster["last_frame"],
                "detections": cluster["detections"],
                # Qué pasó DESPUÉS del contacto, que es lo que decide si esto
                # fue un choque. Se publica para que quien revisa vea en qué
                # se basó la confianza y no solo el número.
                "aftermath": cluster["aftermath"],
                # La puntuación geométrica cruda, antes de aplicar el
                # desenlace. Útil para recalibrar los pesos más adelante
                # contra los veredictos humanos.
                "raw_confidence": round(cluster["confidence"], 3),
                "severity": (
                    "confirmed"
                    if confidence >= self.ALERT_CONFIDENCE
                    else "pending"
                ),
            },
        )

    # ==========================================
    # METRIC CONFLICT
    # ==========================================

    @property
    def metric(self) -> bool:
        """Si esta corrida puede razonar en metros."""
        return self._plane is not None

    def _update_world_positions(self, tracks: list[TrackState]) -> None:
        """Dónde está cada vehículo sobre el asfalto, en metros."""

        if self._plane is None:
            return

        self._world_prev = self._world
        current: dict[int, tuple[float, float]] = {}

        for track in tracks:
            bbox = {
                "x1": track.x1, "y1": track.y1,
                "x2": track.x2, "y2": track.y2,
            }

            try:
                # El punto de contacto con el suelo, no el centro de la caja:
                # el centro flota a media altura del vehículo y proyectarlo
                # lo manda metros más lejos, tanto más cuanto más alto sea.
                current[track.track_id] = self._plane.to_world(
                    ground_point(bbox)
                )
            except CalibrationError:
                # Cae sobre el horizonte: no está sobre la vía.
                continue

        self._world = current

    def _conflict_metrics(
        self,
        track_a: TrackState,
        track_b: TrackState,
    ) -> Optional[dict]:
        """
        Separación, velocidad de cierre y TTC de una pareja, en unidades
        físicas. None si la cámara no está calibrada o falta información.
        """

        if self._plane is None:
            return None

        a = self._world.get(track_a.track_id)
        b = self._world.get(track_b.track_id)

        if a is None or b is None:
            return None

        separation = math.hypot(b[0] - a[0], b[1] - a[1])

        metrics = {
            "separation_m": round(separation, 2),
            "closing_speed_ms": None,
            "ttc_s": None,
        }

        prev_a = self._world_prev.get(track_a.track_id)
        prev_b = self._world_prev.get(track_b.track_id)

        if prev_a is None or prev_b is None:
            return metrics

        previous = math.hypot(prev_b[0] - prev_a[0], prev_b[1] - prev_a[1])

        # Cuánto se acortó la distancia en este frame, llevado a m/s.
        closing = (previous - separation) * self._fps

        metrics["closing_speed_ms"] = round(closing, 2)

        if closing >= self.MIN_CLOSING_SPEED:
            metrics["ttc_s"] = round(separation / closing, 2)

        return metrics

    def _apply_conflict(self, confidence: float, metrics: dict) -> float:
        """
        Corrige la puntuación geométrica con lo que dice la física.

        La geometría de la imagen dice si dos cajas se tocan; eso en una vía
        con tráfico pasa constantemente. Estas señales dicen si además se
        estaban cerrando de forma peligrosa, que es lo que separa un
        conflicto de dos vehículos circulando normalmente uno al lado del
        otro.
        """

        ttc = metrics.get("ttc_s")
        separation = metrics.get("separation_m")
        closing = metrics.get("closing_speed_ms")

        # Un TTC por debajo del umbral es la señal más fuerte que existe sin
        # esperar al desenlace: chocarían en menos de segundo y medio.
        if ttc is not None and ttc <= self.TTC_CONFLICT_S:
            return min(confidence + 0.20, 1.0)

        # Se tocan en la imagen pero no se estaban cerrando y no están
        # pegados sobre el asfalto: es tráfico normal visto desde una cámara
        # que aplasta la escena. Es el caso que más falsos positivos produce.
        if (
            separation is not None
            and separation > self.CONTACT_DISTANCE_M
            and (closing is None or closing < self.MIN_CLOSING_SPEED)
        ):
            return confidence * 0.5

        return confidence

    # ==========================================
    # AFTERMATH
    # ==========================================

    def _effective_confidence(self, cluster: dict) -> float:
        """
        La confianza que sale del motor, ya corregida por el desenlace.

        La puntuación geométrica sola nunca llega a "confirmado": dos cajas
        que se tocan en la imagen son una sospecha, no un hecho.
        """

        raw = cluster["confidence"]
        aftermath = cluster["aftermath"]

        if aftermath == "immobilized":
            return min(raw + self.AFTERMATH_BONUS, 1.0)

        if aftermath == "kept_moving":
            return raw * self.KEPT_MOVING_FACTOR

        return min(raw, self.UNCONFIRMED_CAP)

    def _explained_by_collision(self, track_id: int) -> bool:
        """
        Si la parada de este vehículo ya la explica una colisión reportada.

        Se exige la MISMA relación que usa la confirmación por desenlace: que
        la parada haya empezado a raíz del incidente. Un vehículo que ya
        estaba detenido antes del contacto no queda tapado por él —su parada
        es un hecho independiente— y se reporta como siempre.
        """

        still = self._still_frames.get(track_id, 0)

        if not still:
            return False

        stop_started = self._frame - still

        return any(
            track_id in cluster["track_ids"]
            and stop_started
            >= cluster["first_frame"] - self.AFTERMATH_STOP_TOLERANCE
            for cluster in self._clusters
        )

    def _stopped_since_incident(self, cluster: dict) -> bool:
        """
        Si alguno de los implicados se quedó quieto A RAÍZ del incidente.

        La condición de que la parada haya EMPEZADO cerca del contacto es lo
        que distingue un choque de un carro que ya llevaba rato parqueado
        justo donde otro pasó cerca: sin ella, el parqueadero de una esquina
        confirmaría incidentes toda la tarde.
        """

        for track_id in cluster["track_ids"]:

            still = self._still_frames.get(track_id, 0)

            if still < self.AFTERMATH_STOP_FRAMES:
                continue

            stop_started = self._frame - still

            if stop_started >= cluster["first_frame"] - self.AFTERMATH_STOP_TOLERANCE:
                return True

        return False

    def _update_aftermath(self) -> list[IncidentCandidate]:
        """
        Resuelve los incidentes que siguen a la espera de su desenlace.

        Devuelve los que cambiaron de estado, para que el cliente actualice
        la confianza que ya había mostrado.
        """

        updated: list[IncidentCandidate] = []

        for cluster in self._clusters:

            if cluster["aftermath"] != "pending":
                continue

            if self._stopped_since_incident(cluster):
                cluster["aftermath"] = "immobilized"
                updated.append(self._cluster_incident(cluster))
                continue

            if self._frame - cluster["first_frame"] > self.AFTERMATH_WINDOW:
                cluster["aftermath"] = "kept_moving"
                updated.append(self._cluster_incident(cluster))

        return updated

    # ==========================================
    # PROCESS
    # ==========================================

    def process(
        self,
        tracks: list[TrackState],
        motion_analysis: list[MotionAnalysis],
    ) -> list[IncidentCandidate]:

        self._frame += 1

        incidents: list[IncidentCandidate] = []

        motion_by_id = {m.track_id: m for m in motion_analysis}

        # 1. per-track state (recency, speed peak, stopped streaks)
        self._update_track_state(tracks, motion_by_id)

        # 1b. posiciones en metros, si la cámara está calibrada
        self._update_world_positions(tracks)

        # 2. stopped vehicles
        incidents.extend(self._detect_stopped(tracks, motion_by_id))

        # 3. collision hits (all vehicle pairs)
        active_pairs: set[tuple[int, int]] = set()
        hits: list[dict] = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):

                track_a = tracks[i]
                track_b = tracks[j]

                pair = tuple(sorted([track_a.track_id, track_b.track_id]))
                active_pairs.add(pair)

                motion_a = motion_by_id.get(track_a.track_id)
                motion_b = motion_by_id.get(track_b.track_id)

                if motion_a is None or motion_b is None:
                    continue

                hit = self._collision_hit(
                    track_a=track_a,
                    track_b=track_b,
                    motion_a=motion_a,
                    motion_b=motion_b,
                )

                if hit is not None:
                    hits.append(hit)

        # 4. resolve the aftermath of incidents already reported
        incidents.extend(self._update_aftermath())

        # 5. merge hits into incident clusters
        for hit in hits:

            self.collision_pairs.add(hit["pair"])

            cluster = self._match_cluster(hit["bbox"])

            if cluster is None:

                cluster = {
                    "id": (
                        f"col-{self._frame}-"
                        f"{min(hit['track_ids'])}"
                    ),
                    "track_ids": set(hit["track_ids"]),
                    "bbox": dict(hit["bbox"]),
                    "confidence": hit["confidence"],
                    "first_frame": self._frame,
                    "last_frame": self._frame,
                    "detections": 1,
                    # Nace como sospecha. Lo que pase en los próximos
                    # segundos lo confirma o lo degrada.
                    "aftermath": "pending",
                    "data": dict(hit["data"]),
                }
                self._clusters.append(cluster)
                incidents.append(self._cluster_incident(cluster))

            else:

                before = (
                    len(cluster["track_ids"]),
                    round(cluster["confidence"], 3),
                )

                cluster["track_ids"] |= set(hit["track_ids"])
                cluster["confidence"] = max(
                    cluster["confidence"], hit["confidence"]
                )
                cluster["bbox"] = self._bbox_union(
                    cluster["bbox"], hit["bbox"]
                )
                cluster["last_frame"] = self._frame
                cluster["detections"] += 1

                after = (
                    len(cluster["track_ids"]),
                    round(cluster["confidence"], 3),
                )

                if after != before:
                    incidents.append(self._cluster_incident(cluster))

        # 6. drop candidate counters for pairs no longer present
        for pair in set(self.candidate_frames) - active_pairs:
            self.candidate_frames.pop(pair, None)

        # 7. forget very old clusters (keep a margin past the merge window)
        self._clusters = [
            c
            for c in self._clusters
            if self._frame - c["last_frame"] <= self.MERGE_WINDOW * 4
        ]

        return incidents

    # ==========================================
    # RESET
    # ==========================================

    def reset(self):

        self._frame = 0
        self.collision_pairs.clear()
        self.candidate_frames.clear()
        self._clusters.clear()
        self._abrupt_recency.clear()
        self._speed_peak.clear()
        self._still_frames.clear()
        self._stop_was_abrupt.clear()
        self._stopped_reported.clear()
        self._last_seen.clear()
