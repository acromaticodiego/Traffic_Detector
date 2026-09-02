from typing import Optional
import math

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
    # INIT
    # ==========================================

    def __init__(self):

        # Processed-frame counter (this engine is called once per frame).
        self._frame = 0

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
            # _speed_peak and _stopped_reported are kept on purpose

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

        track_ids = sorted(cluster["track_ids"])

        return IncidentCandidate(
            incident_type="possible_collision",
            incident_id=cluster["id"],
            track_ids=track_ids,
            confidence=cluster["confidence"],
            bbox=dict(cluster["bbox"]),
            data={
                **cluster["data"],
                "involved": track_ids,
                "first_frame": cluster["first_frame"],
                "last_frame": cluster["last_frame"],
                "detections": cluster["detections"],
                "severity": (
                    "confirmed"
                    if cluster["confidence"] >= self.ALERT_CONFIDENCE
                    else "pending"
                ),
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

        self._frame += 1

        incidents: list[IncidentCandidate] = []

        motion_by_id = {m.track_id: m for m in motion_analysis}

        # 1. per-track state (recency, speed peak, stopped streaks)
        self._update_track_state(tracks, motion_by_id)

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

        # 4. merge hits into incident clusters
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

        # 5. drop candidate counters for pairs no longer present
        for pair in set(self.candidate_frames) - active_pairs:
            self.candidate_frames.pop(pair, None)

        # 6. forget very old clusters (keep a margin past the merge window)
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
