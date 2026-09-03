"""
Estimación del nivel de tráfico.

Antes esto contaba vehículos y comparaba contra un umbral fijo. Ese criterio
está mal por construcción: el conteo no sabe qué tan grande es la vía, así
que 6 carros en una avenida de 6 carriles daban "alto" aunque la calzada
estuviera vacía.

El criterio real de ingeniería de tránsito combina dos variables:

    OCUPACIÓN   qué fracción del área de la calzada está cubierta por
                vehículos (ver `road_roi.py`)

    VELOCIDAD   qué tan rápido se mueven respecto al flujo libre, medida en
                ANCHOS DE FRAME POR SEGUNDO. Los px/frame crudos no sirven
                como umbral: dependen de la resolución y de los fps del
                video, así que el mismo trancón daría números distintos en
                dos cámaras. Normalizado, un solo valor sirve para todas.

porque son las dos las que separan los casos que importan:

    ocupación baja                      -> "bajo"
    ocupación alta + velocidad normal   -> "medio"   (denso, pero fluye)
    ocupación alta + velocidad baja     -> "alto"    (congestión real)

Ambas señales se suavizan con una EMA para que el nivel no parpadee cuando el
detector pierde una caja durante un par de frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from ..motion.motion_analyzer import MotionAnalysis
from ..tracking.track_state import TrackState
from .config import settings
from .road_roi import RoadROI, build_road_roi

VEHICLE_CLASSES = {"car", "motorcycle", "truck", "bus"}

# Fracción de vehículos detenidos a partir de la cual hay cola, aunque la
# ocupación todavía no llegue al umbral alto.
_QUEUE_STOPPED_RATIO = 0.6

# Posiciones mínimas en el historial de un track para que su velocidad sea
# creíble. La EMA de MotionAnalyzer necesita un par de frames para arrancar.
_SPEED_WARMUP = 3


@dataclass
class TrafficLevel:
    level: str            # "bajo" | "medio" | "alto"
    vehicles: int         # vehículos sobre la calzada en este frame
    people: int           # peatones / ciclistas en este frame
    occupancy: float      # 0..1, ocupación suavizada de la calzada
    mean_speed: float     # anchos de frame / segundo, corregido por perspectiva
    speed_ratio: float    # 0..1, velocidad respecto al flujo libre
    stopped: float        # 0..1, fracción de vehículos detenidos
    score: float          # ocupación suavizada en %, para lectura rápida


class TrafficLevelEstimator:

    def __init__(
        self,
        roi: Optional[RoadROI] = None,
        medium: Optional[float] = None,
        high: Optional[float] = None,
        free_speed: Optional[float] = None,
        slow_ratio: Optional[float] = None,
        smoothing: Optional[float] = None,
        stride: int = 1,
    ):
        self.roi = roi if roi is not None else build_road_roi(
            settings.road_roi,
            settings.road_perspective,
        )

        self.medium = (
            medium if medium is not None else settings.traffic_occupancy_medium
        )
        self.high = (
            high if high is not None else settings.traffic_occupancy_high
        )
        self.free_speed = (
            free_speed if free_speed is not None else settings.traffic_free_speed
        )
        self.slow_ratio = (
            slow_ratio if slow_ratio is not None else settings.traffic_slow_ratio
        )
        self.smoothing = (
            smoothing if smoothing is not None else settings.traffic_smoothing
        )

        # El desplazamiento se mide entre frames PROCESADOS, así que con
        # stride=N la velocidad sale N veces más grande. Se normaliza para que
        # los umbrales no dependan del stride que pida el frontend.
        self.stride = max(1, int(stride))

        # Se completan con configure() al abrir el video.
        self.width = 0
        self.height = 0
        self.fps = 25.0

        self._occupancy: Optional[float] = None
        self._speed_ratio: Optional[float] = None

    # ------------------------------------------------------------------

    def configure(self, width: int, height: int, fps: float) -> None:
        """
        Dimensiones y fps de la fuente. Hacen falta para expresar posiciones
        en 0..1 y velocidades en anchos de frame por segundo.
        """

        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps) if fps and fps > 0 else 25.0

    # ------------------------------------------------------------------

    def update(
        self,
        tracks: list[TrackState],
        motion: Optional[Iterable[MotionAnalysis]] = None,
    ) -> TrafficLevel:

        speed_by_id = {m.track_id: m for m in motion} if motion else {}

        width = self.width
        height = self.height

        # Sin dimensiones no se puede normalizar nada; se devuelve el estado
        # anterior en vez de inventar una ocupación.
        if width <= 0 or height <= 0:
            return self._result(vehicles=0, people=0, stopped=0.0)

        boxes: list[tuple[float, float, float, float]] = []
        speeds: list[float] = []
        stopped_count = 0
        people = 0

        for track in tracks:

            if track.class_name not in VEHICLE_CLASSES:
                if self._on_road(track, width, height):
                    people += 1
                continue

            if not self._on_road(track, width, height):
                continue

            boxes.append(
                (
                    track.x1 / width,
                    track.y1 / height,
                    track.x2 / width,
                    track.y2 / height,
                )
            )

            analysis = speed_by_id.get(track.track_id)

            # Un track recién aparecido no tiene desplazamiento todavía, así
            # que su velocidad es 0 y contaría como detenido. Sin este guarda,
            # el primer frame de cada sesión (o una tanda de detecciones
            # nuevas) dispararía un "alto" falso por cola.
            if analysis is None or len(track.positions) < _SPEED_WARMUP:
                continue

            _, center_y = track.center

            # px entre frames procesados
            #   / stride          -> px por frame de origen
            #   / escala          -> corrige perspectiva (lejos se ve lento)
            #   * fps / width     -> anchos de frame por segundo
            scale = self.roi.scale_at(center_y / height)
            speed = (
                analysis.speed
                / self.stride
                / scale
                * self.fps
                / width
            )

            speeds.append(speed)

            if speed < self.free_speed * self.slow_ratio:
                stopped_count += 1

        vehicles = len(boxes)

        occupancy = self.roi.occupancy(boxes)

        # Sin vehículos la vía está libre por definición: la velocidad no
        # aporta información, así que se asume flujo libre.
        if speeds:
            mean_speed = sum(speeds) / len(speeds)
            ratio = min(mean_speed / self.free_speed, 1.0) if self.free_speed else 1.0
        else:
            ratio = 1.0

        self._occupancy = self._smooth(self._occupancy, occupancy)
        self._speed_ratio = self._smooth(self._speed_ratio, ratio)

        # Sobre los vehículos con velocidad medible, no sobre todos: los que
        # están calentando el track no son evidencia de cola en ningún sentido.
        stopped = stopped_count / len(speeds) if speeds else 0.0

        return self._result(vehicles=vehicles, people=people, stopped=stopped)

    # ------------------------------------------------------------------

    def _on_road(self, track: TrackState, width: int, height: int) -> bool:
        """
        Se ancla en el centro INFERIOR de la caja: es el punto donde el
        vehículo toca el asfalto. El centro geométrico de un bus alto puede
        caer sobre el andén aunque las ruedas estén en la vía.
        """

        x = (track.x1 + track.x2) / 2 / width
        y = track.y2 / height

        return self.roi.contains(x, y)

    def _smooth(self, previous: Optional[float], value: float) -> float:
        if previous is None:
            return value
        return self.smoothing * value + (1.0 - self.smoothing) * previous

    def _classify(self, occupancy: float, speed_ratio: float, stopped: float) -> str:

        slow = speed_ratio < self.slow_ratio

        # Congestión: la vía está llena Y no avanza.
        if occupancy >= self.high and slow:
            return "alto"

        # Cola: aún no está llena, pero la mayoría está detenida.
        if occupancy >= self.medium and stopped >= _QUEUE_STOPPED_RATIO:
            return "alto"

        # Denso pero fluyendo, o carga intermedia.
        if occupancy >= self.medium:
            return "medio"

        return "bajo"

    def _result(self, vehicles: int, people: int, stopped: float) -> TrafficLevel:

        occupancy = self._occupancy or 0.0
        speed_ratio = self._speed_ratio if self._speed_ratio is not None else 1.0

        return TrafficLevel(
            level=self._classify(occupancy, speed_ratio, stopped),
            vehicles=vehicles,
            people=people,
            occupancy=round(occupancy, 4),
            mean_speed=round(speed_ratio * self.free_speed, 4),
            speed_ratio=round(speed_ratio, 3),
            stopped=round(stopped, 3),
            score=round(occupancy * 100.0, 1),
        )

    def reset(self) -> None:
        self._occupancy = None
        self._speed_ratio = None
