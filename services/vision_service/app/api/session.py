"""
VideoSession: runs the vision pipeline over the
burned-in video and streams per-frame results.

The OpenCV + YOLO loop is blocking, so it runs in a
worker thread and hands messages back to the asyncio
side through per-client queues.

Cada sesión lleva su propio tracker, así que varias pueden
correr a la vez sin mezclarse los IDs. El límite de cuántas
caben lo pone la ruta del WebSocket, no esta clase.

Una cámara se procesa UNA vez y su resultado se reparte a
todos los operarios que la estén mirando. Antes había un solo
cliente y su cola marcaba el ritmo: si el navegador se
atrasaba, el hilo de visión se bloqueaba. Con varios eso deja
de valer —el más lento le impondría su ritmo a todos— así que
ahora cada uno tiene su cola, al que se atrasa se le tiran
frames, y el ritmo lo marca el reloj (ver `_pace`).
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

import cv2

from ..detection.detector import YOLODetector
from ..geometry.homography import CalibrationError, GroundPlane
from .cameras import Camera, registry_source
from .config import settings
from .road_roi import build_road_roi
from .pipeline import build_vision_engine
from .protocol import PROTOCOL_VERSION
from ..db.incident_writer import incident_writer
from .serializers import event_to_dict, incident_to_dict, track_to_dict
from . import metrics
from .playback import Playback
from .subscribers import Subscriber
from .traffic_level import TrafficLevelEstimator

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 25.0

# Cuántos frames entran en la ventana con la que se mide el ritmo. A 25 fps
# son unos doce segundos: suficiente para no bailar con cada bache y corto
# para que una caída se note enseguida.
_RATE_WINDOW = 300


class VideoSession:

    def __init__(
        self,
        detector: YOLODetector,
        loop: asyncio.AbstractEventLoop,
        camera: Camera,
        stride: Optional[int] = None,
    ):
        self._detector = detector
        self._loop = loop
        self._camera = camera
        self._stride = max(1, stride or settings.frame_stride)
        self._lead = settings.pace_lead_seconds

        # El motor se construye en open(), no aquí: necesita los fps del
        # video para poder expresar el TTC en segundos, y esos no se conocen
        # hasta abrir la fuente.
        self._engine = None

        # Igual que el motor: depende de los fps, que no se saben hasta abrir.
        self._playback: Optional[Playback] = None

        # Every scene knob comes from the camera, not from the global config:
        # two cameras in the same deployment have different geometry, so
        # sharing one ROI or one threshold gives one of them a wrong level.
        self._traffic = TrafficLevelEstimator(
            roi=build_road_roi(camera.roi, camera.perspective),
            medium=camera.occupancy_medium,
            high=camera.occupancy_high,
            free_speed=camera.free_speed,
            stride=self._stride,
        )
        self._subscribers: set[Subscriber] = set()
        self._subs_lock = threading.Lock()

        # El `meta` se guarda porque un operario puede entrar a una cámara que
        # ya lleva rato procesándose: hay que poder dárselo sin reabrir nada.
        self.meta: dict[str, Any] = {}

        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.fps: float = _DEFAULT_FPS

        # None cuando la fuente no tiene final que contar.
        self.frame_count: Optional[int] = None
        self.width: int = 0
        self.height: int = 0

        # Los lleva el hilo de visión y los lee quien quiera saber cómo va la
        # cámara: el mensaje `done` y, más adelante, las métricas.
        self.frames_read: int = 0
        self.frames_processed: int = 0

        # Cuántas veces se ha rebobinado el archivo.
        self.laps: int = 0

        # Marcas de tiempo de los últimos frames leídos. El ritmo se mide en
        # ventana y no desde el arranque: un promedio acumulado esconde que la
        # cámara lleva un minuto atascada, que es justo lo que hay que ver.
        self._marks: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._opened_at: float = 0.0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def open(self) -> dict[str, Any]:
        """
        Open the video and return the `meta` message.
        Raises FileNotFoundError / RuntimeError on failure.
        """

        source = self._camera.source

        # A una URL no se le pregunta por el disco: si la cámara no responde,
        # se sabrá al abrirla, unas líneas más abajo.
        if not self._camera.is_stream and not source.exists():
            raise FileNotFoundError(
                f"Fuente no encontrada para '{self._camera.id}': {source}"
            )

        cap = cv2.VideoCapture(str(source))

        if not cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la fuente de '{self._camera.id}': {source}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or math.isnan(fps) or fps <= 0:
            fps = _DEFAULT_FPS

        self._cap = cap
        self.fps = float(fps)

        self._playback = Playback(
            is_stream=self._camera.is_stream,
            loop_enabled=settings.loop_source,
            fps=self.fps,
        )

        self.frame_count = self._playback.frame_count(self._reported_frames(cap))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._traffic.configure(self.width, self.height, self.fps)

        self._engine = build_vision_engine(
            self._detector,
            self._camera.id,
            ground_plane=self._ground_plane(),
            # El motor procesa 1 de cada `stride` frames, así que su reloj
            # corre a esa fracción de los fps del video. Pasarle los fps
            # crudos haría que un TTC de 3 s se reportara como 1 s.
            fps=self.fps / self._stride,
        )

        self.meta = {
            "type": "meta",
            # Lets the frontend notice it is talking to a service older than
            # itself, instead of crashing on a field that is not there.
            "protocol": PROTOCOL_VERSION,
            "camera": self._camera.public(),
            "fps": self.fps,
            # null en una fuente en vivo: no tiene final que contar. El
            # frontend lo distingue de un 0, que significaría video vacío.
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "stride": self._stride,
            "traffic_thresholds": {
                "medium": self._camera.occupancy_medium,
                "high": self._camera.occupancy_high,
            },
            # Normalized polygon of the drivable surface, so the frontend can
            # draw it over the video while calibrating. null = whole frame.
            "road_roi": self._traffic.roi.polygon,
        }

        return self.meta

    @staticmethod
    def _reported_frames(cap: cv2.VideoCapture) -> int:
        """Lo que opencv dice que dura la fuente, saneado.

        Con un stream responde cualquier cosa —cero, un negativo, un NaN— y
        `int(nan)` revienta."""

        reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if not reported or math.isnan(reported):
            return 0

        return int(reported)

    def _ground_plane(self) -> Optional[GroundPlane]:
        """
        La calibración de esta cámara, o None si no tiene o está corrupta.

        Una homografía inválida no puede impedir que la cámara funcione: se
        avisa y se sigue midiendo en píxeles, que es como funcionaba antes de
        que existiera la calibración.
        """

        if not self._camera.homography:
            logger.info(
                "La cámara '%s' no está calibrada: el motor medirá en "
                "píxeles. Calibrar con scripts/homography_picker.py.",
                self._camera.id,
            )
            return None

        try:
            return GroundPlane.from_values(self._camera.homography)
        except CalibrationError as error:
            logger.warning(
                "La homografía de '%s' no es utilizable (%s); se sigue en "
                "píxeles.",
                self._camera.id,
                error,
            )
            return None

    def start(self) -> None:
        self._opened_at = time.monotonic()

        self._warn_if_incidents_will_not_persist()

        incident_writer.start()

        self._thread = threading.Thread(
            target=self._process_video,
            name="vision-session",
            daemon=True,
        )
        self._thread.start()

    def _warn_if_incidents_will_not_persist(self) -> None:
        """
        Avisar ANTES de procesar, no después de perder el histórico.

        `incidents.camera_id` es clave foránea a `cameras.id`. Si el registro
        vigente no salió de Postgres, esta cámara no está en esa tabla y cada
        inserción va a ser rechazada —una por una, en un hilo aparte— mientras
        el servicio sigue respondiendo con total normalidad. El síntoma
        aparece días después, cuando alguien consulta el histórico y no hay
        nada.
        """

        source = registry_source()

        if source == "postgres":
            return

        logger.warning(
            "El registro de cámaras viene de '%s', no de Postgres: la cámara "
            "'%s' probablemente no existe en la tabla `cameras` y los "
            "incidentes de esta sesión se van a rechazar. Correr "
            "scripts/seed_cameras.py para persistirlos.",
            source,
            self._camera.id,
        )

    # ------------------------------------------------------------------
    # suscriptores
    # ------------------------------------------------------------------

    def add_subscriber(self) -> Subscriber:
        """Enganchar un cliente más a esta cámara."""

        subscriber = Subscriber(self._loop)

        with self._subs_lock:
            self._subscribers.add(subscriber)

        return subscriber

    def remove_subscriber(self, subscriber: Subscriber) -> None:
        with self._subs_lock:
            self._subscribers.discard(subscriber)

    @property
    def subscribers(self) -> int:
        with self._subs_lock:
            return len(self._subscribers)

    # ------------------------------------------------------------------
    # observabilidad
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """
        Los números crudos de esta cámara. El juicio lo hace `metrics`.

        Aquí no se decide si la cámara está bien: solo se mide. Separarlo deja
        el criterio en un sitio donde se puede discutir y probar sin abrir un
        video.
        """

        now = time.monotonic()
        marks = list(self._marks)

        with self._subs_lock:
            viewers = len(self._subscribers)
            dropped = sum(s.dropped for s in self._subscribers)
            pending = sum(s.pending for s in self._subscribers)

        uptime = now - self._opened_at if self._opened_at else 0.0

        return {
            "camera_id": self._camera.id,
            "source": "stream" if self._camera.is_stream else "archivo",
            "uptime_s": round(uptime, 1),
            "source_fps": round(self.fps, 1),
            "measured_fps": metrics.rate(marks, now),
            "stride": self._stride,
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            # Cuánto lleva sin producir un frame. Es lo que distingue una
            # cámara lenta de una caída.
            "since_last_frame_s": round(now - marks[-1], 1) if marks else round(uptime, 1),
            "laps": self.laps,
            "viewers": viewers,
            # Mensajes que se le tiraron a algún cliente atrasado. Si sube,
            # el problema es la red del operario, no la cámara.
            "dropped_messages": dropped,
            "pending_messages": pending,
        }

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._engine is not None:
            self._engine.reset()

        # Sin esto, un cliente que estuviera esperando mensajes se queda
        # colgado en el evento hasta que se caiga la conexión por su cuenta.
        self._broadcast(self._done_message())

    def _done_message(self) -> dict[str, Any]:
        return {
            "type": "done",
            "frames": self.frames_read,
            "processed": self.frames_processed,
        }

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------

    def _broadcast(self, message: dict[str, Any]) -> None:
        """
        Repartir el mensaje a todos los clientes conectados.

        No bloquea nunca. Antes sí lo hacía —esa era la contrapresión que
        impedía acumular frames en memoria— pero con varios espectadores
        bloquear significaría que el más lento le marca el ritmo a la cámara,
        y una cámara no puede dejar de mirar la calle porque alguien tenga mal
        wifi. El acumular se acota ahora en la cola de cada uno.
        """

        with self._subs_lock:
            targets = list(self._subscribers)

        for subscriber in targets:
            subscriber.push(message)

    def _pace(self, frame_id: int, started: float) -> None:
        if self._playback is None:
            return

        delay = self._playback.pace_delay(
            frame_id,
            time.monotonic() - started,
            self._lead,
        )

        if delay > 0:
            self._stop.wait(timeout=delay)

    def _rewind(self, cap: cv2.VideoCapture) -> bool:
        """
        Volver al principio del archivo. False si no toca o no se pudo.

        La comprobación con `grab` no sobra: un archivo truncado acepta el
        salto y sigue sin devolver frames, y eso sería un bucle vacío girando
        a toda velocidad sin que nadie lo note.
        """

        if self._playback is None or not self._playback.loops:
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not cap.grab():
            logger.warning(
                "No se pudo rebobinar la fuente de '%s'; se cierra la sesión.",
                self._camera.id,
            )
            return False

        # `grab` ya consumió el primer frame: hay que dejarlo otra vez ahí.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.laps += 1

        logger.info(
            "La fuente de '%s' volvió a empezar (vuelta %d).",
            self._camera.id,
            self.laps,
        )

        return True

    def _process_video(self) -> None:
        cap = self._cap
        assert cap is not None

        # `frame_id` no se reinicia nunca: identifica el frame dentro de la
        # sesión, y el escritor de incidentes lo usa para no confundir dos
        # pasadas. `pass_frame` sí, porque es la posición dentro del video.
        frame_id = 0
        pass_frame = 0
        processed = 0
        started = time.monotonic()

        try:
            while not self._stop.is_set():

                ok, frame = cap.read()

                if not ok:
                    if not self._rewind(cap):
                        break

                    pass_frame = 0
                    continue

                frame_id += 1
                pass_frame += 1
                self.frames_read = frame_id
                self._marks.append(time.monotonic())

                # El ritmo se marca sobre los frames leídos, no sobre los
                # procesados: con stride > 1 el video sigue durando lo mismo.
                self._pace(frame_id, started)

                # honour the stride (process frame 1, 1+stride, ...)
                if (frame_id - 1) % self._stride != 0:
                    continue

                result = self._engine.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=datetime.now(),
                )
                processed += 1
                self.frames_processed = processed

                t = self._playback.timestamp(
                    pass_frame,
                    time.monotonic() - started,
                )

                motion_by_id = {
                    m.track_id: m for m in result.motion
                }

                tracks_payload = [
                    track_to_dict(track, motion_by_id.get(track.track_id))
                    for track in result.tracks
                ]

                incidents_payload = [
                    incident_to_dict(inc, t) for inc in result.incidents
                ]

                events_payload = [
                    event_to_dict(ev) for ev in result.events
                ]

                traffic = self._traffic.update(
                    result.tracks,
                    motion=result.motion,
                )

                self._broadcast(
                    {
                        "type": "frame",
                        "frame_id": frame_id,
                        "t": round(t, 3),
                        "tracks": tracks_payload,
                        "incidents": incidents_payload,
                        "events": events_payload,
                        "traffic": {
                            "level": traffic.level,
                            "vehicles": traffic.vehicles,
                            "people": traffic.people,
                            "occupancy": traffic.occupancy,
                            "mean_speed": traffic.mean_speed,
                            "speed_ratio": traffic.speed_ratio,
                            "stopped": traffic.stopped,
                            "score": traffic.score,
                        },
                    }
                )

                # emit incidents also as standalone messages
                for inc in incidents_payload:
                    self._broadcast({"type": "incident", **inc})
                    # Queued, never written inline: the vision loop must not
                    # wait on Postgres. Repeats of the same incident across
                    # frames are dropped by the writer.
                    incident_writer.submit(
                        self._camera.id, inc, frame_id, self._camera.source_key
                    )

        except Exception as error:  # noqa: BLE001
            self._broadcast({"type": "error", "message": str(error)})
            return

        # Que un archivo se acabe es el final esperado; que una cámara en vivo
        # deje de mandar es una avería. No es el mismo mensaje.
        self._broadcast(self._playback.end_message(frame_id, processed))
