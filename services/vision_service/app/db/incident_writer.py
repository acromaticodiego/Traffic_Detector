"""
Escritura asíncrona de incidentes.

El hilo de visión no puede esperar a Postgres: un pico de latencia o una base
caída se traducirían en frames perdidos y en el video congelándose. Por eso
los incidentes se encolan y los escribe un hilo aparte.

La cola es acotada y descarta lo que no quepa: perder una fila de histórico es
preferible a frenar la inferencia en vivo.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Los incidentes son escasos comparados con los frames; si esto se llena es
# que la base lleva rato sin responder.
_QUEUE_MAXSIZE = 500

# Señal de apagado.
_SENTINEL = object()


class IncidentWriter:
    """Cola de incidentes con un hilo escritor detrás."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()

        # Ids ya escritos en esta corrida. El motor reporta el mismo incidente
        # en frames consecutivos y la tabla guarda uno por evento, no por frame.
        self._seen: set[tuple[str, str]] = set()

        self.dropped = 0
        self.written = 0
        self.failed = 0

    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._started:
                return

            self._thread = threading.Thread(
                target=self._run, name="incident-writer", daemon=True
            )
            self._thread.start()
            self._started = True

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False

        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def reset_seen(self) -> None:
        """Al reiniciar una sesión los ids de agrupación vuelven a empezar."""
        self._seen.clear()

    # ------------------------------------------------------------------

    def submit(self, camera_id: str, incident: dict[str, Any], frame_id: int) -> None:
        """Encola un incidente si es la primera vez que se ve."""

        cluster_id = incident.get("incident_id")

        # Sin id de agrupación no hay forma de deduplicar, así que se usa el
        # tipo más los tracks: el mismo choque entre los mismos vehículos.
        key = (
            camera_id,
            str(cluster_id)
            if cluster_id
            else f"{incident.get('incident_type')}:{sorted(incident.get('track_ids') or [])}",
        )

        if key in self._seen:
            return

        self._seen.add(key)

        try:
            self._queue.put_nowait((camera_id, incident, frame_id))
        except queue.Full:
            self.dropped += 1

            if self.dropped % 50 == 1:
                logger.warning(
                    "Cola de incidentes llena; %d descartados", self.dropped
                )

    # ------------------------------------------------------------------

    def _run(self) -> None:

        while True:
            item = self._queue.get()

            if item is _SENTINEL:
                return

            try:
                self._write(item)
            except Exception as error:  # noqa: BLE001
                self.failed += 1
                logger.warning("No se pudo guardar el incidente: %s", error)

    def _write(self, item: tuple[str, dict[str, Any], int]) -> None:
        from .models import IncidentRow
        from .session import session_scope

        camera_id, incident, frame_id = item

        bbox = incident.get("bbox")

        with session_scope() as session:
            session.add(
                IncidentRow(
                    camera_id=camera_id,
                    cluster_id=incident.get("incident_id"),
                    incident_type=incident["incident_type"],
                    confidence=float(incident["confidence"]),
                    video_t=incident.get("t"),
                    frame_id=frame_id,
                    track_ids=list(incident.get("track_ids") or []),
                    bbox=dict(bbox) if isinstance(bbox, dict) else bbox,
                    data=incident.get("data") or {},
                )
            )

        self.written += 1


# Una sola cola para todo el proceso: hoy corre una sesión a la vez, y cuando
# corran varias comparten el mismo hilo escritor sin cambiar nada.
incident_writer = IncidentWriter()
