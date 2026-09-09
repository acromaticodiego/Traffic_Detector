"""
Escritura asíncrona de incidentes.

El hilo de visión no puede esperar a Postgres: un pico de latencia o una base
caída se traducirían en frames perdidos y en el video congelándose. Por eso
los incidentes se encolan y los escribe un hilo aparte.

La cola es acotada y descarta lo que no quepa: perder una fila de histórico es
preferible a frenar la inferencia en vivo.

Deduplicar es cosa de dos capas. El `set` en memoria evita encolar el mismo
incidente en frames consecutivos, que es lo habitual, pero solo sabe de lo que
ha visto este proceso. La restricción única de la tabla es la que impide de
verdad los duplicados: sin ella, reprocesar un video volvía a insertar su
histórico entero, y la memoria no ayudaba porque cada sesión empezaba en
blanco.
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

        # Ids ya encolados, por cámara y pasada. El motor reporta el mismo
        # incidente en frames consecutivos y la tabla guarda uno por evento,
        # no por frame. Ya no se vacía entre sesiones: hacerlo era lo que
        # dejaba reinsertar el histórico completo en cada reconexión.
        self._seen: set[tuple[str, str, str]] = set()

        self.dropped = 0
        self.written = 0
        self.duplicates = 0
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

    def submit(
        self,
        camera_id: str,
        incident: dict[str, Any],
        frame_id: int,
        source_key: str,
    ) -> None:
        """Encola un incidente si es la primera vez que se ve.

        `source_key` identifica la pasada del pipeline de la que sale. Entra
        en la clave porque el id de agrupación se deriva de los track_id, que
        vuelven a empezar en cada pasada: sin él, dos incidentes distintos de
        dos videos distintos se taparían el uno al otro.
        """

        cluster_id = incident.get("incident_id")

        # Sin id de agrupación no hay forma de deduplicar, así que se usa el
        # tipo más los tracks: el mismo choque entre los mismos vehículos.
        key = (
            camera_id,
            source_key,
            str(cluster_id)
            if cluster_id
            else f"{incident.get('incident_type')}:{sorted(incident.get('track_ids') or [])}",
        )

        if key in self._seen:
            return

        self._seen.add(key)

        try:
            self._queue.put_nowait((camera_id, source_key, incident, frame_id))
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

                if self.failed == 1:
                    # El primer fallo se grita, no se susurra: si la tabla
                    # `cameras` está vacía —se migró pero no se corrió
                    # scripts/seed_cameras.py— la clave foránea rechaza TODOS
                    # los incidentes y el servicio sigue funcionando de lo más
                    # normal, así que el histórico se pierde entero sin que
                    # nadie se entere hasta que va a consultarlo.
                    logger.error(
                        "No se pudo guardar el incidente: %s. Si el error es "
                        "de clave foránea, la cámara no existe en la tabla "
                        "`cameras`: correr scripts/seed_cameras.py. Los "
                        "incidentes siguientes solo se avisarán cada 50.",
                        error,
                    )
                elif self.failed % 50 == 0:
                    logger.warning(
                        "%d incidentes sin guardar; último error: %s",
                        self.failed,
                        error,
                    )

    def _write(self, item: tuple[str, str, dict[str, Any], int]) -> None:
        from sqlalchemy import text
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from .models import IncidentRow
        from .session import session_scope

        camera_id, source_key, incident, frame_id = item

        bbox = incident.get("bbox")

        # ON CONFLICT DO NOTHING, y no un INSERT a secas, porque la base es el
        # único sitio que sabe lo que se escribió en una pasada anterior: el
        # `set` en memoria se va con el proceso. Es lo que hace que reprocesar
        # un video sea idempotente en vez de duplicar su histórico.
        statement = (
            pg_insert(IncidentRow)
            .values(
                camera_id=camera_id,
                source_key=source_key,
                cluster_id=incident.get("incident_id"),
                incident_type=incident["incident_type"],
                confidence=float(incident["confidence"]),
                video_t=incident.get("t"),
                frame_id=frame_id,
                track_ids=list(incident.get("track_ids") or []),
                bbox=dict(bbox) if isinstance(bbox, dict) else bbox,
                data=incident.get("data") or {},
                evidence_path=incident.get("evidence_path"),
            )
            .on_conflict_do_nothing(
                index_elements=["camera_id", "source_key", "cluster_id"],
                index_where=text("cluster_id IS NOT NULL"),
            )
        )

        with session_scope() as session:
            escritas = session.execute(statement).rowcount

        # Un duplicado no es un fallo: es la señal de que este video ya se
        # había analizado. Se cuenta aparte para poder distinguirlo.
        if escritas:
            self.written += 1
        else:
            self.duplicates += 1


# Una sola cola para todo el proceso: hoy corre una sesión a la vez, y cuando
# corran varias comparten el mismo hilo escritor sin cambiar nada.
incident_writer = IncidentWriter()
