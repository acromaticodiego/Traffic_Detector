"""
Cómo va el servicio ahora mismo.

`/health` dice si el proceso está vivo, que es una pregunta distinta: un
servicio con dos cámaras corriendo a media velocidad responde "ok" y se ve
perfectamente sano desde fuera mientras se va quedando atrás de la calle.

Va detrás de `stream:view` y no abierto: quien puede abrir una cámara puede
saber cómo va. Se reutiliza ese permiso a propósito, en vez de inventar uno
nuevo, porque un permiso nuevo obliga a una migración y a repartirlo por los
roles para exponer un número de frames por segundo.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from ...auth.dependencies import require
from ...db.incident_writer import incident_writer
from ...db.models import PERM_STREAM_VIEW
from .. import metrics
from ..config import settings
from . import inference

logger = logging.getLogger("vision_service.watchdog")

router = APIRouter(prefix="/api/metrics", tags=["metrics"])

_STARTED_AT = time.monotonic()


def snapshot() -> dict[str, Any]:
    """
    El estado del servicio, cámara por cámara.

    Se arma aquí y se juzga en `metrics`: esta función solo recoge.
    """

    cameras: list[dict[str, Any]] = []

    for _camera_id, session in inference.active_sessions():
        stats = session.stats()

        stats.update(
            metrics.camera_status(
                measured_fps=stats["measured_fps"],
                source_fps=stats["source_fps"],
                since_last_frame_s=stats["since_last_frame_s"],
                warmup_s=stats["uptime_s"],
            )
        )

        if stats["measured_fps"] is not None:
            stats["measured_fps"] = round(stats["measured_fps"], 1)

        cameras.append(stats)

    return {
        "status": metrics.overall_status([c["status"] for c in cameras]),
        "uptime_s": round(time.monotonic() - _STARTED_AT, 1),
        "sessions": {
            "active": len(cameras),
            "limit": inference.session_limit(),
        },
        "cameras": cameras,
        # Un servicio que detecta bien y no persiste nada se ve sano desde
        # fuera si estos números no están: `failed` sube cuando la base
        # rechaza la fila, y `dropped` cuando la cola se llenó porque
        # Postgres llevaba rato sin responder.
        "incidents": {
            "written": incident_writer.written,
            "duplicates": incident_writer.duplicates,
            "dropped": incident_writer.dropped,
            "failed": incident_writer.failed,
        },
        "gpu": metrics.gpu_snapshot(),
        "evidence": {
            "anonymized": settings.anonymize_evidence,
            "retention_days": settings.evidence_retention_days,
        },
    }


@router.get("", dependencies=[Depends(require(PERM_STREAM_VIEW))])
def service_metrics() -> dict[str, Any]:
    return snapshot()


# Cada cuánto mira el vigilante. Treinta segundos: lo bastante seguido para
# enterarse en el mismo turno, y lo bastante espaciado para no llenar el log.
_WATCHDOG_INTERVAL_SECONDS = 30


async def watchdog() -> None:
    """
    Deja constancia en el log cuando una cámara se retrasa o se cae.

    Existe porque una métrica que hay que ir a consultar no avisa de nada: a
    las tres de la mañana nadie abre /api/metrics. Esto lo deja escrito solo.

    Solo se registran los CAMBIOS de estado. Repetir el mismo aviso cada
    treinta segundos entrena a cualquiera a ignorarlo, y una alerta que se
    ignora es peor que no tenerla: da la impresión de que hay vigilancia.
    """

    previous: dict[str, str] = {}

    while True:
        await asyncio.sleep(_WATCHDOG_INTERVAL_SECONDS)

        try:
            report = snapshot()
        except Exception:  # noqa: BLE001
            logger.exception("El vigilante de cámaras falló; se reintenta.")
            continue

        present = set()

        for camera in report["cameras"]:
            camera_id = camera["camera_id"]
            status = camera["status"]
            present.add(camera_id)

            if previous.get(camera_id) == status:
                continue

            previous[camera_id] = status

            if status == metrics.OK:
                logger.info("Cámara '%s': al día otra vez.", camera_id)
            else:
                logger.warning(
                    "Cámara '%s' %s. %s", camera_id, status, camera["detail"]
                )

        # Una cámara que se cerró deja de tener estado: si vuelve a abrirse,
        # su primer aviso tiene que salir aunque coincida con el último.
        for camera_id in set(previous) - present:
            previous.pop(camera_id, None)
