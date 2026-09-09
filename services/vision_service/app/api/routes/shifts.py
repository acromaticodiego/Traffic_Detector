"""
Turnos de trabajo.

    POST /api/shifts/heartbeat  -> sigo aquí
    POST /api/shifts/close      -> me voy

El latido no pide permiso: llevar la cuenta del propio turno es algo que
cualquier cuenta con sesión abierta tiene que poder hacer, y exigir un permiso
para ello dejaría a alguien trabajando sin que su tiempo se contara.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from ...auth.dependencies import CurrentUserDep
from ...db.session import session_scope
from ...shifts import tracker
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/shifts", tags=["shifts"])


def _publico(turno: Any) -> dict[str, Any]:
    return {
        "started_at": turno.started_at.isoformat(),
        "last_seen_at": turno.last_seen_at.isoformat(),
        "ended_at": turno.ended_at.isoformat() if turno.ended_at else None,
        "active_seconds": turno.active_seconds,
        "gap_count": turno.gap_count,
        "gap_seconds": turno.gap_seconds,
    }


@router.post("/heartbeat")
def heartbeat(user: CurrentUserDep) -> dict[str, Any]:
    """
    Suma el tiempo transcurrido desde el latido anterior.

    Devuelve el turno vigente para que la consola pueda mostrar el contador sin
    una segunda llamada, y de paso el intervalo con el que debería volver a
    latir: así el ritmo se ajusta desde el servidor y no queda clavado en el
    navegador de cada quien.
    """

    try:
        with session_scope() as session:
            turno = tracker.latido(session, user.id)
            datos = _publico(turno)
    except Exception as error:  # noqa: BLE001
        logger.warning("No se pudo registrar el latido: %s", error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se pudo registrar el turno.",
        ) from error

    return {
        "shift": datos,
        # La mitad de la tolerancia: así hacen falta dos latidos perdidos
        # seguidos para que el hueco empiece siquiera a acercarse al límite.
        "next_in_seconds": max(30, settings.shift_grace_minutes * 60 // 2),
    }


@router.post("/close")
def close(user: CurrentUserDep) -> dict[str, Any]:
    """
    Cierra el turno al salir.

    El final es el momento de pulsar salir y no el último latido: la persona
    estuvo trabajando hasta que se fue.
    """

    try:
        with session_scope() as session:
            turno = tracker.cerrar_por_salida(session, user.id)
            datos = _publico(turno) if turno is not None else None
    except Exception as error:  # noqa: BLE001
        # Cerrar sesión no puede fallar por esto: el frontend ya va a
        # descartar el token, y dejar el turno abierto lo cierra después el
        # barrido por inactividad.
        logger.warning("No se pudo cerrar el turno: %s", error)
        return {"shift": None}

    return {"shift": datos}
