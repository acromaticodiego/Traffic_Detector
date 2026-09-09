"""
Analítica de turnos y registros.

    GET /api/analytics/me            -> lo mío
    GET /api/analytics/users         -> el equipo (pide analytics:read_all)
    GET /api/analytics/users/{id}    -> el de otra persona (idem)

Mirar los propios números no pide permiso. Mirar los de otro sí, y es un
permiso aparte de `users:manage`: supervisar un turno y administrar cuentas no
tienen por qué ir en el mismo rol.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from ...analytics import summary
from ...auth.dependencies import CurrentUserDep, require
from ...db.models import PERM_ANALYTICS_READ_ALL, UserRow
from ...db.session import session_scope
from ...shifts import tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _usuario(session, user_id: int) -> UserRow:
    fila = session.get(UserRow, user_id)

    if fila is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ese usuario no existe.",
        )

    return fila


@router.get("/me")
def mi_analitica(user: CurrentUserDep) -> dict[str, Any]:
    try:
        with session_scope() as session:
            # Antes de contar nada se cierran los turnos que quedaron
            # colgados. Si no, quien cerró el portátil sin salir seguiría
            # "en turno" y su tiempo del día sería el de un turno que nadie
            # terminó nunca.
            tracker.cerrar_vencidos(session)

            return summary.resumen(session, _usuario(session, user.id))
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        logger.warning("No se pudo calcular la analítica: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base de datos no está disponible.",
        ) from error


@router.get(
    "/users",
    dependencies=[Depends(require(PERM_ANALYTICS_READ_ALL))],
)
def equipo() -> dict[str, Any]:
    """
    Quién está en turno ahora y cuánto lleva cada quien hoy.

    Es la vista de arriba: una fila por persona con lo justo para decidir a
    quién mirar de cerca, no el dashboard completo de cada una.
    """

    try:
        with session_scope() as session:
            tracker.cerrar_vencidos(session)

            filas = session.scalars(
                select(UserRow).order_by(UserRow.full_name, UserRow.email)
            ).all()

            gente = []

            for fila in filas:
                datos = summary.resumen(session, fila)

                gente.append(
                    {
                        "user": datos["user"],
                        "on_shift": datos["shift"] is not None,
                        "shift": datos["shift"],
                        "today": datos["today"],
                    }
                )

            return {"users": gente}
    except Exception as error:  # noqa: BLE001
        logger.warning("No se pudo calcular la analítica: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base de datos no está disponible.",
        ) from error


@router.get(
    "/users/{user_id}",
    dependencies=[Depends(require(PERM_ANALYTICS_READ_ALL))],
)
def analitica_de(user_id: int) -> dict[str, Any]:
    try:
        with session_scope() as session:
            tracker.cerrar_vencidos(session)

            return summary.resumen(session, _usuario(session, user_id))
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        logger.warning("No se pudo calcular la analítica: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base de datos no está disponible.",
        ) from error
