"""
La analítica de una persona: su turno y sus registros.

Dos fuentes que ya existen y no se tocan: `work_sessions` dice cuánto tiempo
estuvo, e `incidents` —por la columna `reviewed_by`— dice qué revisó y con qué
veredicto. Aquí solo se juntan.

Los días se cortan por el huso horario del despliegue, no por UTC. Con UTC, en
Colombia "hoy" empezaría a las siete de la tarde del día anterior, y un
dashboard de turnos que no coincide con el día del operario no sirve de nada.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..api.config import settings
from ..db.models import IncidentRow, UserRow, WorkSessionRow

#: Cuántos días muestra la serie del dashboard.
DIAS_SERIE = 14

#: Cuántas revisiones recientes se listan.
RECIENTES = 25


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.timezone)


def inicio_del_dia(momento: datetime, tz: ZoneInfo) -> datetime:
    """La medianoche local del día de `momento`, expresada en UTC."""

    local = momento.astimezone(tz)
    medianoche = local.replace(hour=0, minute=0, second=0, microsecond=0)

    return medianoche.astimezone(timezone.utc)


def dia_local(momento: datetime, tz: ZoneInfo) -> date:
    return momento.astimezone(tz).date()


def serie_por_dia(
    eventos: Iterable[tuple[datetime, int, int]],
    ahora: datetime,
    tz: ZoneInfo,
    dias: int = DIAS_SERIE,
) -> list[dict[str, Any]]:
    """
    Los últimos `dias` días, incluidos los vacíos.

    Devolver solo los días con actividad haría que la gráfica mintiera: dos
    barras seguidas parecerían dos días seguidos aunque entre ellas hubiera un
    fin de semana sin trabajar.
    """

    hoy = dia_local(ahora, tz)
    cubo: dict[date, dict[str, int]] = {
        hoy - timedelta(days=n): {"active_seconds": 0, "reviewed": 0}
        for n in range(dias)
    }

    for momento, segundos, revisiones in eventos:
        dia = dia_local(momento, tz)

        if dia in cubo:
            cubo[dia]["active_seconds"] += segundos
            cubo[dia]["reviewed"] += revisiones

    return [
        {"date": dia.isoformat(), **valores} for dia, valores in sorted(cubo.items())
    ]


def validez(confirmados: int, descartados: int) -> Optional[float]:
    """
    Qué fracción de lo que esta persona revisó resultó ser un incidente real.

    `None` cuando todavía no ha emitido ningún veredicto: un cero ahí se leería
    como "se equivoca siempre" en vez de "aún no ha revisado nada".
    """

    total = confirmados + descartados

    return round(confirmados / total, 4) if total else None


def _conteos(session: Session, email: str, desde: Optional[datetime]) -> dict[str, Any]:
    consulta = select(IncidentRow.review_status, func.count()).where(
        IncidentRow.reviewed_by == email,
        IncidentRow.reviewed_at.is_not(None),
    )

    if desde is not None:
        consulta = consulta.where(IncidentRow.reviewed_at >= desde)

    filas = dict(session.execute(consulta.group_by(IncidentRow.review_status)).all())

    confirmados = int(filas.get("confirmado", 0))
    descartados = int(filas.get("descartado", 0))
    archivados = int(filas.get("archivado", 0))

    return {
        "reviewed": confirmados + descartados + archivados,
        "confirmed": confirmados,
        "discarded": descartados,
        "archived": archivados,
        "accuracy": validez(confirmados, descartados),
    }


def _turnos(session: Session, user_id: int, desde: Optional[datetime]) -> dict[str, int]:
    consulta = select(
        func.coalesce(func.sum(WorkSessionRow.active_seconds), 0),
        func.coalesce(func.sum(WorkSessionRow.gap_seconds), 0),
        func.coalesce(func.sum(WorkSessionRow.gap_count), 0),
        func.count(),
    ).where(WorkSessionRow.user_id == user_id)

    if desde is not None:
        consulta = consulta.where(WorkSessionRow.started_at >= desde)

    activos, corte, cortes, turnos = session.execute(consulta).one()

    return {
        "active_seconds": int(activos),
        "gap_seconds": int(corte),
        "gap_count": int(cortes),
        "shifts": int(turnos),
    }


def _bloque(
    session: Session, user: UserRow, desde: Optional[datetime]
) -> dict[str, Any]:
    return {**_turnos(session, user.id, desde), **_conteos(session, user.email, desde)}


def _turno_abierto(session: Session, user_id: int) -> Optional[dict[str, Any]]:
    turno = session.scalars(
        select(WorkSessionRow)
        .where(
            WorkSessionRow.user_id == user_id,
            WorkSessionRow.ended_at.is_(None),
        )
        .limit(1)
    ).one_or_none()

    if turno is None:
        return None

    return {
        "started_at": turno.started_at.isoformat(),
        "last_seen_at": turno.last_seen_at.isoformat(),
        "active_seconds": turno.active_seconds,
        "gap_count": turno.gap_count,
        "gap_seconds": turno.gap_seconds,
    }


def _recientes(session: Session, email: str) -> list[dict[str, Any]]:
    filas = session.scalars(
        select(IncidentRow)
        .where(
            IncidentRow.reviewed_by == email,
            IncidentRow.reviewed_at.is_not(None),
        )
        .order_by(IncidentRow.reviewed_at.desc())
        .limit(RECIENTES)
    ).all()

    return [
        {
            "id": fila.id,
            "camera_id": fila.camera_id,
            "incident_type": fila.incident_type,
            "confidence": fila.confidence,
            "review_status": fila.review_status,
            "review_note": fila.review_note,
            "reviewed_at": fila.reviewed_at.isoformat() if fila.reviewed_at else None,
            "detected_at": fila.detected_at.isoformat(),
        }
        for fila in filas
    ]


def _eventos_de_la_serie(
    session: Session, user: UserRow, desde: datetime
) -> list[tuple[datetime, int, int]]:
    """Turnos y revisiones del periodo, como (momento, segundos, revisiones)."""

    turnos = session.execute(
        select(WorkSessionRow.started_at, WorkSessionRow.active_seconds).where(
            WorkSessionRow.user_id == user.id,
            WorkSessionRow.started_at >= desde,
        )
    ).all()

    revisiones = session.execute(
        select(IncidentRow.reviewed_at).where(
            IncidentRow.reviewed_by == user.email,
            IncidentRow.reviewed_at.is_not(None),
            IncidentRow.reviewed_at >= desde,
        )
    ).all()

    return [(momento, int(segundos), 0) for momento, segundos in turnos] + [
        (momento, 0, 1) for (momento,) in revisiones
    ]


def resumen(session: Session, user: UserRow) -> dict[str, Any]:
    """El dashboard completo de una persona."""

    tz = _tz()
    ahora = datetime.now(timezone.utc)

    hoy = inicio_del_dia(ahora, tz)
    semana = hoy - timedelta(days=6)
    serie_desde = hoy - timedelta(days=DIAS_SERIE - 1)

    return {
        "generated_at": ahora.isoformat(),
        "timezone": settings.timezone,
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role_name,
            "active": user.active,
        },
        "shift": _turno_abierto(session, user.id),
        "today": _bloque(session, user, hoy),
        "week": _bloque(session, user, semana),
        "total": _bloque(session, user, None),
        "series": serie_por_dia(
            _eventos_de_la_serie(session, user, serie_desde), ahora, tz
        ),
        "recent": _recientes(session, user.email),
    }
