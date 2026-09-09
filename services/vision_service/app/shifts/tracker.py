"""
Turnos de trabajo: cuánto tiempo estuvo cada quien delante de la consola.

La consola manda un latido cada minuto y aquí se decide qué hacer con el hueco
entre uno y el siguiente. La decisión no es obvia, así que conviene dejarla
escrita:

- Un hueco por debajo de la TOLERANCIA se acredita entero. Es lo que dura un
  corte de internet, un cambio de red o una pestaña que el navegador congeló
  un momento, y descontarle ese tiempo a alguien que estaba trabajando sería
  castigarlo por su conexión.

- Un hueco mayor, pero por debajo del LÍMITE, acredita la tolerancia y anota
  el resto como corte. No se descuenta en silencio: queda contado y a la vista
  en el dashboard, para que quien supervisa vea "seis horas, con un corte de
  media hora" y no un número más bajo sin explicación.

- Un hueco por encima del LÍMITE se toma como que el turno terminó. Se cierra
  en el último latido más la tolerancia, y el siguiente latido abre un turno
  nuevo. Sin esto, una pestaña olvidada abierta el viernes acumularía horas
  todo el fin de semana.

El tiempo se acumula en la fila a cada latido, ya calculado, en vez de
derivarse de `ended_at - started_at` al consultarlo. Así un reinicio del
servicio no pierde nada y un turno que quedó abierto no puede inflarse solo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..api.config import settings
from ..db.models import WorkSessionRow


@dataclass(frozen=True)
class Credito:
    """Qué hacer con el hueco entre dos latidos."""

    #: Segundos que cuentan como trabajados.
    activos: int
    #: Segundos del hueco que NO cuentan, y que se reportan como corte.
    corte: int
    #: Si el turno se da por terminado en este punto.
    cerrado: bool

    @property
    def hubo_corte(self) -> bool:
        return self.corte > 0


def acreditar(
    transcurrido: float,
    tolerancia_s: int,
    limite_s: int,
) -> Credito:
    """
    El reparto de un hueco entre tiempo trabajado, corte y fin de turno.

    Función pura a propósito: es la regla de negocio del módulo y se puede
    comprobar entera sin base de datos ni reloj.
    """

    # Un reloj que va hacia atrás —ajuste de hora, latido reordenado— no puede
    # restar tiempo trabajado.
    transcurrido = max(0.0, transcurrido)

    if transcurrido <= tolerancia_s:
        return Credito(activos=int(transcurrido), corte=0, cerrado=False)

    if transcurrido <= limite_s:
        return Credito(
            activos=tolerancia_s,
            corte=int(transcurrido) - tolerancia_s,
            cerrado=False,
        )

    # Más allá del límite ya no se estaba trabajando: se acredita la
    # tolerancia y el turno se cierra ahí.
    return Credito(activos=tolerancia_s, corte=0, cerrado=True)


def _ahora() -> datetime:
    return datetime.now(timezone.utc)


def _tolerancia() -> int:
    return settings.shift_grace_minutes * 60


def _limite() -> int:
    return settings.shift_timeout_minutes * 60


def turno_abierto(session: Session, user_id: int) -> Optional[WorkSessionRow]:
    return session.scalars(
        select(WorkSessionRow).where(
            WorkSessionRow.user_id == user_id,
            WorkSessionRow.ended_at.is_(None),
        )
    ).one_or_none()


def _cerrar(turno: WorkSessionRow, cuando: datetime, motivo: str) -> None:
    turno.ended_at = cuando
    turno.closed_by = motivo


def abrir_o_reanudar(session: Session, user_id: int) -> WorkSessionRow:
    """
    El turno vigente de alguien que acaba de entrar.

    Entrar de nuevo con un turno todavía vivo no abre otro: es la misma
    jornada vista desde otra pestaña o después de recargar.
    """

    ahora = _ahora()
    turno = turno_abierto(session, user_id)

    if turno is not None:
        credito = acreditar(
            (ahora - turno.last_seen_at).total_seconds(),
            _tolerancia(),
            _limite(),
        )

        if not credito.cerrado:
            _aplicar(turno, credito, ahora)
            return turno

        # El anterior venía de otra jornada: se cierra donde se le perdió la
        # pista y este inicio de sesión empieza uno nuevo.
        turno.active_seconds += credito.activos
        _cerrar(
            turno,
            turno.last_seen_at + timedelta(seconds=credito.activos),
            "inactividad",
        )
        session.flush()

    nuevo = WorkSessionRow(
        user_id=user_id,
        started_at=ahora,
        last_seen_at=ahora,
        active_seconds=0,
        gap_count=0,
        gap_seconds=0,
    )
    session.add(nuevo)
    session.flush()

    return nuevo


def _aplicar(turno: WorkSessionRow, credito: Credito, ahora: datetime) -> None:
    turno.active_seconds += credito.activos
    turno.last_seen_at = ahora

    if credito.hubo_corte:
        turno.gap_count += 1
        turno.gap_seconds += credito.corte


def latido(session: Session, user_id: int) -> WorkSessionRow:
    """
    Registra que esta persona sigue delante de la consola.

    Devuelve el turno vigente, que puede ser uno nuevo si el anterior llevaba
    demasiado tiempo sin dar señales.
    """

    ahora = _ahora()
    turno = turno_abierto(session, user_id)

    if turno is None:
        return abrir_o_reanudar(session, user_id)

    credito = acreditar(
        (ahora - turno.last_seen_at).total_seconds(),
        _tolerancia(),
        _limite(),
    )

    if credito.cerrado:
        turno.active_seconds += credito.activos
        _cerrar(
            turno,
            turno.last_seen_at + timedelta(seconds=credito.activos),
            "inactividad",
        )
        session.flush()
        return abrir_o_reanudar(session, user_id)

    _aplicar(turno, credito, ahora)
    session.flush()

    return turno


def cerrar_por_salida(session: Session, user_id: int) -> Optional[WorkSessionRow]:
    """Cierra el turno al cerrar sesión. El final es AHORA, no el último
    latido: la persona estuvo hasta que pulsó salir."""

    turno = turno_abierto(session, user_id)

    if turno is None:
        return None

    ahora = _ahora()
    credito = acreditar(
        (ahora - turno.last_seen_at).total_seconds(),
        _tolerancia(),
        _limite(),
    )

    turno.active_seconds += credito.activos

    if credito.hubo_corte:
        turno.gap_count += 1
        turno.gap_seconds += credito.corte

    turno.last_seen_at = ahora
    _cerrar(turno, ahora, "salida")
    session.flush()

    return turno


def cerrar_vencidos(session: Session) -> int:
    """
    Cierra los turnos que dejaron de dar señales hace demasiado.

    Se llama al consultar la analítica: sin esto, quien cerró el portátil sin
    salir aparecería "en turno" indefinidamente, y su tiempo del día sería el
    de un turno que nadie terminó nunca.
    """

    limite = _limite()
    tolerancia = _tolerancia()
    ahora = _ahora()
    cerrados = 0

    for turno in session.scalars(
        select(WorkSessionRow).where(WorkSessionRow.ended_at.is_(None))
    ):
        transcurrido = (ahora - turno.last_seen_at).total_seconds()

        if transcurrido <= limite:
            continue

        turno.active_seconds += tolerancia
        _cerrar(
            turno,
            turno.last_seen_at + timedelta(seconds=tolerancia),
            "inactividad",
        )
        cerrados += 1

    if cerrados:
        session.flush()

    return cerrados
