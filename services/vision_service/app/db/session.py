"""
Motor y fábrica de sesiones.

El motor se crea perezosamente: importar este módulo no debe intentar hablar
con Postgres, porque los scripts de calibración y el propio arranque del
servicio importan el paquete sin necesitar base de datos.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..api.config import settings

# Seconds to wait for a TCP connection before giving up on Postgres.
_CONNECT_TIMEOUT_SECONDS = 3

_engine: Optional[Engine] = None
_factory: Optional[sessionmaker[Session]] = None


def get_engine() -> Engine:
    global _engine

    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            # Long-lived process with idle periods: a stale socket otherwise
            # surfaces as a random error on the first query after a pause.
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            future=True,
            # Without this, an unreachable host does not fail — it HANGS,
            # and the camera registry's fallback to cameras.yaml never runs
            # because the call never returns. The vision pipeline must never
            # block on the database.
            connect_args={"connect_timeout": _CONNECT_TIMEOUT_SECONDS},
        )

    return _engine


def get_session() -> Session:
    global _factory

    if _factory is None:
        _factory = sessionmaker(bind=get_engine(), expire_on_commit=False)

    return _factory()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Sesión con commit al salir y rollback si algo falla."""

    session = get_session()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
