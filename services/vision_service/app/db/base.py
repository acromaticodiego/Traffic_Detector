"""
Base declarativa de SQLAlchemy.

Vive en su propio módulo para que Alembic pueda importar los metadatos sin
arrastrar el motor ni abrir una conexión: `alembic/env.py` solo necesita
`Base.metadata` para comparar contra la base real.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
