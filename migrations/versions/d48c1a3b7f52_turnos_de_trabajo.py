"""Turnos de trabajo y permiso de analítica

Revision ID: d48c1a3b7f52
Revises: c3a71f0d9e28

El servicio no guardaba nada sobre cuánto tiempo trabajó cada persona: solo
`users.last_login_at`, que dice cuándo entró la última vez y nada más. Sin esta
tabla el dashboard del operario no tiene de dónde salir.

El tiempo se acumula en `active_seconds` a cada latido en vez de derivarse de
`ended_at - started_at`, para que una pestaña olvidada abierta no acumule horas
y para que un reinicio del servicio no pierda lo ya contado.

Se añade también el permiso `analytics:read_all`, que separa ver la analítica
propia —que no pide permiso, cualquiera puede mirar su turno— de ver la de los
demás. Va como permiso aparte y no colgado de `users:manage` porque supervisar
y administrar cuentas no tienen por qué ir juntos: mañana puede haber un rol
que mire los números sin poder tocar un usuario.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d48c1a3b7f52"
down_revision: Union[str, Sequence[str], None] = "c3a71f0d9e28"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PERMISO = "analytics:read_all"
_DESCRIPCION = "Ver la analítica de turnos y registros de otras personas"


def upgrade() -> None:
    op.create_table(
        "work_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_by", sa.String(length=16), nullable=True),
        sa.Column("active_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gap_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gap_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_work_sessions_user_started",
        "work_sessions",
        ["user_id", sa.text("started_at DESC")],
    )

    # Un solo turno abierto por persona: dos pestañas son la misma jornada.
    op.create_index(
        "uq_work_sessions_abierto",
        "work_sessions",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("ended_at IS NULL"),
    )

    conexion = op.get_bind()

    conexion.execute(
        sa.text("INSERT INTO permissions (code, description) VALUES (:c, :d)"),
        {"c": _PERMISO, "d": _DESCRIPCION},
    )

    # Solo al administrador. Los roles se editan por fila, así que dárselo a
    # otro más adelante no necesita una migración.
    conexion.execute(
        sa.text(
            "INSERT INTO role_permissions (role_name, permission_code) "
            "VALUES ('admin', :p)"
        ),
        {"p": _PERMISO},
    )


def downgrade() -> None:
    conexion = op.get_bind()
    conexion.execute(
        sa.text("DELETE FROM role_permissions WHERE permission_code = :p"),
        {"p": _PERMISO},
    )
    conexion.execute(
        sa.text("DELETE FROM permissions WHERE code = :p"), {"p": _PERMISO}
    )

    op.drop_index("uq_work_sessions_abierto", table_name="work_sessions")
    op.drop_index("ix_work_sessions_user_started", table_name="work_sessions")
    op.drop_table("work_sessions")
