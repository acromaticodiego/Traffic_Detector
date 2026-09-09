"""
Modelos ORM.

`cameras` replica el vocabulario de cameras.yaml a propósito: el registro
sigue leyéndose igual desde el resto del servicio, solo cambia de dónde salen
las filas. Eso permite migrar sin tocar el pipeline ni las rutas.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

# ----------------------------------------------------------------------
# Estados de revisión
# ----------------------------------------------------------------------
# Se guardan como texto y no como enum de Postgres a propósito: agregar un
# estado nuevo a un enum exige una migración con ALTER TYPE, y estos van a
# cambiar mientras se afina el flujo de trabajo del agente.

REVIEW_PENDING = "pendiente"
REVIEW_CONFIRMED = "confirmado"
REVIEW_DISCARDED = "descartado"
REVIEW_ARCHIVED = "archivado"

REVIEW_STATUSES = (
    REVIEW_PENDING,
    REVIEW_CONFIRMED,
    REVIEW_DISCARDED,
    REVIEW_ARCHIVED,
)


# ----------------------------------------------------------------------
# Roles del sistema
# ----------------------------------------------------------------------

ROLE_ADMIN = "admin"
ROLE_OPERARIO = "operario"
ROLE_ANALISTA = "analista"

# ----------------------------------------------------------------------
# Permisos
# ----------------------------------------------------------------------
# Se nombran <recurso>:<accion> para que un vistazo baste. Viven en su
# propia tabla y no como constantes del código porque el objetivo de
# normalizar esto es poder cambiar qué puede hacer un rol con una fila, sin
# desplegar.

PERM_STREAM_VIEW = "stream:view"
PERM_INCIDENTS_READ = "incidents:read"
PERM_INCIDENTS_REVIEW = "incidents:review"
PERM_INCIDENTS_SUMMARIZE = "incidents:summarize"
PERM_CAMERAS_READ = "cameras:read"
PERM_CAMERAS_WRITE = "cameras:write"
PERM_USERS_MANAGE = "users:manage"


class RoleRow(Base):
    """Un rol y los permisos que trae."""

    __tablename__ = "roles"

    # El nombre es la clave natural: aparece en el token, en los logs y en la
    # interfaz. Un serial solo agregaría un número que nadie usa.
    name: Mapped[str] = mapped_column(String(32), primary_key=True)

    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    permissions: Mapped[list["PermissionRow"]] = relationship(
        secondary="role_permissions", back_populates="roles", lazy="selectin"
    )

    users: Mapped[list["UserRow"]] = relationship(back_populates="role")

    def __repr__(self) -> str:
        return f"<RoleRow {self.name!r}>"


class PermissionRow(Base):
    """Algo que se puede hacer en el sistema."""

    __tablename__ = "permissions"

    code: Mapped[str] = mapped_column(String(64), primary_key=True)

    description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    roles: Mapped[list[RoleRow]] = relationship(
        secondary="role_permissions", back_populates="permissions"
    )

    def __repr__(self) -> str:
        return f"<PermissionRow {self.code!r}>"


class RolePermissionRow(Base):
    """Qué permisos tiene cada rol. La tabla puente."""

    __tablename__ = "role_permissions"

    role_name: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("roles.name", ondelete="CASCADE"),
        primary_key=True,
    )

    permission_code: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("permissions.code", ondelete="CASCADE"),
        primary_key=True,
    )


class UserRow(Base):
    """Una persona que puede entrar al sistema."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # El identificador con el que entra. Se guarda normalizado en minúsculas
    # para que "Juan@x.com" y "juan@x.com" no sean dos cuentas distintas.
    email: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)

    full_name: Mapped[str] = mapped_column(String(160), nullable=False, default="")

    # El hash bcrypt, nunca la contraseña. Si esta tabla se filtra, lo que se
    # filtra son hashes: recuperar las contraseñas de ahí cuesta años por
    # cuenta, que es exactamente el punto de usar bcrypt y no un SHA.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    role_name: Mapped[str] = mapped_column(
        String(32), ForeignKey("roles.name", ondelete="RESTRICT"), nullable=False
    )

    # Desactivar en vez de borrar: un usuario borrado se lleva por delante la
    # trazabilidad de qué incidentes revisó.
    active: Mapped[bool] = mapped_column(nullable=False, default=True)

    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    role: Mapped[RoleRow] = relationship(back_populates="users", lazy="selectin")

    @property
    def permissions(self) -> set[str]:
        """Los códigos de permiso que trae su rol."""
        return {p.code for p in self.role.permissions}

    def __repr__(self) -> str:
        return f"<UserRow {self.email!r} ({self.role_name})>"


class CameraRow(Base):
    """Una cámara y su calibración."""

    __tablename__ = "cameras"

    # El id lo escribe una persona (va en URLs y en el YAML), así que es la
    # clave natural: un serial solo agregaría un número que nadie usa.
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    name: Mapped[str] = mapped_column(String(160), nullable=False)

    # Ruta de archivo o URL RTSP. Nunca sale al frontend.
    source: Mapped[str] = mapped_column(Text, nullable=False)

    lat: Mapped[float | None] = mapped_column(Float)
    lng: Mapped[float | None] = mapped_column(Float)

    # Polígono normalizado "x,y x,y ...". Vacío = frame completo.
    roi: Mapped[str] = mapped_column(Text, nullable=False, default="")
    perspective: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    occupancy_medium: Mapped[float] = mapped_column(Float, nullable=False)
    occupancy_high: Mapped[float] = mapped_column(Float, nullable=False)
    free_speed: Mapped[float] = mapped_column(Float, nullable=False)

    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Homografía imagen -> metros sobre el asfalto: nueve valores, o nulo si
    # la cámara no está calibrada. En JSONB y no en nueve columnas porque
    # siempre se leen y escriben juntos: son una matriz, no nueve ajustes.
    homography: Mapped[list[float] | None] = mapped_column(JSONB)

    # Permite retirar una cámara sin perder su histórico de incidentes.
    enabled: Mapped[bool] = mapped_column(nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    incidents: Mapped[list["IncidentRow"]] = relationship(
        back_populates="camera", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<CameraRow {self.id!r}>"


class IncidentRow(Base):
    """Un incidente detectado, con la evidencia mínima para revisarlo."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    camera_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )

    # Id de agrupación que asigna el motor de incidentes. Puede repetirse
    # entre cámaras, así que no sirve como clave por sí solo.
    cluster_id: Mapped[str | None] = mapped_column(String(128))

    # De qué pasada del pipeline viene este incidente. Para una fuente en
    # fichero es la huella del archivo, así que reprocesar el mismo video
    # reconoce lo ya escrito en vez de duplicarlo, y sustituir el archivo
    # empieza un histórico nuevo.
    #
    # Hace falta junto a `cluster_id` porque ese id se deriva de los
    # track_id, que vuelven a empezar en cada pasada: es único dentro de una
    # corrida y solo dentro de ella. Los dos juntos sí identifican un
    # incidente.
    source_key: Mapped[str] = mapped_column(String(64), nullable=False)

    incident_type: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Segundo del video donde ocurre, y frame correspondiente.
    video_t: Mapped[float | None] = mapped_column(Float)
    frame_id: Mapped[int | None] = mapped_column(Integer)

    track_ids: Mapped[list[int]] = mapped_column(JSONB, nullable=False, default=list)
    bbox: Mapped[dict | None] = mapped_column(JSONB)

    # Lo que el motor haya adjuntado; se guarda tal cual para no perder
    # información al cambiar de versión.
    data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Carpeta con el frame original y el anotado, relativa a la raíz del repo.
    # Va en su propia columna y no solo dentro de `data` porque "¿qué
    # incidentes tienen evidencia?" es una pregunta de todos los días, y
    # dentro del JSONB obliga a un filtro que ningún índice ayuda. Nula
    # cuando la captura está apagada o el incidente se detectó antes de esto.
    evidence_path: Mapped[str | None] = mapped_column(Text)

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ------------------------------------------------------------------
    # Revisión humana
    # ------------------------------------------------------------------
    # El detector propone y una persona dispone. Un incidente recién
    # detectado no es un hecho, es una hipótesis con una confianza; estas
    # columnas guardan el veredicto de quien la revisó.
    #
    # Un falso positivo se marca `descartado`, nunca se borra: esa fila es el
    # único dato que dice cuántas veces se equivoca el modelo, que es
    # justamente lo que hace falta para mejorarlo. Y un clic equivocado del
    # agente se deshace.

    review_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=REVIEW_PENDING,
        default=REVIEW_PENDING,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Quién revisó. Todavía no hay autenticación, así que hoy es lo que diga
    # el cliente: sirve para dejar la traza preparada, no para confiar en ella.
    reviewed_by: Mapped[str | None] = mapped_column(String(120))

    # Por qué se descartó. Leer "no era un choque, era un bus parando" seis
    # meses después vale más que el estado a secas.
    review_note: Mapped[str | None] = mapped_column(Text)

    # ------------------------------------------------------------------
    # Resumen generado por IA
    # ------------------------------------------------------------------
    # Se guarda porque generarlo cuesta dinero y latencia: el agente abre el
    # mismo incidente varias veces mientras decide, y sin caché cada apertura
    # sería otra llamada al modelo. Nulo mientras nadie lo haya pedido.

    ai_summary: Mapped[str | None] = mapped_column(Text)

    ai_summary_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Qué modelo lo escribió. Sin esto, dentro de seis meses no hay forma de
    # saber si un resumen raro salió de un modelo que ya se reemplazó.
    ai_model: Mapped[str | None] = mapped_column(String(64))

    camera: Mapped[CameraRow] = relationship(back_populates="incidents")

    __table_args__ = (
        # La consulta normal es "incidentes de esta cámara, más recientes
        # primero", y el orden descendente evita un sort en cada página.
        Index("ix_incidents_camera_detected", "camera_id", detected_at.desc()),
        Index("ix_incidents_type", "incident_type"),
        # La consulta de la bandeja de revisión: "lo que falta por revisar,
        # de más reciente a más antiguo".
        Index("ix_incidents_review", "review_status", detected_at.desc()),
        # Un incidente por agrupación y por pasada. Es lo que hace que
        # reprocesar un video sea idempotente: sin esto cada reconexión
        # volvía a insertar el histórico entero. Parcial porque un incidente
        # sin id de agrupación no se puede deduplicar, y esas filas no deben
        # bloquearse entre sí.
        Index(
            "uq_incidents_source_cluster",
            "camera_id",
            "source_key",
            "cluster_id",
            unique=True,
            postgresql_where=cluster_id.isnot(None),
        ),
    )

    def __repr__(self) -> str:
        return f"<IncidentRow {self.id} {self.incident_type!r}>"
