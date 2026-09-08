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
    )

    def __repr__(self) -> str:
        return f"<IncidentRow {self.id} {self.incident_type!r}>"
