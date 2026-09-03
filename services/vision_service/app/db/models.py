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

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    camera: Mapped[CameraRow] = relationship(back_populates="incidents")

    __table_args__ = (
        # La consulta normal es "incidentes de esta cámara, más recientes
        # primero", y el orden descendente evita un sort en cada página.
        Index("ix_incidents_camera_detected", "camera_id", detected_at.desc()),
        Index("ix_incidents_type", "incident_type"),
    )

    def __repr__(self) -> str:
        return f"<IncidentRow {self.id} {self.incident_type!r}>"
