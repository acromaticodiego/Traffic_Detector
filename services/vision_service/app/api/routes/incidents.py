"""
Histórico de incidentes y su revisión humana.

    GET   /api/incidents                  -> listado, más recientes primero
    PATCH /api/incidents/{id}/review      -> confirmar / descartar / archivar
    GET   /api/incidents/{id}/evidence    -> la imagen del incidente

Filtros del listado: camera, type, status, min_confidence, since (ISO 8601).
La paginación es por `limit` y `before_id`, no por offset: con inserciones
llegando en vivo, un offset se salta filas o las repite entre páginas.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from ...ai import gemini
from ...db.models import REVIEW_STATUSES, IncidentRow
from ...db.session import session_scope
from ..config import REPO_ROOT, settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


@contextmanager
def _db(doing: str) -> Iterator[None]:
    """
    Traduce un fallo de Postgres en un 503 sin filtrar la cadena de conexión.

    El texto de un error de psycopg trae host, puerto y usuario. Eso va al
    log, donde sirve; no a la respuesta HTTP, donde es una filtración.
    """

    try:
        yield
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        logger.warning("No se pudo %s: %s", doing, error, exc_info=True)

        raise HTTPException(
            status_code=503, detail="La base de datos no está disponible"
        ) from error


def _row_to_dict(row: IncidentRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "camera_id": row.camera_id,
        "cluster_id": row.cluster_id,
        "incident_type": row.incident_type,
        "confidence": row.confidence,
        "video_t": row.video_t,
        "frame_id": row.frame_id,
        "track_ids": row.track_ids,
        "bbox": row.bbox,
        "data": row.data,
        "detected_at": row.detected_at.isoformat(),
        "review_status": row.review_status,
        "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        "reviewed_by": row.reviewed_by,
        "review_note": row.review_note,
        # El panel de revisión decide con esto si dibuja la imagen o un
        # marcador de "sin evidencia", en vez de pedir un archivo que no
        # está y mostrar un roto.
        "has_evidence": bool(row.evidence_path),
        "ai_summary": row.ai_summary,
        "ai_model": row.ai_model,
    }


@router.get("")
def list_incidents(
    camera: Optional[str] = None,
    type: Optional[str] = None,
    status: Optional[str] = Query(
        None, description=f"Uno de: {', '.join(REVIEW_STATUSES)}"
    ),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    since: Optional[datetime] = None,
    before_id: Optional[int] = Query(
        None, description="Página siguiente: el id más bajo de la anterior."
    ),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:

    if status is not None and status not in REVIEW_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Estado desconocido: '{status}'. Válidos: {', '.join(REVIEW_STATUSES)}",
        )

    query = select(IncidentRow).order_by(IncidentRow.id.desc()).limit(limit)

    if camera:
        query = query.where(IncidentRow.camera_id == camera)

    if type:
        query = query.where(IncidentRow.incident_type == type)

    if status:
        query = query.where(IncidentRow.review_status == status)

    if min_confidence is not None:
        query = query.where(IncidentRow.confidence >= min_confidence)

    if since is not None:
        query = query.where(IncidentRow.detected_at >= since)

    if before_id is not None:
        query = query.where(IncidentRow.id < before_id)

    with _db("listar los incidentes"):
        with session_scope() as session:
            rows = session.scalars(query).all()

    items = [_row_to_dict(row) for row in rows]

    return {
        "items": items,
        # Null = no hay más páginas.
        "next_before_id": items[-1]["id"] if len(items) == limit else None,
    }


# ----------------------------------------------------------------------
# Revisión
# ----------------------------------------------------------------------


class ReviewUpdate(BaseModel):
    """El veredicto de la persona que revisó el incidente."""

    status: str

    # Por qué. Importa sobre todo al descartar: "era un bus parando" explica
    # el falso positivo a quien mire la métrica dentro de seis meses.
    note: Optional[str] = Field(default=None, max_length=2000)

    # Sin autenticación todavía, esto es lo que diga el cliente. Queda la
    # traza preparada para cuando haya usuarios de verdad.
    reviewed_by: Optional[str] = Field(default=None, max_length=120)


@router.patch("/{incident_id}/review")
def review_incident(incident_id: int, update: ReviewUpdate) -> dict[str, Any]:
    """
    Cambia el estado de revisión de un incidente.

    No hay borrado: un falso positivo se marca `descartado` y la fila queda.
    Es el único registro de en qué se equivoca el detector, y deshacer un
    clic apurado es cambiar el estado otra vez.
    """

    if update.status not in REVIEW_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Estado desconocido: '{update.status}'. "
                f"Válidos: {', '.join(REVIEW_STATUSES)}"
            ),
        )

    with _db("guardar la revisión"):
        with session_scope() as session:
            row = session.get(IncidentRow, incident_id)

            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"No existe el incidente {incident_id}"
                )

            row.review_status = update.status
            row.reviewed_at = datetime.now(timezone.utc)
            row.reviewed_by = update.reviewed_by
            row.review_note = update.note

            session.flush()

            return _row_to_dict(row)


# ----------------------------------------------------------------------
# Evidencia
# ----------------------------------------------------------------------


def _evidence_file(row: IncidentRow, variant: str) -> Path:
    """
    El archivo de evidencia, verificando que esté donde debe estar.

    La ruta sale de la base, así que en principio la escribimos nosotros;
    pero una ruta de base de datos que se convierte en un archivo servido por
    HTTP es exactamente la forma de un directory traversal, y basta una fila
    editada a mano para que `../../../.env` salga por esta ruta. Se ancla
    dentro del directorio de evidencia y se comprueba después de resolver.
    """

    if not row.evidence_path:
        raise HTTPException(
            status_code=404,
            detail=f"El incidente {row.id} no tiene evidencia guardada",
        )

    stored = Path(row.evidence_path)
    resolved = (stored if stored.is_absolute() else REPO_ROOT / stored).resolve()

    if variant == "original":
        resolved = resolved.parent / "original.jpg"

    root = settings.evidence_dir.resolve()

    if not resolved.is_relative_to(root):
        raise HTTPException(
            status_code=404,
            detail="La evidencia apunta fuera del directorio de evidencia",
        )

    if not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail="La evidencia está registrada pero el archivo ya no está en disco",
        )

    return resolved


# ----------------------------------------------------------------------
# Resumen con IA
# ----------------------------------------------------------------------


@router.post("/{incident_id}/summary")
def incident_summary(incident_id: int, force: bool = False) -> dict[str, Any]:
    """
    El resumen del incidente, generándolo si aún no existe.

    Bajo demanda y con caché: se escribe la primera vez que un agente abre el
    caso y se reutiliza después. Generarlo al detectar cada incidente
    significaría pagar también por todos los falsos positivos que nadie va a
    mirar, que en este detector son la mayoría.

    `force=true` lo vuelve a generar, para cuando el resumen guardado quedó
    mal o se cambió de modelo.
    """

    with _db("leer el incidente"):
        with session_scope() as session:
            row = session.get(IncidentRow, incident_id)

            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"No existe el incidente {incident_id}"
                )

            if row.ai_summary and not force:
                return {
                    "summary": row.ai_summary,
                    "generated_at": row.ai_summary_at.isoformat()
                    if row.ai_summary_at
                    else None,
                    "model": row.ai_model,
                    "cached": True,
                }

            payload = _row_to_dict(row)

            # La imagen es opcional: sin ella el resumen es más pobre pero
            # sigue siendo útil, así que no se aborta por no tenerla.
            try:
                image = _evidence_file(row, "annotated")
            except HTTPException:
                image = None

    try:
        texto = gemini.generate(payload, image)
    except gemini.SummaryUnavailable as error:
        # 503 y no 500: no es que la petición esté mal, es que el servicio de
        # IA no está disponible. El panel lo distingue y ofrece reintentar.
        raise HTTPException(status_code=503, detail=str(error)) from error

    with _db("guardar el resumen"):
        with session_scope() as session:
            row = session.get(IncidentRow, incident_id)

            if row is not None:
                row.ai_summary = texto
                row.ai_summary_at = datetime.now(timezone.utc)
                row.ai_model = settings.gemini_model

    return {
        "summary": texto,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.gemini_model,
        "cached": False,
    }


@router.get("/{incident_id}/evidence")
def incident_evidence(
    incident_id: int,
    variant: str = Query(
        "annotated",
        pattern="^(annotated|original)$",
        description="annotated = con las cajas dibujadas; original = el frame limpio",
    ),
) -> FileResponse:

    with _db("leer el incidente"):
        with session_scope() as session:
            row = session.get(IncidentRow, incident_id)

            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"No existe el incidente {incident_id}"
                )

            path = _evidence_file(row, variant)

    return FileResponse(path, media_type="image/jpeg")
