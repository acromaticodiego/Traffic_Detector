"""
GET /api/incidents  -> histórico de incidentes, más recientes primero.

Filtros: camera, type, min_confidence, since (ISO 8601). La paginación es por
`limit` y `before_id`, no por offset: con inserciones llegando en vivo, un
offset se salta filas o las repite entre páginas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from ...db.models import IncidentRow
from ...db.session import session_scope

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


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
    }


@router.get("")
def list_incidents(
    camera: Optional[str] = None,
    type: Optional[str] = None,
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    since: Optional[datetime] = None,
    before_id: Optional[int] = Query(None, description="Página siguiente: el id más bajo de la anterior."),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:

    query = select(IncidentRow).order_by(IncidentRow.id.desc()).limit(limit)

    if camera:
        query = query.where(IncidentRow.camera_id == camera)

    if type:
        query = query.where(IncidentRow.incident_type == type)

    if min_confidence is not None:
        query = query.where(IncidentRow.confidence >= min_confidence)

    if since is not None:
        query = query.where(IncidentRow.detected_at >= since)

    if before_id is not None:
        query = query.where(IncidentRow.id < before_id)

    try:
        with session_scope() as session:
            rows = session.scalars(query).all()
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"Base de datos no disponible: {error}"
        )

    items = [_row_to_dict(row) for row in rows]

    return {
        "items": items,
        # Null = no hay más páginas.
        "next_before_id": items[-1]["id"] if len(items) == limit else None,
    }
