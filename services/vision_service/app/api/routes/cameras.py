"""
GET /api/cameras       -> el registro de cámaras
GET /api/cameras/{id}  -> una sola

Devuelve la vista pública de cada cámara: nombre, coordenadas, umbrales y si
está calibrada y disponible. La ruta del archivo (o la URL RTSP con sus
credenciales) se queda en el servidor.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..cameras import get_camera, list_cameras

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("")
def all_cameras() -> list[dict]:
    return [camera.public() for camera in list_cameras()]


@router.get("/{camera_id}")
def one_camera(camera_id: str) -> dict:
    try:
        return get_camera(camera_id).public()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Cámara desconocida: {camera_id}")
