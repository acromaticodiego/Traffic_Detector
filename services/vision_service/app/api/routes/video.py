"""
Serve a camera's source video and its metadata.

GET /api/video?camera=<id>       -> the raw mp4 (HTTP Range, needed for <video> seek)
GET /api/video/meta?camera=<id>  -> fps / dimensions / frame count / duration

Omitting `camera` falls back to the first camera in the registry.
"""

from __future__ import annotations

import math

import cv2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ...auth.dependencies import require
from ..cameras import Camera, get_camera

router = APIRouter(
    prefix="/api/video", tags=["video"],
    # Todo el módulo exige el permiso: es más seguro que
    # decorarlas una por una, porque una ruta nueva queda
    # protegida por omisión en vez de quedar abierta por olvido.
    dependencies=[Depends(require("stream:view"))],
)

# Keyed by camera id and the file's modification time, so replacing a source
# (even with the same name) is picked up without a restart, and two cameras
# never hand each other the wrong dimensions.
_meta_cache: dict[str, tuple[float, dict]] = {}


def _resolve(camera_id: str | None) -> Camera:
    try:
        camera = get_camera(camera_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Cámara desconocida: {camera_id}"
        )

    # Una cámara en vivo no tiene archivo que servir ni descargar. El
    # frontend usa esta ruta para el <video> de fondo, así que la respuesta
    # tiene que explicarlo en vez de reventar con un 500 al pedirle `.exists()`
    # a una URL.
    if camera.is_stream:
        raise HTTPException(
            status_code=404,
            detail=(
                f"'{camera.id}' es una cámara en vivo: no hay archivo de "
                f"video que servir, solo el stream de inferencia."
            ),
        )

    if not camera.source.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Fuente no encontrada para '{camera.id}': {camera.source}",
        )

    return camera


def _read_meta(camera: Camera) -> dict:

    mtime = camera.source.stat().st_mtime
    cached = _meta_cache.get(camera.id)

    if cached and cached[0] == mtime:
        return cached[1]

    cap = cv2.VideoCapture(str(camera.source))

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    meta = {
        "camera": camera.id,
        "filename": camera.source.name,
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": round(frame_count / fps, 3) if fps else None,
    }

    _meta_cache[camera.id] = (mtime, meta)

    return meta


@router.get("/meta")
def video_meta(camera: str | None = None) -> dict:
    return _read_meta(_resolve(camera))


@router.get("")
def video_file(camera: str | None = None) -> FileResponse:

    source = _resolve(camera).source

    # no-cache => the browser revalidates every time (ETag/Last-Modified from
    # the file stat), so swapping the video is reflected on a plain refresh
    # instead of serving a stale cached copy under the same URL.
    return FileResponse(
        path=source,
        media_type="video/mp4",
        filename=source.name,
        headers={"Cache-Control": "no-cache"},
    )
