"""
Serve the burned-in demo video and its metadata.

GET /api/video       -> the raw mp4 (supports HTTP Range, needed for <video> seek)
GET /api/video/meta  -> fps / dimensions / frame count / duration
"""

from __future__ import annotations

import math

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings

router = APIRouter(prefix="/api/video", tags=["video"])

# cache keyed on the file's modification time, so replacing the video
# (even with the same name) is picked up without a restart
_meta_cache: dict | None = None
_meta_mtime: float | None = None


def _read_meta() -> dict:
    global _meta_cache, _meta_mtime

    if not settings.video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {settings.video_path}",
        )

    mtime = settings.video_path.stat().st_mtime

    if _meta_cache is not None and _meta_mtime == mtime:
        return _meta_cache

    cap = cv2.VideoCapture(str(settings.video_path))

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    _meta_cache = {
        "filename": settings.video_path.name,
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": round(frame_count / fps, 3) if fps else None,
    }
    _meta_mtime = mtime

    return _meta_cache


@router.get("/meta")
def video_meta() -> dict:
    return _read_meta()


@router.get("")
def video_file() -> FileResponse:

    if not settings.video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {settings.video_path}",
        )

    # no-cache => the browser revalidates every time (ETag/Last-Modified from
    # the file stat), so swapping the video is reflected on a plain refresh
    # instead of serving a stale cached copy under the same URL.
    return FileResponse(
        path=settings.video_path,
        media_type="video/mp4",
        filename=settings.video_path.name,
        headers={"Cache-Control": "no-cache"},
    )
