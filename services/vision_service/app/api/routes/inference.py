"""
WebSocket endpoint that streams live inference over
the burned-in video.

Protocol (server -> client):

    {"type": "meta",     "camera", "fps", "frame_count", "width", "height", "stride"}
    {"type": "frame",    "frame_id", "t", "tracks":[...], "incidents":[...], "events":[...]}
    {"type": "incident", "incident_type", "confidence", "track_ids", "bbox", "data", "t"}
    {"type": "done",     "frames", "processed"}
    {"type": "error",    "message"}

The camera is chosen with ?camera=<id> (see cameras.yaml); omitting it
takes the first one in the registry.

Still only one session at a time, whatever the camera: ByteTrack keeps its
state on the shared YOLO model, so two concurrent sessions would corrupt each
other's tracks. Switching camera therefore restarts the pipeline.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..cameras import get_camera
from ..session import VideoSession

router = APIRouter(tags=["inference"])

_active: VideoSession | None = None
_lock = asyncio.Lock()


def _parse_stride(websocket: WebSocket) -> int | None:
    raw = websocket.query_params.get("stride")

    if not raw:
        return None

    try:
        return max(1, int(raw))
    except ValueError:
        return None


@router.websocket("/ws/inference")
async def inference_ws(websocket: WebSocket) -> None:
    global _active

    await websocket.accept()

    detector = websocket.app.state.detector
    loop = asyncio.get_running_loop()
    stride = _parse_stride(websocket)

    try:
        camera = get_camera(websocket.query_params.get("camera"))
    except KeyError as unknown:
        await websocket.send_json(
            {"type": "error", "message": f"Cámara desconocida: {unknown.args[0]}"}
        )
        await websocket.close()
        return

    async with _lock:
        if _active is not None:
            await asyncio.to_thread(_active.stop)
            _active = None

        session = VideoSession(detector, loop, camera, stride)

        try:
            meta = session.open()
        except Exception as error:  # noqa: BLE001
            await websocket.send_json(
                {"type": "error", "message": str(error)}
            )
            await websocket.close()
            return

        _active = session

    await websocket.send_json(meta)
    session.start()

    try:
        async for message in session.messages():
            await websocket.send_json(message)

    except WebSocketDisconnect:
        pass

    except RuntimeError:
        # send after client already gone
        pass

    finally:
        await asyncio.to_thread(session.stop)

        async with _lock:
            if _active is session:
                _active = None

        try:
            await websocket.close()
        except RuntimeError:
            pass
