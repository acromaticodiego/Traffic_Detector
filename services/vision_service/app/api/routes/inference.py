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

Por ahora sigue habiendo una sola sesión a la vez, aunque ya no por una
limitación del tracker —cada sesión tiene el suyo— sino porque el registro de
sesiones por cámara todavía no está. Cambiar de cámara reinicia el pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ...auth.dependencies import CurrentUser, load_user
from ...db.models import PERM_STREAM_VIEW
from ...db.session import session_scope
from ..cameras import get_camera
from ..session import VideoSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

_active: VideoSession | None = None
_lock = asyncio.Lock()


def _authenticate(websocket: WebSocket) -> Optional[CurrentUser]:
    """El usuario del token de la query, o None si no vale."""

    token = websocket.query_params.get("token", "").strip()

    if not token:
        return None

    try:
        with session_scope() as session:
            user = CurrentUser(load_user(session, token))
    except HTTPException:
        return None
    except Exception as error:  # noqa: BLE001
        logger.warning("No se pudo validar la sesión del WS: %s", error)
        return None

    # Ver el stream es un permiso aparte: un rol podría tener acceso al
    # histórico sin poder abrir la cámara en vivo.
    if not user.can(PERM_STREAM_VIEW):
        return None

    return user


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

    # --------------------------------------------------------------
    # Autenticación
    # --------------------------------------------------------------
    # El token viaja en la query y no en una cabecera porque la API de
    # WebSocket del navegador no permite mandar cabeceras propias al abrir la
    # conexión: no hay forma de enviar `Authorization`. Es una concesión
    # conocida y acotada — la URL puede quedar en logs del servidor, así que
    # el token dura poco y esta es la única ruta que lo acepta así.
    #
    # Se acepta primero y se rechaza después, en vez de cerrar antes del
    # handshake, para poder mandar un mensaje de error que el frontend
    # entienda en vez de una desconexión muda.
    user = _authenticate(websocket)

    if user is None:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Sesión inválida o vencida. Vuelve a entrar.",
                "code": "unauthorized",
            }
        )
        await websocket.close()
        return

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
