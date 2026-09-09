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

Varias cámaras pueden correr a la vez —cada sesión tiene su propio tracker—
pero cada una se procesa UNA sola vez: los operarios que abran la misma cámara
comparten sesión y reciben el mismo stream. Procesar dos veces el mismo video
gastaría el doble de GPU para producir exactamente lo mismo.

El tope de sesiones (`VISION_MAX_SESSIONS`) cuenta cámaras, no espectadores.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ...auth.dependencies import CurrentUser, load_user
from ...db.models import PERM_STREAM_VIEW
from ...db.session import session_scope
from ..cameras import Camera, get_camera
from ..config import settings
from ..registry import Rejected, SessionRegistry
from ..session import VideoSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

_registry = SessionRegistry(settings.max_sessions)


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


def _open_session(
    detector,
    loop: asyncio.AbstractEventLoop,
    camera: Camera,
    stride: int | None,
):
    """La corrutina que el registro usa para levantar esta cámara."""

    async def factory() -> VideoSession:
        session = VideoSession(detector, loop, camera, stride)

        # Abrir el video toca disco (o la red, con RTSP). En un hilo aparte
        # para no dejar clavado el bucle, que ahora atiende varias cámaras.
        try:
            await asyncio.to_thread(session.open)
        except Exception as error:  # noqa: BLE001
            raise Rejected(str(error), "unavailable") from error

        return session

    return factory


async def stop_all() -> None:
    """Apagar todas las cámaras. La llama el apagado del servicio."""

    await _registry.stop_all()


# ----------------------------------------------------------------------
# WebSocket
# ----------------------------------------------------------------------


@router.websocket("/ws/inference")
async def inference_ws(websocket: WebSocket) -> None:
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

    try:
        session, subscriber = await _registry.acquire(
            camera.id,
            _open_session(detector, loop, camera, stride),
        )

    except Rejected as rejection:
        await websocket.send_json(
            {
                "type": "error",
                "message": rejection.message,
                "code": rejection.code,
            }
        )
        await websocket.close()
        return

    # El `meta` puede ser de una sesión que lleva rato corriendo: quien entra
    # tarde se engancha en vivo, no desde el principio del video.
    await websocket.send_json(session.meta)

    try:
        async for message in subscriber.messages():
            await websocket.send_json(message)

    except WebSocketDisconnect:
        pass

    except RuntimeError:
        # send after client already gone
        pass

    finally:
        await _registry.release(camera.id, session, subscriber)

        try:
            await websocket.close()
        except RuntimeError:
            pass
