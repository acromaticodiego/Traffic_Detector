"""
FastAPI application for the vision service.

Wraps the existing VisionEngine pipeline so a
frontend can:

    - fetch the burned-in demo video      (GET  /api/video)
    - stream live inference over it        (WS   /ws/inference)

The YOLO model is loaded once at startup and shared
by every inference session.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .cameras import list_cameras, registry_source
from .config import config_fingerprint, redact_url, settings
from ..db.incident_writer import incident_writer
from .protocol import PROTOCOL_VERSION
from .pipeline import create_detector
from .routes import cameras, incidents, inference, video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision_service")


def _quiet_connection_reset(loop, context):
    """
    On Windows the Proactor event loop logs a noisy traceback
    (ConnectionResetError / WinError 10054) whenever a browser
    drops a socket abruptly - e.g. on page refresh. It is
    harmless; swallow just that case and defer everything else.
    """

    exc = context.get("exception")

    if isinstance(exc, ConnectionResetError):
        return

    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.get_running_loop().set_exception_handler(_quiet_connection_reset)

    logger.info("Vision service config: %s", settings.describe())
    logger.info("Loading YOLO model (once)...")

    app.state.detector = create_detector()

    logger.info("Model ready. Classes: %s", app.state.detector.model.names)

    yield

    incident_writer.stop()
    app.state.detector = None


app = FastAPI(title="Traffic Detector - Vision Service", lifespan=lifespan)

_allow_all = settings.cors_origins == ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # "*" + credentials is rejected by browsers, so only
    # enable credentials when explicit origins are configured.
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cameras.router)
app.include_router(incidents.router)
app.include_router(video.router)
app.include_router(inference.router)


@app.get("/health")
def health() -> dict:
    detector = getattr(app.state, "detector", None)

    registry = list_cameras()

    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "frame_stride": settings.frame_stride,
        # Config identity: two services on the same port used to be
        # indistinguishable from the outside, which cost a long debugging
        # session. These fields make a stale process obvious at a glance.
        "protocol": PROTOCOL_VERSION,
        "cameras": [c.id for c in registry],
        "camera_registry": registry_source(),
        "database": redact_url(settings.database_url),
        "incidents_written": incident_writer.written,
        "incidents_dropped": incident_writer.dropped,
        "config_fingerprint": config_fingerprint(),
    }
