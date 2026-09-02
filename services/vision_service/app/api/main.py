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

from .config import settings
from .pipeline import create_detector
from .routes import inference, video

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

app.include_router(video.router)
app.include_router(inference.router)


@app.get("/health")
def health() -> dict:
    detector = getattr(app.state, "detector", None)

    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "video_path": str(settings.video_path),
        "video_exists": settings.video_path.exists(),
        "frame_stride": settings.frame_stride,
    }
