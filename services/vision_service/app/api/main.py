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
from ..incidents.retention import run_retention
from .protocol import PROTOCOL_VERSION
from .pipeline import create_detector
from .routes import analytics, auth, cameras, incidents, inference, shifts, users, video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision_service")

# Cada cuánto se revisa si hay evidencia vencida. Seis horas: el plazo se mide
# en días, así que revisar más seguido no adelanta nada, y menos seguido
# dejaría un servicio de fin de semana sin limpiar hasta el lunes.
_RETENTION_INTERVAL_SECONDS = 6 * 60 * 60


async def _retention_loop() -> None:
    """
    Borra la evidencia vencida cada pocas horas, mientras el servicio viva.

    Corre en un hilo aparte porque recorre directorios y habla con Postgres, y
    el bucle de asyncio está atendiendo cámaras. Un fallo aquí se registra y se
    reintenta: quedarse sin limpiar unas horas es molesto, pero tumbar el
    servicio por la limpieza sería mucho peor.
    """

    while True:
        try:
            await asyncio.to_thread(
                run_retention,
                settings.evidence_dir,
                settings.evidence_retention_days,
                settings.retention_dry_run,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "La limpieza de evidencia falló; se reintenta en la próxima "
                "pasada."
            )

        await asyncio.sleep(_RETENTION_INTERVAL_SECONDS)


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

    # Una pasada al arrancar y luego cada pocas horas. Al arrancar porque un
    # servicio que estuvo apagado una semana vuelve con evidencia vencida.
    retention = asyncio.create_task(_retention_loop())

    yield

    retention.cancel()

    # Las cámaras corren en hilos daemon, así que morirían solas con el
    # proceso; pararlas a mano libera las capturas y, con RTSP, cierra la
    # conexión en vez de dejarla colgada del lado de la cámara.
    await inference.stop_all()

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

app.include_router(auth.router)
app.include_router(shifts.router)
app.include_router(analytics.router)
app.include_router(users.router)
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
        # Rechazados por la base (típicamente la clave foránea, cuando la
        # cámara no está en la tabla `cameras`). Se muestra junto a los otros
        # dos porque un servicio que detecta bien y no persiste nada se ve
        # perfectamente sano desde afuera si este número no está.
        "incidents_failed": incident_writer.failed,
        "config_fingerprint": config_fingerprint(),
    }
