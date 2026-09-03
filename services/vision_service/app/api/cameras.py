"""
Registro de cámaras.

Hasta ahora el servicio era de una sola cámara por construcción: la ruta del
video, la ROI de la calzada, la perspectiva y los umbrales de tráfico eran
valores únicos en `settings`. Eso no aguanta un despliegue real, y el propio
clip de demo lo demuestra: contiene dos cámaras con geometrías distintas, y
una sola ROI no puede describir las dos.

Aquí cada cámara es una entidad con su fuente y su calibración propia. El
`.env` global queda como respaldo: si no hay `cameras.yaml`, se sintetiza una
cámara única a partir de `settings` y el servicio se comporta como antes.

El formato del archivo es deliberadamente el mismo vocabulario que usan los
scripts de calibración, para que lo que imprime `roi_picker.py` se pueda pegar
tal cual.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import REPO_ROOT, settings

logger = logging.getLogger(__name__)

REGISTRY_PATH = REPO_ROOT / "cameras.yaml"

# La tabla es la fuente de verdad, pero consultarla en cada frame sería
# absurdo. Con esta ventana, recalibrar una cámara se refleja en segundos.
_DB_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class Camera:
    """Una cámara y todo lo que hace falta para interpretarla."""

    id: str
    name: str
    source: Path

    # Calibración de la escena. Son por cámara porque dependen del ángulo y
    # del encuadre: reutilizar las de otra cámara da niveles equivocados.
    roi: str = ""
    perspective: float = 1.0

    occupancy_medium: float = 0.22
    occupancy_high: float = 0.38
    free_speed: float = 0.08

    # Para el mapa del frontend.
    lat: Optional[float] = None
    lng: Optional[float] = None

    notes: str = ""

    @property
    def calibrated(self) -> bool:
        """Sin ROI la ocupación se mide sobre el frame entero: sirve, pero
        el cielo y los andenes la diluyen."""
        return bool(self.roi.strip())

    def public(self) -> dict[str, Any]:
        """Lo que ve el frontend. La ruta del archivo no sale de aquí."""

        return {
            "id": self.id,
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "calibrated": self.calibrated,
            "available": self.source.exists(),
            "thresholds": {
                "medium": self.occupancy_medium,
                "high": self.occupancy_high,
            },
            "notes": self.notes,
        }


def _fallback_camera() -> Camera:
    """La configuración del `.env`, envuelta como cámara única."""

    return Camera(
        id="default",
        name=settings.video_path.stem or "Cámara",
        source=settings.video_path,
        roi=settings.road_roi,
        perspective=settings.road_perspective,
        occupancy_medium=settings.traffic_occupancy_medium,
        occupancy_high=settings.traffic_occupancy_high,
        free_speed=settings.traffic_free_speed,
        lat=None,
        lng=None,
        notes="Sin cameras.yaml: configuración tomada del .env.",
    )


def _resolve_source(raw: str) -> Path:
    """Las rutas relativas del YAML se leen desde la raíz del repo, no desde
    el directorio donde se lanzó el proceso."""

    path = Path(raw)

    return path if path.is_absolute() else REPO_ROOT / path


def _parse(entry: dict[str, Any]) -> Camera:

    cam_id = str(entry.get("id") or "").strip()

    if not cam_id:
        raise ValueError("Cada cámara necesita un 'id'.")

    source = entry.get("source")

    if not source:
        raise ValueError(f"La cámara '{cam_id}' no tiene 'source'.")

    thresholds = entry.get("thresholds") or {}

    return Camera(
        id=cam_id,
        name=str(entry.get("name") or cam_id),
        source=_resolve_source(str(source)),
        roi=str(entry.get("roi") or ""),
        perspective=float(entry.get("perspective", 1.0)),
        occupancy_medium=float(
            thresholds.get("occupancy_medium", settings.traffic_occupancy_medium)
        ),
        occupancy_high=float(
            thresholds.get("occupancy_high", settings.traffic_occupancy_high)
        ),
        free_speed=float(
            thresholds.get("free_speed", settings.traffic_free_speed)
        ),
        lat=entry.get("lat"),
        lng=entry.get("lng"),
        notes=str(entry.get("notes") or ""),
    )


@dataclass
class _Registry:
    cameras: dict[str, Camera] = field(default_factory=dict)
    mtime: Optional[float] = None
    fetched_at: float = 0.0
    source: str = "none"


_registry = _Registry()
_lock = threading.Lock()


def _row_to_camera(row: Any) -> Camera:
    return Camera(
        id=row.id,
        name=row.name,
        source=_resolve_source(row.source),
        roi=row.roi or "",
        perspective=row.perspective,
        occupancy_medium=row.occupancy_medium,
        occupancy_high=row.occupancy_high,
        free_speed=row.free_speed,
        lat=row.lat,
        lng=row.lng,
        notes=row.notes or "",
    )


def _load_from_db() -> Optional[dict[str, Camera]]:
    """Las cámaras habilitadas de la tabla, o None si la base no responde.

    Una base caída no puede dejar ciego al servicio de visión: el pipeline
    sigue sirviendo con lo que haya en cameras.yaml."""

    try:
        # Import perezoso: los scripts que no tocan la base no deberían
        # pagar el arranque de SQLAlchemy ni fallar si falta el driver.
        from sqlalchemy import select

        from ..db.models import CameraRow
        from ..db.session import session_scope

        with session_scope() as session:
            rows = session.scalars(
                select(CameraRow).where(CameraRow.enabled.is_(True)).order_by(CameraRow.id)
            ).all()

        return {row.id: _row_to_camera(row) for row in rows}

    except Exception as error:  # noqa: BLE001
        logger.warning(
            "No se pudo leer las cámaras de la base (%s); se usa cameras.yaml",
            error,
        )
        return None


def _load() -> dict[str, Camera]:
    """
    El registro vigente, en orden de preferencia: tabla `cameras`, luego
    cameras.yaml, luego el `.env`.

    La tabla manda porque es donde se edita en producción; el YAML queda como
    respaldo operativo y como formato para versionar la calibración en Git.
    """

    now = time.monotonic()

    if _registry.cameras and now - _registry.fetched_at < _DB_TTL_SECONDS:
        return _registry.cameras

    from_db = _load_from_db()

    if from_db:
        _registry.cameras = from_db
        _registry.fetched_at = now
        _registry.source = "postgres"
        return from_db

    cameras = load_yaml_cameras()

    if not cameras:
        cameras = {"default": _fallback_camera()}
        _registry.source = "env"
    else:
        _registry.source = "cameras.yaml"

    _registry.cameras = cameras
    _registry.fetched_at = now

    return cameras


def registry_source() -> str:
    """De dónde salió el registro vigente: postgres, cameras.yaml o env."""

    with _lock:
        _load()
        return _registry.source


def load_yaml_cameras() -> dict[str, Camera]:
    """El contenido de cameras.yaml, sin tocar la base de datos.

    Lo usa el script de carga inicial, que necesita el YAML como origen y la
    tabla como destino."""

    if not REGISTRY_PATH.is_file():
        return {}

    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or []

    if isinstance(raw, dict):
        raw = raw.get("cameras") or []

    cameras: dict[str, Camera] = {}

    for entry in raw:
        camera = _parse(entry)

        if camera.id in cameras:
            raise ValueError(f"Id de cámara repetido: '{camera.id}'")

        cameras[camera.id] = camera

    return cameras


def list_cameras() -> list[Camera]:
    with _lock:
        return list(_load().values())


def get_camera(camera_id: Optional[str]) -> Camera:
    """La cámara pedida, o la primera del registro si no se pide ninguna."""

    with _lock:
        cameras = _load()

        # A stray space around the id (a hand-typed URL, a copy-paste) should
        # not read as "unknown camera".
        wanted = (camera_id or "").strip()

        if not wanted:
            return next(iter(cameras.values()))

        camera = cameras.get(wanted)

        if camera is None:
            raise KeyError(wanted)

        return camera
