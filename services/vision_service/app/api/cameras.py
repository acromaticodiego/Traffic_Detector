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

import hashlib
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

    # Un archivo es un Path; una cámara en vivo es la URL tal cual, en str.
    # El tipo ES la marca —ver `is_stream`— porque `Path("rtsp://host/x")` no
    # es una ruta rota que se pueda detectar después: en Windows queda como
    # `rtsp:/host/x`, indistinguible de una ruta relativa cualquiera.
    source: Path | str

    # Calibración de la escena. Son por cámara porque dependen del ángulo y
    # del encuadre: reutilizar las de otra cámara da niveles equivocados.
    roi: str = ""
    perspective: float = 1.0

    # Los nueve valores de la homografía que lleva píxeles a metros sobre el
    # asfalto, o None si la cámara no está calibrada. La calcula
    # scripts/homography_picker.py marcando cuatro esquinas de medidas
    # conocidas. Sin ella el motor sigue midiendo en píxeles, que es lo que
    # hace que dos vehículos de carriles distintos parezcan tocarse.
    homography: Optional[list[float]] = None

    occupancy_medium: float = 0.22
    occupancy_high: float = 0.38
    free_speed: float = 0.08

    # Para el mapa del frontend.
    lat: Optional[float] = None
    lng: Optional[float] = None

    notes: str = ""

    @property
    def is_stream(self) -> bool:
        """
        Si la fuente es una cámara en vivo y no un archivo.

        Cambia casi todo lo que la rodea: no se comprueba en disco, no se
        acaba, no se rebobina y no se le marca el ritmo.
        """

        return isinstance(self.source, str)

    @property
    def metric(self) -> bool:
        """Si esta cámara sabe convertir píxeles en metros."""
        return bool(self.homography)

    @property
    def calibrated(self) -> bool:
        """Sin ROI la ocupación se mide sobre el frame entero: sirve, pero
        el cielo y los andenes la diluyen."""
        return bool(self.roi.strip())

    @property
    def source_key(self) -> str:
        """
        Huella de la fuente: identifica una pasada del pipeline.

        Analizar el mismo archivo dos veces da la misma huella, y por eso
        reprocesarlo reconoce los incidentes ya guardados en vez de
        duplicarlos. Sustituir el archivo —aunque conserve el nombre— la
        cambia, y el histórico nuevo no se mezcla con el viejo.

        Va con la ruta, el tamaño y la fecha en vez de con el contenido
        porque leer decenas de megas para hashearlos, cada vez que arranca
        una sesión, costaría más que todo lo que ahorra. Es el mismo criterio
        que ya usa la caché de /api/video.
        """

        if self.is_stream:
            # Una cámara en vivo no tiene tamaño ni fecha que mirar, pero
            # tampoco los necesita: la URL ya identifica la pasada.
            return hashlib.sha1(str(self.source).encode("utf-8")).hexdigest()[:16]

        try:
            stat = self.source.stat()
            crudo = f"{self.source.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
        except OSError:
            # Fuente ilegible: la ruta sola sigue agrupando lo de una misma
            # cámara, que es mejor que inventar una huella nueva cada vez.
            crudo = str(self.source)

        return hashlib.sha1(crudo.encode("utf-8")).hexdigest()[:16]

    def public(self) -> dict[str, Any]:
        """Lo que ve el frontend. La ruta del archivo no sale de aquí."""

        return {
            "id": self.id,
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "calibrated": self.calibrated,
            "metric": self.metric,
            # De una cámara en vivo no se sabe si responde sin abrirla, y
            # abrir una conexión RTSP por cada listado sería caro y lento: se
            # da por disponible y la verdad se sabe al conectarse.
            "available": True if self.is_stream else self.source.exists(),
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


# Lo que opencv sabe abrir por red. Se listan a propósito en vez de aceptar
# cualquier cosa con "://": un typo tiene que fallar como fuente inexistente,
# no convertirse en un stream que nunca conecta.
_STREAM_SCHEMES = (
    "rtsp://",
    "rtsps://",
    "rtmp://",
    "http://",
    "https://",
    "udp://",
    "tcp://",
)


def _resolve_source(raw: str) -> Path | str:
    """
    La fuente lista para `cv2.VideoCapture`.

    Las rutas relativas del YAML se leen desde la raíz del repo, no desde el
    directorio donde se lanzó el proceso. Las URL se devuelven intactas: meter
    una en un `Path` la deja irreconocible.
    """

    if raw.lower().startswith(_STREAM_SCHEMES):
        return raw

    path = Path(raw)

    return path if path.is_absolute() else REPO_ROOT / path


def _parse_homography(cam_id: str, raw: Any) -> Optional[list[float]]:
    """
    Los nueve valores, validados al cargar y no al usarse.

    Una homografía a medias es peor que ninguna: el servicio arrancaría
    igual y empezaría a medir distancias equivocadas sin que nada falle.
    """

    if raw is None or raw == "":
        return None

    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"La homografía de '{cam_id}' no es una lista de números."
        ) from error

    if len(values) != 9:
        raise ValueError(
            f"La homografía de '{cam_id}' necesita 9 valores, tiene {len(values)}."
        )

    return values


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
        homography=_parse_homography(cam_id, entry.get("homography")),
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
        homography=list(row.homography) if row.homography else None,
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
