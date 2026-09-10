"""
Runtime configuration for the vision service API.

Every value can be overridden with an environment
variable. Sensible defaults point at the burned-in
demo video and the trained model already in the repo.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path


# Repository root: services/vision_service/app/api/config.py -> up 4
REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_env_file(path: Path) -> None:
    """
    Read KEY=value lines from a .env file into the environment.

    Hand-rolled instead of python-dotenv to avoid a dependency for ~15 lines.
    Real environment variables always win, so `set VISION_X=…` in the shell
    still overrides the file.
    """

    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():

        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


# Loaded before Settings reads anything.
_load_env_file(REPO_ROOT / ".env")


def redact_url(url: str) -> str:
    """Hide the password so a connection string can be logged or returned by
    /health without leaking the credential."""

    if "://" not in url:
        return url

    scheme, _, rest = url.partition("://")

    if "@" not in rest:
        return url

    creds, _, host = rest.partition("@")
    user, sep, _password = creds.partition(":")

    return f"{scheme}://{user}{':***' if sep else ''}@{host}"


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    """Acepta 0/1, true/false, yes/no. Cualquier otra cosa cae en el default,
    porque un typo no debería apagar en silencio la captura de evidencia."""

    value = os.getenv(name)

    if not value:
        return default

    lowered = value.strip().lower()

    if lowered in ("1", "true", "yes", "on"):
        return True

    if lowered in ("0", "false", "no", "off"):
        return False

    return default


def _default_video_path() -> Path:
    """
    Use VISION_VIDEO_PATH if given, otherwise the first
    video file found in videos/input/.
    """

    override = os.getenv("VISION_VIDEO_PATH")

    if override:
        return Path(override)

    input_dir = REPO_ROOT / "videos" / "input"

    if input_dir.is_dir():

        for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):

            matches = sorted(input_dir.glob(pattern))

            if matches:
                return matches[0]

    # Fallback (may not exist; surfaced at startup)
    return input_dir / "input.mp4"


def _cors_origins() -> list[str]:
    raw = os.getenv("VISION_CORS_ORIGINS")

    if not raw:
        return ["*"]

    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass
class Settings:

    model_path: Path = field(
        default_factory=lambda: Path(
            _env_str(
                "VISION_MODEL_PATH",
                str(REPO_ROOT / "models" / "detectorfinal.pt"),
            )
        )
    )

    video_path: Path = field(
        default_factory=_default_video_path
    )

    confidence: float = field(
        default_factory=lambda: _env_float("VISION_CONFIDENCE", 0.65)
    )

    iou: float = field(
        default_factory=lambda: _env_float("VISION_IOU", 0.60)
    )

    image_size: int = field(
        default_factory=lambda: _env_int("VISION_IMAGE_SIZE", 640)
    )

    # Process 1 out of every N frames (1 = every frame).
    frame_stride: int = field(
        default_factory=lambda: max(1, _env_int("VISION_FRAME_STRIDE", 1))
    )

    # Cuántas cámaras pueden estar procesándose a la vez. Todas comparten una
    # sola GPU, así que pasada cierta cantidad no se gana nada: se reparten los
    # mismos frames por segundo entre más streams y TODAS van peor, sin que
    # nada lo explique. Es preferible negar la cuarta cámara con un mensaje
    # claro. Los espectadores no cuentan: diez operarios sobre las mismas tres
    # cámaras siguen siendo tres sesiones.
    max_sessions: int = field(
        default_factory=lambda: max(1, _env_int("VISION_MAX_SESSIONS", 3))
    )

    # Volver a empezar cuando se acaba el archivo. Existe porque no hay
    # acceso a las cámaras reales y todo se prueba sobre grabaciones: un clip
    # en bucle es lo más parecido a un stream continuo que se puede tener sin
    # RTSP, y sin esto una demo se queda congelada al minuto. No afecta a las
    # fuentes en vivo, que no se rebobinan.
    loop_source: bool = field(
        default_factory=lambda: _env_bool("VISION_LOOP_SOURCE", False)
    )

    stream_frames: bool = field(
        default_factory=lambda: _env_bool("VISION_STREAM_FRAMES", True)
    )

    stream_quality: int = field(
        default_factory=lambda: min(
            95, max(30, _env_int("VISION_STREAM_QUALITY", 70))
        )
    )

    pace_lead_seconds: float = field(
        default_factory=lambda: max(
            0.0, _env_float("VISION_PACE_LEAD_SECONDS", 5.0)
        )
    )

    cors_origins: list[str] = field(default_factory=_cors_origins)

    # ------------------------------------------------------------------
    # Autenticación
    # ------------------------------------------------------------------
    # Con este secreto se firman los tokens de sesión. Quien lo tenga puede
    # fabricar un token de admin, así que va en el .env y nunca en el código.
    # Sin él el servicio arranca pero rechaza todo inicio de sesión: es
    # preferible a arrancar con un secreto por defecto que alguien olvide
    # cambiar, que es como se filtran estos sistemas.
    jwt_secret: str = field(
        default_factory=lambda: _env_str("JWT_SECRET", "")
    )

    # Cuánto dura una sesión. Un JWT no se puede revocar antes de expirar,
    # así que esto acota el daño de un token robado. 12 horas cubre un turno
    # completo sin obligar a reingresar a media jornada.
    jwt_expire_minutes: int = field(
        default_factory=lambda: _env_int("JWT_EXPIRE_MINUTES", 720)
    )

    # ------------------------------------------------------------------
    # Turnos de trabajo
    # ------------------------------------------------------------------
    # Cuánto silencio se perdona antes de dejar de contar el tiempo como
    # trabajado. Cubre el corte de internet, el cambio de red y la pestaña que
    # el navegador congela: a nadie se le descuenta el tiempo por su conexión.
    shift_grace_minutes: int = field(
        default_factory=lambda: _env_int("SHIFT_GRACE_MINUTES", 10)
    )

    # A partir de aquí se entiende que el turno terminó y el siguiente latido
    # abre uno nuevo. Sin este tope, una pestaña olvidada abierta el viernes
    # acumularía horas todo el fin de semana.
    shift_timeout_minutes: int = field(
        default_factory=lambda: _env_int("SHIFT_TIMEOUT_MINUTES", 60)
    )

    # Huso horario del despliegue. Decide dónde corta "hoy" en el dashboard:
    # con UTC, en Colombia el día empezaría a las siete de la tarde anterior y
    # las horas de un turno aparecerían repartidas entre dos días.
    timezone: str = field(
        default_factory=lambda: _env_str("VISION_TIMEZONE", "America/Bogota")
    )

    # ------------------------------------------------------------------
    # Resumen con IA (Gemini)
    # ------------------------------------------------------------------
    # La clave NO tiene valor por defecto y nunca se escribe en el código:
    # va en el .env, que está en .gitignore. Si falta, el servicio arranca
    # igual y el resumen simplemente no se ofrece — el panel de detalle
    # tiene que funcionar sin depender de un proveedor externo.
    gemini_api_key: str = field(
        default_factory=lambda: _env_str("GEMINI_API_KEY", "")
    )

    # Configurable a propósito: los identificadores de modelo de Google
    # cambian, y quedarse fijado a uno retirado convierte un cambio de
    # proveedor en un cambio de código.
    #
    # Por defecto va un ALIAS y no una versión concreta. No es descuido: el
    # `gemini-2.5-flash` que estaba aquí antes fue retirado y la función se
    # cayó con un 404 en producción. Un alias se mueve solo cuando Google
    # jubila el modelo de debajo. A cambio, el modelo puede cambiar sin aviso
    # y con él el tono del resumen: para reproducibilidad, fijar una versión
    # concreta aquí y aceptar tener que actualizarla.
    gemini_model: str = field(
        default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-flash-latest")
    )

    # Segundos antes de rendirse. Una petición colgada no puede dejar
    # esperando al agente que abrió el incidente.
    #
    # 30 se quedaban cortos: los modelos actuales razonan antes de responder y
    # el SDK reintenta por su cuenta, así que una llamada legítima puede pasar
    # del medio minuto. Con el valor viejo el agente recibía un timeout en vez
    # del motivo real del fallo.
    gemini_timeout: float = field(
        default_factory=lambda: _env_float("GEMINI_TIMEOUT", 60.0)
    )

    # Evidencia de los incidentes: por cada uno se guarda el frame original y
    # una copia anotada con las cajas de los vehículos involucrados. Es lo que
    # permite auditar después si el incidente fue real, así que va encendida
    # por defecto; se apaga con VISION_EVIDENCE=0 cuando el disco es el
    # problema (dos JPEG por incidente, sin límite de tamaño).
    evidence_enabled: bool = field(
        default_factory=lambda: _env_bool("VISION_EVIDENCE", True)
    )

    evidence_dir: Path = field(
        default_factory=lambda: Path(
            _env_str(
                "VISION_EVIDENCE_DIR",
                str(REPO_ROOT / "outputs" / "incidents"),
            )
        )
    )

    # ------------------------------------------------------------------
    # Anonimización de la evidencia (Ley 1581 de 2012, Habeas Data)
    # ------------------------------------------------------------------
    # Placas y rostros se tapan ANTES de escribir la imagen, así que el
    # original identificable no llega a existir en disco. Va encendido por
    # defecto porque el descuido aquí no se nota hasta que lo encuentra un
    # abogado: apagarlo es una decisión que alguien tiene que tomar a
    # conciencia, no algo que pase por olvido.
    anonymize_evidence: bool = field(
        default_factory=lambda: _env_bool("VISION_ANONYMIZE", True)
    )

    # Qué fracción de cada caja se tapa: abajo la placa, arriba la cabeza. Se
    # pueden ajustar porque dependen del ángulo de la cámara —una toma más
    # cenital ve la placa más arriba dentro del recuadro—, pero conviene
    # moverlas hacia arriba, no hacia abajo: quedarse corto deja una placa
    # legible guardada para siempre.
    anonymize_plate_band: float = field(
        default_factory=lambda: _env_float("VISION_ANONYMIZE_PLATE_BAND", 0.35)
    )

    anonymize_face_band: float = field(
        default_factory=lambda: _env_float("VISION_ANONYMIZE_FACE_BAND", 0.30)
    )

    # ------------------------------------------------------------------
    # Retención de la evidencia
    # ------------------------------------------------------------------
    # Cuántos días se conserva la imagen de un incidente. El incidente en sí
    # NO se borra nunca: sigue en la bandeja con su veredicto, que es de donde
    # sale la métrica de precisión. Lo que caduca es la foto, que es el dato
    # personal.
    #
    # Cero o negativo = conservar indefinidamente, que es el valor por defecto
    # a propósito: este número borra archivos, y una variable vacía o un typo
    # tienen que dejar el disco intacto, nunca vaciarlo. Un despliegue real
    # tiene que fijarlo a conciencia.
    evidence_retention_days: int = field(
        default_factory=lambda: _env_int("VISION_EVIDENCE_RETENTION_DAYS", 0)
    )

    # Registra qué borraría, sin borrar nada. Para mirar la primera pasada
    # antes de soltarla sobre evidencia acumulada.
    retention_dry_run: bool = field(
        default_factory=lambda: _env_bool("VISION_RETENTION_DRY_RUN", False)
    )

    # Postgres. The +psycopg suffix picks the psycopg 3 driver explicitly;
    # without it SQLAlchemy still looks for psycopg2, which is not installed.
    database_url: str = field(
        default_factory=lambda: _env_str(
            "VISION_DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/traffic_detector",
        )
    )

    # Road ROI: polygon delimiting the drivable surface, in NORMALIZED 0..1
    # coordinates ("x,y x,y ..."), so the same value works at any resolution.
    # Empty = whole frame (works uncalibrated, but sky/trees/sidewalk dilute
    # the occupancy). Draw one with scripts/roi_picker.py.
    road_roi: str = field(
        default_factory=lambda: _env_str("VISION_ROAD_ROI", "")
    )

    # Apparent size ratio between a vehicle at the bottom of the ROI and one
    # at the top (3.0 = the near one looks 3x longer). Corrects perspective so
    # a far vehicle weighs the same as a near one. 1.0 = no correction.
    road_perspective: float = field(
        default_factory=lambda: _env_float("VISION_ROAD_PERSPECTIVE", 1.0)
    )

    # Traffic level: fraction of the road area covered by vehicles.
    # >= high AND slow -> "alto"; >= medium -> "medio"; otherwise "bajo".
    # Scene-dependent: measure yours with scripts/traffic_calibrate.py.
    traffic_occupancy_medium: float = field(
        default_factory=lambda: _env_float("VISION_TRAFFIC_OCC_MEDIUM", 0.22)
    )

    traffic_occupancy_high: float = field(
        default_factory=lambda: _env_float("VISION_TRAFFIC_OCC_HIGH", 0.38)
    )

    # Free-flow speed in FRAME WIDTHS PER SECOND (perspective-corrected).
    # Not px/frame: those depend on resolution and fps, so the same jam would
    # need different thresholds per camera. Vehicles below
    # slow_ratio * free_speed count as stopped.
    traffic_free_speed: float = field(
        default_factory=lambda: _env_float("VISION_TRAFFIC_FREE_SPEED", 0.08)
    )

    traffic_slow_ratio: float = field(
        default_factory=lambda: _env_float("VISION_TRAFFIC_SLOW_RATIO", 0.35)
    )

    # EMA smoothing for occupancy and speed (0..1, higher = snappier).
    traffic_smoothing: float = field(
        default_factory=lambda: _env_float("VISION_TRAFFIC_SMOOTHING", 0.2)
    )

    def describe(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "video_path": str(self.video_path),
            "confidence": self.confidence,
            "iou": self.iou,
            "image_size": self.image_size,
            "frame_stride": self.frame_stride,
            "max_sessions": self.max_sessions,
            "loop_source": self.loop_source,
            "pace_lead_seconds": self.pace_lead_seconds,
            "stream_frames": self.stream_frames,
            "stream_quality": self.stream_quality,
            "cors_origins": self.cors_origins,
            "evidence_enabled": self.evidence_enabled,
            "evidence_dir": str(self.evidence_dir),
            "anonymize_evidence": self.anonymize_evidence,
            "evidence_retention_days": self.evidence_retention_days,
            # Solo si hay clave o no. El valor jamás se registra ni se
            # devuelve por /health.
            # Solo si hay secreto o no. El valor jamás se registra.
            "auth_configured": bool(self.jwt_secret),
            "timezone": self.timezone,
            "shift_grace_minutes": self.shift_grace_minutes,
            "shift_timeout_minutes": self.shift_timeout_minutes,
            "jwt_expire_minutes": self.jwt_expire_minutes,
            "gemini_configured": bool(self.gemini_api_key),
            "gemini_model": self.gemini_model,
            # redacted: describe() goes to the log on every startup
            "database_url": redact_url(self.database_url),
            "road_roi": self.road_roi,
            "road_perspective": self.road_perspective,
            "traffic_occupancy_medium": self.traffic_occupancy_medium,
            "traffic_occupancy_high": self.traffic_occupancy_high,
            "traffic_free_speed": self.traffic_free_speed,
            "traffic_slow_ratio": self.traffic_slow_ratio,
        }


settings = Settings()


def config_fingerprint() -> str:
    """
    Short hash of the files that decide how the service behaves.

    Two processes serving the same port with different fingerprints means one
    of them is stale — which is exactly the failure that is invisible
    otherwise, since both answer requests perfectly happily.
    """

    digest = hashlib.sha256()

    for name in (".env", "cameras.yaml"):
        path = REPO_ROOT / name

        if path.is_file():
            digest.update(path.read_bytes())

    return digest.hexdigest()[:12]
