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

    cors_origins: list[str] = field(default_factory=_cors_origins)

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
            "cors_origins": self.cors_origins,
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
