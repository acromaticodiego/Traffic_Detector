"""
Runtime configuration for the vision service API.

Every value can be overridden with an environment
variable. Sensible defaults point at the burned-in
demo video and the trained model already in the repo.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


# Repository root: services/vision_service/app/api/config.py -> up 4
REPO_ROOT = Path(__file__).resolve().parents[4]


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
        default_factory=lambda: _env_float("VISION_CONFIDENCE", 0.60)
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

    # Traffic level: smoothed vehicle count >= high -> "alto",
    # >= medium -> "medio", otherwise "bajo". Tune per camera/scene.
    traffic_medium: float = field(
        default_factory=lambda: _env_float("VISION_TRAFFIC_MEDIUM", 3.0)
    )

    traffic_high: float = field(
        default_factory=lambda: _env_float("VISION_TRAFFIC_HIGH", 6.0)
    )

    # EMA smoothing factor for the vehicle count (0..1, higher = snappier).
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
            "traffic_medium": self.traffic_medium,
            "traffic_high": self.traffic_high,
        }


settings = Settings()
