"""
Factory that wires the full vision pipeline.

Mirrors the manual wiring in
scripts/test_vision_engine.py but splits the parts
that can be shared across sessions (the loaded YOLO
model) from the parts that hold per-run state
(track manager, engines, analyzers).
"""

from __future__ import annotations

import numpy as np

from ..detection.detector import YOLODetector
from ..tracking.tracker import ByteTrackTracker
from ..tracking.track_manager import TrackManager
from ..motion.motion_analyzer import MotionAnalyzer
from ..events.event_engine import EventEngine
from ..incidents.incident_engine import IncidentEngine
from ..vision_engine import VisionEngine

from .config import settings


def create_detector() -> YOLODetector:
    """
    Load the YOLO model once. The returned detector
    (and its .model) is reused by every session.
    """

    detector = YOLODetector(
        model_path=settings.model_path,
        confidence=settings.confidence,
        iou=settings.iou,
        image_size=settings.image_size,
    )

    # Warm up CUDA / cuDNN so the first real frames of a session
    # are not slow (which otherwise looks like bad early tracking).
    warmup = np.zeros(
        (settings.image_size, settings.image_size, 3),
        dtype=np.uint8,
    )
    for _ in range(3):
        detector.model.predict(
            warmup,
            imgsz=settings.image_size,
            device=detector.device,
            verbose=False,
        )

    return detector


def build_vision_engine(detector: YOLODetector) -> VisionEngine:
    """
    Build a fresh VisionEngine around an already
    loaded detector. All stateful components are new,
    so each session starts clean.
    """

    tracker = ByteTrackTracker(
        model=detector.model,
        confidence=settings.confidence,
        iou=settings.iou,
        image_size=settings.image_size,
        device=detector.device,
    )

    return VisionEngine(
        detector=detector,
        tracker=tracker,
        track_manager=TrackManager(),
        event_engine=EventEngine(),
        motion_analyzer=MotionAnalyzer(),
        incident_engine=IncidentEngine(),
    )
