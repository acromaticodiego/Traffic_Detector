"""
VideoSession: runs the vision pipeline over the
burned-in video and streams per-frame results.

The OpenCV + YOLO loop is blocking, so it runs in a
worker thread and hands messages back to the asyncio
side through a bounded queue. The bounded queue also
provides natural backpressure: if the browser cannot
keep up, the worker thread blocks instead of piling
frames in memory.

Only ONE session is meant to be active at a time
(ByteTrack keeps state on the shared YOLO model).
The WebSocket route enforces that.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
from datetime import datetime
from typing import Any, Optional

import cv2

from ..detection.detector import YOLODetector
from .config import settings
from .pipeline import build_vision_engine
from .serializers import event_to_dict, incident_to_dict, track_to_dict
from .traffic_level import TrafficLevelEstimator

_QUEUE_MAXSIZE = 120
_DEFAULT_FPS = 25.0


class VideoSession:

    def __init__(
        self,
        detector: YOLODetector,
        loop: asyncio.AbstractEventLoop,
        stride: Optional[int] = None,
    ):
        self._detector = detector
        self._loop = loop
        self._stride = max(1, stride or settings.frame_stride)

        self._engine = build_vision_engine(detector)
        self._traffic = TrafficLevelEstimator()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=_QUEUE_MAXSIZE
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.fps: float = _DEFAULT_FPS
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def open(self) -> dict[str, Any]:
        """
        Open the video and return the `meta` message.
        Raises FileNotFoundError / RuntimeError on failure.
        """

        if not settings.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {settings.video_path}"
            )

        cap = cv2.VideoCapture(str(settings.video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {settings.video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or math.isnan(fps) or fps <= 0:
            fps = _DEFAULT_FPS

        self._cap = cap
        self.fps = float(fps)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return {
            "type": "meta",
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "stride": self._stride,
            "traffic_thresholds": {
                "medium": settings.traffic_medium,
                "high": settings.traffic_high,
            },
        }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._process_video,
            name="vision-session",
            daemon=True,
        )
        self._thread.start()

    async def messages(self):
        """
        Async generator yielding stream messages until
        `done` (or the session is stopped).
        """

        while True:
            msg = await self._queue.get()

            yield msg

            if msg.get("type") in ("done", "error"):
                break

    def stop(self) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._engine.reset()

    # ------------------------------------------------------------------
    # worker thread
    # ------------------------------------------------------------------

    def _put(self, message: dict[str, Any]) -> None:
        """
        Block the worker thread until the asyncio queue
        has room (backpressure), but keep checking the
        stop flag so a dropped client cannot wedge the
        thread forever on a full queue.
        """

        future = asyncio.run_coroutine_threadsafe(
            self._queue.put(message),
            self._loop,
        )

        while not self._stop.is_set():
            try:
                future.result(timeout=0.5)
                return
            except concurrent.futures.TimeoutError:
                continue
            except Exception:
                self._stop.set()
                return

        future.cancel()

    def _process_video(self) -> None:
        cap = self._cap
        assert cap is not None

        frame_id = 0
        processed = 0
        first = True

        try:
            while not self._stop.is_set():

                ok, frame = cap.read()

                if not ok:
                    break

                frame_id += 1

                # honour the stride (process frame 1, 1+stride, ...)
                if (frame_id - 1) % self._stride != 0:
                    continue

                result = self._engine.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=datetime.now(),
                    persist_tracks=not first,
                )
                first = False
                processed += 1

                t = frame_id / self.fps

                motion_by_id = {
                    m.track_id: m for m in result.motion
                }

                tracks_payload = [
                    track_to_dict(track, motion_by_id.get(track.track_id))
                    for track in result.tracks
                ]

                incidents_payload = [
                    incident_to_dict(inc, t) for inc in result.incidents
                ]

                events_payload = [
                    event_to_dict(ev) for ev in result.events
                ]

                traffic = self._traffic.update(result.tracks)

                self._put(
                    {
                        "type": "frame",
                        "frame_id": frame_id,
                        "t": round(t, 3),
                        "tracks": tracks_payload,
                        "incidents": incidents_payload,
                        "events": events_payload,
                        "traffic": {
                            "level": traffic.level,
                            "vehicles": traffic.vehicles,
                            "people": traffic.people,
                            "score": traffic.score,
                        },
                    }
                )

                # emit incidents also as standalone messages
                for inc in incidents_payload:
                    self._put({"type": "incident", **inc})

        except Exception as error:  # noqa: BLE001
            self._put({"type": "error", "message": str(error)})
            return

        self._put(
            {
                "type": "done",
                "frames": frame_id,
                "processed": processed,
            }
        )
