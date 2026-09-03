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
from .cameras import Camera
from .config import settings
from .road_roi import build_road_roi
from .pipeline import build_vision_engine
from .protocol import PROTOCOL_VERSION
from ..db.incident_writer import incident_writer
from .serializers import event_to_dict, incident_to_dict, track_to_dict
from .traffic_level import TrafficLevelEstimator

_QUEUE_MAXSIZE = 120
_DEFAULT_FPS = 25.0


class VideoSession:

    def __init__(
        self,
        detector: YOLODetector,
        loop: asyncio.AbstractEventLoop,
        camera: Camera,
        stride: Optional[int] = None,
    ):
        self._detector = detector
        self._loop = loop
        self._camera = camera
        self._stride = max(1, stride or settings.frame_stride)

        self._engine = build_vision_engine(detector)

        # Every scene knob comes from the camera, not from the global config:
        # two cameras in the same deployment have different geometry, so
        # sharing one ROI or one threshold gives one of them a wrong level.
        self._traffic = TrafficLevelEstimator(
            roi=build_road_roi(camera.roi, camera.perspective),
            medium=camera.occupancy_medium,
            high=camera.occupancy_high,
            free_speed=camera.free_speed,
            stride=self._stride,
        )
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

        source = self._camera.source

        if not source.exists():
            raise FileNotFoundError(
                f"Fuente no encontrada para '{self._camera.id}': {source}"
            )

        cap = cv2.VideoCapture(str(source))

        if not cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la fuente de '{self._camera.id}': {source}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or math.isnan(fps) or fps <= 0:
            fps = _DEFAULT_FPS

        self._cap = cap
        self.fps = float(fps)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._traffic.configure(self.width, self.height, self.fps)

        return {
            "type": "meta",
            # Lets the frontend notice it is talking to a service older than
            # itself, instead of crashing on a field that is not there.
            "protocol": PROTOCOL_VERSION,
            "camera": self._camera.public(),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "stride": self._stride,
            "traffic_thresholds": {
                "medium": self._camera.occupancy_medium,
                "high": self._camera.occupancy_high,
            },
            # Normalized polygon of the drivable surface, so the frontend can
            # draw it over the video while calibrating. null = whole frame.
            "road_roi": self._traffic.roi.polygon,
        }

    def start(self) -> None:
        incident_writer.reset_seen()
        incident_writer.start()

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

                traffic = self._traffic.update(
                    result.tracks,
                    motion=result.motion,
                )

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
                            "occupancy": traffic.occupancy,
                            "mean_speed": traffic.mean_speed,
                            "speed_ratio": traffic.speed_ratio,
                            "stopped": traffic.stopped,
                            "score": traffic.score,
                        },
                    }
                )

                # emit incidents also as standalone messages
                for inc in incidents_payload:
                    self._put({"type": "incident", **inc})
                    # Queued, never written inline: the vision loop must not
                    # wait on Postgres. Repeats of the same incident across
                    # frames are dropped by the writer.
                    incident_writer.submit(self._camera.id, inc, frame_id)

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
