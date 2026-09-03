"""
Calibra los umbrales de nivel de tráfico contra un video real.

La ocupación y la velocidad de flujo libre dependen de la escena: la altura de
la cámara, cuántos carriles se ven, qué tan lejos llega la toma. En vez de
adivinar los umbrales, se procesa un tramo del video y se miran los percentiles
de lo que realmente ocurre.

Uso:

    .venv\\Scripts\\python scripts\\traffic_calibrate.py
    .venv\\Scripts\\python scripts\\traffic_calibrate.py --frames 600 --roi "0.0,0.6 0.3,0.1 ..."

El clip que se procese determina qué significan los umbrales, así que el
script imprime DOS sugerencias y hay que elegir según lo que muestre el video:

  * Clip de FLUJO LIBRE (como los de demo): los umbrales van por encima de
    todo lo observado, para que "medio" y "alto" solo aparezcan cuando la vía
    se cargue más de lo que se vio.

  * Clip REPRESENTATIVO (horas pico y valle incluidas): los umbrales salen de
    los percentiles de la distribución real.

FREE_SPEED sale del percentil 85 de la velocidad en ambos casos: el flujo
libre de la escena es, por definición, lo más rápido que se ve rodar.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "vision_service"))

from app.api.cameras import get_camera, list_cameras
from app.api.config import settings  # noqa: E402
from app.api.pipeline import build_vision_engine  # noqa: E402
from app.api.road_roi import build_road_roi  # noqa: E402
from app.api.traffic_level import TrafficLevelEstimator  # noqa: E402
from app.detection.detector import YOLODetector  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(p / 100.0 * len(ordered)), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        default=None,
        help="Id en cameras.yaml: toma su fuente, su ROI y su perspectiva.",
    )
    parser.add_argument("--video", default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Primer frame a medir. Útil si el clip cambia de cámara.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Polígono normalizado; por defecto el de la cámara.",
    )
    parser.add_argument(
        "--perspective",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    try:
        camera = get_camera(args.camera)
    except KeyError:
        known = ", ".join(c.id for c in list_cameras())
        raise SystemExit(
            f"Cámara desconocida: {args.camera}. Disponibles: {known}"
        )

    # Explicit flags win, so a polygon can be tried out before committing it
    # to the registry.
    video = args.video or str(camera.source)
    roi_spec = args.roi if args.roi is not None else camera.roi
    perspective = (
        args.perspective if args.perspective is not None else camera.perspective
    )

    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    roi = build_road_roi(roi_spec, perspective)

    estimator = TrafficLevelEstimator(roi=roi)
    estimator.configure(width, height, fps)

    engine = build_vision_engine(
        YOLODetector(model_path=str(settings.model_path))
    )

    print(f"Cámara      : {camera.id}  ({camera.name})")
    print(f"Video       : {video}")
    print(f"Resolución  : {width}x{height} @ {fps:.1f} fps")
    print(f"ROI         : {'polígono' if roi.calibrated else 'FRAME COMPLETO (sin calibrar)'}")
    print(f"Perspectiva : {roi.perspective}")
    print(f"Procesando {args.frames} frames…\n")

    occupancies: list[float] = []
    speeds: list[float] = []
    vehicles: list[int] = []
    levels: list[str] = []

    for index in range(args.frames):

        ok, frame = cap.read()

        if not ok:
            break

        result = engine.process_frame(
            frame=frame,
            frame_id=index + 1,
            timestamp=datetime.now(),
            persist_tracks=index > 0,
        )

        traffic = estimator.update(result.tracks, motion=result.motion)

        occupancies.append(traffic.occupancy)
        vehicles.append(traffic.vehicles)
        levels.append(traffic.level)

        # solo cuenta la velocidad cuando hay vehículos que medir
        if traffic.vehicles:
            speeds.append(traffic.mean_speed)

    cap.release()

    if not occupancies:
        raise SystemExit("No se procesó ningún frame.")

    print(f"Frames procesados : {len(occupancies)}")
    print(
        "Vehículos en vía  : min %d · media %.1f · max %d"
        % (min(vehicles), statistics.mean(vehicles), max(vehicles))
    )
    print(
        "Ocupación         : p10 %.3f · p50 %.3f · p60 %.3f · p90 %.3f · max %.3f"
        % (
            percentile(occupancies, 10),
            percentile(occupancies, 50),
            percentile(occupancies, 60),
            percentile(occupancies, 90),
            max(occupancies),
        )
    )

    if speeds:
        print(
            "Velocidad (ancho/s): p15 %.4f · p50 %.4f · p85 %.4f"
            % (
                percentile(speeds, 15),
                percentile(speeds, 50),
                percentile(speeds, 85),
            )
        )

    print(f"Niveles con los umbrales actuales: {dict(Counter(levels))}\n")

    free_speed = (
        f"VISION_TRAFFIC_FREE_SPEED={percentile(speeds, 85):.3f}"
        if speeds
        else "VISION_TRAFFIC_FREE_SPEED=  (sin vehículos medidos)"
    )

    peak = max(occupancies)

    print("─" * 62)
    print("Si este clip es de FLUJO LIBRE (debería dar 'bajo' casi siempre):\n")
    print(f"VISION_TRAFFIC_OCC_MEDIUM={peak * 1.15:.2f}")
    print(f"VISION_TRAFFIC_OCC_HIGH={peak * 2.0:.2f}")
    print(free_speed)

    print("\nSi el clip es REPRESENTATIVO (incluye pico y valle):\n")
    print(f"VISION_TRAFFIC_OCC_MEDIUM={percentile(occupancies, 60):.2f}")
    print(f"VISION_TRAFFIC_OCC_HIGH={percentile(occupancies, 90):.2f}")
    print(free_speed)
    print("─" * 62)


if __name__ == "__main__":
    main()
