"""
Dibuja el polígono de la calzada (ROI) haciendo clic sobre un frame.

El nivel de tráfico se calcula como ocupación del área de la vía, así que hay
que decirle al servicio cuál es la vía. Este script abre un frame del video,
deja marcar el contorno de la calzada con clics e imprime la línea lista para
pegar en el `.env`.

Uso:

    .venv\\Scripts\\python scripts\\roi_picker.py
    .venv\\Scripts\\python scripts\\roi_picker.py --video videos/input/otro.mp4 --frame 120

Controles:

    clic izquierdo   agregar punto
    clic derecho / z quitar el último punto
    r                empezar de nuevo
    ENTER            confirmar e imprimir
    ESC / q          salir sin guardar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "vision_service"))

from app.api.cameras import get_camera, list_cameras  # noqa: E402
from app.api.config import settings  # noqa: E402

WINDOW = "ROI de la calzada  |  clic: punto  ·  z: deshacer  ·  r: reiniciar  ·  ENTER: listo"


def read_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {video_path}")

    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"No se pudo leer el frame {frame_index} de {video_path}")

    return frame


def draw(frame, points: list[tuple[int, int]]):
    canvas = frame.copy()

    if len(points) >= 2:
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, (0, 220, 120), 2)

    if len(points) >= 3:
        # cierre del polígono, punteado visualmente con un color distinto
        cv2.line(canvas, points[-1], points[0], (0, 140, 255), 1)

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.array(points, dtype="int32")], (0, 220, 120))
        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    for i, point in enumerate(points):
        cv2.circle(canvas, point, 5, (255, 255, 255), -1)
        cv2.circle(canvas, point, 5, (0, 120, 255), 2)
        cv2.putText(
            canvas,
            str(i + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        default=None,
        help="Id en cameras.yaml. Determina la fuente y el nombre a mostrar.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Video de entrada. Ignora --camera si se pasa.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Índice del frame a mostrar (elige uno representativo).",
    )
    args = parser.parse_args()

    camera = None

    if args.video:
        video_path = Path(args.video)
    else:
        try:
            camera = get_camera(args.camera)
        except KeyError:
            known = ", ".join(c.id for c in list_cameras())
            raise SystemExit(
                f"Cámara desconocida: {args.camera}. Disponibles: {known}"
            )
        video_path = camera.source

    if camera is not None:
        print(f"Cámara: {camera.id}  ({camera.name})")

    frame = read_frame(video_path, args.frame)

    height, width = frame.shape[:2]
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _param):
        del flags
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(width, 1280), min(height, 720))
    cv2.setMouseCallback(WINDOW, on_mouse)

    print(f"Video : {video_path}")
    print(f"Frame : {args.frame}  ({width}x{height})")
    print("Marca el contorno de la CALZADA (solo el asfalto por donde pasan")
    print("los vehículos: sin cielo, sin árboles, sin andén) y pulsa ENTER.\n")

    while True:
        cv2.imshow(WINDOW, draw(frame, points))
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            print("Cancelado.")
            break

        if key == ord("z") and points:
            points.pop()

        if key == ord("r"):
            points.clear()

        if key in (13, 10):  # ENTER
            if len(points) < 3:
                print("Se necesitan al menos 3 puntos.")
                continue

            polygon = " ".join(
                f"{x / width:.4f},{y / height:.4f}" for x, y in points
            )

            print("\nPega esto en el .env del servicio:\n")
            print(f"VISION_ROAD_ROI={polygon}")
            print(
                "\nY estima la perspectiva: cuántas veces más largo se ve un"
                "\nvehículo abajo de la ROI que arriba (1.0 = toma cenital).\n"
            )
            print("VISION_ROAD_PERSPECTIVE=2.5")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
