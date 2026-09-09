r"""
Calibra el plano de la vía marcando cuatro puntos del pavimento.

Por qué
-------

El motor de colisiones decide si dos vehículos chocaron mirando si sus cajas
se tocan EN LA IMAGEN, y eso no es una medida de distancia: la cámara aplasta
la escena contra un plano, así que un bus del carril lejano y un carro del
cercano pueden solaparse estando a diez metros. Es el origen de la mayoría de
los falsos positivos.

Calibrando la cámara contra el asfalto, cada vehículo pasa a tener una
posición en METROS y las distancias vuelven a significar algo.

Cómo
----

1. Elige un rectángulo del pavimento cuyas medidas reales puedas conocer.
   Lo más práctico en una vía urbana:

     - el ANCHO lo dan los carriles (en Colombia, entre 3,0 y 3,5 m cada uno);
     - el LARGO lo puedes medir sobre la vista satelital de Google Maps,
       usando la coordenada de la cámara que está en cameras.yaml.

   Sirve cualquier cuadrilátero del que sepas las medidas: un paso de cebra,
   dos postes, el tramo entre dos líneas discontinuas.

2. Haz clic en sus cuatro esquinas. EL ORDEN DA IGUAL: el programa las
   acomoda solo. Lo unico que importa es que sean las cuatro esquinas del
   mismo rectangulo.

        +------------+      (lejos, al fondo)
        |            |         el lado corto/largo lo decides tu
        +------------+      (cerca, al frente)

   --width  es el lado que CRUZA la via (de izquierda a derecha)
   --length es el lado que se ALEJA de la camara (hacia el fondo)

3. Comprueba la calibración: marca dos puntos de los que sepas la distancia
   y verifica que el número que sale sea el correcto. Este paso importa -
   una calibración mala no falla, solo miente.

Uso:

    .venv\Scripts\python scripts\homography_picker.py --camera cra55_cl37 --width 7 --length 20

Controles:

    clic izquierdo   marca las 4 esquinas; despues de la cuarta, cada par
                     de clics mide la distancia real entre dos puntos
    z                quitar el último punto
    r                empezar de nuevo
    ENTER            imprimir la calibración
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
from app.geometry.homography import (  # noqa: E402
    CalibrationError,
    GroundPlane,
    order_quad,
)

WINDOW = "Calibracion del plano de la via"

# El orden lo acomoda order_quad(), asi que aqui solo se cuenta.
ESQUINAS = [
    "Marca la 1a esquina del rectangulo (el orden da igual)",
    "Marca la 2a esquina",
    "Marca la 3a esquina",
    "Marca la 4a y ultima esquina",
]

# Largo de un carro de referencia. Sirve para traducir la calibracion a algo
# que se pueda comprobar mirando la pantalla.
LARGO_CARRO_M = 4.5

VERDE = (120, 224, 95)
AMBAR = (66, 194, 245)
BLANCO = (255, 255, 255)


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


def draw(frame, corners, measure, plane, width_m, length_m):
    canvas = frame.copy()

    # --- el cuadrilátero de referencia ---
    if len(corners) >= 2:
        for a, b in zip(corners, corners[1:]):
            cv2.line(canvas, a, b, VERDE, 2)

    if len(corners) == 4:
        cv2.line(canvas, corners[3], corners[0], VERDE, 2)

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [np.array(corners, dtype="int32")], VERDE)
        canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)

        # Las medidas reales, escritas sobre los lados que representan.
        medio_abajo = (
            (corners[0][0] + corners[1][0]) // 2,
            (corners[0][1] + corners[1][1]) // 2,
        )
        medio_lado = (
            (corners[1][0] + corners[2][0]) // 2,
            (corners[1][1] + corners[2][1]) // 2,
        )
        cv2.putText(canvas, f"{width_m} m", medio_abajo,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERDE, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{length_m} m", medio_lado,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERDE, 2, cv2.LINE_AA)

    for i, point in enumerate(corners):
        cv2.circle(canvas, point, 5, BLANCO, -1)
        cv2.circle(canvas, point, 5, VERDE, 2)
        cv2.putText(canvas, str(i + 1), (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1, cv2.LINE_AA)

    # --- la regla de comprobación ---
    for point in measure:
        cv2.circle(canvas, point, 5, AMBAR, -1)

    if len(measure) == 2 and plane is not None:
        cv2.line(canvas, measure[0], measure[1], AMBAR, 2)

        try:
            metros = plane.distance_m(measure[0], measure[1])
            texto = f"{metros:.2f} m"
        except CalibrationError:
            texto = "fuera del plano"

        medio = (
            (measure[0][0] + measure[1][0]) // 2,
            (measure[0][1] + measure[1][1]) // 2,
        )
        cv2.putText(canvas, texto, (medio[0] + 8, medio[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBAR, 2, cv2.LINE_AA)

    # --- instrucción de arriba ---
    if len(corners) < 4:
        ayuda = ESQUINAS[len(corners)]
    elif plane is None:
        ayuda = "Puntos invalidos: pulsa r y vuelve a marcar"
    else:
        ayuda = "Listo. Mide dos puntos conocidos para comprobar  |  ENTER: imprimir"

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (31, 24, 7), -1)
    cv2.putText(canvas, ayuda, (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLANCO, 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default=None,
                        help="Id en cameras.yaml.")
    parser.add_argument("--video", default=None,
                        help="Video de entrada. Ignora --camera si se pasa.")
    parser.add_argument("--frame", type=int, default=0,
                        help="Indice del frame a mostrar.")
    parser.add_argument("--width", type=float, required=True,
                        help="Ancho REAL en metros del rectangulo (lado 1-2).")
    parser.add_argument("--length", type=float, required=True,
                        help="Largo REAL en metros del rectangulo (lado 2-3).")
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
                f"Camara desconocida: '{args.camera}'. Conocidas: {known}"
            )
        video_path = camera.source

    frame = read_frame(video_path, args.frame)

    corners: list[tuple[int, int]] = []
    measure: list[tuple[int, int]] = []
    plane: GroundPlane | None = None

    def rebuild():
        """Recalcula el plano cada vez que cambian las esquinas."""
        nonlocal plane

        if len(corners) != 4:
            plane = None
            return

        try:
            plane = GroundPlane.from_quad(corners, args.width, args.length)
        except CalibrationError as error:
            plane = None
            print(f"  {error}")

    def on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(corners) < 4:
            corners.append((x, y))
            rebuild()
        else:
            # Con las esquinas puestas, los clics miden.
            if len(measure) == 2:
                measure.clear()
            measure.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        cv2.imshow(WINDOW, draw(frame, corners, measure, plane,
                                args.width, args.length))

        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelado, no se imprimio nada.")

        if key == ord("r"):
            corners.clear()
            measure.clear()
            plane = None

        if key == ord("z"):
            if measure:
                measure.pop()
            elif corners:
                corners.pop()
                rebuild()

        if key in (13, 10):  # ENTER
            if plane is None:
                print("Faltan las cuatro esquinas, o no forman un cuadrilatero.")
                continue
            break

    cv2.destroyAllWindows()

    valores = ", ".join(f"{v:.10g}" for v in plane.values())
    nombre = camera.id if camera else "tu_camara"

    print()
    print("Pega esto en la entrada de la camara en cameras.yaml:")
    print()
    print(f"  # {nombre}: rectangulo de {args.width} x {args.length} m")
    print(f"  homography: [{valores}]")
    print()
    print("Y despues carga el registro en la base:")
    print(r"  .venv\Scripts\python scripts\seed_cameras.py")
    print()

    # Una comprobación automática además de la manual: las esquinas tienen
    # que devolver justo las medidas que se declararon. Si esto no cuadra,
    # el orden de los clics estaba mal.
    ordenadas = order_quad(corners)
    ancho = plane.distance_m(ordenadas[0], ordenadas[1])
    largo = plane.distance_m(ordenadas[1], ordenadas[2])

    print(f"Comprobacion: lado 1-2 = {ancho:.2f} m (declaraste {args.width})")
    print(f"              lado 2-3 = {largo:.2f} m (declaraste {args.length})")

    if abs(ancho - args.width) > 0.01 or abs(largo - args.length) > 0.01:
        print()
        print("  AVISO: no coinciden, y no deberia pasar. Reporta este caso.")

    _diagnostico(plane, frame.shape[1], frame.shape[0])


def _diagnostico(plane: GroundPlane, width_px: int, height_px: int) -> None:
    """
    Qué escena describe esta calibración, en metros.

    La comprobación de las esquinas es circular: siempre devuelve las medidas
    declaradas, porque es lo que se le pidió ajustar. Lo que NO puede
    detectar es que esas medidas fueran equivocadas. Esto sí: si el
    cuadrilátero marcado era grande y se le declararon pocos metros, la
    escena completa sale absurdamente pequeña y se ve aquí.
    """

    margen = 10
    abajo = height_px - margen
    centro_x = width_px // 2

    # Hasta donde llega la via visible: la fila mas alta que siga cayendo por
    # debajo del horizonte. Medir "hasta media altura" subestimaba muchisimo
    # -la perspectiva comprime el fondo, asi que la mitad superior es la que
    # tiene casi todos los metros- y podia hacer descartar una calibracion
    # correcta.
    arriba = None

    for fila in range(margen, abajo):
        try:
            plane.to_world((centro_x, fila))
        except CalibrationError:
            continue
        arriba = fila
        break

    if arriba is None:
        print()
        print("  AVISO: el encuadre cae fuera del plano de la via; la")
        print("  calibracion no es utilizable.")
        return

    try:
        ancho_cerca = plane.distance_m((margen, abajo), (width_px - margen, abajo))
        fondo = plane.distance_m((centro_x, abajo), (centro_x, arriba))
    except CalibrationError:
        print()
        print("  AVISO: no se pudo medir el encuadre; calibracion dudosa.")
        return

    # Cuantos pixeles deberia ocupar un carro cerca del borde inferior. Es el
    # numero mas util del diagnostico porque se comprueba a ojo contra el
    # video, sin herramientas: si dice 90 px y en pantalla los carros miden
    # 200, las medidas declaradas estaban mal.
    referencia = int(height_px * 0.95)
    px_carro = None

    for d in range(1, height_px):
        try:
            if plane.distance_m((centro_x, referencia),
                                (centro_x, referencia - d)) >= LARGO_CARRO_M:
                px_carro = d
                break
        except CalibrationError:
            break

    print()
    print("Segun esta calibracion, el encuadre completo abarca:")
    print(f"  {ancho_cerca:.0f} m de ancho en primer plano")
    print(f"  {fondo:.0f} m de via visible hacia el fondo")

    if px_carro:
        print()
        print(f"  Un carro ({LARGO_CARRO_M} m) cerca del borde inferior deberia")
        print(f"  ocupar unos {px_carro} px de alto en pantalla.")

    print()

    # Solo se avisa de lo absurdo. El aviso anterior comparaba ancho contra
    # profundidad para detectar ejes girados, pero eso ya no puede pasar:
    # order_quad ordena las esquinas. Y cerca del horizonte la profundidad
    # tiende a infinito, asi que como criterio tampoco servia.
    if fondo < 15:
        print("  AVISO: muy poca via visible para una camara de trafico.")
        print("  Revisa las medidas que declaraste.")
        print()
    elif px_carro and (px_carro < 15 or px_carro > height_px * 0.45):
        print("  AVISO: el tamano de carro que implica no es creible.")
        print("  Revisa las medidas que declaraste.")
        print()

    print("La comprobacion definitiva, con la regla del tool:")
    print("  marca la base delantera y la base trasera de un vehiculo.")
    print(f"  Un carro debe dar ~{LARGO_CARRO_M} m; un bus urbano ~12 m.")


if __name__ == "__main__":
    main()
