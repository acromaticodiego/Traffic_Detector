"""
El plano de la vía: convierte píxeles de la imagen en metros sobre el asfalto.

Por qué hace falta
------------------

Una cámara aplasta una escena tridimensional contra un plano, y en esa
proyección las distancias dejan de significar lo que parecen. Dos vehículos
cuyas cajas se tocan en la imagen pueden estar a quince metros uno del otro
si van por carriles distintos: uno está lejos y se ve pequeño, el otro cerca
y se ve grande, y sus recuadros se solapan sin que nada haya pasado.

Ese es el origen de la mayoría de los falsos positivos del motor de
colisiones, y no se arregla ajustando umbrales en píxeles: `bbox_gap_px`
simplemente no es una medida de distancia.

Lo que sí lo arregla es calibrar la cámara contra el plano del pavimento.
Como la vía es plana, la relación entre la imagen y el suelo es una
**homografía**: una matriz de 3×3 que se calcula una sola vez marcando cuatro
puntos del asfalto de los que se conozcan las distancias reales. A partir de
ahí, cada vehículo tiene una posición en metros, las distancias son
comparables, las velocidades salen en unidades físicas y "estos dos están a
0,4 m" quiere decir algo.

Por qué numpy y no OpenCV
-------------------------

`cv2.getPerspectiveTransform` haría esto en una línea, pero opencv no está en
`requirements-dev.txt`: el CI corre sin él a propósito, para no bajar un par
de gigas por unos tests de lógica. Esta es justamente la clase de código que
más necesita pruebas —un error de signo aquí envenena en silencio todo lo que
venga después— así que se resuelve con numpy, que sí está.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Point = tuple[float, float]


class CalibrationError(ValueError):
    """Los puntos dados no definen una homografía utilizable."""


# Área mínima (relativa al tamaño del cuadrilátero) de los triángulos que
# forman cada tres esquinas. Por debajo de esto hay tres puntos casi
# alineados. Un cuadrilátero sano da valores del orden de 1.
_MIN_TRIANGLE_RATIO = 0.02


def _reject_degenerate_quad(points: Sequence[Point]) -> None:
    """
    Rechaza cuatro puntos que no formen un cuadrilátero de verdad.

    La comprobación es geométrica y no numérica, y eso importa: con tres
    puntos alineados el sistema lineal SÍ tiene solución y está perfectamente
    bien condicionado. Lo que devuelve es una homografía degenerada, que
    aplasta el plano contra una recta y mide cualquier cosa. Mirar el
    condicionamiento de la matriz no lo detecta; mirar los puntos, sí.
    """

    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    scale = float(np.hypot(*(pts - centroid).T).mean())

    if scale < 1e-12:
        raise CalibrationError("Los cuatro puntos son el mismo punto.")

    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]

        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        if abs(cross) / (scale * scale) < _MIN_TRIANGLE_RATIO:
            raise CalibrationError(
                "Los cuatro puntos no forman un cuadrilátero válido: hay "
                "tres casi alineados o dos repetidos. Marca las esquinas de "
                "un rectángulo del pavimento, no puntos sobre una misma "
                "línea."
            )


def _normalize(points: Sequence[Point]) -> tuple[list[Point], np.ndarray]:
    """
    Centra los puntos en su centroide y los escala a distancia media √2.

    Sin esto el sistema es numéricamente singular en cuanto los datos son
    reales: los píxeles valen cientos y los metros unidades, así que las
    columnas de la matriz difieren en varios órdenes de magnitud y la
    solución se pierde en el redondeo. Es la normalización de Hartley, y en
    la práctica es la diferencia entre una calibración correcta y uno de
    esos errores que no fallan, solo mienten.
    """

    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    shifted = pts - centroid

    mean_distance = float(np.hypot(shifted[:, 0], shifted[:, 1]).mean())

    if mean_distance < 1e-12:
        raise CalibrationError("Los cuatro puntos son el mismo punto.")

    scale = np.sqrt(2.0) / mean_distance

    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    normalized = [(float(x), float(y)) for x, y in shifted * scale]

    return normalized, transform


def homography_from_points(
    image_points: Sequence[Point],
    world_points: Sequence[Point],
) -> np.ndarray:
    """
    La matriz que lleva un punto de la IMAGEN a metros sobre el suelo.

    Se resuelve el sistema lineal de 8 incógnitas que sale de las cuatro
    correspondencias, fijando h33 = 1 (una homografía está definida salvo
    escala, así que esa normalización no pierde nada), sobre los puntos ya
    normalizados.
    """

    if len(image_points) != 4 or len(world_points) != 4:
        raise CalibrationError(
            "Hacen falta exactamente 4 puntos en la imagen y 4 en el suelo."
        )

    _reject_degenerate_quad(image_points)
    _reject_degenerate_quad(world_points)

    norm_image, t_image = _normalize(image_points)
    norm_world, t_world = _normalize(world_points)

    rows = []
    targets = []

    for (u, v), (x, y) in zip(norm_image, norm_world):
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        targets.append(x)

        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y])
        targets.append(y)

    a = np.array(rows, dtype=float)
    b = np.array(targets, dtype=float)

    try:
        solution = np.linalg.solve(a, b)
    except np.linalg.LinAlgError as error:
        raise CalibrationError(
            "Los cuatro puntos no forman un cuadrilátero válido."
        ) from error

    normalized_h = np.append(solution, 1.0).reshape(3, 3)

    # Deshacer las dos normalizaciones para volver a píxeles y metros.
    return np.linalg.inv(t_world) @ normalized_h @ t_image


def order_quad(points: Sequence[Point]) -> list[Point]:
    """
    Ordena cuatro esquinas marcadas en cualquier orden.

    Devuelve siempre: cerca-izquierda, cerca-derecha, lejos-derecha,
    lejos-izquierda. "Cerca" es abajo en la imagen, que en una cámara de vía
    es lo próximo a la cámara.

    Existe porque el orden de los clics era la única forma de arruinar una
    calibración sin que nada avisara. Marcando las esquinas rotadas, el ancho
    declarado se aplica a la profundidad y la escena entera queda deformada
    —y la comprobación contra las esquinas sigue dando el resultado
    declarado, porque es circular—. Pedirle a una persona que haga cuatro
    clics en un orden exacto es pedirle que no se equivoque; mejor no
    necesitarlo.
    """

    if len(points) != 4:
        raise CalibrationError("Hacen falta exactamente 4 esquinas.")

    # Las dos de mayor Y están abajo en la imagen: son las cercanas.
    por_profundidad = sorted(points, key=lambda p: p[1], reverse=True)

    cercanas = sorted(por_profundidad[:2], key=lambda p: p[0])
    lejanas = sorted(por_profundidad[2:], key=lambda p: p[0])

    return [cercanas[0], cercanas[1], lejanas[1], lejanas[0]]


def rectangle_world_points(width_m: float, length_m: float) -> list[Point]:
    """
    Las esquinas del rectángulo de referencia, en metros.

    El orden es el mismo en el que la herramienta pide los clics —cerca
    izquierda, cerca derecha, lejos derecha, lejos izquierda— y define el
    sistema de coordenadas: X cruza la vía, Y se aleja de la cámara, el
    origen queda en la esquina cercana izquierda.
    """

    if width_m <= 0 or length_m <= 0:
        raise CalibrationError("El ancho y el largo deben ser positivos.")

    return [
        (0.0, 0.0),
        (width_m, 0.0),
        (width_m, length_m),
        (0.0, length_m),
    ]


@dataclass(frozen=True)
class GroundPlane:
    """Una cámara ya calibrada contra el pavimento."""

    matrix: np.ndarray

    # ------------------------------------------------------------------

    @classmethod
    def from_quad(
        cls,
        image_points: Sequence[Point],
        width_m: float,
        length_m: float,
    ) -> "GroundPlane":
        """A partir de cuatro clics sobre el asfalto y las medidas reales."""

        return cls(
            homography_from_points(
                order_quad(image_points),
                rectangle_world_points(width_m, length_m),
            )
        )

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "GroundPlane":
        """Desde los nueve números guardados en el registro de cámaras."""

        flat = [float(v) for v in values]

        if len(flat) != 9:
            raise CalibrationError(
                f"La homografía necesita 9 valores, llegaron {len(flat)}."
            )

        return cls(np.array(flat, dtype=float).reshape(3, 3))

    def values(self) -> list[float]:
        """Los nueve números, para guardarlos."""
        return [float(v) for v in self.matrix.reshape(9)]

    # ------------------------------------------------------------------

    def to_world(self, point: Point) -> Point:
        """Un punto de la imagen, en metros sobre el suelo."""

        u, v = point
        m = self.matrix

        denominator = m[2, 0] * u + m[2, 1] * v + m[2, 2]

        if abs(denominator) < 1e-9:
            # El punto cae sobre la línea del horizonte: ahí el plano del
            # suelo se proyecta al infinito y no hay respuesta finita.
            raise CalibrationError(
                "El punto está sobre el horizonte y no cae en el plano de la "
                "vía."
            )

        x = (m[0, 0] * u + m[0, 1] * v + m[0, 2]) / denominator
        y = (m[1, 0] * u + m[1, 1] * v + m[1, 2]) / denominator

        return (float(x), float(y))

    def distance_m(self, a: Point, b: Point) -> float:
        """Metros entre dos puntos de la imagen, medidos sobre el suelo."""

        ax, ay = self.to_world(a)
        bx, by = self.to_world(b)

        return float(np.hypot(bx - ax, by - ay))


def ground_point(bbox: dict) -> Point:
    """
    El punto donde un vehículo toca el suelo: el centro del borde INFERIOR.

    No el centro de la caja. El centro flota a media altura del vehículo, y
    proyectarlo sobre el plano del asfalto lo coloca varios metros más lejos
    de lo que está —tanto más cuanto más alto sea, así que un bus se
    "desplazaría" mucho más que una moto—. Es un error sistemático que
    ninguna calibración corrige después.
    """

    return ((float(bbox["x1"]) + float(bbox["x2"])) / 2.0, float(bbox["y2"]))
