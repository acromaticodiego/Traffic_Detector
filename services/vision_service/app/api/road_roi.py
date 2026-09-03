"""
Región de interés (ROI) de la calzada.

El nivel de tráfico no se puede medir contando vehículos: 6 carros en una
avenida de 6 carriles es flujo libre, y en una calle de un carril es un trancón.
Lo que sí se puede medir es la **ocupación**: qué fracción del área de la vía
está cubierta por vehículos (el equivalente al "occupancy %" que reportan los
lazos inductivos del pavimento).

Este módulo aporta la parte geométrica de ese cálculo:

  * un polígono que delimita la calzada, en coordenadas NORMALIZADAS 0..1
    (independientes de la resolución, así la misma ROI sirve para cualquier
    cámara o para el mismo video reescalado),
  * una máscara rasterizada de ese polígono, y
  * un peso por fila que compensa la perspectiva.

Lo de la perspectiva importa: en una toma oblicua un carro cercano ocupa
muchos más píxeles que uno lejano, así que contar píxeles crudos sobrevalora
el tráfico del primer plano. `VISION_ROAD_PERSPECTIVE` es la razón de tamaño
aparente entre un vehículo abajo de la ROI y uno arriba (3.0 = el de abajo se
ve 3 veces más largo). Con eso el peso de cada fila es 1/escala², que es
exactamente lo que hace falta para que un carro pese igual esté donde esté.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Resolución de la rasterización. La ocupación es una fracción de área, así que
# una grilla gruesa basta y cuesta microsegundos por frame.
GRID_W = 192
GRID_H = 108


def parse_polygon(raw: str) -> list[tuple[float, float]] | None:
    """
    Convierte "0.02,0.55 0.45,0.12 0.98,0.30" en [(0.02, 0.55), ...].

    Acepta puntos separados por espacios o ';' y coordenadas por ','.
    Devuelve None si la cadena está vacía o no deja al menos 3 puntos
    válidos (un polígono con menos de 3 vértices no encierra área).
    """

    if not raw or not raw.strip():
        return None

    points: list[tuple[float, float]] = []

    for chunk in raw.replace(";", " ").split():

        parts = chunk.split(",")

        if len(parts) != 2:
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue

        points.append(
            (
                min(max(x, 0.0), 1.0),
                min(max(y, 0.0), 1.0),
            )
        )

    return points if len(points) >= 3 else None


def _polygon_mask(
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Rasteriza el polígono con ray casting vectorizado (regla par/impar).

    Se evalúa el centro de cada celda para no sesgar el borde hacia adentro
    ni hacia afuera.
    """

    xs = (np.arange(width) + 0.5) / width
    ys = (np.arange(height) + 0.5) / height

    grid_x, grid_y = np.meshgrid(xs, ys)

    inside = np.zeros((height, width), dtype=bool)

    count = len(polygon)

    for i in range(count):

        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]

        # ¿La arista cruza la horizontal que pasa por esta celda?
        straddles = (y1 > grid_y) != (y2 > grid_y)

        if not straddles.any():
            continue

        # x del cruce. y2 != y1 siempre que `straddles` sea True, así que la
        # división solo puede dar NaN donde la máscara ya la descarta.
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = x1 + (grid_y - y1) * (x2 - x1) / (y2 - y1)

        inside ^= straddles & (grid_x < crossing_x)

    return inside


@dataclass
class RoadROI:
    """
    Calzada rasterizada + pesos de perspectiva.

    Sin polígono configurado la ROI es el frame completo: funciona sin
    calibrar nada, aunque el cielo, los árboles y el andén diluyen la
    ocupación. Para medir de verdad hay que dibujar el polígono
    (ver `scripts/roi_picker.py`).
    """

    polygon: list[tuple[float, float]] | None
    perspective: float

    mask: np.ndarray            # (GRID_H, GRID_W) bool
    weights: np.ndarray         # (GRID_H, GRID_W) float, 0 fuera de la ROI
    weight_total: float

    y_top: float                # extremo superior de la ROI (0..1)
    y_bottom: float             # extremo inferior de la ROI (0..1)

    @property
    def calibrated(self) -> bool:
        """True si hay un polígono de calzada definido."""
        return self.polygon is not None

    # ------------------------------------------------------------------
    # perspectiva
    # ------------------------------------------------------------------

    def scale_at(self, y_norm: float) -> float:
        """
        Tamaño aparente relativo a esa altura del frame: 1.0 en el borde
        superior de la ROI y `perspective` en el inferior.

        Sirve tanto para normalizar áreas (dividiendo por el cuadrado) como
        velocidades en px/frame (dividiendo por el valor directo).
        """

        if self.perspective == 1.0 or self.y_bottom <= self.y_top:
            return 1.0

        t = (y_norm - self.y_top) / (self.y_bottom - self.y_top)
        t = min(max(t, 0.0), 1.0)

        return 1.0 + (self.perspective - 1.0) * t

    # ------------------------------------------------------------------
    # consultas
    # ------------------------------------------------------------------

    def contains(self, x_norm: float, y_norm: float) -> bool:
        """
        ¿Ese punto cae sobre la calzada?

        Se espera el centro inferior de la caja (donde el vehículo toca el
        piso), no el centro geométrico: un camión alto tiene el centro muy
        por encima del asfalto.
        """

        col = int(x_norm * GRID_W)
        row = int(y_norm * GRID_H)

        col = min(max(col, 0), GRID_W - 1)
        row = min(max(row, 0), GRID_H - 1)

        return bool(self.mask[row, col])

    def occupancy(
        self,
        boxes: list[tuple[float, float, float, float]],
    ) -> float:
        """
        Fracción de la calzada cubierta por las cajas dadas (0..1).

        `boxes` son (x1, y1, x2, y2) NORMALIZADAS. Se rasteriza la UNIÓN, no
        la suma: dos vehículos superpuestos no ocupan doble asfalto.
        """

        if not boxes or self.weight_total <= 0.0:
            return 0.0

        covered = np.zeros_like(self.mask)

        for x1, y1, x2, y2 in boxes:

            col_from = int(math.floor(min(x1, x2) * GRID_W))
            col_to = int(math.ceil(max(x1, x2) * GRID_W))
            row_from = int(math.floor(min(y1, y2) * GRID_H))
            row_to = int(math.ceil(max(y1, y2) * GRID_H))

            col_from = min(max(col_from, 0), GRID_W)
            col_to = min(max(col_to, col_from + 1), GRID_W)
            row_from = min(max(row_from, 0), GRID_H)
            row_to = min(max(row_to, row_from + 1), GRID_H)

            covered[row_from:row_to, col_from:col_to] = True

        occupied = float((covered * self.weights).sum())

        return min(occupied / self.weight_total, 1.0)


def build_road_roi(
    raw_polygon: str = "",
    perspective: float = 1.0,
) -> RoadROI:
    """
    Construye la ROI una sola vez (al abrir la sesión) y la reutiliza en
    todos los frames.
    """

    perspective = max(float(perspective), 1.0)

    polygon = parse_polygon(raw_polygon)

    if polygon is None:
        mask = np.ones((GRID_H, GRID_W), dtype=bool)
        y_top = 0.0
        y_bottom = 1.0
    else:
        mask = _polygon_mask(polygon, GRID_W, GRID_H)
        y_top = min(y for _, y in polygon)
        y_bottom = max(y for _, y in polygon)

        # Un polígono degenerado (o mal escrito) dejaría la máscara vacía y
        # la ocupación siempre en 0. Mejor caer al frame completo que
        # reportar "tráfico bajo" para siempre.
        if not mask.any():
            mask = np.ones((GRID_H, GRID_W), dtype=bool)
            polygon = None
            y_top = 0.0
            y_bottom = 1.0

    rows = (np.arange(GRID_H) + 0.5) / GRID_H

    if perspective == 1.0 or y_bottom <= y_top:
        row_scale = np.ones(GRID_H)
    else:
        t = np.clip((rows - y_top) / (y_bottom - y_top), 0.0, 1.0)
        row_scale = 1.0 + (perspective - 1.0) * t

    # Área aparente ∝ escala² -> pesar por 1/escala² iguala el aporte de un
    # vehículo cercano y uno lejano.
    weights = mask * (1.0 / (row_scale**2))[:, None]

    return RoadROI(
        polygon=polygon,
        perspective=perspective,
        mask=mask,
        weights=weights,
        weight_total=float(weights.sum()),
        y_top=y_top,
        y_bottom=y_bottom,
    )
