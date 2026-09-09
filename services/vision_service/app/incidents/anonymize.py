"""
Qué hay que tapar en una imagen de evidencia, y dónde.

Ley 1581 de 2012 (Habeas Data): la evidencia que se guarda en disco no puede
permitir identificar a una persona ni a un vehículo concreto. Aquí se decide
la geometría; el difuminado en sí lo aplica `evidence.py`.

La decisión de fondo, que explica todo lo demás:

    NO se detecta la placa. Se tapa la franja donde una placa TIENE que estar.

Un detector de placas falla abierto: la que no reconoce se queda legible en
disco para siempre y nadie se entera. Una franja geométrica falla cerrada:
se tapa haya placa o no, y da igual el ángulo, la distancia o la suciedad
sobre el vidrio. Para un requisito legal se quiere el modo que falla del lado
seguro, no el que tiene mejor F1. Además evita cargar un segundo modelo, que
en esta GPU ya va justa con dos cámaras.

Esto NO sirve para leer placas ni para identificar vehículos: el producto
mide conflictos viales, no pone comparendos. El día que alguien pida eso,
hace falta otro modelo y, sobre todo, otro marco legal.

Vive fuera de `evidence.py`, que importa cv2, para que el CI pueda probar la
cobertura sin opencv: es aritmética sobre rectángulos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

# Una caja ya normalizada: la clase y las cuatro esquinas en píxeles.
Box = tuple[str, float, float, float, float]


# Vehículos: llevan placa, siempre en la parte baja de la carrocería. La moto
# está en los dos conjuntos porque tiene placa abajo y conductor arriba.
CON_PLACA = frozenset(
    {"ambulance", "bus", "car", "motorcycle", "truck"}
)

# Todo lo que lleva a una persona encima o dentro del recuadro.
CON_PERSONA = frozenset(
    {"pedestrian", "ciclist", "monopatin", "motorcycle"}
)

# Fracción de la caja que se tapa. Son generosas a propósito: la caja del
# detector oscila entre frames, y quedarse corto significa una placa legible
# guardada para siempre. Pasarse solo cuesta un poco de imagen.
PLATE_BAND = 0.35
FACE_BAND = 0.30


@dataclass(frozen=True)
class Region:
    """Un rectángulo a tapar, en píxeles enteros y dentro del frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def empty(self) -> bool:
        return self.x2 <= self.x1 or self.y2 <= self.y1


def boxes_from_tracks(tracks: Iterable[Any]) -> list[Box]:
    """Las cajas de los objetos seguidos (`TrackState`, coordenadas planas)."""

    return [
        (track.class_name, track.x1, track.y1, track.x2, track.y2)
        for track in tracks
    ]


def boxes_from_detections(detections: Iterable[Any]) -> list[Box]:
    """Las cajas crudas del detector (`Detection`, con la caja dentro)."""

    return [
        (
            detection.class_name,
            detection.bbox.x1,
            detection.bbox.y1,
            detection.bbox.x2,
            detection.bbox.y2,
        )
        for detection in detections
    ]


def regions_for(
    boxes: Iterable[Box],
    width: int,
    height: int,
    plate_band: float = PLATE_BAND,
    face_band: float = FACE_BAND,
) -> list[Region]:
    """
    Los rectángulos a tapar en un frame completo.

    Se pasan TODAS las cajas del frame, no solo las del incidente: la
    evidencia es la imagen entera, y ahí salen las placas de los carros que
    iban al lado y la cara de quien cruzaba por detrás. Anonimizar solo a los
    implicados dejaría al resto identificable, que es exactamente lo que la
    ley no permite.
    """

    regions: list[Region] = []

    for class_name, x1, y1, x2, y2 in boxes:
        box = _clamp(x1, y1, x2, y2, width, height)

        if box is None:
            continue

        left, top, right, bottom = box
        alto = bottom - top

        name = (class_name or "").strip().lower()

        # Una clase que no conocemos se trata como si fuera las dos cosas. Si
        # mañana se reentrena el modelo y aparece una clase nueva, el fallo
        # tiene que ser "tapé de más", no "dejé una placa legible".
        known = name in CON_PLACA or name in CON_PERSONA

        if name in CON_PERSONA or not known:
            regions.append(
                Region(left, top, right, top + _band(alto, face_band))
            )

        if name in CON_PLACA or not known:
            regions.append(
                Region(left, bottom - _band(alto, plate_band), right, bottom)
            )

    return [region for region in regions if not region.empty]


def _band(alto: int, fraction: float) -> int:
    """
    Cuántos píxeles de alto tiene la franja.

    Al menos uno: una caja pequeña sigue siendo una caja, y devolver cero la
    dejaría sin tapar justo en el caso en que nadie lo va a revisar a ojo.
    """

    fraction = min(1.0, max(0.0, fraction))

    return max(1, int(alto * fraction + 0.5))


def _clamp(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """
    La caja en enteros, recortada al frame. None si no queda nada.

    Redondea hacia afuera —suelo a la izquierda, techo a la derecha— porque
    medio píxel de menos en el borde de una placa se lee igual.
    """

    left = max(0, math.floor(min(x1, x2)))
    top = max(0, math.floor(min(y1, y2)))
    right = min(width, math.ceil(max(x1, x2)))
    bottom = min(height, math.ceil(max(y1, y2)))

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom
