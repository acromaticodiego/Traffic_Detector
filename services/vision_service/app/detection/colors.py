"""
Colores para etiquetas y bounding boxes de detección y seguimiento.

OpenCV utiliza el orden BGR (Blue, Green, Red).
Cada etiqueta/clase de vehículo y actor vial tiene asignado un color distintivo
de alto contraste para diferenciar claramente cada tipo en pantalla.
"""

from __future__ import annotations

import unicodedata

# ============================================================
# PALETA DE COLORES POR CLASE (BGR)
# ============================================================
#
# Azul Real (Carro)          = (255, 100, 0)
# Naranja (Moto)             = (0, 140, 255)
# Magenta / Fucsia (Camión)  = (255, 0, 255)
# Amarillo (Bus)             = (0, 235, 255)
# Rojo Carmesí (Ambulancia)  = (40, 40, 245)
# Verde Lima (Ciclista)      = (50, 220, 50)
# Morado / Violeta (Patineta)= (220, 80, 160)
# Cian / Turquesa (Peatón)   = (255, 255, 0)
#
# ============================================================

COLOR_CAR = (255, 100, 0)          # Royal / Dodger Blue
COLOR_MOTORCYCLE = (0, 140, 255)   # Vivid Orange
COLOR_TRUCK = (255, 0, 255)        # Magenta / Fuchsia
COLOR_BUS = (0, 235, 255)          # Bright Yellow
COLOR_AMBULANCE = (40, 40, 245)    # Crimson / Emergency Red
COLOR_CYCLIST = (50, 220, 50)      # Lime Green
COLOR_SCOOTER = (220, 80, 160)     # Purple / Violet
COLOR_PEDESTRIAN = (255, 255, 0)   # Cyan / Teal

DEFAULT_COLOR = (200, 200, 200)    # Gris claro neutro


CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    # Carros / Vehículos livianos
    "car": COLOR_CAR,
    "carro": COLOR_CAR,
    "auto": COLOR_CAR,
    "automovil": COLOR_CAR,

    # Motocicletas
    "motorcycle": COLOR_MOTORCYCLE,
    "moto": COLOR_MOTORCYCLE,

    # Camiones / Vehículos pesados
    "truck": COLOR_TRUCK,
    "camion": COLOR_TRUCK,

    # Buses / Transporte público
    "bus": COLOR_BUS,

    # Ambulancias / Emergencias
    "ambulance": COLOR_AMBULANCE,
    "ambulancia": COLOR_AMBULANCE,

    # Ciclistas / Bicicletas
    "ciclist": COLOR_CYCLIST,
    "cyclist": COLOR_CYCLIST,
    "ciclista": COLOR_CYCLIST,
    "bicycle": COLOR_CYCLIST,
    "bicicleta": COLOR_CYCLIST,

    # Monopatines / Patinetas
    "monopatin": COLOR_SCOOTER,
    "patineta": COLOR_SCOOTER,
    "scooter": COLOR_SCOOTER,

    # Peatones
    "pedestrian": COLOR_PEDESTRIAN,
    "peaton": COLOR_PEDESTRIAN,
}

# Paleta rotativa para etiquetas desconocidas o dinámicas
FALLBACK_PALETTE: list[tuple[int, int, int]] = [
    (255, 140, 0),    # Azul profundo
    (0, 165, 255),    # Naranja cálido
    (180, 105, 255),  # Rosa
    (0, 215, 255),    # Oro
    (75, 180, 60),    # Verde bosque
    (204, 50, 153),   # Violeta
    (255, 191, 0),    # Celeste
    (0, 128, 255),    # Ámbar
]


def _normalize_name(name: str | None) -> str:
    """Normaliza un nombre de clase quitando espacios, mayúsculas y tildes."""
    if not name:
        return ""
    text = name.strip().lower()
    # Eliminar acentos diacríticos (á -> a, ó -> o, etc.)
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def get_class_color(class_name: str | None) -> tuple[int, int, int]:
    """
    Retorna el color BGR correspondiente a la etiqueta o clase indicada.

    Si la clase no se encuentra en el mapa directo, se le asigna de manera
    determinística un color llamativo y consistente de la paleta rotativa.
    """
    key = _normalize_name(class_name)
    if not key:
        return DEFAULT_COLOR

    if key in CLASS_COLORS:
        return CLASS_COLORS[key]

    # Asignación determinista por hash del nombre para etiquetas no mapeadas
    palette_index = abs(hash(key)) % len(FALLBACK_PALETTE)
    return FALLBACK_PALETTE[palette_index]


def get_text_color(bgr_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Retorna (0, 0, 0) para fondos claros y (255, 255, 255) para fondos oscuros,
    garantizando un contraste óptimo y legibilidad del texto en la etiqueta.
    """
    b, g, r = bgr_color
    # Luminancia perceptual estándar
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    if luminance > 160:
        return (0, 0, 0)  # Texto negro
    return (255, 255, 255)  # Texto blanco
