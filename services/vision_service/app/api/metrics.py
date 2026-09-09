"""
Cómo se traduce un puñado de números en "esta cámara está bien" o no.

Hoy el servicio se ve igual de sano por fuera vaya al día o vaya media hora
por detrás de la calle. Y ese es el fallo que más caro sale: nadie lo nota.
Con dos cámaras a la vez sobre una RTX 3050 el pipeline ya corre a 0,69x del
tiempo real, así que no es una hipótesis.

La clasificación vive aquí, separada de quien recoge los números, por dos
razones: es la parte que tiene criterio —y el criterio hay que poder
discutirlo— y así el CI la prueba sin opencv ni torch.

Tres estados y nada más. Un semáforo con siete colores no lo mira nadie:

    ok         va al día
    retrasada  procesa, pero más lento que la calle
    detenida   lleva segundos sin producir un frame
"""

from __future__ import annotations

from typing import Any, Optional

# Segundos sin un solo frame antes de dar la cámara por caída. A 25 fps son
# más de cien frames perdidos: ya no es un bache, es que algo se rompió.
STALL_SECONDS = 5.0

# Por debajo de esta fracción del ritmo de la fuente, la cámara va retrasada.
# No es 1.0 a propósito: el ritmo real oscila y marcar en rojo por un 2% de
# diferencia entrena a todo el mundo a ignorar el aviso.
BEHIND_RATIO = 0.85

OK = "ok"
BEHIND = "retrasada"
STALLED = "detenida"

# De menos a más grave. El estado global es el peor de las cámaras.
_SEVERITY = {OK: 0, BEHIND: 1, STALLED: 2}


def camera_status(
    measured_fps: Optional[float],
    source_fps: float,
    since_last_frame_s: float,
    warmup_s: float = 0.0,
) -> dict[str, Any]:
    """
    El estado de una cámara y, si no está bien, por qué.

    `warmup_s` son los segundos que lleva abierta: recién arrancada todavía no
    hay ventana suficiente para medir nada, y llamarla "retrasada" en su primer
    segundo sería ruido garantizado.
    """

    if since_last_frame_s >= STALL_SECONDS:
        return {
            "status": STALLED,
            "detail": (
                f"Sin frames desde hace {since_last_frame_s:.0f} s."
            ),
        }

    # Todavía arrancando, o sin dos marcas con las que calcular un ritmo.
    if measured_fps is None or warmup_s < STALL_SECONDS:
        return {"status": OK, "detail": "Arrancando."}

    if source_fps <= 0:
        return {"status": OK, "detail": ""}

    ratio = measured_fps / source_fps

    if ratio < BEHIND_RATIO:
        return {
            "status": BEHIND,
            "detail": (
                f"Va a {ratio:.0%} del tiempo real "
                f"({measured_fps:.1f} de {source_fps:.0f} fps). "
                f"Baja VISION_MAX_SESSIONS o sube VISION_FRAME_STRIDE."
            ),
        }

    return {"status": OK, "detail": ""}


def overall_status(statuses: list[str]) -> str:
    """El peor estado de las cámaras. Sin cámaras abiertas, no hay nada malo."""

    if not statuses:
        return OK

    return max(statuses, key=lambda status: _SEVERITY.get(status, 0))


def rate(marks: list[float], now: float) -> Optional[float]:
    """
    Frames por segundo a partir de las marcas de tiempo de una ventana.

    None cuando no hay con qué: una sola marca no es un ritmo. Se mide contra
    `now` y no contra la última marca para que una cámara atascada vea caer su
    ritmo en vez de conservar para siempre el que tenía al pararse.
    """

    if len(marks) < 2:
        return None

    span = max(now, marks[-1]) - marks[0]

    if span <= 0:
        return None

    return (len(marks) - 1) / span


def gpu_snapshot() -> Optional[dict[str, Any]]:
    """
    Memoria de la GPU, o None si no hay CUDA o no hay torch.

    Import perezoso y a prueba de todo: las métricas tienen que responder
    aunque el stack de visión no esté instalado, que es como corre el CI.
    """

    try:
        import torch
    except ImportError:
        return None

    try:
        if not torch.cuda.is_available():
            return None

        free, total = torch.cuda.mem_get_info()

        return {
            "device": torch.cuda.get_device_name(0),
            "memory_used_mb": round((total - free) / (1024 * 1024)),
            "memory_total_mb": round(total / (1024 * 1024)),
        }
    except Exception:  # noqa: BLE001
        # Un driver a medias no puede tumbar la única ruta que sirve para
        # saber si el servicio está sano.
        return None
