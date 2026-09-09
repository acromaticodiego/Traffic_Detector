"""
Qué significa "se acabó la fuente" y "qué hora es" según de dónde venga el video.

Un archivo y una cámara en vivo se comportan al revés en casi todo: el archivo
tiene principio y fin, se puede rebobinar y hay que marcarle el ritmo; el
stream no se acaba nunca —si se acaba, es que se cayó— y se marca su propio
ritmo. El pipeline es el mismo, así que la diferencia se concentra aquí en vez
de repartirse en condicionales por todo el bucle de visión.

Vive fuera de `session.py`, que importa cv2, para que el CI lo pruebe sin
opencv: son decisiones puras sobre números y banderas.
"""

from __future__ import annotations

from typing import Any, Optional


class Playback:
    """La política de reproducción de una fuente concreta."""

    def __init__(
        self,
        is_stream: bool,
        loop_enabled: bool,
        fps: float,
    ):
        self.is_stream = is_stream
        self.loop_enabled = loop_enabled
        self.fps = fps

    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        """
        Si al acabarse el video hay que volver a empezar.

        Rebobinar un stream no significa nada: no hay nada a lo que volver.
        """

        return self.loop_enabled and not self.is_stream

    @property
    def paced(self) -> bool:
        """
        Si hay que frenar a propósito para no adelantar al reloj.

        Un archivo se procesa tan rápido como dé la GPU, así que sin freno el
        operario vería la calle en cámara rápida. Una cámara en vivo entrega
        un frame cuando lo tiene, y frenarla de más solo acumularía retraso.
        """

        return not self.is_stream

    def frame_count(self, reported: int) -> Optional[int]:
        """
        Cuántos frames tiene la fuente, o None si la pregunta no aplica.

        Un stream no tiene final, y opencv responde a esa pregunta con basura
        (un 0 o un negativo). Decir `null` es lo honesto: 0 se lee como "un
        video vacío", que es otra cosa.
        """

        if self.is_stream or reported <= 0:
            return None

        return reported

    def timestamp(self, pass_frame: int, elapsed: float) -> float:
        """
        El `t` que va en cada mensaje, en segundos.

        En un archivo es la posición DENTRO de la pasada actual, no el tiempo
        que lleva corriendo la sesión: el frontend lo usa para saltar a ese
        punto del video (`video.currentTime`), y un valor que crece sin fin
        —o que se pasa de la duración cuando el video va por su tercera
        vuelta— no lo puede ubicar.

        En vivo no hay video que rebobinar, así que la única lectura posible
        es cuánto lleva abierta la sesión.
        """

        if self.is_stream:
            return elapsed

        if self.fps <= 0:
            return 0.0

        return pass_frame / self.fps

    def end_message(self, frames: int, processed: int) -> dict[str, Any]:
        """
        Cómo se cierra la sesión cuando la fuente deja de dar frames.

        En un archivo es el final esperado. En una cámara en vivo NO: que deje
        de mandar es una avería, y anunciarla como un final normal haría que
        el operario viera "listo" donde debería ver que perdió la cámara.
        """

        if self.is_stream:
            return {
                "type": "error",
                "message": (
                    "Se perdió la conexión con la cámara. Vuelve a abrirla "
                    "para reintentar."
                ),
                "code": "source_lost",
            }

        return {
            "type": "done",
            "frames": frames,
            "processed": processed,
        }
