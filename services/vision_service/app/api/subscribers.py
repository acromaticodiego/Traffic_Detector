"""
La cola de salida de un cliente conectado a una cámara.

Vive fuera de `session.py` porque ese módulo importa cv2 y el CI corre sin
opencv: aquí no hay nada del stack de visión, así que la política de descarte
—la parte que decide qué mensajes se pierden cuando alguien se atrasa— sí
queda cubierta por los tests.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any, Optional


# Cuántos mensajes se le guardan a un cliente antes de empezar a tirarle
# frames. A 25 fps son casi cinco segundos de margen: aguanta un bache de red
# sin que se note, y más que eso tampoco serviría, porque a un operario no le
# sirve ver un incidente con medio minuto de retraso.
_QUEUE_MAXSIZE = 120

# Techo absoluto. Los incidentes no se tiran, pero un cliente que dejó de leer
# y no se ha desconectado tampoco puede crecer sin fin en memoria.
_HARD_LIMIT = 4 * _QUEUE_MAXSIZE


class Subscriber:
    """
    Un cliente mirando una sesión.

    Cada uno tiene su propia cola: que un operario esté con mala conexión no
    puede frenar la cámara para los demás. Cuando se atrasa se le tiran los
    frames más viejos —son reemplazables, el siguiente llega en 40 ms— pero
    NUNCA los incidentes: un choque no se puede perder porque a alguien se le
    cayera el wifi.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._pending: deque[dict[str, Any]] = deque()

        # El hilo de visión escribe y el bucle de asyncio lee. Cada operación
        # suelta de un deque es atómica, pero podar no lo es.
        self._lock = threading.Lock()
        self._wakeup = asyncio.Event()

        self.dropped = 0

    # ------------------------------------------------------------------

    def push(self, message: dict[str, Any]) -> None:
        """Encolar desde el hilo de visión. No bloquea nunca."""

        with self._lock:
            self._pending.append(message)

            if len(self._pending) > _QUEUE_MAXSIZE:
                self._prune()

        try:
            self._loop.call_soon_threadsafe(self._wakeup.set)
        except RuntimeError:
            # El bucle ya se cerró: el cliente se fue y no hay a quién avisar.
            pass

    def _prune(self) -> None:
        """El frame más viejo se va; los incidentes se quedan."""

        for index, message in enumerate(self._pending):
            if message.get("type") == "frame":
                del self._pending[index]
                self.dropped += 1
                return

        if len(self._pending) > _HARD_LIMIT:
            self._pending.popleft()
            self.dropped += 1

    @property
    def pending(self) -> int:
        """Cuántos mensajes se le deben. Sube cuando el cliente se atrasa."""

        with self._lock:
            return len(self._pending)

    def _next(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._pending.popleft() if self._pending else None

    async def messages(self):
        """
        Async generator yielding stream messages until
        `done` (or the session is stopped).
        """

        while True:
            message = self._next()

            if message is None:
                # Limpiar el evento ANTES de volver a mirar la cola: al revés
                # se pierde un push que caiga justo entre las dos, y el
                # cliente se queda esperando con mensajes ya encolados.
                self._wakeup.clear()
                message = self._next()

                if message is None:
                    await self._wakeup.wait()
                    continue

            yield message

            if message.get("type") in ("done", "error"):
                break
