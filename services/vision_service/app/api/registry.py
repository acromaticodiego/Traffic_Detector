"""
Qué cámaras se están procesando ahora mismo y quién las está mirando.

Vive aparte de la ruta del WebSocket, y sin importar nada del stack de visión,
para que el CI —que corre sin opencv ni ultralytics— pueda probar justo lo que
tiene riesgo aquí: que dos operarios en la misma cámara compartan sesión en vez
de procesarla dos veces, que el tope se respete, y que la última desconexión no
deje una cámara huérfana ni apague una que otro acaba de abrir.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Crea y abre una sesión. Es una corrutina porque abrir la fuente toca disco o
# red, y eso no puede bloquear el bucle que atiende a las demás cámaras.
SessionFactory = Callable[[], Awaitable[Any]]


class Rejected(Exception):
    """No se pudo enganchar al cliente; lleva el motivo que él va a leer."""

    def __init__(self, message: str, code: str):
        super().__init__(message)

        self.message = message
        self.code = code


class SessionRegistry:
    """
    Las sesiones vivas, indexadas por cámara.

    Una cámara se procesa UNA sola vez aunque la miren diez personas: procesar
    el mismo video dos veces gastaría el doble de GPU para producir exactamente
    lo mismo. El tope cuenta cámaras, no espectadores.
    """

    def __init__(self, max_sessions: int):
        self._max = max(1, max_sessions)
        self._sessions: dict[str, Any] = {}

        # Todo el ciclo de vida pasa por aquí. Sin el lock, entre comprobar si
        # la cámara está viva y arrancarla caben dos clientes, y acabaríamos
        # con dos sesiones sobre la misma cámara —justo lo que evitamos.
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return len(self._sessions)

    @property
    def limit(self) -> int:
        return self._max

    def camera_ids(self) -> list[str]:
        return sorted(self._sessions)

    def snapshot(self) -> list[tuple[str, Any]]:
        """
        Las sesiones vivas, para observabilidad.

        Sin tomar el lock a propósito: solo lee, y la ruta de métricas no
        puede quedarse esperando a que arranque una cámara. Como mucho ve el
        registro un instante antes o después de un cambio, que para medir da
        igual.
        """

        return sorted(self._sessions.items())

    # ------------------------------------------------------------------

    async def acquire(
        self,
        camera_id: str,
        factory: SessionFactory,
    ) -> tuple[Any, Any]:
        """
        Enganchar un cliente a la cámara, arrancándola si no estaba.

        Devuelve la sesión y el suscriptor del cliente.
        """

        async with self._lock:
            session = self._sessions.get(camera_id)

            if session is None:
                if len(self._sessions) >= self._max:
                    raise Rejected(
                        f"Ya hay {len(self._sessions)} cámaras procesándose, "
                        f"que es el máximo configurado. Cierra una para poder "
                        f"abrir esta.",
                        "capacity",
                    )

                session = await factory()

                try:
                    session.start()
                except Exception:
                    # La fuente ya está abierta: si no se cierra queda un
                    # archivo (o una conexión RTSP) colgando sin dueño.
                    await asyncio.to_thread(session.stop)
                    raise

                self._sessions[camera_id] = session

                logger.info(
                    "Cámara '%s' arrancada (%d de %d en curso).",
                    camera_id,
                    len(self._sessions),
                    self._max,
                )

            return session, session.add_subscriber()

    async def release(
        self,
        camera_id: str,
        session: Any,
        subscriber: Any,
    ) -> None:
        """
        Soltar a un cliente y apagar la cámara si era el último.

        Bajo el mismo lock que `acquire`: entre contar los que quedan y borrar
        la sesión cabría un cliente nuevo, que se quedaría enganchado a una
        cámara a punto de pararse.
        """

        async with self._lock:
            session.remove_subscriber(subscriber)

            if session.subscribers:
                return

            # Puede no ser la que está registrada si la cámara ya se reinició.
            if self._sessions.get(camera_id) is session:
                del self._sessions[camera_id]

            await asyncio.to_thread(session.stop)

            logger.info(
                "Cámara '%s' detenida: no queda nadie mirándola.", camera_id
            )

    async def stop_all(self) -> None:
        """Apagar todas las cámaras. La llama el apagado del servicio."""

        async with self._lock:
            sessions = list(self._sessions.items())
            self._sessions.clear()

        for camera_id, session in sessions:
            logger.info("Deteniendo la cámara '%s'...", camera_id)
            await asyncio.to_thread(session.stop)
