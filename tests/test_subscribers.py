"""
La cola de salida de un cliente: qué se le guarda y qué se le tira.

Lo que se prueba aquí es la promesa que sostiene el fan-out: un operario con
mala conexión pierde frames, pero no pierde incidentes ni frena la cámara.
"""

import asyncio
import threading
import time

from services.vision_service.app.api.subscribers import (
    _HARD_LIMIT,
    _QUEUE_MAXSIZE,
    Subscriber,
)

DONE = {"type": "done", "frames": 0, "processed": 0}


def frame(number: int) -> dict:
    return {"type": "frame", "frame_id": number}


def incident(number: int) -> dict:
    return {"type": "incident", "incident_type": "colision", "frame_id": number}


def deliver(messages: list[dict]) -> list[dict]:
    """
    Encolar todo y devolver lo que el cliente acaba recibiendo.

    El suscriptor se construye dentro del bucle porque se despierta con
    `call_soon_threadsafe`: creado sobre un bucle que no corre, nadie lo
    despertaría nunca.
    """

    async def main() -> list[dict]:
        subscriber = Subscriber(asyncio.get_running_loop())

        for message in [*messages, DONE]:
            subscriber.push(message)

        received = []

        async for message in subscriber.messages():
            received.append(message)

        return received

    return asyncio.run(main())


class TestEntrega:

    def test_entrega_los_mensajes_en_orden(self):
        received = deliver([frame(1), frame(2), incident(3)])

        assert [m.get("frame_id") for m in received[:3]] == [1, 2, 3]

    def test_termina_en_done(self):
        received = deliver([frame(1)])

        assert received[-1]["type"] == "done"

    def test_termina_tambien_con_un_error(self):
        async def main():
            subscriber = Subscriber(asyncio.get_running_loop())
            subscriber.push({"type": "error", "message": "la fuente se cayó"})
            subscriber.push(frame(99))

            return [m async for m in subscriber.messages()]

        received = asyncio.run(main())

        # El frame de después no se entrega: la sesión ya terminó.
        assert len(received) == 1
        assert received[0]["type"] == "error"

    def test_espera_a_que_el_hilo_de_vision_encole(self):
        """El cliente se queda esperando, no se le corta el stream."""

        async def main():
            subscriber = Subscriber(asyncio.get_running_loop())

            def worker():
                time.sleep(0.05)
                subscriber.push(frame(1))
                subscriber.push(DONE)

            threading.Thread(target=worker, daemon=True).start()

            async def collect():
                return [m async for m in subscriber.messages()]

            return await asyncio.wait_for(collect(), timeout=5.0)

        received = asyncio.run(main())

        assert [m["type"] for m in received] == ["frame", "done"]


class TestDescarte:

    def _cargar(self, messages: list[dict]):
        """
        Encolar y leer en el MISMO bucle.

        Medir en un bucle y drenar en otro parece funcionar, pero deja al
        suscriptor apuntando a un bucle ya cerrado: el test pasaría por el
        camino equivocado.
        """

        async def main():
            subscriber = Subscriber(asyncio.get_running_loop())

            for message in messages:
                subscriber.push(message)

            pending, dropped = subscriber.pending, subscriber.dropped

            subscriber.push(DONE)
            received = [m async for m in subscriber.messages()]

            return pending, dropped, received

        return asyncio.run(main())

    def test_al_atrasarse_pierde_los_frames_mas_viejos(self):
        extra = 10

        pending, dropped, received = self._cargar(
            [frame(n) for n in range(_QUEUE_MAXSIZE + extra)]
        )

        assert pending == _QUEUE_MAXSIZE
        assert dropped == extra

        # Se pierden los viejos y se conservan los recientes: lo último que
        # pasó en la calle es lo que le importa a quien está mirando. El
        # `done` con el que se cierra la lectura desaloja un frame más, que
        # es de dónde sale el +1.
        assert received[0]["frame_id"] == extra + 1
        assert received[-2]["frame_id"] == _QUEUE_MAXSIZE + extra - 1

    def test_un_incidente_no_se_tira_nunca(self):
        # Un choque sepultado bajo un montón de frames: sobreviva quien
        # sobreviva, el incidente tiene que llegar.
        messages = [incident(0)] + [
            frame(n) for n in range(1, _QUEUE_MAXSIZE * 2)
        ]

        pending, dropped, received = self._cargar(messages)

        assert pending == _QUEUE_MAXSIZE
        assert dropped == len(messages) - _QUEUE_MAXSIZE
        assert received[0]["type"] == "incident"

    def test_un_cliente_muerto_no_crece_sin_fin(self):
        # Solo incidentes: no hay frames que tirar, pero la memoria tampoco
        # puede crecer sola mientras el cliente no se desconecta.
        extra = 50

        pending, dropped, _ = self._cargar(
            [incident(n) for n in range(_HARD_LIMIT + extra)]
        )

        assert pending == _HARD_LIMIT
        assert dropped == extra
