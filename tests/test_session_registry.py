"""
El registro de cámaras vivas: quién comparte sesión y cuándo se apaga una.

Es la parte con carreras de todo el multi-cámara, así que se prueba con
sesiones falsas —sin video ni modelo— para que corra en el CI.
"""

import asyncio

import pytest

from services.vision_service.app.api.registry import Rejected, SessionRegistry


class FakeSession:
    """Una sesión que cuenta lo que le hacen, sin abrir nada."""

    def __init__(self, fails_to_start: bool = False):
        self.started = 0
        self.stopped = 0
        self.fails_to_start = fails_to_start

        self._subscribers: set[object] = set()

    def start(self) -> None:
        if self.fails_to_start:
            raise RuntimeError("no arrancó el hilo")

        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def add_subscriber(self) -> object:
        subscriber = object()
        self._subscribers.add(subscriber)

        return subscriber

    def remove_subscriber(self, subscriber: object) -> None:
        self._subscribers.discard(subscriber)

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)


def factory_for(session: FakeSession):
    async def factory() -> FakeSession:
        return session

    return factory


def failing_factory(error: Exception):
    async def factory():
        raise error

    return factory


class TestFanOut:

    def test_dos_operarios_en_la_misma_camara_comparten_sesion(self):
        """Procesar dos veces el mismo video gastaría el doble para nada."""

        async def main():
            registry = SessionRegistry(max_sessions=3)
            camera = FakeSession()

            first, sub_a = await registry.acquire("cra55", factory_for(camera))
            second, sub_b = await registry.acquire("cra55", factory_for(camera))

            return registry, camera, first is second, sub_a is sub_b

        registry, camera, same_session, same_subscriber = asyncio.run(main())

        assert same_session
        # Cada uno con su cola: si compartieran una, leer uno le robaría los
        # mensajes al otro.
        assert not same_subscriber
        assert camera.started == 1
        assert registry.active == 1

    def test_el_primero_en_irse_no_apaga_la_camara(self):
        async def main():
            registry = SessionRegistry(max_sessions=3)
            camera = FakeSession()

            _, sub_a = await registry.acquire("cra55", factory_for(camera))
            _, sub_b = await registry.acquire("cra55", factory_for(camera))

            await registry.release("cra55", camera, sub_a)
            quedaba_viva = camera.stopped == 0 and registry.active == 1

            await registry.release("cra55", camera, sub_b)

            return quedaba_viva, camera.stopped, registry.active

        quedaba_viva, stopped, active = asyncio.run(main())

        assert quedaba_viva
        assert stopped == 1
        assert active == 0

    def test_soltar_una_sesion_ya_reemplazada_no_apaga_la_nueva(self):
        """
        La cámara se cerró y se volvió a abrir; el cliente viejo se desconecta
        tarde. Su adiós no puede llevarse por delante la sesión nueva.
        """

        async def main():
            registry = SessionRegistry(max_sessions=3)
            vieja, nueva = FakeSession(), FakeSession()

            _, sub_vieja = await registry.acquire("cra55", factory_for(vieja))
            await registry.release("cra55", vieja, sub_vieja)

            _, _sub_nueva = await registry.acquire("cra55", factory_for(nueva))

            await registry.release("cra55", vieja, sub_vieja)

            return registry.active, nueva.stopped

        active, nueva_stopped = asyncio.run(main())

        assert active == 1
        assert nueva_stopped == 0


class TestTope:

    def test_rechaza_la_camara_que_pasa_del_tope(self):
        async def main():
            registry = SessionRegistry(max_sessions=2)

            await registry.acquire("uno", factory_for(FakeSession()))
            await registry.acquire("dos", factory_for(FakeSession()))

            tercera = FakeSession()

            with pytest.raises(Rejected) as rejection:
                await registry.acquire("tres", factory_for(tercera))

            return rejection.value, tercera.started, registry.camera_ids()

        error, tercera_started, camera_ids = asyncio.run(main())

        assert error.code == "capacity"
        # Ni siquiera se intenta abrir: se rechaza antes de tocar la fuente.
        assert tercera_started == 0
        assert camera_ids == ["dos", "uno"]

    def test_el_tope_cuenta_camaras_y_no_espectadores(self):
        """Diez operarios sobre una cámara siguen siendo una sola sesión."""

        async def main():
            registry = SessionRegistry(max_sessions=1)
            camera = FakeSession()

            for _ in range(10):
                await registry.acquire("cra55", factory_for(camera))

            return registry.active, camera.started

        active, started = asyncio.run(main())

        assert active == 1
        assert started == 1


class TestArranqueFallido:

    def test_una_camara_que_no_abre_no_queda_registrada(self):
        async def main():
            registry = SessionRegistry(max_sessions=2)

            with pytest.raises(Rejected):
                await registry.acquire(
                    "rota",
                    failing_factory(Rejected("no se pudo abrir", "unavailable")),
                )

            libre = registry.active

            # Y el hueco sigue disponible para el siguiente intento.
            await registry.acquire("buena", factory_for(FakeSession()))

            return libre, registry.camera_ids()

        libre, camera_ids = asyncio.run(main())

        assert libre == 0
        assert camera_ids == ["buena"]

    def test_si_falla_el_arranque_se_cierra_la_fuente(self):
        """
        La fuente ya está abierta cuando se arranca el hilo. Si eso revienta y
        nadie la cierra, queda un archivo —o una conexión RTSP— sin dueño.
        """

        async def main():
            registry = SessionRegistry(max_sessions=2)
            camera = FakeSession(fails_to_start=True)

            with pytest.raises(RuntimeError):
                await registry.acquire("cra55", factory_for(camera))

            return camera.stopped, registry.active

        stopped, active = asyncio.run(main())

        assert stopped == 1
        assert active == 0


class TestApagado:

    def test_stop_all_apaga_todas_las_camaras(self):
        async def main():
            registry = SessionRegistry(max_sessions=3)
            cameras = [FakeSession() for _ in range(3)]

            for index, camera in enumerate(cameras):
                await registry.acquire(f"cam{index}", factory_for(camera))

            await registry.stop_all()

            return [c.stopped for c in cameras], registry.active

        stopped, active = asyncio.run(main())

        assert stopped == [1, 1, 1]
        assert active == 0
