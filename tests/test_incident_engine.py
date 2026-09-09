"""
Motor de incidentes.

Lo que se prueba aquí es el criterio, no la geometría: cuándo un vehículo
quieto es un incidente y cuándo es un semáforo en rojo, y cuándo dos cajas
juntas son un choque y cuándo son dos carros parqueados.
"""

import pytest

from services.vision_service.app.geometry.homography import GroundPlane
from services.vision_service.app.incidents.incident_engine import IncidentEngine

from helpers import make_motion, make_track

MOVING = 5.0       # > MOVING_SPEED (2.5): el track cuenta como "ya se movió"
STOPPED = 0.4      # <= STOP_SPEED (0.6)


def feed(engine, track, speed, frames, abrupt=False):
    """Procesa el mismo track durante N frames y devuelve los incidentes."""

    incidents = []

    for _ in range(frames):
        incidents.extend(
            engine.process(
                [track],
                [
                    make_motion(
                        track.track_id,
                        speed=speed,
                        abrupt_change=abrupt,
                        acceleration=-5.0 if abrupt else 0.0,
                    )
                ],
            )
        )

    return incidents


class TestVehiculoDetenido:

    def test_un_frenazo_seguido_de_parada_se_reporta(self):
        engine = IncidentEngine()
        track = make_track(1)

        feed(engine, track, MOVING, frames=5)
        feed(engine, track, MOVING, frames=1, abrupt=True)

        assert feed(engine, track, STOPPED, frames=24) == []

        incidents = feed(engine, track, STOPPED, frames=1)

        assert len(incidents) == 1
        assert incidents[0].incident_type == "vehiculo_detenido"
        assert incidents[0].data["abrupt_stop"] is True
        assert incidents[0].confidence == 0.85

    def test_una_parada_normal_aguanta_mucho_mas_antes_de_avisar(self):
        # Un semáforo en rojo no es un incidente: sin frenazo previo hacen
        # falta 150 frames quieto, no 25.
        engine = IncidentEngine()
        track = make_track(1)

        feed(engine, track, MOVING, frames=5)

        assert feed(engine, track, STOPPED, frames=100) == []

        incidents = feed(engine, track, STOPPED, frames=50)

        assert len(incidents) == 1
        assert incidents[0].data["abrupt_stop"] is False
        assert incidents[0].confidence == 0.65

    def test_un_carro_parqueado_no_es_un_incidente(self):
        # Nunca se movió, así que no se "detuvo": está estacionado.
        engine = IncidentEngine()

        assert feed(engine, make_track(1), STOPPED, frames=300) == []

    def test_un_peaton_quieto_no_es_un_incidente(self):
        engine = IncidentEngine()
        track = make_track(1, class_name="pedestrian")

        feed(engine, track, MOVING, frames=5)

        assert feed(engine, track, STOPPED, frames=300) == []

    def test_se_reporta_una_sola_vez_mientras_siga_quieto(self):
        engine = IncidentEngine()
        track = make_track(1)

        feed(engine, track, MOVING, frames=5)
        feed(engine, track, MOVING, frames=1, abrupt=True)
        primera = feed(engine, track, STOPPED, frames=25)

        assert len(primera) == 1
        assert feed(engine, track, STOPPED, frames=200) == []

    def test_el_id_es_estable_para_que_el_cliente_haga_upsert(self):
        engine = IncidentEngine()
        track = make_track(3)

        feed(engine, track, MOVING, frames=5)
        feed(engine, track, MOVING, frames=1, abrupt=True)
        incidents = feed(engine, track, STOPPED, frames=25)

        assert incidents[0].incident_id == "stop-3"
        assert incidents[0].track_ids == [3]

    def test_si_arranca_y_vuelve_a_pararse_se_reporta_de_nuevo(self):
        engine = IncidentEngine()
        track = make_track(1)

        feed(engine, track, MOVING, frames=5)
        feed(engine, track, MOVING, frames=1, abrupt=True)
        assert len(feed(engine, track, STOPPED, frames=25)) == 1

        # Arranca otra vez: es un evento nuevo, no el mismo.
        feed(engine, track, MOVING, frames=5)
        feed(engine, track, MOVING, frames=1, abrupt=True)

        assert len(feed(engine, track, STOPPED, frames=25)) == 1

    def test_la_caja_del_incidente_es_la_del_vehiculo(self):
        engine = IncidentEngine()
        track = make_track(1, box=(10, 20, 50, 60))

        feed(engine, track, MOVING, frames=5)
        feed(engine, track, MOVING, frames=1, abrupt=True)
        incidents = feed(engine, track, STOPPED, frames=25)

        assert incidents[0].bbox == {"x1": 10.0, "y1": 20.0, "x2": 50.0, "y2": 60.0}


def acercar(engine, track_a, track_b, offsets):
    """
    Acerca de frente los dos vehículos y devuelve lo que reporte el motor.

    El último paso es corto a propósito: si se cruzan, la geometría deja de
    decir que se estaban acercando, que no es el caso que se quiere probar.
    """

    incidents = []

    for offset in offsets:
        track_a.update(0.9, 100 + offset, 100, 140 + offset, 140)
        track_b.update(0.9, 200 - offset, 100, 240 - offset, 140)

        incidents.extend(
            engine.process(
                [track_a, track_b],
                [
                    make_motion(1, speed=6.0, dx=20, abrupt_change=True,
                                acceleration=-6.0),
                    make_motion(2, speed=6.0, dx=-20, abrupt_change=True,
                                acceleration=-6.0),
                ],
            )
        )

    return incidents


# Frenada de frente: se tocan en el segundo paso y el tercero confirma.
IMPACTO = (20, 40, 45)


def seguir(engine, track_a, track_b, speed, frames):
    """Deja correr N frames más con los dos vehículos a una velocidad dada.

    Es lo que ocurre DESPUÉS del contacto, que es lo que decide si hubo
    choque o si simplemente pasaron cerca."""

    incidents = []

    for _ in range(frames):
        incidents.extend(
            engine.process(
                [track_a, track_b],
                [
                    make_motion(1, speed=speed),
                    make_motion(2, speed=speed),
                ],
            )
        )

    return incidents


def colisiones(incidents):
    """Solo los `possible_collision`.

    Tras un choque que inmoviliza, el motor emite además un
    `vehiculo_detenido` por cada implicado —lo cual es correcto: quedaron
    parados— pero aquí lo que se está probando es el veredicto del choque.
    """

    return [i for i in incidents if i.incident_type == "possible_collision"]


class TestDesenlace:
    """
    Dos cajas que se tocan en la imagen no prueban un choque: la cámara
    aplasta la escena contra un plano, así que un bus del carril lejano y un
    carro del cercano pueden solaparse estando a metros de distancia. Es el
    origen de casi todos los falsos positivos del motor.

    Lo que sí es inequívoco es la consecuencia: un choque de verdad deja
    algún vehículo inmovilizado. Estas pruebas fijan que el contacto por sí
    solo ya no confirma nada.
    """

    def test_el_contacto_solo_no_llega_a_confirmado(self):
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        incidents = acercar(engine, a, b, IMPACTO)

        assert incidents[0].data["aftermath"] == "pending"
        assert incidents[0].confidence <= IncidentEngine.UNCONFIRMED_CAP
        assert incidents[0].confidence < IncidentEngine.ALERT_CONFIDENCE
        assert incidents[0].data["severity"] == "pending"

    def test_si_un_vehiculo_queda_inmovil_el_incidente_se_confirma(self):
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        primero = acercar(engine, a, b, IMPACTO)[0]

        confirmados = colisiones(seguir(engine, a, b, STOPPED, frames=25))

        assert len(confirmados) == 1
        assert confirmados[0].incident_id == primero.incident_id
        assert confirmados[0].data["aftermath"] == "immobilized"
        assert confirmados[0].confidence > primero.confidence
        assert confirmados[0].data["severity"] == "confirmed"

    def test_si_ambos_siguen_circulando_el_incidente_se_degrada(self):
        # El falso positivo típico: dos vehículos de carriles distintos cuyas
        # cajas se solapan en la imagen y siguen su camino tan tranquilos.
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        primero = acercar(engine, a, b, IMPACTO)[0]

        degradados = colisiones(
            seguir(engine, a, b, MOVING, frames=IncidentEngine.AFTERMATH_WINDOW + 2)
        )

        assert len(degradados) == 1
        assert degradados[0].data["aftermath"] == "kept_moving"
        assert degradados[0].confidence < primero.confidence

    def test_un_vehiculo_ya_parqueado_no_confirma_nada(self):
        # Si bastara con "hay alguien quieto", una esquina con un carro
        # estacionado confirmaría incidentes toda la tarde. La parada tiene
        # que EMPEZAR con el incidente.
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        # b lleva rato quieto antes de que a se le acerque.
        for _ in range(60):
            engine.process(
                [a, b],
                [make_motion(1, speed=6.0), make_motion(2, speed=0.0)],
            )

        incidents = acercar(engine, a, b, IMPACTO)

        assert incidents[0].data["aftermath"] == "pending"
        assert incidents[0].confidence < IncidentEngine.ALERT_CONFIDENCE

    def test_se_conserva_la_puntuacion_cruda_para_recalibrar(self):
        # El veredicto humano de la bandeja de revisión solo sirve para
        # reajustar los pesos si se sabe qué puntuación geométrica dio el
        # motor antes de corregirla por el desenlace.
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        incidents = acercar(engine, a, b, IMPACTO)

        assert incidents[0].data["raw_confidence"] > incidents[0].confidence


class TestConflictoEnMetros:
    """
    Con la cámara calibrada el motor razona en metros y segundos.

    Y el dato que importa NO es la distancia. Medidos sobre el asfalto, los
    pares que este motor venía marcando estaban a metros o menos unos de
    otros — que es lo normal en tráfico urbano. Lo que separa un conflicto
    del tráfico corriente es a qué velocidad se cierran: el TTC.
    """

    def plano(self):
        # Cámara cenital de 10 px = 1 m: las respuestas se calculan a mano.
        return GroundPlane.from_quad(
            [(0, 100), (100, 100), (100, 0), (0, 0)],
            width_m=10.0,
            length_m=10.0,
        )

    def motor(self):
        return IncidentEngine(ground_plane=self.plano(), fps=10.0)

    def acercar_metros(self, engine, separaciones):
        """Dos vehículos que se aproximan, con la separación dada en píxeles
        (que a 10 px/m son décimas de metro)."""

        a = make_track(1, box=(0, 60, 20, 80))
        b = make_track(2, box=(0, 0, 20, 20))

        incidents = []

        for sep in separaciones:
            a.update(0.9, 0, 60, 20, 80)
            b.update(0.9, 0, 60 - 20 - sep, 20, 80 - sep)

            incidents.extend(
                engine.process(
                    [a, b],
                    [
                        make_motion(1, speed=6.0, dy=0, abrupt_change=True,
                                    acceleration=-6.0),
                        make_motion(2, speed=6.0, dy=sep, abrupt_change=True,
                                    acceleration=-6.0),
                    ],
                )
            )

        return incidents

    def test_el_motor_sabe_si_esta_calibrado(self):
        assert self.motor().metric is True
        assert IncidentEngine().metric is False

    def test_reporta_la_separacion_en_metros(self):
        engine = self.motor()
        incidents = self.acercar_metros(engine, [40, 20, 5, 0])

        assert incidents
        # 10 px = 1 m en este plano.
        assert incidents[0].data["separation_m"] < 3.0

    def test_calcula_la_velocidad_de_cierre_y_el_ttc(self):
        engine = self.motor()
        incidents = self.acercar_metros(engine, [40, 20, 5, 0])

        data = incidents[0].data

        assert data["closing_speed_ms"] is not None
        # Se estaban acercando, así que el cierre es positivo.
        assert data["closing_speed_ms"] > 0

    def test_sin_calibrar_no_hay_metricas_fisicas(self):
        # Una cámara sin calibrar tiene que seguir detectando igual que antes.
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        incidents = acercar(engine, a, b, IMPACTO)

        assert len(incidents) == 1
        assert "separation_m" not in incidents[0].data

    def test_dos_vehiculos_que_no_se_cierran_pierden_confianza(self):
        # El falso positivo dominante: circulan juntos, sin acercarse. Sus
        # cajas se tocan en la imagen, pero no está pasando nada.
        engine = self.motor()

        antes = engine._apply_conflict(
            0.8, {"separation_m": 2.5, "closing_speed_ms": 0.0, "ttc_s": None}
        )

        assert antes == pytest.approx(0.4)

    def test_un_ttc_bajo_sube_la_confianza(self):
        engine = self.motor()

        subida = engine._apply_conflict(
            0.6, {"separation_m": 1.5, "closing_speed_ms": 3.0, "ttc_s": 0.5}
        )

        assert subida > 0.6

    def test_estar_cerca_sin_cerrarse_no_basta(self):
        # Dos metros de separación es tráfico normal, no un conflicto.
        engine = self.motor()

        igual = engine._apply_conflict(
            0.7, {"separation_m": 0.5, "closing_speed_ms": 0.1, "ttc_s": None}
        )

        # Están pegados: no se penaliza, pero tampoco se premia.
        assert igual == pytest.approx(0.7)


class TestColision:

    def test_dos_carros_parqueados_pegados_no_son_un_choque(self):
        # Se tocan, pero ninguno se ha movido nunca.
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(141, 100, 181, 140))

        incidents = []
        for _ in range(10):
            incidents.extend(
                engine.process(
                    [a, b],
                    [make_motion(1, speed=0.0), make_motion(2, speed=0.0)],
                )
            )

        assert incidents == []

    def test_dos_vehiculos_que_se_acercan_y_chocan_se_reportan(self):
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        incidents = acercar(engine, a, b, IMPACTO)

        assert len(incidents) == 1
        assert incidents[0].incident_type == "possible_collision"
        assert sorted(incidents[0].track_ids) == [1, 2]
        assert incidents[0].confidence >= 0.60
        assert incidents[0].data["approaching"] is True
        # La caja del incidente cubre a los dos implicados.
        assert incidents[0].bbox["x1"] <= 145
        assert incidents[0].bbox["x2"] >= 185

    def test_un_solo_frame_de_contacto_no_basta(self):
        # Dos cajas que se rozan un frame suelen ser un error del detector,
        # no un choque: hacen falta dos frames seguidos cumpliendo todo.
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        assert acercar(engine, a, b, IMPACTO[:2]) == []

    def test_una_pareja_ya_reportada_no_vuelve_a_dispararse(self):
        engine = IncidentEngine()

        a = make_track(1, box=(100, 100, 140, 140))
        b = make_track(2, box=(200, 100, 240, 140))

        assert len(acercar(engine, a, b, IMPACTO)) == 1

        # El mismo par, ya registrado: no debe generar un incidente nuevo.
        repetido = [
            inc
            for inc in acercar(engine, a, b, IMPACTO)
            if inc.incident_type == "possible_collision"
        ]
        assert repetido == []


def test_reset_deja_el_motor_como_recien_creado():
    engine = IncidentEngine()
    track = make_track(1)

    feed(engine, track, MOVING, frames=5)
    feed(engine, track, MOVING, frames=1, abrupt=True)
    feed(engine, track, STOPPED, frames=25)

    engine.reset()

    feed(engine, track, MOVING, frames=5)
    feed(engine, track, MOVING, frames=1, abrupt=True)

    assert len(feed(engine, track, STOPPED, frames=25)) == 1


class TestOlvido:
    """
    Lo que se guarda entre oclusiones tiene que vencer.

    `_speed_peak`, `_stopped_reported` y `collision_pairs` se conservan a
    propósito cuando un track desaparece unos frames. Sin vencimiento eso es
    una fuga: una entrada por cada vehículo que pasa, en una cámara que con la
    fuente en bucle puede llevar semanas abierta.
    """

    def vacio(self, engine, frames):
        """Frames sin ningún track, como cuando la vía queda despejada."""

        for _ in range(frames):
            engine.process([], [])

    def test_lo_que_ya_no_puede_volver_se_olvida(self):
        engine = IncidentEngine()

        feed(engine, make_track(1), MOVING, frames=5)

        assert 1 in engine._speed_peak

        self.vacio(engine, engine.FORGET_AFTER_FRAMES + engine.FORGET_EVERY_FRAMES)

        assert engine._speed_peak == {}
        assert engine._last_seen == {}

    def test_una_oclusion_corta_no_borra_nada(self):
        # Un vehículo tapado por un bus vuelve con el mismo id: si se le
        # olvida que ya se movió, deja de poder reportarse como detenido.
        engine = IncidentEngine()

        feed(engine, make_track(1), MOVING, frames=5)

        self.vacio(engine, engine.FORGET_EVERY_FRAMES * 2)

        assert engine._ever_moved(1)

    def test_las_parejas_de_colision_tambien_vencen(self):
        engine = IncidentEngine()

        feed(engine, make_track(1), MOVING, frames=1)
        engine.collision_pairs.add((1, 2))

        self.vacio(engine, engine.FORGET_AFTER_FRAMES + engine.FORGET_EVERY_FRAMES)

        assert engine.collision_pairs == set()

    def test_una_pareja_sobrevive_mientras_sus_dos_vehiculos_sigan_ahi(self):
        engine = IncidentEngine()

        engine.collision_pairs.add((1, 2))

        for _ in range(engine.FORGET_AFTER_FRAMES + engine.FORGET_EVERY_FRAMES):
            engine.process(
                [make_track(1), make_track(2)],
                [make_motion(1, speed=MOVING), make_motion(2, speed=MOVING)],
            )

        assert (1, 2) in engine.collision_pairs

    def test_el_estado_no_crece_sin_techo(self):
        # Tráfico continuo con vehículos siempre distintos, que es lo que
        # hace una cámara real: cada uno pasa y no vuelve.
        engine = IncidentEngine()

        for track_id in range(1, 400):
            engine.process(
                [make_track(track_id)],
                [make_motion(track_id, speed=MOVING)],
            )

        techo = engine.FORGET_AFTER_FRAMES + engine.FORGET_EVERY_FRAMES

        assert len(engine._speed_peak) <= techo
        assert len(engine._last_seen) <= techo
