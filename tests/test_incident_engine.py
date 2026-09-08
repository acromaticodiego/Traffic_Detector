"""
Motor de incidentes.

Lo que se prueba aquí es el criterio, no la geometría: cuándo un vehículo
quieto es un incidente y cuándo es un semáforo en rojo, y cuándo dos cajas
juntas son un choque y cuándo son dos carros parqueados.
"""

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
