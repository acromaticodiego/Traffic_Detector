"""Análisis de movimiento: velocidad suavizada, dirección y frenazos."""

import pytest

from services.vision_service.app.motion.motion_analyzer import MotionAnalyzer

from helpers import make_track


def test_el_primer_frame_de_un_track_no_tiene_velocidad():
    # Con una sola posición no hay desplazamiento del que hablar.
    analysis = MotionAnalyzer().analyze(make_track(1))

    assert analysis.speed == 0.0
    assert analysis.acceleration == 0.0
    assert analysis.moving is False
    assert analysis.abrupt_change is False


def test_mide_el_desplazamiento_entre_las_dos_ultimas_posiciones():
    analyzer = MotionAnalyzer()
    track = make_track(1, box=(0, 0, 10, 10))

    analyzer.analyze(track)
    track.update(0.9, 30, 40, 40, 50)   # centro: (5,5) -> (35,45)

    analysis = analyzer.analyze(track)

    assert analysis.dx == pytest.approx(30.0)
    assert analysis.dy == pytest.approx(40.0)
    # El desplazamiento crudo es 50 (3-4-5), pero el primer analyze() ya
    # había fijado la velocidad previa en 0, así que la EMA la arranca desde
    # abajo: el track tarda unos frames en reportar su velocidad real.
    assert analysis.speed == pytest.approx(0.35 * 50.0)
    assert analysis.moving is True


def test_la_direccion_es_el_angulo_del_desplazamiento():
    analyzer = MotionAnalyzer()
    track = make_track(1, box=(0, 0, 10, 10))

    analyzer.analyze(track)
    track.update(0.9, 10, 0, 20, 10)   # se mueve solo en x

    assert analyzer.analyze(track).direction == pytest.approx(0.0)


def test_el_suavizado_amortigua_un_salto_del_tracker():
    analyzer = MotionAnalyzer()
    track = make_track(1, box=(0, 0, 10, 10))

    analyzer.analyze(track)
    track.update(0.9, 10, 0, 20, 10)    # 10 px
    analyzer.analyze(track)

    track.update(0.9, 110, 0, 120, 10)  # salto de 100 px
    analysis = analyzer.analyze(track)

    # La velocidad reportada queda muy por debajo del salto crudo.
    assert 10.0 < analysis.speed < 50.0


def test_un_frenazo_se_marca_como_cambio_abrupto():
    analyzer = MotionAnalyzer()
    track = make_track(1, box=(0, 0, 10, 10))

    analyzer.analyze(track)

    for _ in range(6):
        track.update(0.9, track.x1 + 40, 0, track.x2 + 40, 10)
        analyzer.analyze(track)

    # Se detiene en seco.
    track.update(0.9, track.x1, 0, track.x2, 10)
    analysis = analyzer.analyze(track)

    assert analysis.acceleration < 0
    assert analysis.abrupt_change is True


def test_olvidar_un_track_reinicia_su_historial():
    analyzer = MotionAnalyzer()
    track = make_track(7, box=(0, 0, 10, 10))

    analyzer.analyze(track)
    track.update(0.9, 40, 0, 50, 10)
    analyzer.analyze(track)

    assert 7 in analyzer.previous_speeds

    analyzer.remove_track(7)
    assert 7 not in analyzer.previous_speeds

    analyzer.reset()
    assert analyzer.previous_speeds == {}
