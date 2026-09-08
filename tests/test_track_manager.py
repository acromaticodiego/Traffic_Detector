"""Ciclo de vida de los tracks: alta, actualización y expiración."""

from services.vision_service.app.tracking.track_manager import TrackManager

from helpers import make_detection


def test_una_deteccion_nueva_crea_un_track():
    manager = TrackManager()

    active = manager.update([make_detection(track_id=1)])

    assert len(active) == 1
    assert manager.get_track(1) is not None
    assert active[0].track_id == 1


def test_las_detecciones_sin_track_id_se_ignoran():
    # El detector puede dar cajas sin identidad; sin id no hay historia que
    # mantener, y meterlas rompería el conteo de tracks activos.
    manager = TrackManager()

    assert manager.update([make_detection(track_id=None)]) == []
    assert manager.get_active_tracks() == []


def test_volver_a_ver_un_track_actualiza_su_caja_y_su_historial():
    manager = TrackManager()

    manager.update([make_detection(track_id=1, box=(0, 0, 10, 10))])
    manager.update([make_detection(track_id=1, box=(20, 20, 30, 30))])

    track = manager.get_track(1)

    assert track.x1 == 20
    assert len(track.positions) == 2
    assert track.previous_position == (5.0, 5.0)


def test_un_track_que_desaparece_sobrevive_unos_frames():
    # Una oclusión de medio segundo no debe borrar el objeto: al reaparecer
    # tiene que seguir siendo el mismo, con su historia intacta.
    manager = TrackManager(max_missing_frames=3)

    manager.update([make_detection(track_id=1)])

    for _ in range(3):
        manager.update([])
        assert manager.get_track(1) is not None

    manager.update([])
    assert manager.get_track(1) is None


def test_reaparecer_reinicia_la_cuenta_de_ausencias():
    manager = TrackManager(max_missing_frames=2)

    manager.update([make_detection(track_id=1)])
    manager.update([])
    manager.update([make_detection(track_id=1)])
    manager.update([])
    manager.update([])

    assert manager.get_track(1) is not None


def test_clear_deja_el_gestor_vacio():
    manager = TrackManager()
    manager.update([make_detection(track_id=1), make_detection(track_id=2)])

    manager.clear()

    assert manager.get_active_tracks() == []
    assert manager.missing_frames == {}
