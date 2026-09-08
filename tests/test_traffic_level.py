"""
Nivel de tráfico.

El criterio del servicio es ocupación + velocidad, no conteo de vehículos:
6 carros en una avenida de 6 carriles es flujo libre y en una calle de un
carril es un trancón. Estos tests fijan esa regla.
"""

import pytest

from services.vision_service.app.api.road_roi import build_road_roi
from services.vision_service.app.api.traffic_level import TrafficLevelEstimator

from helpers import make_motion, make_track

# Umbrales explícitos: los del .env cambian por cámara y por máquina, y un
# test que dependiera de ellos fallaría según quién lo corra.
MEDIUM = 0.20
HIGH = 0.40
FREE_SPEED = 0.08
SLOW_RATIO = 0.35

WIDTH = 1000
HEIGHT = 1000
FPS = 25.0

# speed en px/frame -> anchos de frame por segundo: px * fps / width.
# Con estos números, 4.0 px/frame es flujo libre y 0.4 px/frame es detenido.
FLOWING = 4.0
STOPPED = 0.4


@pytest.fixture
def estimator():
    est = TrafficLevelEstimator(
        roi=build_road_roi(""),
        medium=MEDIUM,
        high=HIGH,
        free_speed=FREE_SPEED,
        slow_ratio=SLOW_RATIO,
        smoothing=1.0,   # sin EMA: cada test mide un solo frame
    )
    est.configure(WIDTH, HEIGHT, FPS)
    return est


def warmed_track(track_id, box, class_name="car"):
    """Track con historial suficiente para que su velocidad sea creíble."""

    track = make_track(track_id=track_id, class_name=class_name, box=box)
    # El estimador ignora los tracks con menos de 3 posiciones.
    track.update(0.9, *box)
    track.update(0.9, *box)
    return track


class TestClasificacion:

    def test_via_vacia_es_bajo(self, estimator):
        result = estimator.update([], [])

        assert result.level == "bajo"
        assert result.vehicles == 0
        assert result.occupancy == 0.0

    def test_poca_ocupacion_es_bajo(self, estimator):
        track = warmed_track(1, (400, 400, 500, 500))
        result = estimator.update([track], [make_motion(1, speed=FLOWING)])

        assert result.vehicles == 1
        assert result.occupancy < MEDIUM
        assert result.level == "bajo"

    def test_via_llena_pero_fluyendo_es_medio(self, estimator):
        track = warmed_track(1, (0, 0, 1000, 600))
        result = estimator.update([track], [make_motion(1, speed=FLOWING)])

        assert result.occupancy >= HIGH
        assert result.speed_ratio == 1.0
        assert result.level == "medio"

    def test_via_llena_y_detenida_es_alto(self, estimator):
        track = warmed_track(1, (0, 0, 1000, 600))
        result = estimator.update([track], [make_motion(1, speed=STOPPED)])

        assert result.occupancy >= HIGH
        assert result.speed_ratio < SLOW_RATIO
        assert result.level == "alto"

    def test_cola_detenida_es_alto_aunque_no_llene_la_via(self, estimator):
        # Ocupación intermedia, pero todo el mundo quieto: es una cola.
        tracks = [
            warmed_track(1, (0, 0, 500, 500)),
            warmed_track(2, (520, 0, 800, 400)),
        ]
        motion = [make_motion(1, speed=STOPPED), make_motion(2, speed=STOPPED)]

        result = estimator.update(tracks, motion)

        assert MEDIUM <= result.occupancy < HIGH
        assert result.stopped >= 0.6
        assert result.level == "alto"


class TestConteos:

    def test_los_peatones_no_cuentan_como_vehiculos(self, estimator):
        tracks = [
            warmed_track(1, (400, 400, 500, 500), class_name="car"),
            warmed_track(2, (600, 400, 640, 500), class_name="pedestrian"),
            warmed_track(3, (700, 400, 740, 500), class_name="ciclist"),
        ]
        motion = [make_motion(i, speed=FLOWING) for i in (1, 2, 3)]

        result = estimator.update(tracks, motion)

        assert result.vehicles == 1
        assert result.people == 2

    def test_lo_que_esta_fuera_de_la_calzada_no_suma(self):
        # ROI: solo la mitad inferior del frame.
        est = TrafficLevelEstimator(
            roi=build_road_roi("0.0,0.5 1.0,0.5 1.0,1.0 0.0,1.0"),
            medium=MEDIUM,
            high=HIGH,
            free_speed=FREE_SPEED,
            slow_ratio=SLOW_RATIO,
            smoothing=1.0,
        )
        est.configure(WIDTH, HEIGHT, FPS)

        en_via = warmed_track(1, (400, 700, 500, 800))
        en_andén = warmed_track(2, (400, 100, 500, 200))

        result = est.update(
            [en_via, en_andén],
            [make_motion(1, speed=FLOWING), make_motion(2, speed=FLOWING)],
        )

        assert result.vehicles == 1

    def test_sin_dimensiones_no_inventa_ocupacion(self):
        # configure() todavía no se ha llamado: el video no está abierto.
        est = TrafficLevelEstimator(roi=build_road_roi(""), smoothing=1.0)
        result = est.update([warmed_track(1, (0, 0, 1000, 900))], [])

        assert result.vehicles == 0
        assert result.occupancy == 0.0
        assert result.level == "bajo"


class TestVelocidad:

    def test_un_track_recien_aparecido_no_dispara_una_cola_falsa(self, estimator):
        # Sin historial la velocidad es 0, que se vería como "detenido".
        nuevo = make_track(track_id=1, box=(0, 0, 1000, 600))

        result = estimator.update([nuevo], [make_motion(1, speed=0.0)])

        assert result.occupancy >= HIGH
        assert result.stopped == 0.0
        assert result.level == "medio"

    def test_el_stride_no_cambia_el_nivel(self):
        # Con stride=2 el desplazamiento medido es el doble, porque hay el
        # doble de frames de origen entre dos frames procesados.
        def medir(stride, speed):
            est = TrafficLevelEstimator(
                roi=build_road_roi(""),
                medium=MEDIUM,
                high=HIGH,
                free_speed=FREE_SPEED,
                slow_ratio=SLOW_RATIO,
                smoothing=1.0,
                stride=stride,
            )
            est.configure(WIDTH, HEIGHT, FPS)
            track = warmed_track(1, (0, 0, 1000, 600))
            return est.update([track], [make_motion(1, speed=speed)])

        assert medir(1, STOPPED).level == medir(2, STOPPED * 2).level == "alto"
        assert medir(1, FLOWING).level == medir(2, FLOWING * 2).level == "medio"

    def test_la_ema_amortigua_un_frame_perdido(self):
        est = TrafficLevelEstimator(
            roi=build_road_roi(""),
            medium=MEDIUM,
            high=HIGH,
            free_speed=FREE_SPEED,
            slow_ratio=SLOW_RATIO,
            smoothing=0.2,
        )
        est.configure(WIDTH, HEIGHT, FPS)

        lleno = warmed_track(1, (0, 0, 1000, 900))
        for _ in range(10):
            est.update([lleno], [make_motion(1, speed=STOPPED)])

        antes = est.update([lleno], [make_motion(1, speed=STOPPED)])

        # El detector pierde la caja un frame: la ocupación no se desploma.
        despues = est.update([], [])

        assert despues.occupancy > antes.occupancy * 0.7
        assert despues.level == antes.level

    def test_score_es_la_ocupacion_en_porcentaje(self, estimator):
        track = warmed_track(1, (0, 0, 1000, 600))
        result = estimator.update([track], [make_motion(1, speed=FLOWING)])

        assert result.score == pytest.approx(result.occupancy * 100, abs=0.1)
