"""
Registro de cámaras.

Incluye un test sobre el `cameras.yaml` real del repo: una calibración mal
pegada (un id repetido, una ROI con un punto de menos) no se nota hasta que
el servicio arranca y sirve el nivel de tráfico equivocado.
"""

import pytest

from services.vision_service.app.api import cameras as registry
from services.vision_service.app.api.road_roi import parse_polygon


def parse(**entry):
    return registry._parse(entry)


class TestParseoDeUnaEntrada:

    def test_lee_la_calibracion_de_la_camara(self):
        camera = parse(
            id="cra64c_cl78",
            name="Cra 64C x Cl 78",
            source="videos/input/demo.mp4",
            roi="0.0,0.5 1.0,0.5 1.0,1.0",
            perspective=2.5,
            lat=6.2716,
            lng=-75.59,
            thresholds={
                "occupancy_medium": 0.24,
                "occupancy_high": 0.41,
                "free_speed": 0.065,
            },
            notes="Calibrada con el clip de demo.",
        )

        assert camera.id == "cra64c_cl78"
        assert camera.perspective == 2.5
        assert camera.occupancy_medium == 0.24
        assert camera.occupancy_high == 0.41
        assert camera.free_speed == 0.065
        assert camera.calibrated is True

    def test_las_rutas_relativas_se_leen_desde_la_raiz_del_repo(self):
        # El servicio se puede lanzar desde cualquier directorio; la ruta del
        # YAML no puede depender de eso.
        camera = parse(id="c1", source="videos/input/demo.mp4")

        assert camera.source.is_absolute()
        assert camera.source == registry.REPO_ROOT / "videos/input/demo.mp4"

    def test_una_camara_sin_id_o_sin_source_es_un_error_de_configuracion(self):
        with pytest.raises(ValueError):
            parse(name="Sin id", source="videos/input/demo.mp4")

        with pytest.raises(ValueError):
            parse(id="c1", name="Sin source")

    def test_sin_roi_la_camara_queda_marcada_como_no_calibrada(self):
        camera = parse(id="c1", source="videos/input/demo.mp4")

        assert camera.roi == ""
        assert camera.calibrated is False
        assert camera.public()["calibrated"] is False

    def test_lo_publico_no_expone_la_ruta_del_video(self):
        # `public()` va tal cual al frontend: la ruta en disco (o la URL RTSP
        # con credenciales) no tiene por qué salir de la máquina.
        camera = parse(
            id="c1",
            name="Cámara 1",
            source="videos/input/demo.mp4",
            thresholds={"occupancy_medium": 0.2, "occupancy_high": 0.4},
        )

        publico = camera.public()

        assert "source" not in publico
        assert publico["thresholds"] == {"medium": 0.2, "high": 0.4}
        assert set(publico) == {
            "id",
            "name",
            "lat",
            "lng",
            "calibrated",
            "available",
            "thresholds",
            "notes",
        }


class TestCamerasYamlDelRepo:

    def test_el_registro_del_repo_se_parsea_entero(self):
        cameras = registry.load_yaml_cameras()

        assert cameras, "cameras.yaml no debería quedar vacío"

        for camera_id, camera in cameras.items():
            assert camera_id == camera.id
            assert camera.name
            assert camera.source.name

    def test_las_roi_configuradas_son_poligonos_validos(self):
        for camera in registry.load_yaml_cameras().values():
            if not camera.roi:
                continue

            polygon = parse_polygon(camera.roi)

            assert polygon is not None, f"ROI inválida en '{camera.id}'"
            assert len(polygon) >= 3

    def test_los_umbrales_van_en_orden(self):
        # occupancy_high por debajo de occupancy_medium haría que "alto" no
        # se alcanzara nunca, y el nivel se quedaría pegado en "medio".
        for camera in registry.load_yaml_cameras().values():
            assert 0.0 < camera.occupancy_medium < camera.occupancy_high <= 1.0
            assert camera.free_speed > 0.0
            assert camera.perspective >= 1.0
