"""
Geometría de la anonimización de la evidencia (Ley 1581).

Lo que se prueba aquí no es que quede bonito, sino que no quede nada sin
tapar: una placa legible guardada en disco no la detecta nadie hasta que la
encuentra un abogado.
"""

from services.vision_service.app.incidents.anonymize import (
    CON_PERSONA,
    CON_PLACA,
    Region,
    boxes_from_detections,
    boxes_from_tracks,
    regions_for,
)

ANCHO, ALTO = 1920, 1080


def caja(class_name: str, x1=100.0, y1=100.0, x2=200.0, y2=200.0):
    return (class_name, x1, y1, x2, y2)


def cubre(region: Region, x: int, y: int) -> bool:
    return region.x1 <= x < region.x2 and region.y1 <= y < region.y2


def alguna_cubre(regions: list[Region], x: int, y: int) -> bool:
    return any(cubre(r, x, y) for r in regions)


class TestVehiculos:

    def test_tapa_la_franja_baja_de_un_carro(self):
        # La placa está abajo, sea de frente o de espaldas a la cámara.
        regions = regions_for([caja("car", 100, 100, 200, 200)], ANCHO, ALTO)

        assert len(regions) == 1
        assert regions[0].y2 == 200
        assert alguna_cubre(regions, 150, 199)

    def test_no_tapa_el_carro_entero(self):
        # El operario tiene que poder juzgar el incidente: necesita ver el
        # vehículo, solo no su placa.
        regions = regions_for([caja("car", 100, 100, 200, 200)], ANCHO, ALTO)

        assert not alguna_cubre(regions, 150, 110)

    def test_todas_las_clases_con_placa_se_tapan(self):
        for class_name in CON_PLACA:
            regions = regions_for([caja(class_name)], ANCHO, ALTO)

            assert regions, f"'{class_name}' quedó sin tapar"
            assert alguna_cubre(regions, 150, 199), class_name


class TestPersonas:

    def test_tapa_la_franja_alta_de_un_peaton(self):
        regions = regions_for(
            [caja("pedestrian", 100, 100, 140, 300)], ANCHO, ALTO
        )

        assert len(regions) == 1
        assert regions[0].y1 == 100
        assert alguna_cubre(regions, 120, 101)

    def test_todas_las_clases_con_persona_se_tapan(self):
        for class_name in CON_PERSONA:
            regions = regions_for([caja(class_name)], ANCHO, ALTO)

            assert regions, f"'{class_name}' quedó sin tapar"
            assert alguna_cubre(regions, 150, 101), class_name

    def test_la_moto_se_tapa_por_arriba_y_por_abajo(self):
        # Tiene placa abajo y conductor arriba: las dos cosas a la vez.
        regions = regions_for(
            [caja("motorcycle", 100, 100, 200, 200)], ANCHO, ALTO
        )

        assert len(regions) == 2
        assert alguna_cubre(regions, 150, 101)
        assert alguna_cubre(regions, 150, 199)


class TestCobertura:

    def test_se_tapan_todos_los_vehiculos_del_frame_no_solo_los_del_incidente(
        self,
    ):
        # La evidencia es el frame entero: ahí salen las placas de los que
        # pasaban al lado, que no tienen nada que ver con el incidente.
        boxes = [
            caja("car", 100, 100, 200, 200),
            caja("truck", 400, 100, 600, 300),
            caja("bus", 800, 50, 1100, 400),
        ]

        regions = regions_for(boxes, ANCHO, ALTO)

        assert alguna_cubre(regions, 150, 199)
        assert alguna_cubre(regions, 500, 299)
        assert alguna_cubre(regions, 950, 399)

    def test_una_clase_desconocida_se_tapa_por_las_dos_puntas(self):
        # Si mañana se reentrena el modelo con una clase nueva, el fallo tiene
        # que ser "tapé de más", nunca "dejé una placa legible".
        regions = regions_for(
            [caja("camion_de_bomberos", 100, 100, 200, 200)], ANCHO, ALTO
        )

        assert len(regions) == 2
        assert alguna_cubre(regions, 150, 101)
        assert alguna_cubre(regions, 150, 199)

    def test_una_clase_vacia_tambien_se_tapa(self):
        assert regions_for([caja("")], ANCHO, ALTO)

    def test_el_nombre_de_la_clase_no_distingue_mayusculas(self):
        assert len(regions_for([caja("CAR")], ANCHO, ALTO)) == 1


class TestBordes:

    def test_una_caja_que_se_sale_del_frame_se_recorta(self):
        regions = regions_for(
            [caja("car", -50, -50, 100, 100)], ANCHO, ALTO
        )

        assert all(r.x1 >= 0 and r.y1 >= 0 for r in regions)
        assert all(r.x2 <= ANCHO and r.y2 <= ALTO for r in regions)

    def test_una_caja_pegada_al_borde_inferior_llega_hasta_el_final(self):
        regions = regions_for(
            [caja("car", 100, 1000, 200, 1200)], ANCHO, ALTO
        )

        assert regions[0].y2 == ALTO

    def test_una_caja_entera_fuera_del_frame_se_descarta(self):
        assert regions_for([caja("car", 3000, 3000, 3100, 3100)], ANCHO, ALTO) == []

    def test_una_caja_degenerada_no_produce_region(self):
        assert regions_for([caja("car", 100, 100, 100, 100)], ANCHO, ALTO) == []

    def test_una_caja_con_las_esquinas_al_reves_se_endereza(self):
        # Nunca debería llegar así, pero una caja invertida que se descarte en
        # silencio es una placa sin tapar.
        regions = regions_for([caja("car", 200, 200, 100, 100)], ANCHO, ALTO)

        assert regions
        assert regions[0].y2 == 200

    def test_una_caja_diminuta_se_tapa_igual(self):
        # Con una franja de 0 píxeles la caja quedaría intacta, y las cajas
        # pequeñas son justo las que nadie va a revisar a ojo.
        regions = regions_for([caja("car", 100, 100, 102, 102)], ANCHO, ALTO)

        assert regions and regions[0].y2 > regions[0].y1


class TestAdaptadores:

    def test_lee_las_cajas_de_los_tracks(self):
        class FakeTrack:
            class_name = "car"
            x1, y1, x2, y2 = 10.0, 20.0, 30.0, 40.0

        assert boxes_from_tracks([FakeTrack()]) == [
            ("car", 10.0, 20.0, 30.0, 40.0)
        ]

    def test_lee_las_cajas_de_las_detecciones(self):
        class FakeBox:
            x1, y1, x2, y2 = 10.0, 20.0, 30.0, 40.0

        class FakeDetection:
            class_name = "bus"
            bbox = FakeBox()

        assert boxes_from_detections([FakeDetection()]) == [
            ("bus", 10.0, 20.0, 30.0, 40.0)
        ]
