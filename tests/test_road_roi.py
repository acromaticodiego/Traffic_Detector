"""Geometría de la calzada: parseo del polígono, ocupación y perspectiva."""

from services.vision_service.app.api.road_roi import (
    build_road_roi,
    parse_polygon,
)


class TestParsePolygon:

    def test_lee_puntos_separados_por_espacios_o_punto_y_coma(self):
        assert parse_polygon("0.0,0.0 1.0,0.0 1.0,1.0") == [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ]
        assert parse_polygon("0.0,0.0;1.0,0.0;1.0,1.0") is not None

    def test_recorta_las_coordenadas_al_rango_normalizado(self):
        assert parse_polygon("-0.5,2.0 1.0,0.0 1.0,1.0")[0] == (0.0, 1.0)

    def test_descarta_lo_que_no_encierra_area(self):
        # Menos de 3 vértices no es un polígono.
        assert parse_polygon("0.1,0.1 0.9,0.9") is None
        assert parse_polygon("") is None
        assert parse_polygon("   ") is None

    def test_ignora_los_trozos_malformados_en_vez_de_reventar(self):
        # Una ROI copiada a medias no debe tumbar el arranque del servicio.
        assert parse_polygon("0.0,0.0 basura 1.0,0.0 x,y 1.0,1.0") == [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ]


class TestRoadROI:

    def test_sin_poligono_la_roi_es_el_frame_completo(self):
        roi = build_road_roi("")

        assert roi.calibrated is False
        assert roi.contains(0.5, 0.5)
        assert roi.contains(0.01, 0.99)

    def test_un_poligono_degenerado_cae_al_frame_completo(self):
        # Una máscara vacía dejaría la ocupación en 0 para siempre, que se ve
        # igual que "tráfico bajo" y no se nota que está roto.
        roi = build_road_roi("0.5,0.5 0.5,0.5 0.5,0.5")

        assert roi.calibrated is False
        assert roi.contains(0.2, 0.8)

    def test_delimita_la_mitad_inferior_del_frame(self):
        roi = build_road_roi("0.0,0.5 1.0,0.5 1.0,1.0 0.0,1.0")

        assert roi.calibrated is True
        assert roi.contains(0.5, 0.8) is True
        assert roi.contains(0.5, 0.2) is False

    def test_la_ocupacion_es_la_fraccion_de_calzada_cubierta(self):
        roi = build_road_roi("")

        assert roi.occupancy([]) == 0.0

        media = roi.occupancy([(0.0, 0.0, 1.0, 0.5)])
        assert 0.45 < media < 0.55

        assert roi.occupancy([(0.0, 0.0, 1.0, 1.0)]) == 1.0

    def test_dos_vehiculos_superpuestos_no_ocupan_doble_asfalto(self):
        roi = build_road_roi("")

        solo = roi.occupancy([(0.0, 0.0, 0.5, 1.0)])
        con_solape = roi.occupancy(
            [(0.0, 0.0, 0.5, 1.0), (0.1, 0.0, 0.4, 1.0)]
        )

        assert con_solape == solo

    def test_la_perspectiva_iguala_el_peso_de_cerca_y_de_lejos(self):
        # Con perspective=3 un carro del fondo se ve 3 veces más pequeño; su
        # aporte a la ocupación debe ser comparable al de uno del frente.
        roi = build_road_roi("0.0,0.0 1.0,0.0 1.0,1.0 0.0,1.0", perspective=3.0)

        cerca = roi.occupancy([(0.4, 0.85, 0.6, 0.95)])
        lejos = roi.occupancy([(0.4, 0.05, 0.6, 0.15)])

        # Misma caja en píxeles: la de arriba pesa más porque representa un
        # vehículo real más grande.
        assert lejos > cerca

    def test_scale_at_va_de_1_en_el_borde_superior_a_perspective_abajo(self):
        roi = build_road_roi("0.0,0.0 1.0,0.0 1.0,1.0 0.0,1.0", perspective=2.5)

        assert roi.scale_at(0.0) == 1.0
        assert roi.scale_at(1.0) == 2.5
        assert roi.scale_at(0.5) == 1.75
        # Fuera del rango se satura, no extrapola.
        assert roi.scale_at(-1.0) == 1.0
        assert roi.scale_at(5.0) == 2.5

    def test_sin_correccion_de_perspectiva_la_escala_es_plana(self):
        roi = build_road_roi("", perspective=1.0)

        assert roi.scale_at(0.1) == 1.0
        assert roi.scale_at(0.9) == 1.0
