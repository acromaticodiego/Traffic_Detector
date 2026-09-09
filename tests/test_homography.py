"""
Calibración del plano de la vía.

Un error de signo o un eje cambiado aquí no se nota: el servicio sigue
funcionando y simplemente reporta distancias equivocadas, que es peor que
fallar. De ahí que esto se pruebe con casos donde la respuesta correcta se
puede calcular a mano.
"""

import numpy as np
import pytest

from services.vision_service.app.geometry.homography import (
    CalibrationError,
    GroundPlane,
    ground_point,
    homography_from_points,
    order_quad,
    rectangle_world_points,
)


def plano_simple(escala=10.0):
    """
    Una cámara cenital imaginaria: 10 px = 1 m, sin perspectiva.

    Con este plano las respuestas son evidentes, así que sirve para
    comprobar que la mecánica es correcta antes de meter perspectiva.

    Ojo con el eje Y: en una imagen crece HACIA ABAJO, así que la fila
    y = escala es la cercana a la cámara y es la que queda en el origen del
    mundo, mientras que y = 0 (arriba en la imagen) es la lejana.
    """

    imagen = [(0, 0), (escala, 0), (escala, escala), (0, escala)]

    return GroundPlane.from_quad(imagen, width_m=1.0, length_m=1.0)


class TestMecanica:

    def test_las_esquinas_caen_donde_se_dijo(self):
        plano = plano_simple()

        # Abajo en la imagen (y=10) es lo CERCANO, y va al origen del mundo.
        assert plano.to_world((0, 10)) == pytest.approx((0.0, 0.0), abs=1e-6)
        assert plano.to_world((10, 10)) == pytest.approx((1.0, 0.0), abs=1e-6)
        # Arriba en la imagen (y=0) es lo lejano: un metro más allá.
        assert plano.to_world((10, 0)) == pytest.approx((1.0, 1.0), abs=1e-6)
        assert plano.to_world((0, 0)) == pytest.approx((0.0, 1.0), abs=1e-6)

    def test_mide_distancias_en_metros(self):
        plano = plano_simple()

        # 10 px en horizontal = 1 m.
        assert plano.distance_m((0, 0), (10, 0)) == pytest.approx(1.0, abs=1e-6)
        # La diagonal del cuadrado de 1 m.
        assert plano.distance_m((0, 0), (10, 10)) == pytest.approx(
            np.sqrt(2), abs=1e-6
        )

    def test_sobrevive_a_guardarse_y_releerse(self):
        # Los nueve números van a cameras.yaml y a la base; si el viaje de
        # ida y vuelta pierde precisión, la calibración se degrada sola.
        plano = plano_simple()
        vuelta = GroundPlane.from_values(plano.values())

        assert vuelta.to_world((7, 3)) == pytest.approx(
            plano.to_world((7, 3)), abs=1e-9
        )


class TestOrdenDeLasEsquinas:
    """
    El orden de los clics era la única forma de arruinar una calibración sin
    que nada fallara: marcándolas rotadas, el ancho declarado se aplica a la
    profundidad y toda la escena queda deformada. Ahora el orden lo pone el
    programa.
    """

    ESPERADO = [(100, 400), (500, 400), (400, 200), (200, 200)]

    def test_ordena_esquinas_marcadas_al_reves(self):
        revés = list(reversed(self.ESPERADO))

        assert order_quad(revés) == self.ESPERADO

    def test_ordena_esquinas_marcadas_rotadas(self):
        # El error real: empezar por una esquina cercana y seguir hacia el
        # fondo en vez de cruzar primero.
        rotadas = [(100, 400), (200, 200), (400, 200), (500, 400)]

        assert order_quad(rotadas) == self.ESPERADO

    def test_el_orden_ya_correcto_no_se_toca(self):
        assert order_quad(self.ESPERADO) == self.ESPERADO

    def test_la_calibracion_sale_igual_sea_cual_sea_el_orden(self):
        # Lo que de verdad importa: el resultado no depende de por dónde
        # empezó a hacer clic la persona.
        derecho = GroundPlane.from_quad(self.ESPERADO, 7.0, 20.0)
        torcido = GroundPlane.from_quad(
            [(400, 200), (500, 400), (100, 400), (200, 200)], 7.0, 20.0
        )

        assert torcido.to_world((300, 300)) == pytest.approx(
            derecho.to_world((300, 300)), abs=1e-9
        )

    def test_sin_cuatro_esquinas_avisa(self):
        with pytest.raises(CalibrationError):
            order_quad([(0, 0), (1, 1)])


class TestPerspectiva:
    """
    Lo que de verdad importa: con perspectiva, la misma distancia en píxeles
    vale metros muy distintos según dónde esté en la imagen. Es exactamente
    el efecto que hace que dos vehículos de carriles distintos parezcan
    tocarse.
    """

    def plano_oblicuo(self):
        # Un trapecio: la parte de arriba (lejos) es más angosta en la
        # imagen que la de abajo (cerca), como en cualquier cámara de vía.
        imagen = [(100, 400), (500, 400), (400, 200), (200, 200)]

        return GroundPlane.from_quad(imagen, width_m=7.0, length_m=20.0)

    def test_arriba_los_pixeles_valen_mas_metros(self):
        plano = self.plano_oblicuo()

        cerca = plano.distance_m((300, 400), (350, 400))
        lejos = plano.distance_m((300, 200), (350, 200))

        # Los mismos 50 px cubren bastante más terreno al fondo.
        assert lejos > cerca * 1.5

    def test_dos_vehiculos_de_carriles_distintos_no_estan_pegados(self):
        # El falso positivo que motivó todo esto: en la imagen las cajas se
        # tocan, sobre el asfalto están a varios metros.
        plano = self.plano_oblicuo()

        cerca = (300, 395)   # vehículo del carril cercano
        lejos = (305, 205)   # vehículo del carril lejano, casi encima en la imagen

        # Apenas 5 px de separación horizontal en la imagen...
        assert abs(cerca[0] - lejos[0]) < 10

        # ...pero metros de distancia real sobre la vía.
        assert plano.distance_m(cerca, lejos) > 5.0


class TestErrores:

    def test_tres_puntos_en_linea_se_rechazan(self):
        # No definen un cuadrilátero: no hay plano que ajustar. Mejor fallar
        # ruidosamente que devolver una matriz basura.
        with pytest.raises(CalibrationError):
            homography_from_points(
                [(0, 0), (10, 0), (20, 0), (5, 10)],
                rectangle_world_points(1.0, 1.0),
            )

    def test_faltan_puntos(self):
        with pytest.raises(CalibrationError):
            homography_from_points([(0, 0)], [(0, 0)])

    def test_medidas_no_positivas(self):
        with pytest.raises(CalibrationError):
            rectangle_world_points(0.0, 10.0)

    def test_una_matriz_incompleta_se_rechaza(self):
        with pytest.raises(CalibrationError):
            GroundPlane.from_values([1, 2, 3])

    def test_un_punto_en_el_horizonte_no_devuelve_infinito(self):
        # Sobre la línea del horizonte el plano se proyecta al infinito.
        # Devolver un número enorme sería peor que avisar.
        plano = GroundPlane.from_values([1, 0, 0, 0, 1, 0, 1, 0, 0])

        with pytest.raises(CalibrationError):
            plano.to_world((0, 5))


class TestPuntoDeContacto:

    def test_usa_el_borde_inferior_y_no_el_centro(self):
        # El centro de la caja flota a media altura del vehículo. Proyectarlo
        # sobre el asfalto lo manda metros más lejos, y tanto más cuanto más
        # alto sea: un bus se desplazaría mucho más que una moto.
        bbox = {"x1": 100.0, "y1": 200.0, "x2": 200.0, "y2": 260.0}

        assert ground_point(bbox) == (150.0, 260.0)
