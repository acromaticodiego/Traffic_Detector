"""
El reparto del tiempo de un turno.

Es la regla que decide cuánto se le cuenta a alguien como trabajado, así que
conviene que sea explícita y esté probada: el usuario pidió expresamente que un
corte de internet no lo perjudicara.

`acreditar` es pura, así que se comprueba entera sin base de datos ni reloj.
"""

from services.vision_service.app.shifts.tracker import acreditar

TOLERANCIA = 10 * 60  # 10 minutos
LIMITE = 60 * 60  # 1 hora


def reparto(segundos):
    return acreditar(segundos, TOLERANCIA, LIMITE)


class TestLatidoNormal:

    def test_el_intervalo_entre_latidos_cuenta_entero(self):
        c = reparto(60)

        assert c.activos == 60
        assert c.corte == 0
        assert c.cerrado is False

    def test_un_corte_corto_no_le_cuesta_tiempo_a_nadie(self):
        """Cuatro minutos sin señal es un cambio de red o un wifi que se cayó,
        no una ausencia."""

        c = reparto(4 * 60)

        assert c.activos == 4 * 60
        assert c.hubo_corte is False

    def test_justo_en_la_tolerancia_todavia_cuenta_entero(self):
        c = reparto(TOLERANCIA)

        assert c.activos == TOLERANCIA
        assert c.hubo_corte is False


class TestCortesLargos:

    def test_pasada_la_tolerancia_se_acredita_hasta_ella_y_se_anota_el_resto(self):
        """Media hora sin internet: se cuentan diez minutos y los otros veinte
        quedan registrados como corte, a la vista, en vez de descontarse en
        silencio."""

        c = reparto(30 * 60)

        assert c.activos == TOLERANCIA
        assert c.corte == 20 * 60
        assert c.hubo_corte is True
        assert c.cerrado is False

    def test_el_turno_sigue_abierto_tras_un_corte_largo(self):
        """Lo que el usuario pidió: que perder la conexión no parta el turno
        ni le quite las horas trabajadas."""

        assert reparto(LIMITE).cerrado is False


class TestFinDeTurno:

    def test_pasado_el_limite_el_turno_se_da_por_terminado(self):
        c = reparto(LIMITE + 1)

        assert c.cerrado is True

    def test_al_cerrarse_se_acredita_la_tolerancia_y_no_el_hueco_entero(self):
        """Una pestaña olvidada abierta el viernes no puede acumular horas
        todo el fin de semana."""

        c = reparto(3 * 24 * 3600)

        assert c.activos == TOLERANCIA
        assert c.corte == 0


class TestRelojesRaros:

    def test_un_reloj_que_va_hacia_atras_no_resta_tiempo(self):
        """Un ajuste de hora o dos latidos que llegan desordenados no pueden
        dejar el contador en negativo."""

        c = reparto(-500)

        assert c.activos == 0
        assert c.corte == 0
        assert c.cerrado is False
