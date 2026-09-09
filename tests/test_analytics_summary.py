"""
Los cálculos del dashboard.

Se prueban las dos piezas puras: el corte de los días por huso horario local y
la tasa de validez. Las consultas se dejan fuera porque necesitan Postgres.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from services.vision_service.app.analytics.summary import (
    dia_local,
    inicio_del_dia,
    serie_por_dia,
    validez,
)

BOGOTA = ZoneInfo("America/Bogota")


class TestCorteDelDia:

    def test_el_dia_se_corta_a_medianoche_local_y_no_en_utc(self):
        """A las 22:00 de Bogotá ya es el día siguiente en UTC. Si el corte
        fuera UTC, el turno de la noche aparecería repartido en dos días."""

        momento = datetime(2026, 9, 8, 3, 0, tzinfo=timezone.utc)  # 22:00 del 7

        assert dia_local(momento, BOGOTA).isoformat() == "2026-09-07"

    def test_el_inicio_del_dia_es_medianoche_local_expresada_en_utc(self):
        momento = datetime(2026, 9, 8, 18, 0, tzinfo=timezone.utc)  # 13:00 local
        inicio = inicio_del_dia(momento, BOGOTA)

        assert inicio.astimezone(BOGOTA).hour == 0
        assert inicio.astimezone(BOGOTA).date().isoformat() == "2026-09-08"


class TestSeriePorDia:

    def test_incluye_los_dias_sin_actividad(self):
        """Una gráfica que solo trae los días con trabajo pinta un fin de
        semana como si no existiera, y dos lunes seguidos parecen consecutivos."""

        ahora = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
        serie = serie_por_dia([], ahora, BOGOTA, dias=5)

        assert len(serie) == 5
        assert [d["active_seconds"] for d in serie] == [0, 0, 0, 0, 0]

    def test_va_del_mas_antiguo_al_mas_reciente(self):
        ahora = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
        serie = serie_por_dia([], ahora, BOGOTA, dias=3)

        assert [d["date"] for d in serie] == ["2026-09-06", "2026-09-07", "2026-09-08"]

    def test_suma_turnos_y_revisiones_en_su_dia(self):
        ahora = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
        eventos = [
            (ahora, 3600, 0),
            (ahora, 1800, 0),
            (ahora, 0, 1),
            (ahora - timedelta(days=1), 600, 0),
        ]

        serie = serie_por_dia(eventos, ahora, BOGOTA, dias=3)
        por_dia = {d["date"]: d for d in serie}

        assert por_dia["2026-09-08"]["active_seconds"] == 5400
        assert por_dia["2026-09-08"]["reviewed"] == 1
        assert por_dia["2026-09-07"]["active_seconds"] == 600

    def test_lo_anterior_al_periodo_se_descarta(self):
        ahora = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
        viejo = [(ahora - timedelta(days=40), 9999, 0)]

        assert sum(d["active_seconds"] for d in serie_por_dia(viejo, ahora, BOGOTA)) == 0


class TestValidez:

    def test_sin_veredictos_todavia_no_hay_tasa(self):
        """Un cero se leería como "se equivoca siempre" en vez de "aún no ha
        revisado nada"."""

        assert validez(0, 0) is None

    def test_es_la_fraccion_de_confirmados_sobre_lo_decidido(self):
        assert validez(3, 1) == 0.75

    def test_los_archivados_no_entran_porque_no_son_un_veredicto(self):
        # archivar no dice si el incidente era real; por eso la función solo
        # recibe confirmados y descartados.
        assert validez(1, 1) == 0.5
