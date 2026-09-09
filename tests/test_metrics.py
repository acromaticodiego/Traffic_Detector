"""
Cuándo se declara que una cámara va mal.

Es la parte con criterio de la observabilidad, y el criterio hay que poder
discutirlo: un umbral demasiado sensible produce avisos que todo el mundo
aprende a ignorar, y una alerta ignorada es peor que ninguna porque da la
impresión de que hay vigilancia.
"""

from services.vision_service.app.api.metrics import (
    BEHIND,
    OK,
    STALLED,
    camera_status,
    overall_status,
    rate,
)

FPS = 30.0
CALIENTE = 60.0  # segundos abierta: ya no está arrancando


def estado(**kwargs):
    base = {
        "measured_fps": FPS,
        "source_fps": FPS,
        "since_last_frame_s": 0.1,
        "warmup_s": CALIENTE,
    }
    base.update(kwargs)

    return camera_status(**base)


class TestEstadoDeUnaCamara:

    def test_al_dia_es_ok(self):
        assert estado()["status"] == OK

    def test_una_camara_sin_frames_esta_detenida(self):
        resultado = estado(since_last_frame_s=30.0)

        assert resultado["status"] == STALLED
        assert "30" in resultado["detail"]

    def test_procesar_mas_lento_que_la_calle_es_ir_retrasada(self):
        # El caso medido de verdad: dos cámaras sobre una RTX 3050 corren a
        # 0.69x del tiempo real.
        resultado = estado(measured_fps=FPS * 0.69)

        assert resultado["status"] == BEHIND
        assert "69%" in resultado["detail"]

    def test_el_aviso_dice_que_hacer(self):
        # Un aviso que no propone nada obliga a ir a buscar a quien sepa.
        detalle = estado(measured_fps=FPS * 0.5)["detail"]

        assert "VISION_MAX_SESSIONS" in detalle
        assert "VISION_FRAME_STRIDE" in detalle

    def test_una_diferencia_pequena_no_alarma(self):
        # El ritmo real oscila. Marcar en rojo por un 5% entrena a ignorar.
        assert estado(measured_fps=FPS * 0.95)["status"] == OK

    def test_detenida_pesa_mas_que_retrasada(self):
        # Sin frames Y lenta: lo que hay que decir es que no llegan frames.
        resultado = estado(measured_fps=1.0, since_last_frame_s=30.0)

        assert resultado["status"] == STALLED


class TestArranque:

    def test_recien_abierta_no_se_declara_retrasada(self):
        # En el primer segundo no hay ventana con la que medir nada, y el
        # modelo todavía está calentando: sería un aviso garantizado y falso.
        assert estado(measured_fps=1.0, warmup_s=0.5)["status"] == OK

    def test_sin_ritmo_medido_tampoco(self):
        assert estado(measured_fps=None)["status"] == OK

    def test_pero_una_camara_recien_abierta_que_no_da_frames_si_alarma(self):
        # Arrancar no justifica cinco segundos sin un solo frame.
        assert estado(warmup_s=0.5, since_last_frame_s=30.0)["status"] == STALLED

    def test_una_fuente_sin_fps_conocidos_no_alarma(self):
        assert estado(source_fps=0.0)["status"] == OK


class TestEstadoGlobal:

    def test_sin_camaras_abiertas_no_hay_nada_malo(self):
        assert overall_status([]) == OK

    def test_es_el_peor_de_todas(self):
        assert overall_status([OK, OK]) == OK
        assert overall_status([OK, BEHIND]) == BEHIND
        assert overall_status([OK, BEHIND, STALLED]) == STALLED


class TestRitmo:

    def test_calcula_los_frames_por_segundo(self):
        # Once marcas separadas 0,1 s = diez intervalos en un segundo.
        marks = [i * 0.1 for i in range(11)]

        assert rate(marks, now=1.0) == 10.0

    def test_una_sola_marca_no_es_un_ritmo(self):
        assert rate([1.0], now=2.0) is None
        assert rate([], now=2.0) is None

    def test_una_camara_atascada_ve_caer_su_ritmo(self):
        # Se mide contra AHORA, no contra la última marca: si no, una cámara
        # parada conservaría para siempre el ritmo que tenía al pararse.
        marks = [i * 0.1 for i in range(11)]

        assert rate(marks, now=1.0) == 10.0
        assert rate(marks, now=10.0) < 1.5
