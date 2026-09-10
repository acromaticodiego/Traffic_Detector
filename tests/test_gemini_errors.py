"""
Qué se le dice al operario cuando el resumen con IA falla.

La distinción que importa no es técnica sino operativa: un fallo que se
arregla esperando cinco minutos y uno que necesita que alguien toque la
configuración no pueden verse igual desde la interfaz. Con un único mensaje
para los dos, el operario se queda reintentando contra un modelo retirado.

Esto no llegó como hipótesis: `gemini-2.5-flash` fue retirado por Google y
"Analizar el caso" empezó a fallar con un mensaje que no decía nada.
"""

from services.vision_service.app.ai.gemini import (
    _explain,
    _retryable,
    _truncated,
)

MODELO = "gemini-flash-latest"


class ErrorDeApi(Exception):
    """Como los errores del SDK de Google: traen el estado HTTP en `code`."""

    def __init__(self, code: int):
        super().__init__(f"{code} algo pasó")
        self.code = code


class ReadTimeout(Exception):
    """El nombre es lo que importa: así lo llama httpx."""


class TestMotivo:

    def test_un_modelo_retirado_dice_que_reintentar_no_sirve(self):
        mensaje = _explain(ErrorDeApi(404), MODELO)

        assert MODELO in mensaje
        assert "GEMINI_MODEL" in mensaje
        assert "reintentar" in mensaje.lower()

    def test_una_clave_invalida_se_distingue(self):
        assert "clave" in _explain(ErrorDeApi(401), MODELO).lower()
        assert "clave" in _explain(ErrorDeApi(403), MODELO).lower()

    def test_la_cuota_agotada_se_distingue(self):
        assert "cuota" in _explain(ErrorDeApi(429), MODELO).lower()

    def test_un_fallo_del_servidor_invita_a_reintentar(self):
        # 503 es lo que devuelve Google cuando el modelo está saturado, que
        # es temporal y no requiere tocar nada.
        for code in (500, 502, 503):
            mensaje = _explain(ErrorDeApi(code), MODELO).lower()

            assert "satur" in mensaje
            assert "intent" in mensaje

    def test_un_timeout_se_distingue_por_el_nombre(self):
        assert "tardó" in _explain(ReadTimeout(), MODELO)

    def test_lo_desconocido_cae_en_un_mensaje_generico(self):
        class RaroInesperado(Exception):
            pass

        assert "RaroInesperado" in _explain(RaroInesperado(), MODELO)


class TestNoFiltrarLaClave:

    def test_el_texto_crudo_del_sdk_nunca_sale(self):
        # El mensaje de error de un SDK puede traer la URL de la petición, y
        # en esa URL viaja la clave. Este texto va a la pantalla del operario.
        secreto = "AIzaSyFALSA_clave_de_prueba"
        error = ErrorDeApi(404)
        error.args = (f"404 https://api.google.com/v1?key={secreto}",)

        mensaje = _explain(error, MODELO)

        assert secreto not in mensaje
        assert "https://" not in mensaje


class TestRespuestaCortada:
    """
    Un texto cortado no es un resumen a medias que sirva igual.

    Se guarda en la base y se muestra como la lectura del caso, así que
    parece un análisis sin serlo. Eso ya pasó: con el límite de tokens que
    había, el modelo gastaba el presupuesto razonando y lo que se cacheaba
    era un fragmento como '). * Sudden braking/change: "'.
    """

    def respuesta(self, reason, texto="algo"):
        class Candidate:
            finish_reason = reason

        class Response:
            candidates = [Candidate()]
            text = texto

        return Response()

    def test_detecta_que_se_quedo_sin_espacio(self):
        assert _truncated(self.respuesta("FinishReason.MAX_TOKENS"))

    def test_una_respuesta_completa_no_lo_esta(self):
        assert not _truncated(self.respuesta("FinishReason.STOP"))

    def test_sin_candidatos_no_revienta(self):
        class Vacia:
            candidates = []
            text = ""

        assert not _truncated(Vacia())

    def test_sin_el_campo_tampoco(self):
        assert not _truncated(object())


class TestReintentos:
    """
    Cuándo insistir solo y cuándo no.

    Google devuelve 503 de forma intermitente y el siguiente intento suele
    pasar: que insista el servicio y no el operario a base de clics. Pero
    insistir tiene coste —el agente espera— así que solo donde sirve.
    """

    def test_reintenta_un_error_del_servidor(self):
        for code in (500, 502, 503, 504):
            assert _retryable(ErrorDeApi(code)), code

    def test_reintenta_un_fallo_de_conexion(self):
        class ConnectError(Exception):
            pass

        assert _retryable(ConnectError())

    def test_no_reintenta_un_timeout(self):
        # Ya esperó un minuto; volver a intentarlo solo multiplica la espera.
        class ReadTimeout(Exception):
            pass

        assert not _retryable(ReadTimeout())

    def test_no_reintenta_lo_permanente(self):
        # Un modelo retirado o una clave mala no se arreglan insistiendo, y
        # con la cuota agotada insistir la empeora.
        for code in (401, 403, 404, 429):
            assert not _retryable(ErrorDeApi(code)), code
