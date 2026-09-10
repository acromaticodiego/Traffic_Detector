"""
Política de reproducción: archivo contra cámara en vivo.

Es lo que decide si la fuente se rebobina, a qué ritmo se lee, qué `t` viaja
en cada mensaje y cómo se cierra la sesión. Se prueba aquí porque en el bucle
de visión no se puede: ese importa cv2 y el CI corre sin opencv.
"""

from services.vision_service.app.api.playback import Playback


def archivo(loop_enabled: bool = False, fps: float = 25.0) -> Playback:
    return Playback(is_stream=False, loop_enabled=loop_enabled, fps=fps)


def vivo(loop_enabled: bool = False, fps: float = 25.0) -> Playback:
    return Playback(is_stream=True, loop_enabled=loop_enabled, fps=fps)


class TestBucle:

    def test_un_archivo_se_rebobina_solo_si_se_pide(self):
        assert archivo(loop_enabled=True).loops is True
        assert archivo(loop_enabled=False).loops is False

    def test_una_camara_en_vivo_no_se_rebobina_aunque_se_pida(self):
        # No hay nada a lo que volver: rebobinar un stream no significa nada.
        assert vivo(loop_enabled=True).loops is False


class TestRitmo:

    def test_al_archivo_se_le_marca_el_ritmo(self):
        # Si no, se procesaría tan rápido como dé la GPU y el operario vería
        # la calle en cámara rápida.
        assert archivo().paced is True

    def test_a_la_camara_en_vivo_no(self):
        # Ya entrega un frame cuando lo tiene; frenarla solo acumula retraso.
        assert vivo().paced is False


class TestDuracion:

    def test_un_archivo_reporta_sus_frames(self):
        assert archivo().frame_count(1500) == 1500

    def test_una_camara_en_vivo_no_tiene_final_que_contar(self):
        assert vivo().frame_count(1500) is None

    def test_una_duracion_absurda_se_reporta_como_desconocida(self):
        # opencv contesta 0 o un negativo cuando no lo sabe. Mandar ese 0 tal
        # cual diría "video vacío", que es una afirmación distinta.
        assert archivo().frame_count(0) is None
        assert archivo().frame_count(-1) is None


class TestTiempo:

    def test_en_un_archivo_t_es_la_posicion_dentro_del_video(self):
        # El frontend lo usa para saltar a ese punto (`video.currentTime`).
        assert archivo(fps=25.0).timestamp(pass_frame=50, elapsed=999.0) == 2.0

    def test_al_dar_la_vuelta_t_vuelve_a_empezar(self):
        # Si siguiera creciendo, en la segunda vuelta apuntaría más allá del
        # final del video y no habría a dónde saltar.
        source = archivo(loop_enabled=True, fps=25.0)

        assert source.timestamp(pass_frame=1, elapsed=120.0) == 0.04

    def test_en_vivo_t_es_lo_que_lleva_abierta_la_sesion(self):
        # No hay video que rebobinar, así que la posición no significa nada.
        assert vivo().timestamp(pass_frame=50, elapsed=7.5) == 7.5

    def test_sin_fps_no_divide_por_cero(self):
        assert archivo(fps=0.0).timestamp(pass_frame=50, elapsed=1.0) == 0.0


class TestCierre:

    def test_un_archivo_que_se_acaba_termina_normal(self):
        message = archivo().end_message(frames=900, processed=900)

        assert message["type"] == "done"
        assert message["frames"] == 900

    def test_una_camara_que_deja_de_mandar_es_una_averia(self):
        # Anunciarlo como un final normal le mostraría "listo" al operario
        # justo cuando acaba de perder la cámara.
        message = vivo().end_message(frames=900, processed=900)

        assert message["type"] == "error"
        assert message["code"] == "source_lost"


class TestRitmoConVentaja:
    """
    El limitador tiene que correr ADELANTADO, no al día.

    Sin ventaja el pipeline produce el frame N justo en el segundo N/fps, así
    que nunca se adelanta y no hay colchón. El frontend, que sincroniza el
    video con la frontera de inferencia, alcanza esa frontera al menor
    tropiezo y pausa el video: el operario ve un tartamudeo constante.
    """

    def test_sin_ventaja_va_al_dia(self):
        # Frame 300 a 30 fps = segundo 10 del video. Si van 4 s de reloj,
        # hay que esperar 6.
        assert archivo(fps=30.0).pace_delay(300, elapsed=4.0) == 6.0

    def test_con_ventaja_se_adelanta(self):
        # Con 5 s de ventaja, ese mismo frame sale en el segundo 5.
        assert archivo(fps=30.0).pace_delay(300, elapsed=4.0, lead=5.0) == 1.0

    def test_mientras_construye_la_ventaja_no_frena(self):
        # Al arrancar corre suelto hasta ponerse por delante.
        assert archivo(fps=30.0).pace_delay(30, elapsed=0.5, lead=5.0) == 0.0

    def test_si_va_atrasado_no_espera(self):
        # La GPU no da abasto: se queda atrás, que es lo honesto, en vez de
        # fingir que va al día.
        assert archivo(fps=30.0).pace_delay(300, elapsed=60.0) == 0.0

    def test_a_una_camara_en_vivo_no_se_le_marca_el_ritmo(self):
        assert vivo(fps=30.0).pace_delay(300, elapsed=0.0, lead=5.0) == 0.0

    def test_sin_fps_no_divide_por_cero(self):
        assert archivo(fps=0.0).pace_delay(300, elapsed=0.0) == 0.0
