"""Configuración del servicio: lectura del .env y redacción de credenciales."""

import os

from services.vision_service.app.api.config import _load_env_file, redact_url


class TestRedactUrl:

    def test_esconde_la_contrasena_de_la_cadena_de_conexion(self):
        # describe() se imprime en cada arranque y /health la devuelve: la
        # contraseña de Postgres no puede salir en claro en un log.
        redacted = redact_url(
            "postgresql+psycopg://postgres:s3cr3t@localhost:5432/traffic"
        )

        assert "s3cr3t" not in redacted
        assert redacted == (
            "postgresql+psycopg://postgres:***@localhost:5432/traffic"
        )

    def test_deja_intacto_lo_que_no_lleva_credenciales(self):
        url = "postgresql+psycopg://localhost:5432/traffic"

        assert redact_url(url) == url
        assert redact_url("sqlite:///local.db") == "sqlite:///local.db"
        assert redact_url("no-es-una-url") == "no-es-una-url"

    def test_un_usuario_sin_contrasena_se_conserva(self):
        assert redact_url("postgresql://postgres@localhost/traffic") == (
            "postgresql://postgres@localhost/traffic"
        )


class TestLoadEnvFile:

    def test_lee_pares_clave_valor_y_quita_las_comillas(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text(
            "\n".join(
                [
                    "# un comentario",
                    "",
                    'VISION_TEST_A="con comillas"',
                    "VISION_TEST_B = 42",
                    "linea sin igual",
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.delenv("VISION_TEST_A", raising=False)
        monkeypatch.delenv("VISION_TEST_B", raising=False)

        _load_env_file(env)

        assert os.environ["VISION_TEST_A"] == "con comillas"
        assert os.environ["VISION_TEST_B"] == "42"

    def test_el_entorno_real_le_gana_al_archivo(self, tmp_path, monkeypatch):
        # `set VISION_X=…` en la terminal tiene que poder sobrescribir el
        # .env, si no no hay forma de probar otra configuración a mano.
        env = tmp_path / ".env"
        env.write_text("VISION_TEST_C=del-archivo", encoding="utf-8")

        monkeypatch.setenv("VISION_TEST_C", "del-entorno")

        _load_env_file(env)

        assert os.environ["VISION_TEST_C"] == "del-entorno"

    def test_sin_archivo_no_pasa_nada(self, tmp_path):
        # Un despliegue que use solo variables de entorno no trae .env.
        _load_env_file(tmp_path / "no-existe.env")
