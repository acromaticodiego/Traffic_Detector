"""
Autenticación y permisos.

Es el código donde un error no se nota: el sistema sigue funcionando y
simplemente deja entrar a quien no debía. No hay forma de darse cuenta
mirándolo, así que se prueba a conciencia.
"""

import time

import pytest

from services.vision_service.app.api.config import settings
from services.vision_service.app.auth import tokens
from services.vision_service.app.auth.passwords import (
    MAX_BYTES,
    PasswordError,
    hash_password,
    validate,
    verify,
)


@pytest.fixture(autouse=True)
def secreto(monkeypatch):
    """Un secreto fijo para las pruebas, sin depender del .env."""
    monkeypatch.setattr(
        settings, "jwt_secret", "secreto-de-prueba-largo-para-hmac-sha256-ok"
    )
    monkeypatch.setattr(settings, "jwt_expire_minutes", 60)


class TestContrasenas:

    def test_el_hash_no_contiene_la_contrasena(self):
        # Obvio, pero es LA propiedad: si esta tabla se filtra, lo que se
        # filtra son hashes y no las contraseñas de la gente.
        hashed = hash_password("caballo-correcto-grapa")

        assert "caballo" not in hashed
        assert hashed.startswith("$2b$")

    def test_verifica_la_correcta_y_rechaza_la_incorrecta(self):
        hashed = hash_password("caballo-correcto-grapa")

        assert verify("caballo-correcto-grapa", hashed) is True
        assert verify("caballo-incorrecto-grapa", hashed) is False

    def test_dos_personas_con_la_misma_contrasena_tienen_hashes_distintos(self):
        # bcrypt guarda su propia sal, así que una tabla precalculada no
        # sirve y romper una cuenta no rompe las demás.
        assert hash_password("la-misma-clave") != hash_password("la-misma-clave")

    def test_un_hash_corrupto_no_revienta(self):
        # Devolver False y no lanzar: un 500 aquí delataría que ese usuario
        # existe, además de tumbar el login.
        assert verify("lo-que-sea", "esto-no-es-un-hash") is False
        assert verify("lo-que-sea", "") is False

    def test_rechaza_contrasenas_muy_cortas(self):
        with pytest.raises(PasswordError):
            validate("corta")

    def test_rechaza_contrasenas_que_bcrypt_truncaria(self):
        # bcrypt ignora en silencio lo que pase de 72 bytes. Aceptarla sería
        # darle a alguien una falsa sensación de seguridad.
        with pytest.raises(PasswordError):
            validate("a" * (MAX_BYTES + 1))

    def test_los_acentos_cuentan_como_bytes_no_como_letras(self):
        # 'ñ' ocupa dos bytes en UTF-8: una contraseña de 40 caracteres en
        # español puede pasarse del límite sin que se note contando letras.
        assert len("ñ" * 40) == 40
        with pytest.raises(PasswordError):
            validate("ñ" * 40)


class TestTokens:

    def datos(self):
        return {
            "user_id": 7,
            "email": "ana@ejemplo.com",
            "role": "analista",
            "permissions": {"incidents:read", "incidents:review"},
        }

    def test_ida_y_vuelta(self):
        token, _ = tokens.create_access_token(**self.datos())
        payload = tokens.decode_access_token(token)

        assert tokens.user_id_from(payload) == 7
        assert payload["role"] == "analista"
        assert "incidents:review" in payload["permissions"]

    def test_un_token_firmado_con_otro_secreto_se_rechaza(self, monkeypatch):
        token, _ = tokens.create_access_token(**self.datos())

        monkeypatch.setattr(
            settings, "jwt_secret", "otro-secreto-distinto-igual-de-largo-que-el-otro"
        )

        with pytest.raises(tokens.TokenError):
            tokens.decode_access_token(token)

    def test_un_token_manipulado_se_rechaza(self):
        # Cambiar un carácter del payload invalida la firma. Es lo que
        # impide que alguien se ascienda a admin editando el token.
        token, _ = tokens.create_access_token(**self.datos())

        cabecera, payload, firma = token.split(".")
        alterado = f"{cabecera}.{payload[:-2]}XX.{firma}"

        with pytest.raises(tokens.TokenError):
            tokens.decode_access_token(alterado)

    def test_un_token_vencido_se_rechaza(self, monkeypatch):
        monkeypatch.setattr(settings, "jwt_expire_minutes", -1)

        token, _ = tokens.create_access_token(**self.datos())

        with pytest.raises(tokens.TokenError, match="expir"):
            tokens.decode_access_token(token)

    def test_un_secreto_corto_se_rechaza(self, monkeypatch):
        # Un secreto corto es adivinable por fuerza bruta, y adivinarlo
        # equivale a poder firmarse un token de administrador. Se rechaza en
        # vez de avisar: los avisos del log de arranque no los lee nadie.
        monkeypatch.setattr(settings, "jwt_secret", "corto")

        with pytest.raises(tokens.TokenError, match="corto"):
            tokens.create_access_token(**self.datos())

    def test_sin_secreto_no_se_valida_nada(self, monkeypatch):
        # Si alguien despliega sin JWT_SECRET, el sistema tiene que quedarse
        # cerrado, no abierto.
        token, _ = tokens.create_access_token(**self.datos())

        monkeypatch.setattr(settings, "jwt_secret", "")

        with pytest.raises(tokens.TokenError):
            tokens.decode_access_token(token)

    def test_el_algoritmo_none_no_cuela(self):
        # El ataque clásico contra JWT: cambiar el algoritmo a "none" para
        # que la firma deje de comprobarse. Se fija el algoritmo esperado
        # justamente para esto.
        import base64
        import json

        def b64(dato):
            return base64.urlsafe_b64encode(
                json.dumps(dato).encode()
            ).rstrip(b"=").decode()

        falso = (
            f"{b64({'alg': 'none', 'typ': 'JWT'})}."
            f"{b64({'sub': '1', 'role': 'admin', 'exp': int(time.time()) + 999})}."
        )

        with pytest.raises(tokens.TokenError):
            tokens.decode_access_token(falso)

    def test_el_token_dice_cuando_expira(self):
        _, expira = tokens.create_access_token(**self.datos())

        assert expira.tzinfo is not None, "debe llevar zona horaria"
