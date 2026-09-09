"""
Normalización de cédula, teléfono y correo.

Esto existe porque el fallo aquí es invisible: "1.234.567.890" y
"1234567890" son dos filas distintas para la base de datos y la misma
persona para cualquiera que las lea. La restricción de unicidad deja de
servir sin que nadie se entere, hasta que hay dos cuentas del mismo empleado.
"""

import pytest

from services.vision_service.app.auth.identity import (
    IdentityError,
    clean_cedula,
    clean_email,
    clean_name,
    clean_phone,
    format_cedula,
    format_phone,
)


class TestCedula:

    def test_todas_las_formas_de_escribirla_dan_lo_mismo(self):
        # Es LA prueba: si estas cuatro no coinciden, la unicidad no protege
        # de duplicar a una persona.
        formas = [
            "1234567890",
            "1.234.567.890",
            "1 234 567 890",
            "1-234-567-890",
        ]

        assert {clean_cedula(f) for f in formas} == {"1234567890"}

    def test_una_cedula_vacia_se_rechaza(self):
        with pytest.raises(IdentityError, match="obligatoria"):
            clean_cedula("")

    def test_algo_que_no_tiene_digitos_se_rechaza(self):
        with pytest.raises(IdentityError):
            clean_cedula("no-es-una-cedula")

    def test_demasiado_corta_o_larga(self):
        with pytest.raises(IdentityError, match="dígitos"):
            clean_cedula("123")

        with pytest.raises(IdentityError, match="dígitos"):
            clean_cedula("1" * 20)

    def test_solo_ceros_no_es_una_cedula(self):
        # Es justo el relleno que puso la migración en las cuentas viejas:
        # tiene que seguir siendo imposible escribirlo a mano.
        with pytest.raises(IdentityError, match="ceros"):
            clean_cedula("0000000000")

    def test_se_muestra_con_puntos(self):
        assert format_cedula("1234567890") == "1.234.567.890"


class TestTelefono:

    def test_normaliza_los_formatos_habituales(self):
        formas = ["3001234567", "300 123 4567", "(300) 123-4567"]

        assert {clean_phone(f) for f in formas} == {"3001234567"}

    def test_acepta_el_indicativo_internacional(self):
        assert clean_phone("+57 300 123 4567") == "573001234567"

    def test_obligatorio_por_defecto_y_opcional_si_se_pide(self):
        with pytest.raises(IdentityError, match="obligatorio"):
            clean_phone("")

        assert clean_phone("", required=False) == ""

    def test_se_muestra_agrupado(self):
        assert format_phone("3001234567") == "300 123 4567"


class TestCorreo:

    def test_normaliza_a_minusculas(self):
        # "Juan@X.com" y "juan@x.com" tienen que ser la misma cuenta.
        assert clean_email("  Juan@Ejemplo.COM ") == "juan@ejemplo.com"

    def test_rechaza_lo_que_claramente_no_es_un_correo(self):
        for malo in ["sin-arroba", "@sin-usuario.com", "sin@dominio", "a b@c.com"]:
            with pytest.raises(IdentityError):
                clean_email(malo)


class TestNombre:

    def test_colapsa_los_espacios_de_sobra(self):
        assert clean_name("  Ana   María   Gómez ") == "Ana María Gómez"

    def test_un_nombre_vacio_es_aceptable(self):
        # La ruta lo sustituye por la parte del correo: no es un dato del que
        # dependa nada.
        assert clean_name("") == ""
