"""
Normalización y validación de los datos de identidad de una persona.

Vive aparte de las rutas porque es la parte que se equivoca en silencio: una
cédula guardada como "1.234.567.890" y la misma como "1234567890" son dos
filas distintas para la base de datos y la misma persona para todo el mundo,
así que la restricción de unicidad deja de servir sin que nadie lo note hasta
que hay dos cuentas del mismo empleado.

La regla es normalizar al guardar, no al leer: en la base solo hay dígitos.
"""

from __future__ import annotations

import re

# En Colombia la cédula de ciudadanía va hasta 10 dígitos, y la de extranjería
# puede ser más corta. Se acota por longitud y no por formato porque los
# documentos cambian y rechazar uno legítimo deja a alguien sin poder entrar.
CEDULA_MIN = 5
CEDULA_MAX = 15

# Un celular colombiano son 10 dígitos (3XXXXXXXXX); un fijo con indicativo,
# también. Con prefijo internacional +57 llegan a 12.
PHONE_MIN = 7
PHONE_MAX = 15

_NOT_DIGITS = re.compile(r"\D+")

# Suficiente para descartar lo que claramente no es un correo, sin pretender
# validar el RFC: la comprobación de verdad es mandarle un mensaje.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


class IdentityError(ValueError):
    """Un dato de identidad no es utilizable."""


def normalize_digits(value: str) -> str:
    """Solo los dígitos. '1.234.567-890' y '1 234 567 890' dan lo mismo."""
    return _NOT_DIGITS.sub("", value or "")


def clean_email(value: str) -> str:
    """El correo en minúsculas y sin espacios alrededor."""

    email = (value or "").strip().lower()

    if not _EMAIL.match(email):
        raise IdentityError(f"'{value}' no parece un correo válido.")

    if len(email) > 160:
        raise IdentityError("El correo es demasiado largo.")

    return email


def clean_cedula(value: str) -> str:
    """La cédula, solo dígitos y validada."""

    cedula = normalize_digits(value)

    if not cedula:
        raise IdentityError("La cédula es obligatoria.")

    if len(cedula) < CEDULA_MIN or len(cedula) > CEDULA_MAX:
        raise IdentityError(
            f"La cédula debe tener entre {CEDULA_MIN} y {CEDULA_MAX} dígitos; "
            f"esta tiene {len(cedula)}."
        )

    if set(cedula) == {"0"}:
        raise IdentityError("La cédula no puede ser solo ceros.")

    return cedula


def clean_phone(value: str, *, required: bool = True) -> str:
    """El teléfono, solo dígitos y validado."""

    phone = normalize_digits(value)

    if not phone:
        if required:
            raise IdentityError("El teléfono es obligatorio.")
        return ""

    if len(phone) < PHONE_MIN or len(phone) > PHONE_MAX:
        raise IdentityError(
            f"El teléfono debe tener entre {PHONE_MIN} y {PHONE_MAX} dígitos; "
            f"este tiene {len(phone)}."
        )

    return phone


def clean_name(value: str) -> str:
    """El nombre para mostrar, sin espacios de sobra."""

    name = " ".join((value or "").split())

    if len(name) > 160:
        raise IdentityError("El nombre es demasiado largo.")

    return name


def format_phone(phone: str) -> str:
    """
    El teléfono agrupado para leerlo: '3001234567' -> '300 123 4567'.

    Solo para mostrar. En la base siempre están los dígitos pelados, que es
    lo que hace que buscar y comparar funcione.
    """

    digits = normalize_digits(phone)

    if len(digits) == 10:
        return f"{digits[:3]} {digits[3:6]} {digits[6:]}"

    return digits


def format_cedula(cedula: str) -> str:
    """La cédula con puntos de millar, como se escribe en Colombia."""

    digits = normalize_digits(cedula)

    return f"{int(digits):,}".replace(",", ".") if digits else ""
