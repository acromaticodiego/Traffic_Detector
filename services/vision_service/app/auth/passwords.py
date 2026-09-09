"""
Contraseñas.

La regla es una: en la base nunca hay una contraseña, hay un hash bcrypt.
Si la tabla `users` se filtra, lo que se filtra son hashes, y recuperar de
ahí una contraseña cuesta años por cuenta. Ese es exactamente el punto de
usar bcrypt y no un SHA-256, que una GPU prueba a miles de millones por
segundo.

bcrypt además guarda su propia sal dentro del hash, así que dos personas con
la misma contraseña tienen hashes distintos y una tabla precalculada no
sirve de nada.
"""

from __future__ import annotations

import bcrypt

# Cuánto trabajo cuesta calcular un hash. Cada punto DUPLICA el tiempo, y
# ese tiempo es lo que protege: 12 son unos cientos de milisegundos en un
# servidor normal, imperceptible al entrar y carísimo para quien pruebe
# millones de combinaciones. Subirlo con los años es lo que mantiene la
# defensa al día conforme el hardware mejora.
ROUNDS = 12

# bcrypt trunca en silencio a 72 bytes. Una contraseña más larga que eso
# quedaría con su cola ignorada, así que se rechaza en vez de aceptar algo
# que no protege lo que el usuario cree.
MAX_BYTES = 72

MIN_LENGTH = 8


class PasswordError(ValueError):
    """La contraseña no se puede usar."""


def validate(password: str) -> None:
    """Rechaza lo que no sirve, con un motivo que se pueda mostrar."""

    if len(password) < MIN_LENGTH:
        raise PasswordError(
            f"La contraseña debe tener al menos {MIN_LENGTH} caracteres."
        )

    if len(password.encode("utf-8")) > MAX_BYTES:
        raise PasswordError(
            f"La contraseña no puede superar los {MAX_BYTES} bytes "
            "(bcrypt ignora el resto en silencio)."
        )


def hash_password(password: str) -> str:
    """El hash que se guarda en la base."""

    validate(password)

    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=ROUNDS)
    ).decode("ascii")


def verify(password: str, hashed: str) -> bool:
    """
    Si la contraseña corresponde al hash.

    Nunca lanza: un hash corrupto en la base tiene que dar "no coincide" y
    no un error 500 que además delataría que ese usuario existe.
    """

    try:
        return bcrypt.checkpw(
            password.encode("utf-8"), hashed.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False
