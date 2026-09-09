"""
Tokens de sesión (JWT).

Un JWT es un texto firmado que dice quién eres y hasta cuándo. El servidor no
guarda sesiones: verifica la firma y confía en el contenido. Eso es lo que lo
hace escalar —cualquier instancia del servicio puede validar cualquier token
sin consultar nada— y también lo que hay que tener presente:

  - **Un token válido no se puede revocar** antes de que expire. Por eso duran
    poco. Desactivar a un usuario surte efecto cuando su token caduca, no al
    instante; para lo inmediato está la comprobación de `active` que hace la
    dependencia en cada petición.

  - **El contenido es legible por cualquiera.** Va firmado, no cifrado. Aquí
    solo viajan el id, el correo, el rol y los permisos: nada que no pueda
    ver quien ya es dueño de la sesión.

  - **La firma depende del secreto.** Si `JWT_SECRET` se filtra, cualquiera
    puede fabricar tokens de admin. Va en el .env, nunca en el código.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from ..api.config import settings

ALGORITHM = "HS256"

# Longitud mínima del secreto. Por debajo de esto la firma HMAC-SHA256 se
# debilita —lo advierte la propia PyJWT, citando el RFC 7518— y un secreto
# corto es adivinable por fuerza bruta, lo que equivale a que cualquiera
# pueda firmarse un token de administrador. Se rechaza en vez de avisar:
# un aviso en el log de arranque no lo lee nadie.
MIN_SECRET_BYTES = 32


class TokenError(ValueError):
    """El token no es válido, está vencido o está mal firmado."""


def _secret() -> str:
    """El secreto de firma, validado."""

    secret = settings.jwt_secret

    if not secret:
        raise TokenError(
            "No hay JWT_SECRET configurado; la autenticación está apagada."
        )

    if len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise TokenError(
            f"JWT_SECRET es demasiado corto ({len(secret)} caracteres). "
            f"Debe tener al menos {MIN_SECRET_BYTES} bytes. Generar uno con: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )

    return secret


def create_access_token(
    *,
    user_id: int,
    email: str,
    role: str,
    permissions: set[str] | list[str],
) -> tuple[str, datetime]:
    """
    El token y su instante de expiración.

    Los permisos viajan dentro para que el frontend sepa qué dibujar sin una
    segunda petición. El backend NO se fía de ellos para autorizar: los
    vuelve a leer de la base en cada petición, porque un permiso revocado
    hace cinco minutos sigue estando escrito en un token emitido hace diez.
    """

    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=settings.jwt_expire_minutes)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "permissions": sorted(permissions),
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }

    return jwt.encode(payload, _secret(), algorithm=ALGORITHM), expires


def decode_access_token(token: str) -> dict[str, Any]:
    """
    El contenido del token, o TokenError.

    Se fija el algoritmo esperado a propósito: aceptar el que venga en la
    cabecera del propio token es la vulnerabilidad clásica de JWT, donde un
    atacante lo cambia a "none" y se firma sus propios permisos.
    """

    secret = _secret()

    try:
        return jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as error:
        raise TokenError("La sesión expiró; vuelve a entrar.") from error
    except jwt.InvalidTokenError as error:
        raise TokenError("Token inválido.") from error


def user_id_from(payload: dict[str, Any]) -> int:
    """El id del usuario, validado. `sub` es texto por el estándar de JWT."""

    try:
        return int(payload["sub"])
    except (KeyError, TypeError, ValueError) as error:
        raise TokenError("El token no identifica a ningún usuario.") from error
