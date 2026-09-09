"""
Quién eres y qué puedes hacer, comprobado en cada petición.

Aquí es donde vive la seguridad de verdad. El frontend esconde los botones
que un rol no puede usar, pero eso es comodidad: cualquiera puede llamar a la
API con `curl`. Lo único que impide que un operario descarte un incidente es
que esta capa lo rechace.

Dos decisiones deliberadas:

  - **El rol y los permisos se releen de la base en cada petición**, aunque
    vengan escritos en el token. Un JWT no se puede revocar antes de que
    expire, así que si no se consultara, degradar a alguien de admin no
    surtiría efecto hasta doce horas después. Cuesta una consulta; vale la
    pena.

  - **Un usuario desactivado queda fuera al instante**, por la misma razón.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..db.models import UserRow
from ..db.session import session_scope
from .tokens import TokenError, decode_access_token, user_id_from

# Cabecera estándar: "Authorization: Bearer <token>".
_PREFIX = "bearer "


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        # Se lo dice al cliente en el idioma del estándar, para que sepa que
        # tiene que autenticarse y no que el recurso no existe.
        headers={"WWW-Authenticate": "Bearer"},
    )


def token_from_request(request: Request) -> str:
    """
    El token, de la cabecera o de la query.

    La query se acepta porque un WebSocket del navegador NO puede llevar
    cabeceras propias: no hay forma de mandar `Authorization` al abrirlo. Es
    una concesión conocida, y por eso solo la usa la ruta del stream.
    """

    header = request.headers.get("Authorization", "")

    if header.lower().startswith(_PREFIX):
        return header[len(_PREFIX):].strip()

    query = request.query_params.get("token")

    if query:
        return query.strip()

    raise _unauthorized("Falta el token de sesión.")


def load_user(session: Session, token: str) -> UserRow:
    """El usuario que el token identifica, validado contra la base."""

    try:
        payload = decode_access_token(token)
        user_id = user_id_from(payload)
    except TokenError as error:
        raise _unauthorized(str(error)) from error

    user = session.get(UserRow, user_id)

    if user is None:
        # El token es válido pero la cuenta ya no está. Mismo mensaje que un
        # token inválido: no hay que confirmarle a nadie qué ids existieron.
        raise _unauthorized("Token inválido.")

    if not user.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Esta cuenta está desactivada.",
        )

    return user


class CurrentUser:
    """
    Los datos del usuario ya extraídos de la sesión de base de datos.

    Se copian a un objeto plano porque la sesión se cierra al salir de la
    dependencia, y una fila del ORM desligada de su sesión falla al tocar
    cualquier relación perezosa — un error que aparecería en runtime y solo
    en algunas rutas.
    """

    def __init__(self, row: UserRow) -> None:
        self.id = row.id
        self.email = row.email
        self.full_name = row.full_name
        self.role = row.role_name
        self.permissions = row.permissions

    def can(self, permission: str) -> bool:
        return permission in self.permissions

    def public(self) -> dict:
        """Lo que se le devuelve al frontend."""
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "permissions": sorted(self.permissions),
        }


def current_user(request: Request) -> CurrentUser:
    """Dependencia base: exige una sesión válida."""

    token = token_from_request(request)

    with session_scope() as session:
        return CurrentUser(load_user(session, token))


CurrentUserDep = Annotated[CurrentUser, Depends(current_user)]


def require(*permissions: str):
    """
    Dependencia que exige TODOS los permisos indicados.

        @router.patch("/{id}/review", dependencies=[Depends(require("incidents:review"))])

    Devuelve 403 y no 404: el recurso existe, lo que falta es autorización.
    Confundir los dos códigos es lo que hace que un cliente reintente
    eternamente algo que nunca va a funcionar.
    """

    def dependency(user: CurrentUserDep) -> CurrentUser:
        faltantes = [p for p in permissions if not user.can(p)]

        if faltantes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Tu rol ({user.role}) no tiene permiso para esto: "
                    f"{', '.join(faltantes)}."
                ),
            )

        return user

    return dependency


def optional_user(request: Request) -> Optional[CurrentUser]:
    """Como `current_user`, pero devuelve None en vez de rechazar."""

    try:
        return current_user(request)
    except HTTPException:
        return None
