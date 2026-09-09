"""
Inicio de sesión.

    POST /api/auth/login  -> token de sesión
    GET  /api/auth/me     -> quién soy y qué puedo hacer
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from ...auth.dependencies import CurrentUserDep
from ...auth.passwords import verify
from ...auth.tokens import create_access_token
from ...db.models import UserRow
from ...db.session import session_scope
from ...shifts import tracker
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str = Field(max_length=160)
    password: str = Field(max_length=200)


@router.post("/login")
def login(body: LoginRequest) -> dict[str, Any]:
    """
    Entrega un token si las credenciales son correctas.

    El mensaje de error es el MISMO para un correo que no existe y para una
    contraseña equivocada. Distinguirlos convierte este endpoint en una
    forma de averiguar qué cuentas existen, que es el primer paso de
    cualquier ataque dirigido.
    """

    if not settings.jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El servicio no tiene JWT_SECRET configurado; no puede "
                "emitir sesiones."
            ),
        )

    email = body.email.strip().lower()

    try:
        with session_scope() as session:
            user = session.scalars(
                select(UserRow).where(UserRow.email == email)
            ).one_or_none()

            # Se verifica el hash incluso cuando el usuario no existe, contra
            # un hash de mentira, para que responder tarde lo mismo en los
            # dos casos. Si no, el tiempo de respuesta delata qué correos
            # están registrados.
            hashed = user.password_hash if user else _DUMMY_HASH
            correcta = verify(body.password, hashed)

            if user is None or not correcta:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Correo o contraseña incorrectos.",
                )

            if not user.active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Esta cuenta está desactivada.",
                )

            permissions = user.permissions
            token, expires = create_access_token(
                user_id=user.id,
                email=user.email,
                role=user.role_name,
                permissions=permissions,
            )

            user.last_login_at = datetime.now(timezone.utc)

            # El turno arranca al entrar. Si ya había uno vivo se reanuda en
            # vez de abrir otro: recargar la página o abrir una segunda
            # pestaña es la misma jornada, y contarla dos veces duplicaría las
            # horas de quien trabaja con la consola en dos monitores.
            tracker.abrir_o_reanudar(session, user.id)

            perfil = {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role_name,
                "permissions": sorted(permissions),
            }

    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        logger.warning("Fallo al iniciar sesión: %s", error, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base de datos no está disponible.",
        ) from error

    logger.info("Sesión iniciada: %s (%s)", perfil["email"], perfil["role"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires.isoformat(),
        "user": perfil,
    }


@router.get("/me")
def me(user: CurrentUserDep) -> dict[str, Any]:
    """
    El perfil vigente.

    El frontend lo llama al arrancar para saber si el token guardado sigue
    sirviendo, y para releer los permisos: pueden haber cambiado desde que
    se emitió el token.
    """

    return user.public()


# Un hash bcrypt real de una contraseña que nadie conoce. Existe solo para
# que comprobar credenciales cueste lo mismo exista o no el usuario.
_DUMMY_HASH = (
    "$2b$12$C6UzMDM.H6dfI/f/IKcEe.9Zx0Q7bTUyfrJlBRe1SDvQAJ8N1o5Lq"
)
