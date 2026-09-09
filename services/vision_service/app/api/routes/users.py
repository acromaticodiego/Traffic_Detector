"""
Administración de usuarios.

    GET   /api/users          -> listado
    POST  /api/users          -> crear cuenta
    PATCH /api/users/{id}     -> editar datos, rol o estado
    GET   /api/users/roles    -> roles disponibles y qué permite cada uno

Todo el módulo exige `users:manage`, así que solo un administrador entra.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ...auth.dependencies import CurrentUser, require
from ...auth.identity import (
    IdentityError,
    clean_cedula,
    clean_email,
    clean_name,
    clean_phone,
)
from ...auth.passwords import PasswordError, hash_password
from ...db.models import PERM_USERS_MANAGE, RoleRow, UserRow
from ...db.session import session_scope

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/users",
    tags=["users"],
    # A nivel de módulo: una ruta nueva queda protegida por omisión en vez de
    # quedar abierta por olvido.
    dependencies=[Depends(require(PERM_USERS_MANAGE))],
)

AdminDep = Annotated[CurrentUser, Depends(require(PERM_USERS_MANAGE))]


def _row_to_dict(row: UserRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "email": row.email,
        "full_name": row.full_name,
        "cedula": row.cedula,
        "phone": row.phone,
        "role": row.role_name,
        "active": row.active,
        "last_login_at": row.last_login_at.isoformat() if row.last_login_at else None,
        "created_at": row.created_at.isoformat(),
    }


@router.get("")
def list_users() -> dict[str, Any]:
    with session_scope() as session:
        rows = session.scalars(select(UserRow).order_by(UserRow.id)).all()

        return {"items": [_row_to_dict(r) for r in rows]}


@router.get("/roles")
def list_roles() -> dict[str, Any]:
    """
    Los roles y sus permisos.

    El formulario los pide en vez de traerlos escritos: si mañana se agrega
    un rol en la base, aparece solo. Y de paso el administrador ve qué está
    concediendo antes de asignarlo.
    """

    with session_scope() as session:
        roles = session.scalars(select(RoleRow).order_by(RoleRow.name)).all()

        return {
            "items": [
                {
                    "name": r.name,
                    "description": r.description,
                    "permissions": sorted(p.code for p in r.permissions),
                }
                for r in roles
            ]
        }


class CreateUser(BaseModel):
    email: str = Field(max_length=200)
    full_name: str = Field(default="", max_length=200)
    cedula: str = Field(max_length=30)
    phone: str = Field(max_length=30)
    role: str = Field(max_length=32)
    password: str = Field(max_length=200)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_user(body: CreateUser) -> dict[str, Any]:
    try:
        email = clean_email(body.email)
        cedula = clean_cedula(body.cedula)
        phone = clean_phone(body.phone)
        full_name = clean_name(body.full_name)
        password_hash = hash_password(body.password)
    except (IdentityError, PasswordError) as error:
        # 422 y no 400: el problema es el contenido de un campo, y el
        # frontend lo muestra tal cual junto al formulario.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error

    with session_scope() as session:
        if session.get(RoleRow, body.role) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Rol desconocido: '{body.role}'.",
            )

        user = UserRow(
            email=email,
            full_name=full_name or email.split("@")[0],
            cedula=cedula,
            phone=phone,
            password_hash=password_hash,
            role_name=body.role,
        )

        session.add(user)

        try:
            session.flush()
        except IntegrityError as error:
            # La unicidad la impone la base, no una consulta previa: entre
            # comprobar y escribir cabe otra petición creando el mismo
            # correo, y esa carrera solo la gana el índice.
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=_conflict_message(error),
            ) from error

        creado = _row_to_dict(user)

    logger.info("Usuario creado: %s (%s)", creado["email"], creado["role"])

    return creado


class UpdateUser(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=200)
    cedula: Optional[str] = Field(default=None, max_length=30)
    phone: Optional[str] = Field(default=None, max_length=30)
    role: Optional[str] = Field(default=None, max_length=32)
    active: Optional[bool] = None
    password: Optional[str] = Field(default=None, max_length=200)


@router.patch("/{user_id}")
def update_user(user_id: int, body: UpdateUser, admin: AdminDep) -> dict[str, Any]:
    with session_scope() as session:
        user = session.get(UserRow, user_id)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No existe el usuario {user_id}.",
            )

        # Nadie se quita a sí mismo el acceso: un administrador que se
        # desactiva o se degrada por error deja el sistema sin quien lo
        # arregle, y recuperarlo exige entrar a la base a mano.
        if user.id == admin.id:
            if body.active is False:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No puedes desactivar tu propia cuenta.",
                )
            if body.role is not None and body.role != user.role_name:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="No puedes cambiarte el rol a ti mismo.",
                )

        try:
            if body.full_name is not None:
                user.full_name = clean_name(body.full_name)
            if body.cedula is not None:
                user.cedula = clean_cedula(body.cedula)
            if body.phone is not None:
                user.phone = clean_phone(body.phone, required=False)
            if body.password:
                user.password_hash = hash_password(body.password)
        except (IdentityError, PasswordError) as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
            ) from error

        if body.role is not None:
            if session.get(RoleRow, body.role) is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Rol desconocido: '{body.role}'.",
                )
            user.role_name = body.role

        if body.active is not None:
            user.active = body.active

        try:
            session.flush()
        except IntegrityError as error:
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=_conflict_message(error),
            ) from error

        return _row_to_dict(user)


def _conflict_message(error: IntegrityError) -> str:
    """Traduce la violación de unicidad a algo que se pueda mostrar."""

    texto = str(error.orig) if error.orig else str(error)

    if "cedula" in texto:
        return "Ya existe una cuenta con esa cédula."

    if "email" in texto:
        return "Ya existe una cuenta con ese correo."

    return "Ese usuario choca con uno que ya existe."
