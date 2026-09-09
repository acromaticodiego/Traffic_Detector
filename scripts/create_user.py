r"""
Crea o actualiza un usuario del sistema.

Es la puerta de entrada: sin esto no existe ninguna cuenta y nadie puede
iniciar sesion. El primer admin se crea con este script; los demas usuarios
tambien, hasta que exista el panel de administracion.

Uso:

    .venv\Scripts\python scripts\create_user.py --email juan@ejemplo.com --role admin
    .venv\Scripts\python scripts\create_user.py --email ana@ejemplo.com --role analista --name "Ana Gomez"

La contrasena NO se pasa por argumento a proposito: quedaria en el historial
del shell y en la lista de procesos de la maquina. Se pide por teclado, o se
lee de la variable de entorno NEW_USER_PASSWORD para poder automatizarlo.

Cambiar la contrasena de alguien que ya existe es correr lo mismo otra vez.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "vision_service"))

from sqlalchemy import select  # noqa: E402

from app.auth.identity import (  # noqa: E402
    IdentityError,
    clean_cedula,
    clean_phone,
)
from app.auth.passwords import PasswordError, hash_password  # noqa: E402
from app.db.models import RoleRow, UserRow  # noqa: E402
from app.db.session import session_scope  # noqa: E402


def pedir_password() -> str:
    """Del entorno si esta, y si no por teclado, dos veces."""

    del_entorno = os.getenv("NEW_USER_PASSWORD")

    if del_entorno:
        return del_entorno

    primera = getpass.getpass("Contrasena: ")
    segunda = getpass.getpass("Reptela: ")

    if primera != segunda:
        raise SystemExit("Las contrasenas no coinciden.")

    return primera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", required=True,
                        help="admin, operario o analista")
    parser.add_argument("--name", default="", help="Nombre para mostrar")
    parser.add_argument("--cedula", default="", help="Cedula (solo digitos)")
    parser.add_argument("--phone", default="", help="Telefono")
    parser.add_argument("--deactivate", action="store_true",
                        help="Desactiva la cuenta en vez de crearla")
    args = parser.parse_args()

    email = args.email.strip().lower()

    with session_scope() as session:

        rol = session.get(RoleRow, args.role)

        if rol is None:
            conocidos = ", ".join(
                r.name for r in session.scalars(select(RoleRow)).all()
            )
            raise SystemExit(
                f"Rol desconocido: '{args.role}'. Disponibles: {conocidos}\n"
                "Si no aparece ninguno, falta correr: alembic upgrade head"
            )

        existente = session.scalars(
            select(UserRow).where(UserRow.email == email)
        ).one_or_none()

        if args.deactivate:
            if existente is None:
                raise SystemExit(f"No existe ningun usuario con {email}.")
            existente.active = False
            print(f"- desactivado  {email}")
            return

        try:
            hashed = hash_password(pedir_password())
        except PasswordError as error:
            raise SystemExit(str(error))

        if existente is None and not args.cedula:
            raise SystemExit(
                "Falta --cedula: es obligatoria para crear una cuenta nueva."
            )

        try:
            cedula = clean_cedula(args.cedula) if args.cedula else ""
            phone = clean_phone(args.phone, required=False)
        except IdentityError as error:
            raise SystemExit(str(error))

        if existente is None:
            session.add(
                UserRow(
                    email=email,
                    full_name=args.name or email.split("@")[0],
                    password_hash=hashed,
                    role_name=rol.name,
                    cedula=cedula,
                    phone=phone,
                )
            )
            print(f"+ creado       {email}  ({rol.name})")
        else:
            existente.password_hash = hashed
            existente.role_name = rol.name
            existente.active = True
            if args.name:
                existente.full_name = args.name
            if cedula:
                existente.cedula = cedula
            if phone:
                existente.phone = phone
            print(f"~ actualizado  {email}  ({rol.name})")

        print()
        print("Permisos de ese rol:")
        for permiso in sorted(p.code for p in rol.permissions):
            print(f"  - {permiso}")


if __name__ == "__main__":
    main()
