r"""
Carga cameras.yaml en la tabla `cameras`.

Es idempotente: vuelve a correrlo tras editar el YAML y actualiza las filas
existentes en vez de duplicarlas. Sirve tanto para la migración inicial como
para versionar la calibración en Git y aplicarla a un despliegue.

    .venv\Scripts\python scripts\seed_cameras.py
    .venv\Scripts\python scripts\seed_cameras.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "vision_service"))

from app.api.cameras import load_yaml_cameras  # noqa: E402
from app.db.models import CameraRow  # noqa: E402
from app.db.session import session_scope  # noqa: E402

FIELDS = (
    "name",
    "source",
    "lat",
    "lng",
    "roi",
    "perspective",
    "occupancy_medium",
    "occupancy_high",
    "free_speed",
    "notes",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra lo que haría sin escribir nada.",
    )
    args = parser.parse_args()

    cameras = load_yaml_cameras()

    if not cameras:
        raise SystemExit("cameras.yaml no tiene cámaras.")

    with session_scope() as session:

        for camera in cameras.values():

            row = session.get(CameraRow, camera.id)
            values = {
                "name": camera.name,
                # La ruta se guarda relativa a la raíz cuando está dentro del
                # repo, para que la fila sirva igual en otra máquina.
                "source": _portable_source(camera.source),
                "lat": camera.lat,
                "lng": camera.lng,
                "roi": camera.roi,
                "perspective": camera.perspective,
                "occupancy_medium": camera.occupancy_medium,
                "occupancy_high": camera.occupancy_high,
                "free_speed": camera.free_speed,
                "notes": camera.notes,
            }

            if row is None:
                print(f"+ alta    {camera.id}")
                if not args.dry_run:
                    session.add(CameraRow(id=camera.id, **values))
                continue

            changed = [f for f in FIELDS if getattr(row, f) != values[f]]

            if not changed:
                print(f"= igual   {camera.id}")
                continue

            print(f"~ cambia  {camera.id}  ({', '.join(changed)})")

            if not args.dry_run:
                for field in changed:
                    setattr(row, field, values[field])

        if args.dry_run:
            session.rollback()
            print("\n(dry-run: no se escribió nada)")


def _portable_source(source: Path) -> str:
    try:
        return source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(source)


if __name__ == "__main__":
    main()
