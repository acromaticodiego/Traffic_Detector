"""
Cuánto tiempo se conserva la evidencia, y qué se borra cuando vence.

Ley 1581 de 2012: los datos personales no se pueden conservar indefinidamente,
hay que guardarlos solo el tiempo necesario para la finalidad declarada. Hoy el
disco crece sin techo —dos JPEG por incidente, unos 215 KB, para siempre.

La regla resuelve la tensión con el principio de "aquí no se borra nada":

    EL INCIDENTE SE QUEDA. LA IMAGEN CADUCA.

El dato personal es la foto, no la fila. La bandeja de revisión conserva su
histórico completo y sus veredictos —que es de donde va a salir la métrica de
precisión— y a los N días la imagen desaparece y `evidence_path` queda en null.

Este módulo BORRA ARCHIVOS, así que está escrito para equivocarse del lado de
conservar de más:

- Sin plazo configurado no borra nada. Una variable vacía o un cero significan
  "conservar para siempre", nunca "borrar todo": un despliegue mal configurado
  no puede vaciar el disco.
- Solo toca directorios que estén bajo la raíz de la evidencia y cuyo
  contenido sean únicamente imágenes. Cualquier otra cosa se deja intacta.
- Tiene modo de simulación, para poder mirar qué haría antes de dejarlo
  suelto sobre un año de evidencia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lo único que se considera evidencia. Un directorio que contenga cualquier
# otra cosa se deja en paz: no sabemos qué es y no nos toca decidirlo.
_IMAGENES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


@dataclass
class PurgeReport:
    """Qué se borró (o qué se habría borrado, en simulación)."""

    directories: int = 0
    files: int = 0
    freed_bytes: int = 0
    dry_run: bool = False

    @property
    def freed_mb(self) -> float:
        return round(self.freed_bytes / (1024 * 1024), 2)


def cutoff_for(days: int, now: Optional[datetime] = None) -> Optional[datetime]:
    """
    La fecha antes de la cual la evidencia vence. None = no vence nunca.

    Un plazo de cero o negativo desactiva la limpieza en vez de caducarlo
    todo. Es la diferencia entre un despliegue sin configurar y un despliegue
    que se borra a sí mismo.
    """

    if days <= 0:
        return None

    return (now or datetime.now(timezone.utc)) - timedelta(days=days)


def is_evidence_dir(path: Path) -> bool:
    """
    Si este directorio es una carpeta de evidencia y nada más.

    Se exige que TODO lo que hay dentro sea una imagen. Un directorio con un
    archivo suelto —un .txt, un subdirectorio, cualquier cosa que no pusimos
    nosotros— no se borra: puede ser de otro, y equivocarse aquí no tiene
    vuelta atrás.
    """

    if not path.is_dir() or path.is_symlink():
        return False

    entries = list(path.iterdir())

    if not entries:
        return False

    return all(
        entry.is_file()
        and not entry.is_symlink()
        and entry.suffix.lower() in _IMAGENES
        for entry in entries
    )


def newest_mtime(path: Path) -> Optional[datetime]:
    """La fecha del archivo más reciente del directorio."""

    stamps = [
        entry.stat().st_mtime for entry in path.iterdir() if entry.is_file()
    ]

    if not stamps:
        return None

    return datetime.fromtimestamp(max(stamps), tz=timezone.utc)


def expired_directories(root: Path, cutoff: datetime) -> list[Path]:
    """
    Las carpetas de evidencia cuya imagen más reciente ya venció.

    Se mira el archivo más nuevo y no la fecha del directorio: en Windows la
    fecha de un directorio no siempre sigue a su contenido, y una carpeta que
    parezca vieja por eso se llevaría por delante evidencia todavía vigente.
    """

    if not root.exists():
        return []

    expired: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not is_evidence_dir(path):
            continue

        stamp = newest_mtime(path)

        if stamp is not None and stamp < cutoff:
            expired.append(path)

    return expired


def purge_evidence(
    root: Path,
    cutoff: Optional[datetime],
    dry_run: bool = False,
) -> PurgeReport:
    """
    Borrar la evidencia vencida bajo `root`.

    Sin `cutoff` no hace nada: es el caso de "conservar para siempre".
    """

    report = PurgeReport(dry_run=dry_run)

    if cutoff is None:
        return report

    for directory in expired_directories(Path(root), cutoff):
        files = [entry for entry in directory.iterdir() if entry.is_file()]

        size = sum(entry.stat().st_size for entry in files)

        if not dry_run:
            try:
                for entry in files:
                    entry.unlink()

                directory.rmdir()
            except OSError as error:
                # Un archivo abierto o sin permisos no puede tumbar la
                # limpieza entera: se salta y se reintenta en la próxima
                # pasada, dentro de unas horas.
                logger.warning(
                    "No se pudo borrar la evidencia de '%s': %s",
                    directory,
                    error,
                )
                continue

        report.directories += 1
        report.files += len(files)
        report.freed_bytes += size

    return report


def forget_missing_paths(cutoff: Optional[datetime]) -> int:
    """
    Poner en null el `evidence_path` de los incidentes cuya imagen ya no está.

    Se decide mirando el disco y no la fecha: así el null llega exactamente
    cuando la imagen dejó de existir, aunque la haya borrado otra cosa —una
    limpieza a mano, un despliegue nuevo— y no solo cuando la cuenta de días
    dice que debería. La fila del incidente NO se toca: sigue en la bandeja,
    con su veredicto, solo que ya sin foto.

    Se acota a lo anterior al vencimiento para no recorrer todo el histórico
    cada pocas horas.
    """

    if cutoff is None:
        return 0

    # Import perezoso, igual que el escritor: importar este módulo no debe
    # obligar a que haya Postgres delante.
    from sqlalchemy import select

    from ..api.config import REPO_ROOT
    from ..db.models import IncidentRow
    from ..db.session import session_scope

    cleared = 0

    with session_scope() as session:
        rows = session.execute(
            select(IncidentRow)
            .where(IncidentRow.evidence_path.is_not(None))
            .where(IncidentRow.detected_at < cutoff)
        ).scalars()

        for row in rows:
            path = Path(row.evidence_path)

            if not path.is_absolute():
                path = REPO_ROOT / path

            if not path.exists():
                row.evidence_path = None
                cleared += 1

    return cleared


def run_retention(
    root: Path,
    days: int,
    dry_run: bool = False,
    now: Optional[datetime] = None,
) -> PurgeReport:
    """
    Una pasada completa: borrar lo vencido del disco y sincronizar la base.

    Es idempotente. Si la base no responde, el disco se limpia igual: dejar
    crecer el disco porque Postgres está caído sería el peor de los dos
    fallos posibles.
    """

    cutoff = cutoff_for(days, now)

    if cutoff is None:
        logger.info(
            "Retención de evidencia desactivada: se conserva indefinidamente. "
            "Para un despliegue real hay que fijar VISION_EVIDENCE_RETENTION_DAYS."
        )
        return PurgeReport(dry_run=dry_run)

    report = purge_evidence(root, cutoff, dry_run=dry_run)

    if report.directories:
        logger.info(
            "Retención: %s %d incidentes (%d archivos, %s MB) anteriores a %s.",
            "se borrarían" if dry_run else "borrados",
            report.directories,
            report.files,
            report.freed_mb,
            cutoff.date(),
        )

    if dry_run:
        return report

    try:
        cleared = forget_missing_paths(cutoff)
    except Exception as error:  # noqa: BLE001
        logger.warning(
            "No se pudo sincronizar la base tras la limpieza (%s). El disco "
            "sí quedó limpio; las rutas huérfanas se corrigen en la próxima "
            "pasada.",
            error,
        )
        return report

    if cleared:
        logger.info(
            "Retención: %d incidentes se quedaron sin foto en la base.", cleared
        )

    return report
