"""limpiar ruta absoluta de evidencia en data

Revision ID: cfb468600785
Revises: 610e9dfc2b32
Create Date: 2026-09-08 18:21:06.729356

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cfb468600785'
down_revision: Union[str, Sequence[str], None] = '610e9dfc2b32'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Quita `evidence_path` del JSONB `data` de los incidentes ya guardados.

    El motor escribe ahí la ruta ABSOLUTA del archivo, con el árbol de
    directorios completo del servidor, y ese campo viaja entero al navegador
    dentro de la respuesta del histórico. Además deja de resolver en cuanto
    el despliegue cambia de máquina.

    No se pierde nada: la ruta utilizable, relativa a la raíz del repo, vive
    en la columna `evidence_path` desde la migración 18d76af8c987. El
    serializador ya no la incluye para los incidentes nuevos; esto arregla
    los viejos.
    """

    op.execute(
        """
        UPDATE incidents
           SET data = data - 'evidence_path'
         WHERE data ? 'evidence_path'
        """
    )


def downgrade() -> None:
    """
    No se restaura.

    Sería reconstruir una ruta absoluta a partir de la relativa usando la
    ubicación del repo en ESTA máquina, que es justamente el dato equivocado
    que la subida vino a quitar. La información no se pierde: sigue en la
    columna `evidence_path`.
    """

    pass
