"""Deduplicar incidentes por pasada del pipeline

Revision ID: c3a71f0d9e28
Revises: ba242350f7da

Cada conexión al WebSocket reprocesaba el video entero y volvía a insertar los
mismos incidentes: la deduplicación vivía en un `set` en memoria que se vaciaba
al arrancar cada sesión, así que sobrevivía dentro de una sesión y en ninguna
otra parte. En la base de desarrollo eso dejó 275 filas para 15 incidentes
reales, una de ellas con 35 copias.

Esta migración añade `source_key` —de qué pasada del pipeline viene cada
incidente—, limpia lo que ya se duplicó y crea el índice único que impide que
vuelva a pasar.

Sobre la limpieza
-----------------
BORRA filas, y el downgrade no puede devolverlas: solo quita el índice y la
columna. Antes de aplicarla donde el histórico importe, sacar copia.

De cada grupo sobrevive una sola fila, elegida en este orden: la que tenga
veredicto humano (y entre varias, la revisada más recientemente), luego la que
tenga evidencia, y a igualdad la más antigua. Después se le rellena la
evidencia desde sus copias si ella no la tenía — al reprocesar, la primera
inserción de un incidente suele quedarse sin imagen y la reciben las copias
siguientes, así que quedarse con la más antigua sin más perdería la foto de
todos los incidentes.

Las filas que ya existen se marcan como pasada `legacy`, todas en el mismo
grupo. Si un despliegue hubiera analizado varios videos distintos con la misma
cámara, dos incidentes de videos distintos que compartan id de agrupación se
fusionarían en esta limpieza. Es asumible porque el id solo empezó a repetirse
al reprocesar la misma fuente, que es justo lo que hay que limpiar.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3a71f0d9e28"
down_revision: Union[str, Sequence[str], None] = "ba242350f7da"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# La fila que se queda de cada grupo. `review_status` se compara con IS NOT
# NULL delante porque un NULL en la expresión ordenaría antes que TRUE con
# DESC, y dejaría ganar a una fila sin revisar.
_SUPERVIVIENTES = """
    CREATE TEMP TABLE _incidentes_a_conservar AS
    SELECT DISTINCT ON (camera_id, cluster_id)
           id, camera_id, cluster_id
      FROM incidents
     WHERE cluster_id IS NOT NULL
     ORDER BY camera_id,
              cluster_id,
              (review_status IS NOT NULL AND review_status <> 'pendiente') DESC,
              reviewed_at DESC NULLS LAST,
              (evidence_path IS NOT NULL) DESC,
              id ASC
"""

# La evidencia de las copias, rescatada para el superviviente que no la tenga.
_RESCATAR_EVIDENCIA = """
    UPDATE incidents AS superviviente
       SET evidence_path = copia.evidence_path
      FROM (
            SELECT DISTINCT ON (camera_id, cluster_id)
                   camera_id, cluster_id, evidence_path
              FROM incidents
             WHERE cluster_id IS NOT NULL
               AND evidence_path IS NOT NULL
             ORDER BY camera_id, cluster_id, id
           ) AS copia
     WHERE superviviente.id IN (SELECT id FROM _incidentes_a_conservar)
       AND superviviente.evidence_path IS NULL
       AND superviviente.camera_id = copia.camera_id
       AND superviviente.cluster_id = copia.cluster_id
"""

_BORRAR_COPIAS = """
    DELETE FROM incidents
     WHERE cluster_id IS NOT NULL
       AND id NOT IN (SELECT id FROM _incidentes_a_conservar)
"""


def upgrade() -> None:
    conn = op.get_bind()

    # Nula al principio: las filas que ya existen no vienen de ninguna pasada
    # identificable, y hay que rellenarlas antes de poder exigir el valor.
    op.add_column(
        "incidents",
        sa.Column("source_key", sa.String(length=64), nullable=True),
    )

    conn.execute(sa.text("UPDATE incidents SET source_key = 'legacy'"))

    conn.execute(sa.text(_SUPERVIVIENTES))

    sobrantes = conn.execute(
        sa.text(
            """
            SELECT count(*) FROM incidents
             WHERE cluster_id IS NOT NULL
               AND id NOT IN (SELECT id FROM _incidentes_a_conservar)
            """
        )
    ).scalar()

    conn.execute(sa.text(_RESCATAR_EVIDENCIA))
    conn.execute(sa.text(_BORRAR_COPIAS))
    conn.execute(sa.text("DROP TABLE _incidentes_a_conservar"))

    print(f"Incidentes duplicados eliminados: {sobrantes}")

    op.alter_column("incidents", "source_key", nullable=False)

    op.create_index(
        "uq_incidents_source_cluster",
        "incidents",
        ["camera_id", "source_key", "cluster_id"],
        unique=True,
        postgresql_where=sa.text("cluster_id IS NOT NULL"),
    )


def downgrade() -> None:
    # Las filas borradas no vuelven. Esto deshace el esquema, no la limpieza.
    op.drop_index("uq_incidents_source_cluster", table_name="incidents")
    op.drop_column("incidents", "source_key")
