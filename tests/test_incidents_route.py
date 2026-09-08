"""
El guardia de rutas de la evidencia.

`GET /api/incidents/{id}/evidence` toma una ruta guardada en la base y
devuelve ese archivo por HTTP. Esa forma —dato de la base convertido en
archivo servido— es exactamente la de un directory traversal: basta una fila
editada a mano, o un `evidence_path` construido con un id de cámara malicioso,
para que un archivo de fuera del directorio de evidencia salga por Internet.

Estos tests fijan que solo salga lo que está dentro del directorio.
"""

import types

import pytest
from fastapi import HTTPException

from services.vision_service.app.api.config import settings
from services.vision_service.app.api.routes.incidents import _evidence_file


def make_row(evidence_path, incident_id=1):
    """Una fila mínima: la función solo mira estos dos campos."""
    return types.SimpleNamespace(id=incident_id, evidence_path=evidence_path)


@pytest.fixture
def evidencia(tmp_path, monkeypatch):
    """Un directorio de evidencia de mentira, con un incidente dentro."""

    root = tmp_path / "incidents"
    incidente = root / "cam1" / "col-1-2_42"
    incidente.mkdir(parents=True)

    (incidente / "annotated.jpg").write_bytes(b"jpeg-anotado")
    (incidente / "original.jpg").write_bytes(b"jpeg-original")

    monkeypatch.setattr(settings, "evidence_dir", root)

    return incidente


class TestEvidencia:

    def test_devuelve_la_imagen_anotada(self, evidencia):
        path = _evidence_file(make_row(evidencia / "annotated.jpg"), "annotated")

        assert path.read_bytes() == b"jpeg-anotado"

    def test_la_variante_original_es_el_frame_limpio(self, evidencia):
        # El frontend pide el original cuando el agente quiere ver la escena
        # sin las cajas encima tapándole medio choque.
        path = _evidence_file(make_row(evidencia / "annotated.jpg"), "original")

        assert path.read_bytes() == b"jpeg-original"

    def test_un_incidente_sin_evidencia_da_404(self):
        with pytest.raises(HTTPException) as fallo:
            _evidence_file(make_row(None), "annotated")

        assert fallo.value.status_code == 404

    def test_una_ruta_registrada_sin_archivo_da_404(self, evidencia):
        # Pasa de verdad: alguien limpia outputs/ para liberar disco y las
        # filas quedan apuntando a archivos que ya no están.
        borrado = evidencia / "annotated.jpg"
        borrado.unlink()

        with pytest.raises(HTTPException) as fallo:
            _evidence_file(make_row(borrado), "annotated")

        assert fallo.value.status_code == 404

    def test_no_sirve_archivos_de_fuera_del_directorio(self, evidencia, tmp_path):
        secreto = tmp_path / ".env"
        secreto.write_text("VISION_DATABASE_URL=postgresql://usuario:clave@host/db")

        with pytest.raises(HTTPException) as fallo:
            _evidence_file(make_row(secreto), "annotated")

        assert fallo.value.status_code == 404

    def test_no_se_escapa_con_saltos_hacia_atras(self, evidencia, tmp_path):
        # El caso clásico: la ruta empieza dentro del directorio permitido y
        # se sale a fuerza de "..". Se comprueba DESPUÉS de resolver, que es
        # la única forma de atraparlo.
        secreto = tmp_path / ".env"
        secreto.write_text("secreto")

        escape = evidencia / ".." / ".." / ".." / ".env"

        with pytest.raises(HTTPException) as fallo:
            _evidence_file(make_row(escape), "annotated")

        assert fallo.value.status_code == 404
