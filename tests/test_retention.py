"""
Retención de la evidencia (Ley 1581).

Este código borra archivos, así que la mitad de los tests no comprueban que
borre, sino que NO borre: un plazo sin configurar, un directorio que no es
nuestro, evidencia todavía vigente.
"""

import os
from datetime import datetime, timedelta, timezone

from services.vision_service.app.incidents.retention import (
    cutoff_for,
    expired_directories,
    is_evidence_dir,
    purge_evidence,
)

AHORA = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def carpeta(root, nombre: str, dias: float, archivos=("original.jpg", "annotated.jpg")):
    """Una carpeta de evidencia con la antigüedad pedida."""

    directorio = root / nombre
    directorio.mkdir(parents=True)

    stamp = (AHORA - timedelta(days=dias)).timestamp()

    for archivo in archivos:
        destino = directorio / archivo
        destino.write_bytes(b"x" * 1024)
        os.utime(destino, (stamp, stamp))

    return directorio


class TestPlazo:

    def test_sin_plazo_no_vence_nunca(self):
        # Una variable vacía o un typo no pueden vaciar el disco.
        assert cutoff_for(0, AHORA) is None
        assert cutoff_for(-30, AHORA) is None

    def test_con_plazo_vence_a_los_n_dias(self):
        assert cutoff_for(30, AHORA) == AHORA - timedelta(days=30)


class TestQueEsEvidencia:

    def test_una_carpeta_de_puras_imagenes_lo_es(self, tmp_path):
        assert is_evidence_dir(carpeta(tmp_path, "inc_1", dias=1))

    def test_una_carpeta_con_algo_que_no_es_imagen_no_lo_es(self, tmp_path):
        directorio = carpeta(tmp_path, "inc_1", dias=99)
        (directorio / "notas.txt").write_text("algo de alguien")

        # No sabemos qué es ese archivo y no nos toca decidirlo.
        assert not is_evidence_dir(directorio)

    def test_una_carpeta_con_subcarpetas_no_lo_es(self, tmp_path):
        directorio = carpeta(tmp_path, "inc_1", dias=99)
        (directorio / "dentro").mkdir()

        assert not is_evidence_dir(directorio)

    def test_una_carpeta_vacia_no_lo_es(self, tmp_path):
        vacia = tmp_path / "vacia"
        vacia.mkdir()

        assert not is_evidence_dir(vacia)


class TestVencimiento:

    def test_encuentra_solo_lo_vencido(self, tmp_path):
        carpeta(tmp_path, "camara/vieja", dias=40)
        carpeta(tmp_path, "camara/reciente", dias=3)

        vencidas = expired_directories(tmp_path, cutoff_for(30, AHORA))

        assert [d.name for d in vencidas] == ["vieja"]

    def test_una_raiz_que_no_existe_no_revienta(self, tmp_path):
        assert expired_directories(tmp_path / "no_existe", AHORA) == []


class TestBorrado:

    def test_borra_lo_vencido_y_conserva_lo_vigente(self, tmp_path):
        vieja = carpeta(tmp_path, "camara/vieja", dias=40)
        reciente = carpeta(tmp_path, "camara/reciente", dias=3)

        report = purge_evidence(tmp_path, cutoff_for(30, AHORA))

        assert not vieja.exists()
        assert reciente.exists()
        assert report.directories == 1
        assert report.files == 2
        assert report.freed_bytes == 2048

    def test_sin_plazo_no_borra_nada(self, tmp_path):
        antigua = carpeta(tmp_path, "camara/antiquisima", dias=3650)

        report = purge_evidence(tmp_path, cutoff_for(0, AHORA))

        assert antigua.exists()
        assert report.directories == 0

    def test_la_simulacion_informa_pero_no_borra(self, tmp_path):
        vieja = carpeta(tmp_path, "camara/vieja", dias=40)

        report = purge_evidence(tmp_path, cutoff_for(30, AHORA), dry_run=True)

        assert vieja.exists()
        assert report.dry_run is True
        assert report.directories == 1
        assert report.files == 2

    def test_no_toca_una_carpeta_ajena_por_vieja_que_sea(self, tmp_path):
        ajena = carpeta(tmp_path, "camara/ajena", dias=3650)
        (ajena / "config.yaml").write_text("no es nuestro")

        purge_evidence(tmp_path, cutoff_for(30, AHORA))

        assert ajena.exists()
        assert (ajena / "config.yaml").exists()

    def test_es_idempotente(self, tmp_path):
        carpeta(tmp_path, "camara/vieja", dias=40)

        primera = purge_evidence(tmp_path, cutoff_for(30, AHORA))
        segunda = purge_evidence(tmp_path, cutoff_for(30, AHORA))

        assert primera.directories == 1
        assert segunda.directories == 0

    def test_no_borra_la_raiz_ni_aunque_quede_vacia(self, tmp_path):
        carpeta(tmp_path, "camara/vieja", dias=40)

        purge_evidence(tmp_path, cutoff_for(30, AHORA))

        assert tmp_path.exists()
