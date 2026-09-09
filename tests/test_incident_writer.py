"""
La cola de escritura de incidentes.

Lo que se prueba aquí es la deduplicación en memoria, que es la primera de las
dos capas: evita encolar el mismo incidente una vez por frame. La segunda —la
restricción única de la tabla, que es la que impide de verdad que reprocesar un
video duplique su histórico— vive en la base y no se puede comprobar sin ella.

Estos tests no tocan Postgres: `submit` solo encola, y el hilo escritor no
arranca hasta que se llama a `start()`.
"""

from services.vision_service.app.db.incident_writer import IncidentWriter

HUELLA = "a1b2c3d4e5f60718"
OTRA_HUELLA = "0f1e2d3c4b5a6978"


def incidente(**campos):
    base = {
        "incident_type": "possible_collision",
        "incident_id": "col-41-2",
        "confidence": 0.85,
        "track_ids": [41, 2],
    }
    base.update(campos)
    return base


def encolados(writer):
    return list(writer._queue.queue)


class TestDeduplicacionEnMemoria:

    def test_el_mismo_incidente_solo_se_encola_una_vez(self):
        """El motor lo reporta en cada frame mientras dura; la tabla guarda
        uno por evento, no uno por frame."""

        writer = IncidentWriter()

        writer.submit("cra55_cl37", incidente(), 41, HUELLA)
        writer.submit("cra55_cl37", incidente(), 42, HUELLA)
        writer.submit("cra55_cl37", incidente(), 43, HUELLA)

        assert len(encolados(writer)) == 1

    def test_la_huella_viaja_con_el_incidente(self):
        writer = IncidentWriter()

        writer.submit("cra55_cl37", incidente(), 41, HUELLA)

        camera_id, source_key, _, frame_id = encolados(writer)[0]

        assert (camera_id, source_key, frame_id) == ("cra55_cl37", HUELLA, 41)

    def test_otra_camara_con_el_mismo_id_de_agrupacion_no_se_pisa(self):
        """Los ids salen de los track_id, que cada cámara numera por su
        cuenta: `col-41-2` en dos cámaras son dos choques distintos."""

        writer = IncidentWriter()

        writer.submit("cra55_cl37", incidente(), 41, HUELLA)
        writer.submit("cra64c_cl78", incidente(), 41, HUELLA)

        assert len(encolados(writer)) == 2

    def test_otra_pasada_del_pipeline_vuelve_a_encolar(self):
        """Cambiar el video de una cámara empieza un histórico nuevo, aunque
        el motor reutilice los mismos ids de agrupación."""

        writer = IncidentWriter()

        writer.submit("cra55_cl37", incidente(), 41, HUELLA)
        writer.submit("cra55_cl37", incidente(), 41, OTRA_HUELLA)

        assert len(encolados(writer)) == 2

    def test_la_memoria_ya_no_se_vacia_entre_sesiones(self):
        """Era el origen del problema: cada reconexión reprocesaba el video y
        volvía a insertar el histórico entero porque la memoria se limpiaba
        al arrancar la sesión."""

        writer = IncidentWriter()

        writer.submit("cra55_cl37", incidente(), 41, HUELLA)

        # El hilo escritor se lleva lo encolado; lo que no puede olvidarse es
        # lo ya visto, o la sesión siguiente lo encolaría otra vez.
        writer._queue.queue.clear()

        writer.submit("cra55_cl37", incidente(), 41, HUELLA)

        assert encolados(writer) == []
        assert not hasattr(writer, "reset_seen")


class TestIncidentesSinIdDeAgrupacion:
    """Un incidente sin id se deduplica por tipo y vehículos: es lo único que
    lo identifica, y sin ello se guardaría uno por frame."""

    def test_se_deduplican_por_tipo_y_tracks(self):
        writer = IncidentWriter()

        writer.submit("cra55_cl37", incidente(incident_id=None), 41, HUELLA)
        writer.submit("cra55_cl37", incidente(incident_id=None), 42, HUELLA)

        assert len(encolados(writer)) == 1

    def test_otros_vehiculos_son_otro_incidente(self):
        writer = IncidentWriter()

        writer.submit(
            "cra55_cl37", incidente(incident_id=None, track_ids=[41, 2]), 41, HUELLA
        )
        writer.submit(
            "cra55_cl37", incidente(incident_id=None, track_ids=[70, 77]), 41, HUELLA
        )

        assert len(encolados(writer)) == 2
