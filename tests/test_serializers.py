"""
Serialización de los mensajes del WebSocket.

Este módulo es el contrato con el frontend (`src/lib/types.ts` lo refleja
campo por campo). Renombrar o quitar una clave aquí no rompe nada del lado
Python: rompe el panel, en silencio y solo en runtime.
"""

import json

import numpy as np

from services.vision_service.app.api.config import REPO_ROOT
from services.vision_service.app.api.serializers import (
    incident_to_dict,
    track_to_dict,
)
from services.vision_service.app.incidents.schemas import IncidentCandidate

from helpers import make_motion, make_track


class TestTrack:

    def test_manda_los_campos_que_el_overlay_dibuja(self):
        track = make_track(1, box=(10.04, 20.06, 50.0, 60.0))

        data = track_to_dict(track)

        assert set(data) == {
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "bbox",
            "center",
            "trail",
        }
        assert data["bbox"] == [10.0, 20.1, 50.0, 60.0]
        assert data["center"] == [30.0, 40.0]
        assert data["trail"] == [[30.0, 40.0]]

    def test_los_campos_de_movimiento_solo_van_si_hay_analisis(self):
        track = make_track(1)

        assert "speed" not in track_to_dict(track)

        con_motion = track_to_dict(track, make_motion(1, speed=3.14159))

        assert con_motion["speed"] == 3.14
        assert con_motion["moving"] is True
        assert con_motion["abrupt_change"] is False

    def test_una_aceleracion_desconocida_viaja_como_null(self):
        data = track_to_dict(make_track(1), make_motion(1, acceleration=None))

        assert data["acceleration"] is None

    def test_el_rastro_acompana_al_track(self):
        track = make_track(1, box=(0, 0, 10, 10))
        track.update(0.9, 10, 10, 20, 20)

        assert track_to_dict(track)["trail"] == [[5.0, 5.0], [15.0, 15.0]]


class TestIncidente:

    def test_manda_los_campos_que_la_lista_de_incidentes_usa(self):
        incident = IncidentCandidate(
            incident_type="possible_collision",
            incident_id="col-1-2",
            track_ids=[1, 2],
            confidence=0.912345,
            bbox={"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0},
            data={"iou": 0.42},
        )

        data = incident_to_dict(incident, t=12.34567)

        assert data["incident_id"] == "col-1-2"
        assert data["incident_type"] == "possible_collision"
        assert data["track_ids"] == [1, 2]
        assert data["confidence"] == 0.912
        assert data["t"] == 12.346

    def test_un_incidente_sin_tiempo_viaja_como_null(self):
        incident = IncidentCandidate(
            incident_type="vehiculo_detenido",
            track_ids=[1],
            confidence=0.85,
        )

        assert incident_to_dict(incident)["t"] is None

    def test_la_evidencia_viaja_relativa_a_la_raiz_del_repo(self):
        # Absoluta filtraría el árbol de directorios del servidor al
        # navegador, y dejaría de resolver apenas el despliegue cambie de
        # máquina: la misma fila apuntando a un disco que ya no existe.
        absoluta = REPO_ROOT / "outputs" / "incidents" / "cam1" / "col-1-2_42"

        incident = IncidentCandidate(
            incident_type="possible_collision",
            track_ids=[1, 2],
            confidence=0.9,
            data={"evidence_path": str(absoluta / "annotated.jpg")},
        )

        data = incident_to_dict(incident)

        assert data["evidence_path"] == (
            "outputs/incidents/cam1/col-1-2_42/annotated.jpg"
        )

    def test_un_incidente_sin_evidencia_viaja_como_null(self):
        # Pasa siempre que VISION_EVIDENCE esté apagada, y en todo incidente
        # anterior a que existiera la captura.
        incident = IncidentCandidate(
            incident_type="vehiculo_detenido",
            track_ids=[1],
            confidence=0.85,
        )

        assert incident_to_dict(incident)["evidence_path"] is None

    def test_los_escalares_de_numpy_no_rompen_el_json(self):
        # El motor mete valores calculados con numpy en `data`; json.dumps no
        # sabe serializarlos y la conexión se caería a mitad del stream.
        incident = IncidentCandidate(
            incident_type="possible_collision",
            track_ids=[1, 2],
            confidence=0.9,
            data={
                "occupancy": np.float64(0.42),
                "vehiculos": np.int64(3),
                "tupla": (np.float32(1.5), 2),
            },
        )

        data = incident_to_dict(incident)

        assert json.dumps(data)
        assert data["data"]["occupancy"] == 0.42
        assert data["data"]["vehiculos"] == 3.0
        assert data["data"]["tupla"] == [1.5, 2]
