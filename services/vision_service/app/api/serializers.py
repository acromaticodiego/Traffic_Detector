"""
Convert the vision dataclasses into plain,
JSON-serializable dicts for the WebSocket stream.

Coordinates are kept in the ORIGINAL video pixel
space (same as the source frame). The frontend
scales them to the rendered <video> size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..events.schemas import Event
from ..incidents.schemas import IncidentCandidate
from ..motion.motion_analyzer import MotionAnalysis
from ..tracking.track_state import TrackState
from .config import REPO_ROOT


def _relative_evidence(raw: Any) -> Optional[str]:
    """
    La ruta de la evidencia, relativa a la raíz del repo y con "/".

    Absoluta no sirve: queda guardada en la base y dentro de un JSON que va al
    navegador, así que filtra el árbol de directorios del servidor y deja de
    resolver apenas el despliegue cambia de máquina. Una ruta de fuera del
    repo se deja como está —es una configuración deliberada— pero normalizada
    a barras para que la base no dependa del sistema operativo.
    """

    if not raw:
        return None

    path = Path(str(raw))

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def track_to_dict(
    track: TrackState,
    motion: Optional[MotionAnalysis] = None,
) -> dict[str, Any]:

    center_x, center_y = track.center

    data: dict[str, Any] = {
        "track_id": track.track_id,
        "class_id": track.class_id,
        "class_name": track.class_name,
        "confidence": round(float(track.confidence), 3),
        "bbox": [
            round(float(track.x1), 1),
            round(float(track.y1), 1),
            round(float(track.x2), 1),
            round(float(track.y2), 1),
        ],
        "center": [round(center_x, 1), round(center_y, 1)],
        "trail": [
            [round(float(x), 1), round(float(y), 1)]
            for (x, y) in track.positions
        ],
    }

    if motion is not None:
        data["speed"] = round(float(motion.speed), 2)
        data["direction"] = round(float(motion.direction), 1)
        data["moving"] = bool(motion.moving)
        data["abrupt_change"] = bool(motion.abrupt_change)
        data["acceleration"] = (
            round(float(motion.acceleration), 2)
            if motion.acceleration is not None
            else None
        )

    return data


def incident_to_dict(
    incident: IncidentCandidate,
    t: float | None = None,
) -> dict[str, Any]:

    return {
        "incident_id": incident.incident_id,
        "incident_type": incident.incident_type,
        "track_ids": list(incident.track_ids),
        "confidence": round(float(incident.confidence), 3),
        "bbox": incident.bbox,
        # `data` es el diccionario libre del motor y se manda tal cual, salvo
        # la ruta de la evidencia: el motor la guarda absoluta y ese campo
        # viaja al navegador y a la base. Una ruta absoluta ahí filtra el
        # árbol de directorios del servidor y además deja de resolver si el
        # despliegue cambia de máquina. La versión utilizable va al primer
        # nivel, abajo.
        "data": _jsonify(
            {k: v for k, v in incident.data.items() if k != "evidence_path"}
        ),
        # Al primer nivel además de dentro de `data`: es lo que necesitan
        # tanto el escritor —que la guarda en su propia columna— como el
        # frontend, y ninguno de los dos debería tener que saber cómo se
        # llama la clave dentro del diccionario libre del motor.
        "evidence_path": _relative_evidence(
            incident.data.get("evidence_path")
        ),
        "t": round(t, 3) if t is not None else None,
    }


def event_to_dict(event: Event) -> dict[str, Any]:

    return {
        "event_type": event.event_type,
        "timestamp": event.timestamp.isoformat(),
        "confidence": round(float(event.confidence), 3),
        "track_ids": list(event.track_ids),
        "data": _jsonify(event.data),
    }


def _jsonify(value: Any) -> Any:
    """
    Best-effort conversion of nested values (tuples,
    sets, numpy scalars) into JSON-safe primitives.
    """

    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # numpy scalar or other -> fall back to float/str
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
