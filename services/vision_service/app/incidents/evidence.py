import re
from pathlib import Path

import cv2

from ..detection.schemas import Detection
from ..tracking.track_state import TrackState
from .anonymize import (
    FACE_BAND,
    PLATE_BAND,
    Region,
    boxes_from_detections,
    boxes_from_tracks,
    regions_for,
)
from .schemas import IncidentCandidate

# Cuántos píxeles del original colapsan en cada bloque del mosaico, y cuántos
# bloques como mucho a lo ancho. El tope importa: sin él, una franja grande
# —un bus de cerca— saldría con bloques finos y una placa podría sobrevivir.
_PIXEL_BLOCK = 6
_MAX_BLOCKS = 12

# Lo que no sea esto se reemplaza por "_": el id del incidente y el de la
# cámara terminan siendo nombres de carpeta, y ambos los escribe una persona.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _slug(value: str) -> str:
    return _UNSAFE.sub("_", value).strip("_") or "sin_id"


class IncidentEvidence:
    """
    Generates visual evidence for detected incidents.

    For each incident, it saves:

    - original.jpg
    - annotated.jpg

    The annotated image contains the bounding
    boxes of the vehicles involved in the incident.
    """

    def __init__(
        self,
        output_dir: Path,
        scope: str = "",
        anonymize: bool = True,
        plate_band: float = PLATE_BAND,
        face_band: float = FACE_BAND,
    ):
        """
        `scope` separa la evidencia por cámara. Sin él, dos cámaras que
        procesan el mismo número de frame escriben en la misma carpeta y la
        segunda pisa a la primera.

        `anonymize` tapa placas y rostros ANTES de escribir (Ley 1581). Se
        puede apagar para calibrar, pero apagarlo en operación real deja datos
        personales en disco sin base legal para conservarlos.
        """

        self.anonymize = anonymize
        self.plate_band = plate_band
        self.face_band = face_band

        self.output_dir = Path(output_dir)

        if scope:
            self.output_dir = self.output_dir / _slug(scope)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Lo ya guardado en esta corrida. El motor reporta el mismo incidente
        # durante todos los frames que dura, así que sin esto un solo choque
        # deja decenas de carpetas idénticas en disco —una por frame— de las
        # que la base referencia exactamente una. Se guarda la primera, que
        # además es la del instante del impacto.
        self._saved: dict[str, Path] = {}

    def save(
        self,
        frame,
        incident: IncidentCandidate,
        tracks: list[TrackState],
        frame_id: int,
        detections: list[Detection] | None = None,
    ) -> Path:
        """
        `detections` es TODO lo que vio el detector en el frame, tenga track o
        no. Se usa para anonimizar: la evidencia es la imagen entera, así que
        ahí salen las placas de los que pasaban al lado, y un vehículo que
        acaba de entrar al frame aún no tiene track pero su placa se lee igual.
        """

        # ==================================================
        # INCIDENT DIRECTORY
        # ==================================================

        # El frame por sí solo no identifica un incidente: en un mismo frame
        # puede haber dos choques distintos, y el motor reporta el mismo
        # incidente durante varios frames seguidos. Con el id de agrupación
        # delante, cada incidente tiene su carpeta y las repeticiones caen
        # sobre la suya en vez de sobre la del vecino.
        name = (
            _slug(incident.incident_id)
            if incident.incident_id
            else _slug(incident.incident_type)
        )

        # Misma clave que usa el escritor de la base para deduplicar, para que
        # el disco y la tabla cuenten la misma historia. Sin id de agrupación,
        # el tipo y los vehículos involucrados identifican el incidente.
        key = (
            incident.incident_id
            or f"{incident.incident_type}:{sorted(incident.track_ids)}"
        )

        if key in self._saved:
            return self._saved[key]

        incident_dir = (
            self.output_dir
            / f"{name}_{frame_id}"
        )

        incident_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==================================================
        # SAVE ORIGINAL FRAME
        # ==================================================

        # Se anonimiza UNA vez y las dos imágenes salen de ahí. El original
        # sin tapar no llega a tocar el disco en ningún momento: si se
        # escribiera primero y se limpiara después, bastaría con que el
        # proceso muriera en medio para dejarlo ahí para siempre.
        frame = self._anonymized(frame, tracks, detections)

        original_path = (
            incident_dir
            / "original.jpg"
        )

        cv2.imwrite(
            str(original_path),
            frame,
        )

        # ==================================================
        # CREATE ANNOTATED FRAME
        # ==================================================

        annotated = frame.copy()

        incident_track_ids = set(
            incident.track_ids
        )

        # ==================================================
        # DRAW INVOLVED TRACKS
        # ==================================================

        for track in tracks:

            if track.track_id not in incident_track_ids:
                continue

            x1 = int(track.x1)
            y1 = int(track.y1)
            x2 = int(track.x2)
            y2 = int(track.y2)

            # ----------------------------------------------
            # Bounding box
            # ----------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3,
            )

            # ----------------------------------------------
            # Label
            # ----------------------------------------------

            label = (
                f"{track.class_name} "
                f"ID:{track.track_id}"
            )

            cv2.putText(
                annotated,
                label,
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # ----------------------------------------------
            # Center point
            # ----------------------------------------------

            center_x = int(
                track.center[0]
            )

            center_y = int(
                track.center[1]
            )

            cv2.circle(
                annotated,
                (
                    center_x,
                    center_y,
                ),
                5,
                (0, 0, 255),
                -1,
            )

        # ==================================================
        # INCIDENT INFORMATION
        # ==================================================

        title = (
            f"INCIDENT: "
            f"{incident.incident_type}"
        )

        confidence_text = (
            f"Confidence: "
            f"{incident.confidence:.2f}"
        )

        frame_text = (
            f"Frame: {frame_id}"
        )

        # ==================================================
        # DRAW TITLE
        # ==================================================

        cv2.putText(
            annotated,
            title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )

        # ==================================================
        # DRAW CONFIDENCE
        # ==================================================

        cv2.putText(
            annotated,
            confidence_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # ==================================================
        # DRAW FRAME
        # ==================================================

        cv2.putText(
            annotated,
            frame_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        # ==================================================
        # SAVE ANNOTATED FRAME
        # ==================================================

        annotated_path = (
            incident_dir
            / "annotated.jpg"
        )

        cv2.imwrite(
            str(annotated_path),
            annotated,
        )

        self._saved[key] = annotated_path

        return annotated_path

    # ==================================================
    # ANONIMIZACIÓN
    # ==================================================

    def _anonymized(
        self,
        frame,
        tracks: list[TrackState],
        detections: list[Detection] | None,
    ):
        """
        Una copia del frame con placas y rostros tapados.

        Se unen las detecciones y los tracks a propósito, aunque casi siempre
        se solapen: un track puede seguir vivo durante una oclusión sin que
        haya detección ese frame, y ahí sigue habiendo un vehículo que tapar.
        Tapar dos veces la misma franja no cuesta nada; dejarla sin tapar sí.
        """

        if not self.anonymize:
            return frame

        height, width = frame.shape[:2]

        boxes = boxes_from_tracks(tracks)

        if detections:
            boxes = boxes_from_detections(detections) + boxes

        regions = regions_for(
            boxes,
            width,
            height,
            plate_band=self.plate_band,
            face_band=self.face_band,
        )

        if not regions:
            return frame

        safe = frame.copy()

        for region in regions:
            _pixelate(safe, region)

        return safe


def _pixelate(image, region: Region) -> None:
    """
    Mosaico grueso sobre una región, in situ.

    Mosaico y no desenfoque: un gaussiano suave conserva bastante señal como
    para que se pueda intentar revertir. Al reducir y volver a ampliar, la
    información simplemente ya no está en el archivo.
    """

    piece = image[region.y1:region.y2, region.x1:region.x2]

    if piece.size == 0:
        return

    height, width = piece.shape[:2]

    small = cv2.resize(
        piece,
        (
            max(1, min(width // _PIXEL_BLOCK, _MAX_BLOCKS)),
            max(1, min(height // _PIXEL_BLOCK, _MAX_BLOCKS)),
        ),
        interpolation=cv2.INTER_AREA,
    )

    image[region.y1:region.y2, region.x1:region.x2] = cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )