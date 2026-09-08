"""
Resumen de un incidente escrito por Gemini.

El motor de visión produce números —IoU, distancia en píxeles, aceleración—
que describen la geometría pero no cuentan qué pasó. Un agente que abre un
caso a las tres de la mañana necesita lo segundo. Esto toma los datos
guardados y la imagen de la evidencia y devuelve un párrafo en español.

Tres decisiones de diseño que conviene no deshacer:

- **La imagen va en la petición.** Es la razón de usar un modelo multimodal:
  sin ella, el resumen no puede hacer más que parafrasear los números que ya
  están en pantalla. Con ella puede decir que uno de los dos vehículos es un
  bus de servicio público, o que la vía está mojada.

- **El modelo NO decide.** Se le pide describir y evaluar, nunca cambiar el
  estado de revisión. Quien confirma o descarta es la persona; el resumen es
  material para esa decisión, no un sustituto de ella.

- **Se le exige admitir cuando no puede ver.** Un frame borroso o un choque
  fuera de cuadro tienen que producir un "no se distingue", no una narración
  segura de algo que el modelo no vio. Es la diferencia entre una ayuda y una
  fuente de errores con aire de autoridad.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..api.config import settings

logger = logging.getLogger(__name__)


class SummaryUnavailable(RuntimeError):
    """No se puede generar el resumen ahora (falta clave, falla la red…)."""


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------

_SYSTEM = """\
Eres un analista de seguridad vial que revisa alertas automáticas de cámaras \
de tránsito en Colombia. Un detector de visión artificial marcó un posible \
incidente y una persona tiene que decidir si fue real.

Escribe en español, en 3 o 4 frases, en tono profesional y directo. Nada de \
listas ni de encabezados: un párrafo corrido.

Tienes que cubrir:
1. Qué se ve en la imagen (vehículos, posición en la vía, contexto).
2. Si lo que muestran los datos y la imagen es consistente con un choque \
real, o si parece una falsa alarma, y por qué.
3. Qué debería verificar la persona antes de confirmarlo.

Reglas que no puedes romper:
- Si la imagen no permite distinguir lo que pasó, dilo con esas palabras. No \
inventes daños, heridos ni maniobras que no se vean.
- No afirmes que hubo un accidente: el detector propone y la persona decide. \
Habla de lo que es "consistente con" o "poco compatible con" un choque.
- No repitas los números tal cual; interprétalos.\
"""


def _describe(incident: dict[str, Any]) -> str:
    """Los datos del incidente, en texto plano para el prompt."""

    data = incident.get("data") or {}

    lineas = [
        f"Tipo detectado: {incident.get('incident_type')}",
        f"Confianza del detector: {round(float(incident.get('confidence', 0)) * 100)}%",
        f"Cámara: {incident.get('camera_id')}",
    ]

    if incident.get("video_t") is not None:
        lineas.append(f"Segundo del video: {incident['video_t']}")

    # Colisión: los dos vehículos y su movimiento.
    if data.get("class_a"):
        lineas += [
            f"Vehículo A: {data.get('class_a')}, velocidad {data.get('speed_a')}, "
            f"aceleración {data.get('acceleration_a')}",
            f"Vehículo B: {data.get('class_b')}, velocidad {data.get('speed_b')}, "
            f"aceleración {data.get('acceleration_b')}",
            f"Separación entre cajas: {data.get('bbox_gap_px')} px "
            f"(tamaño de referencia del vehículo: {data.get('ref_size_px')} px)",
            f"Solape de las cajas (IoU): {data.get('iou')}",
            f"¿Se estaban acercando?: {'sí' if data.get('approaching') else 'no'}",
            f"¿Hubo frenazo o cambio brusco reciente?: "
            f"{'sí' if data.get('recent_crash') else 'no'}",
        ]

    # Vehículo detenido.
    if data.get("still_frames") is not None:
        lineas += [
            f"Clase: {data.get('class')}",
            f"Frames consecutivo detenido: {data.get('still_frames')}",
            f"¿La parada fue brusca?: {'sí' if data.get('abrupt_stop') else 'no'}",
            f"Velocidad máxima que alcanzó antes: {data.get('peak_speed')}",
        ]

    lineas.append(
        "Nota sobre las unidades: las velocidades están en anchos de frame por "
        "segundo, no en km/h; sirven para comparar entre sí, no como magnitud "
        "absoluta."
    )

    return "\n".join(str(line) for line in lineas)


# ----------------------------------------------------------------------


def available() -> bool:
    """Si hay clave configurada. No comprueba que sea válida."""
    return bool(settings.gemini_api_key)


def generate(incident: dict[str, Any], image: Optional[Path] = None) -> str:
    """
    El resumen del incidente, o SummaryUnavailable si no se pudo.

    Nunca devuelve una cadena vacía: o hay texto útil o hay excepción, para
    que no se guarde en la base un resumen en blanco que luego parezca
    generado.
    """

    if not available():
        raise SummaryUnavailable(
            "Falta GEMINI_API_KEY en el .env; el resumen con IA está apagado."
        )

    try:
        # Import perezoso: el servicio y sus tests no deben necesitar el SDK
        # de Google instalado para arrancar.
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise SummaryUnavailable(
            "Falta el paquete google-genai (pip install -r requirements.txt)."
        ) from error

    partes: list[Any] = [_describe(incident)]

    if image is not None and image.is_file():
        try:
            partes.append(
                types.Part.from_bytes(
                    data=image.read_bytes(), mime_type="image/jpeg"
                )
            )
        except Exception as error:  # noqa: BLE001
            # Sin imagen el resumen es más pobre, pero sigue siendo útil:
            # perderla no debe cancelar la generación.
            logger.warning("No se pudo adjuntar la evidencia: %s", error)

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=partes,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                # Bajo a propósito: esto es análisis de un caso real, no
                # redacción creativa. Interesa que dos lecturas del mismo
                # incidente digan lo mismo.
                temperature=0.2,
                max_output_tokens=400,
                http_options=types.HttpOptions(
                    timeout=int(settings.gemini_timeout * 1000)
                ),
            ),
        )

        texto = (response.text or "").strip()

    except Exception as error:  # noqa: BLE001
        # El mensaje de error de un SDK puede traer la URL con la clave.
        # Al log va completo; hacia arriba va algo que se puede mostrar.
        logger.warning("Gemini falló: %s", error, exc_info=True)
        raise SummaryUnavailable(
            f"El servicio de IA no respondió ({type(error).__name__})."
        ) from error

    if not texto:
        raise SummaryUnavailable(
            "El modelo respondió vacío; puede haberlo bloqueado un filtro de "
            "contenido."
        )

    return texto
