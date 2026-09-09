"""
Versión del protocolo que habla el WebSocket.

Existe porque un servicio desactualizado y uno al día son indistinguibles
desde afuera: los dos aceptan la conexión y los dos mandan mensajes con la
misma pinta. La única forma de notar la diferencia era inspeccionar el payload
campo por campo.

Súbela cuando cambie la FORMA de los mensajes (campos nuevos que el frontend
dé por hechos, campos renombrados o eliminados). El frontend la compara contra
la suya y avisa si no coinciden.

Historial:
    1 -> nivel de tráfico por conteo de vehículos
    2 -> ocupación de calzada + velocidad, ROI por cámara, meta.camera
    3 -> multi-cámara: meta.frame_count puede ser null (fuente en vivo, que
         no tiene final que contar)
"""

PROTOCOL_VERSION = 3
