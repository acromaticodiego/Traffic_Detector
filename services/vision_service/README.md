# Vision Service

Pipeline de visión (YOLO + ByteTrack + motion/eventos/incidentes) expuesto como
API para el frontend.

## Requisitos

- Python 3.12, el `.venv` de la raíz del repo
- `models/detectorfinal.pt` y un video en `videos/input/`

## Instalar dependencias

```powershell
# desde la raíz del repo
.venv\Scripts\python -m pip install -r services\vision_service\requirements.txt
```

(En un entorno ya montado, la única lib nueva es `websockets`.)

## Ejecutar la API

```powershell
# desde la raíz del repo (para que resuelva el paquete services.*)
.venv\Scripts\python -m uvicorn services.vision_service.app.api.main:app --host 0.0.0.0 --port 8000
```

o el atajo:

```powershell
.venv\Scripts\python -m services.vision_service.app.api
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | estado, modelo cargado, video detectado |
| GET | `/api/video` | mp4 crudo (soporta `Range` para seek) |
| GET | `/api/video/meta` | fps, dimensiones, nº de frames, duración |
| WS  | `/ws/inference?stride=N` | stream de inferencia (ver `../../ARCHITECTURE.md`) |

## Variables de entorno

| Variable | Default |
|---|---|
| `VISION_MODEL_PATH` | `models/detectorfinal.pt` |
| `VISION_VIDEO_PATH` | primer video de `videos/input/` |
| `VISION_CONFIDENCE` / `VISION_IOU` / `VISION_IMAGE_SIZE` | `0.70` / `0.60` / `640` |
| `VISION_FRAME_STRIDE` | `1` (procesar 1 de cada N frames) |
| `VISION_TRAFFIC_MEDIUM` / `VISION_TRAFFIC_HIGH` | `3` / `6` (nº de vehículos suavizado para nivel medio/alto; ajustar por escena) |
| `VISION_TRAFFIC_SMOOTHING` | `0.2` (EMA del conteo; mayor = reacciona más rápido) |
| `VISION_CORS_ORIGINS` | `*` (lista separada por comas) |
| `VISION_HOST` / `VISION_PORT` | `0.0.0.0` / `8000` (solo con el atajo `-m`) |

## Prueba rápida del WebSocket

```powershell
.venv\Scripts\python scripts\ws_smoke.py   # si se agrega; ver scratchpad
```

## Limitaciones (MVP)

- Una sola sesión de inferencia simultánea (ByteTrack comparte estado con el
  modelo YOLO). Una conexión nueva cancela la anterior.
- La inferencia recorre el video desde el inicio en cada conexión; no sigue el
  `currentTime` del navegador (el frontend cachea los resultados y los sincroniza).
- Sin persistencia: los incidentes se emiten en vivo (el guardado a disco en
  `outputs/` solo lo hace el script `scripts/test_vision_engine.py`).
