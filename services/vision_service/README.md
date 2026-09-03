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
| `VISION_ROAD_ROI` | vacío = frame completo. Polígono de la calzada en coords normalizadas `"x,y x,y …"` |
| `VISION_ROAD_PERSPECTIVE` | `1.0` (cuántas veces más grande se ve un vehículo abajo de la ROI que arriba) |
| `VISION_TRAFFIC_OCC_MEDIUM` / `VISION_TRAFFIC_OCC_HIGH` | `0.22` / `0.38` (ocupación de la calzada para medio/alto; **calibrar por escena**) |
| `VISION_TRAFFIC_FREE_SPEED` | `0.08` (velocidad de flujo libre en anchos de frame por segundo) |
| `VISION_TRAFFIC_SLOW_RATIO` | `0.35` (por debajo de esta fracción del flujo libre, el vehículo cuenta como detenido) |
| `VISION_TRAFFIC_SMOOTHING` | `0.2` (EMA de ocupación y velocidad; mayor = reacciona más rápido) |
| `VISION_CORS_ORIGINS` | `*` (lista separada por comas) |
| `VISION_HOST` / `VISION_PORT` | `0.0.0.0` / `8000` (solo con el atajo `-m`) |

## Nivel de tráfico

No se calcula contando vehículos: el conteo no sabe qué tan grande es la vía,
así que 6 carros daban "alto" en una avenida vacía. Se calcula con la
**ocupación** de la calzada (fracción del área cubierta por vehículos, la
misma medida que reportan los lazos inductivos) combinada con la **velocidad**
respecto al flujo libre:

| Ocupación | Velocidad | Nivel |
|---|---|---|
| < `OCC_MEDIUM` | cualquiera | `bajo` |
| ≥ `OCC_MEDIUM` | normal | `medio` — denso, pero fluye |
| ≥ `OCC_MEDIUM` | ≥ 60 % detenidos | `alto` — cola |
| ≥ `OCC_HIGH` | < `SLOW_RATIO` × flujo libre | `alto` — congestión |

Ambas señales se suavizan con una EMA para que el nivel no parpadee cuando el
detector pierde una caja un par de frames.

### Calibrar una cámara

Los umbrales dependen de la escena (altura de la cámara, carriles visibles,
qué tan lejos llega la toma). Son dos pasos:

```powershell
# 1. dibujar la calzada con clics -> imprime VISION_ROAD_ROI
.venv\Scripts\python scripts\roi_picker.py

# 2. medir el video con esa ROI -> sugiere los umbrales
.venv\Scripts\python scripts\traffic_calibrate.py --frames 400
```

El polígono va en coordenadas normalizadas `0..1`, así que la misma ROI sirve
si cambia la resolución. La velocidad se expresa en anchos de frame por
segundo por la misma razón: en px/frame dependería de la resolución y los fps,
y el mismo trancón necesitaría umbrales distintos en cada cámara.

Sin `VISION_ROAD_ROI` el servicio mide sobre el frame completo: funciona sin
calibrar, pero el cielo, los árboles y el andén diluyen el porcentaje.

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
