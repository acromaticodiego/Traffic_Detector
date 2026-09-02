# Traffic Detector

Plataforma de detección y análisis de tráfico mediante visión artificial,
construida con Python, YOLO y ByteTrack. El proyecto procesa video de tráfico
frame por frame para detectar vehículos y peatones, mantener su seguimiento y
generar eventos e incidentes a partir del comportamiento observado.

Este repo es el **backend / servicio de visión**. El frontend (React) vive en
`../traffic_detector_front`. Arquitectura y hoja de ruta hacia microservicios en
[`ARCHITECTURE.md`](ARCHITECTURE.md).

📌 **Versiones de librerías fijadas:** ver [`STACK.md`](STACK.md) antes de instalar
o actualizar cualquier dependencia.

## Arquitectura actual

```
Video
  ▼
YOLO Detection  →  ByteTrack  →  Track Manager  →  Motion Analyzer
  ▼
Event Engine  →  Incident Engine  →  Evidence
```

Expuesto como API (FastAPI) para el frontend:

- `GET /api/video` — el video (con soporte Range)
- `GET /api/video/meta` — fps / dimensiones / duración
- `WS /ws/inference` — inferencia en vivo por WebSocket (tracks, nivel de tráfico,
  incidentes)

## Funcionalidades

- Detección de objetos con YOLO. Clases: `bus`, `car`, `ciclist`, `monopatin`,
  `motorcycle`, `pedestrian`, `truck`.
- Tracking mediante ByteTrack con `track_id` persistente.
- Análisis de movimiento (velocidad, dirección, cambios bruscos).
- Eventos: `vehicle_detected`, `vehicle_proximity`.
- Incidentes:
  - `possible_collision` — contacto entre vehículos + firma de frenazo; las
    detecciones solapadas de un mismo evento se fusionan en un incidente.
  - `vehiculo_detenido` — vehículo que venía moviéndose y se detiene.
- Nivel de tráfico por frame (`bajo` / `medio` / `alto`).
- Generación de evidencia (imágenes) al detectar un incidente.

La inferencia usa GPU NVIDIA vía CUDA cuando está disponible.

## Tecnologías

Python 3.12 · FastAPI · Uvicorn · OpenCV · PyTorch · Ultralytics YOLO · ByteTrack
· NumPy. Versiones exactas en [`STACK.md`](STACK.md).

## Quickstart

### 1. Backend

```powershell
# desde la raíz del repo, con el .venv
.venv\Scripts\python -m pip install -r services\vision_service\requirements.txt
.venv\Scripts\python -m uvicorn services.vision_service.app.api.main:app --host 0.0.0.0 --port 8000
```

Comprobar: http://localhost:8000/health ·
detalle en [`services/vision_service/README.md`](services/vision_service/README.md).

### 2. Frontend

```bash
cd ../traffic_detector_front
npm install
cp .env.example .env.local
npm run dev            # http://localhost:5173
```

### 3. (Opcional) Exponer con ngrok

```powershell
ngrok http 8000
```

Poner la URL en `traffic_detector_front/.env.local`
(`VITE_API_BASE` / `VITE_WS_BASE`) y `VISION_CORS_ORIGINS` en el backend.

## Pipeline de visión standalone (sin API)

```powershell
.venv\Scripts\python scripts\test_vision_engine.py
```

Requiere `models/detectorfinal.pt` y un video en `videos/input/`. Genera
`outputs/videos/vision_result.mp4` y evidencias en `outputs/incidents/`.

## Estructura

```
traffic_detector/
├── models/                     # pesos YOLO (.pt, git-ignored)
├── videos/input/               # video de entrada (git-ignored)
├── outputs/                    # video anotado + evidencias (git-ignored)
├── scripts/                    # pruebas manuales del pipeline
├── services/vision_service/
│   ├── app/
│   │   ├── detection/ tracking/ motion/ events/ incidents/
│   │   ├── vision_engine.py    # orquestador
│   │   └── api/                # capa FastAPI
│   └── requirements.txt
├── ARCHITECTURE.md
└── STACK.md
```

## Estado

El pipeline de visión está implementado end-to-end (detección, tracking, movimiento,
eventos, incidentes) y expuesto por API para el frontend. El proyecto continúa en
desarrollo hacia un sistema completo de detección y gestión de incidentes
(gateway NestJS, persistencia, audio, fusión multimodal).
