# Traffic Detector

Plataforma de detección y gestión de incidentes de tráfico mediante visión
artificial. Procesa video de una cámara frame por frame para detectar vehículos y
peatones, seguirlos, analizar su comportamiento y generar **incidentes** (choques,
vehículos detenidos) con evidencia, mostrándolos en una interfaz web en vivo.

- **Este repo** = backend / servicio de visión (Python).
- **Frontend** (React) = `../traffic_detector_front` (repo aparte).
- **Arquitectura y hoja de ruta a microservicios:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Versiones de librerías fijadas:** [`STACK.md`](STACK.md) — leer antes de
  instalar o actualizar cualquier dependencia.

---

## Estado actual

| Área | Estado |
|---|---|
| Pipeline de visión (YOLO → ByteTrack → movimiento → eventos → incidentes) | ✅ funcional |
| API FastAPI (video + inferencia en vivo por WebSocket) | ✅ funcional |
| Nivel de tráfico por frame | ✅ |
| Detección de colisión + vehículo detenido, con fusión de detecciones | ✅ |
| Frontend React (mapa + paneles flotantes + overlay en vivo) | ✅ funcional |
| Gateway NestJS, persistencia, audio (Deepgram), fusión (Gemini) | ⏳ pendiente — ver `ARCHITECTURE.md` |

---

## Arquitectura

```
┌───────────────────────────┐        ┌──────────────────────────────┐
│  Frontend (React + Vite)  │  HTTP  │  Vision Service (FastAPI)     │
│  traffic_detector_front   │ <────> │  services/vision_service      │
│                           │  WS    │                              │
│  - mapa a pantalla completa       │  GET /api/video   (mp4 + Range)│
│  - paneles flotantes de vidrio    │  GET /api/video/meta          │
│  - <video> + overlay canvas       │  WS  /ws/inference            │
│  - lista + detalle de incidentes  │        └─ VisionEngine        │
└───────────────────────────┘        │           YOLO + ByteTrack   │
                                     │           TrackManager       │
        ngrok http 8000  ───────────▶│           MotionAnalyzer     │
      (un túnel gratis, opcional)    │           EventEngine        │
                                     │           IncidentEngine     │
                                     │           IncidentEvidence   │
                                     └──────────────────────────────┘
```

### Pipeline de visión (`services/vision_service/app/`)

```
frame
  ▼
YOLO + ByteTrack        detection/ + tracking/tracker.py   (una sola inferencia por frame)
  ▼
TrackManager            tracking/track_manager.py          estado temporal de cada track
  ▼
MotionAnalyzer          motion/                            velocidad, dirección, cambios bruscos
  ▼
EventEngine             events/                            vehicle_detected, vehicle_proximity
  ▼
IncidentEngine          incidents/incident_engine.py       colisión, vehículo detenido, clustering
  ▼
IncidentEvidence        incidents/evidence.py              guarda imágenes del incidente
  ▼
VisionResult            vision_engine.py                   orquestador
```

Clases del modelo: `bus`, `car`, `ciclist`, `monopatin`, `motorcycle`,
`pedestrian`, `truck`. Inferencia en GPU NVIDIA vía CUDA cuando está disponible.

### Incidentes

| Tipo | Cómo se detecta |
|---|---|
| `possible_collision` | Dos vehículos en **contacto** (cajas tocándose / IoU alto, relativo al tamaño del vehículo) junto con una **firma de frenazo** (cambio brusco reciente). Las detecciones solapadas en espacio y tiempo se **fusionan en un solo incidente** (`incident_id` estable, une los objetos involucrados y toma la confianza máxima). |
| `vehiculo_detenido` | Un vehículo que venía moviéndose y queda a velocidad ~0 (parada brusca, o parada muy prolongada). |

**Severidad** (según confianza): ≥ 80 % → *Confirmado*, 60–80 % → *Por confirmar*,
< 60 % → no se emite.

### Nivel de tráfico

Conteo de vehículos activos por frame con suavizado EMA → `bajo` / `medio` / `alto`
(umbrales configurables).
└── README.md
```
---

## Calidad (lint y tests)

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m ruff check .      # lint
.venv\Scripts\python -m pytest            # tests
```

`requirements-dev.txt` a propósito **no** trae torch, ultralytics ni opencv:
los tests cubren la lógica pura —nivel de tráfico, ROI de la calzada, análisis
de movimiento, motor de incidentes, registro de cámaras y serialización del
WebSocket—, que es la que decide qué se reporta y no necesita ni el modelo ni
el video. Así la suite corre en menos de un segundo y el CI en menos de un
minuto, en vez de bajar un par de gigas de wheels.

Un test lee el `cameras.yaml` real del repo y comprueba que cada entrada
parsea, que las ROI son polígonos válidos y que los umbrales van en orden. Una
calibración mal pegada, si no, no se nota hasta que el servicio ya está
sirviendo el nivel equivocado.

`ruff` se configura en `pyproject.toml` con un conjunto de reglas conservador
(pyflakes y errores de pycodestyle): atrapa imports muertos, nombres
inexistentes y sintaxis inválida, sin obligar a reescribir el estilo del código
ya escrito.

`.github/workflows/ci.yml` corre lint y tests en cada push a `master` y en cada
pull request.

---

Imagen del funcionamiento
<img width="1917" height="1028" alt="image" src="https://github.com/user-attachments/assets/c59d558c-1c4a-4106-947c-42972a250b59" />
<img width="1916" height="1030" alt="image" src="https://github.com/user-attachments/assets/0686ac31-cc8d-4ee2-9c25-5eb39be9cf52" />


