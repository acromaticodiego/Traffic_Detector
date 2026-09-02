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
