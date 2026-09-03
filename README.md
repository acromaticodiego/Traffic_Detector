# Traffic Detector

Plataforma de detección y gestión de incidentes de tráfico mediante visión
artificial. Procesa video de cámaras de vía frame por frame para detectar
vehículos y peatones, seguirlos, analizar su comportamiento y generar
**incidentes** (choques, vehículos detenidos) con evidencia, mostrándolos en una
interfaz web en vivo y guardándolos en PostgreSQL.

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
| Registro de cámaras con calibración independiente por cámara | ✅ funcional |
| Nivel de tráfico por ocupación de calzada y velocidad | ✅ funcional |
| Detección de colisión + vehículo detenido, con fusión de detecciones | ✅ funcional |
| Persistencia en PostgreSQL (cámaras + incidentes) | ✅ funcional |
| Frontend React (mapa + paneles flotantes + overlay + registro de actividad) | ✅ funcional |
| Varias cámaras en simultáneo | ⛔ una sesión de inferencia a la vez |
| Histórico de incidentes en la interfaz | ⏳ el endpoint existe, el frontend aún no lo consume |
| Gateway NestJS, audio (Deepgram), fusión (Gemini) | ⏳ pendiente — ver `ARCHITECTURE.md` |

---

## Arquitectura

```
┌───────────────────────────┐        ┌──────────────────────────────┐
│  Frontend (React + Vite)  │  HTTP  │  Vision Service (FastAPI)    │
│  traffic_detector_front   │ <────> │  services/vision_service     │
│                           │  WS    │                              │
│  - mapa a pantalla completa        │  GET /api/cameras            │
│  - selector de cámara              │  GET /api/video   (mp4+Range)│
│  - paneles flotantes de vidrio     │  GET /api/video/meta         │
│  - <video> + overlay canvas        │  GET /api/incidents          │
│  - lista + detalle de incidentes   │  WS  /ws/inference           │
│  - registro de actividad           │        └─ VisionEngine       │
└───────────────────────────┘        │           YOLO + ByteTrack   │
                                     │           TrackManager       │
        ngrok http 8000  ───────────▶│           MotionAnalyzer     │
      (un túnel gratis, opcional)    │           EventEngine        │
                                     │           IncidentEngine     │
                                     │           IncidentEvidence   │
                                     │           TrafficLevel       │
                                     └───────────────┬──────────────┘
                                                     │ hilo escritor
                                                     ▼
                                            ┌──────────────────┐
                                            │   PostgreSQL     │
                                            │   cameras        │
                                            │   incidents      │
                                            └──────────────────┘
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
TrafficLevelEstimator   api/traffic_level.py               ocupación de calzada + velocidad
  ▼
VisionResult            vision_engine.py                   orquestador
```

Clases del modelo: `bus`, `car`, `ciclist`, `monopatin`, `motorcycle`,
`pedestrian`, `truck`. Inferencia en GPU NVIDIA vía CUDA cuando está disponible.

---

## Cámaras

El sistema trabaja sobre un **registro de cámaras**. Cada cámara tiene su fuente
(archivo de video o URL RTSP), su ubicación en el mapa y su propia calibración:
el polígono de la calzada, la corrección de perspectiva y los umbrales de nivel
de tráfico.

La calibración es por cámara porque depende del ángulo y del encuadre: el mismo
polígono aplicado a otra cámara cubre la vía equivocada, y los mismos umbrales
producen niveles distintos.

El registro se resuelve en este orden, cayendo al siguiente si el anterior no
está disponible:

1. Tabla **`cameras`** de PostgreSQL — la fuente de verdad.
2. **`cameras.yaml`** en la raíz — permite versionar la calibración en Git y
   sirve de respaldo si la base no responde.
3. Variables del **`.env`** — configuración de una sola cámara.

```yaml
# cameras.yaml
cameras:
  - id: cra64c_cl78
    name: "Cra 64C × Cl 78"
    source: "videos/input/mi_video.mp4"     # o una URL RTSP
    lat: 6.2716
    lng: -75.5900
    roi: "0.0021,0.9891 0.0031,0.5043 ..."  # polígono de la calzada, 0..1
    perspective: 2.5
    thresholds:
      occupancy_medium: 0.24
      occupancy_high: 0.41
      free_speed: 0.065
```

Para aplicar el YAML a la base:

```powershell
.venv\Scripts\python scripts\seed_cameras.py --dry-run   # ver qué cambiaría
.venv\Scripts\python scripts\seed_cameras.py
```

---

## Nivel de tráfico

El nivel se calcula combinando dos señales, que juntas distinguen una vía llena
que avanza de una congestionada de verdad:

| Señal | Qué mide |
|---|---|
| **Ocupación** | Fracción del área de la **calzada** cubierta por vehículos, corregida por perspectiva. Al medir sobre el área de la vía y no contar unidades, el resultado no depende de cuántos carriles tenga. |
| **Velocidad** | Qué tan rápido avanzan respecto al flujo libre, expresada en **anchos de frame por segundo**, de modo que el mismo umbral sirve para cámaras con distinta resolución y fps. |

```
ocupación baja                      ->  bajo
ocupación alta + velocidad normal   ->  medio    (denso, pero fluye)
ocupación alta + velocidad baja     ->  alto     (congestión real)
ocupación media + mayoría detenida  ->  alto     (cola)
```

Ambas señales se suavizan con una EMA para que el nivel no parpadee cuando el
detector pierde una caja durante un par de frames.

### Calibrar una cámara

**1. Dibujar la ROI de la calzada.** Se encierra *todo el asfalto por donde
circulan los vehículos*, como si la vía estuviera vacía: es el área contra la que
se mide la ocupación. Quedan fuera el cielo, los árboles, los andenes y los
separadores.

```powershell
.venv\Scripts\python scripts\roi_picker.py --camera cra64c_cl78
.venv\Scripts\python scripts\roi_picker.py --camera cra55_cl37 --frame 900
```

Clic en cada vértice, `z` deshace, `r` reinicia, `ENTER` termina. Imprime el
bloque YAML listo para pegar. Las coordenadas se guardan normalizadas (0..1),
así que la ROI sigue siendo válida si cambia la resolución del video.

**2. Medir los umbrales** sobre material real de esa cámara:

```powershell
.venv\Scripts\python scripts\traffic_calibrate.py --camera cra64c_cl78 --frames 400
.venv\Scripts\python scripts\traffic_calibrate.py --camera cra55_cl37 --start 600 --frames 800
```

`--start` permite calibrar un tramo concreto, útil cuando un clip cambia de
cámara a mitad. El script imprime dos juegos de umbrales: uno para material de
**flujo libre** (que nunca debería dar «alto») y otro para material
**representativo** (que incluye pico y valle).

---

## Incidentes

| Tipo | Cómo se detecta |
|---|---|
| `possible_collision` | Dos vehículos en **contacto** (cajas tocándose / IoU alto, relativo al tamaño del vehículo) junto con una **firma de frenazo** (cambio brusco reciente). Las detecciones solapadas en espacio y tiempo se **fusionan en un solo incidente** (`incident_id` estable, une los objetos involucrados y toma la confianza máxima). |
| `vehiculo_detenido` | Un vehículo que venía moviéndose y queda a velocidad ~0 (parada brusca, o parada muy prolongada). |

**Severidad** (según confianza): ≥ 80 % → *Confirmado*, 60–80 % → *Por confirmar*,
< 60 % → no se emite. En el frontend, el marcador sobre el video se dibuja desde
90 % y se queda fijo desde 95 %.

Cada incidente se guarda en la tabla `incidents` **una vez por evento**, no una
vez por frame: el motor lo reporta durante varios frames consecutivos y el
escritor deduplica por `incident_id`.

La escritura ocurre en un hilo aparte con cola acotada, de modo que el hilo de
visión nunca queda esperando a la base. Si PostgreSQL se pone lento se descartan
filas de histórico antes que frenar la inferencia en vivo; `/health` reporta
cuántas se han escrito y cuántas se han descartado.

---

## API

| Endpoint | Qué hace |
|---|---|
| `GET /health` | Estado, versión de protocolo, origen del registro de cámaras y huella de configuración. |
| `GET /api/cameras` | Registro de cámaras (sin la ruta ni las credenciales de la fuente). |
| `GET /api/cameras/{id}` | Una cámara. |
| `GET /api/video?camera=<id>` | El mp4, con soporte de HTTP Range para el seek del `<video>`. |
| `GET /api/video/meta?camera=<id>` | fps, dimensiones, número de frames, duración. |
| `GET /api/incidents` | Histórico. Filtros: `camera`, `type`, `min_confidence`, `since`. Paginación por `before_id`. |
| `WS /ws/inference?camera=<id>&stride=N` | Inferencia en vivo. |

Omitir `camera` toma la primera del registro. La paginación de incidentes usa
`before_id` en lugar de *offset* porque las inserciones llegan en vivo y un
offset se saltaría filas entre páginas.

### Protocolo del WebSocket

```
{"type": "meta",     "protocol", "camera", "fps", "frame_count", "width",
                     "height", "stride", "traffic_thresholds", "road_roi"}
{"type": "frame",    "frame_id", "t", "tracks", "incidents", "events", "traffic"}
{"type": "incident", "incident_type", "confidence", "track_ids", "bbox", "data", "t"}
{"type": "done",     "frames", "processed"}
{"type": "error",    "message"}
```

`protocol` (ver `api/protocol.py`) identifica la forma de los mensajes. El
frontend la compara con la suya y avisa si no coinciden, lo que permite detectar
un servicio y una interfaz de versiones distintas — algo que de otro modo no da
ningún error visible.

---

## Puesta en marcha

### 1. Dependencias

```powershell
# desde la raíz del repo, con el .venv
.venv\Scripts\python -m pip install -r services\vision_service\requirements.txt
```

### 2. PostgreSQL

```powershell
psql -U postgres -c "CREATE DATABASE traffic_detector"
```

Copiar `.env.example` a `.env` y poner la cadena de conexión:

```ini
VISION_DATABASE_URL=postgresql+psycopg://usuario:contraseña@localhost:5432/traffic_detector
```

> `.env` está en `.gitignore`, así que la contraseña no se versiona, y `/health`
> la devuelve redactada. El sufijo `+psycopg` selecciona el driver psycopg 3;
> sin él SQLAlchemy busca psycopg2, que no está instalado.

Crear el esquema y cargar las cámaras:

```powershell
.venv\Scripts\python -m alembic upgrade head
.venv\Scripts\python scripts\seed_cameras.py
```

### 3. Backend

```powershell
.venv\Scripts\python -m uvicorn services.vision_service.app.api.main:app --host 0.0.0.0 --port 8000
```

Comprobar: http://localhost:8000/health ·
detalle en [`services/vision_service/README.md`](services/vision_service/README.md).

### 4. Frontend

```bash
cd ../traffic_detector_front
npm install
cp .env.example .env.local
npm run dev            # http://localhost:5173
```

### 5. (Opcional) Exponer con ngrok

```powershell
ngrok http 8000
```

Poner la URL en `traffic_detector_front/.env.local`
(`VITE_API_BASE` / `VITE_WS_BASE`) y `VISION_CORS_ORIGINS` en el backend.

---

## Migraciones

Alembic toma la URL de conexión de `VISION_DATABASE_URL`, no de `alembic.ini`,
para no duplicar la configuración ni versionar la contraseña.

```powershell
.venv\Scripts\python -m alembic upgrade head                        # aplicar
.venv\Scripts\python -m alembic revision --autogenerate -m "motivo" # generar
.venv\Scripts\python -m alembic downgrade -1                        # revertir
```

Conviene revisar la migración generada antes de aplicarla: el autogenerate
interpreta un renombrado de columna como un borrar más un crear.

---

## Pipeline de visión standalone (sin API)

```powershell
.venv\Scripts\python scripts\test_vision_engine.py
```

Requiere `models/detectorfinal.pt` y un video en `videos/input/`. Genera
`outputs/videos/vision_result.mp4` y evidencias en `outputs/incidents/`.

---

## Estructura

```
traffic_detector/
├── models/                     # pesos YOLO (.pt, git-ignored)
├── videos/input/               # video de entrada (git-ignored)
├── outputs/                    # video anotado + evidencias (git-ignored)
├── migrations/                 # Alembic (env.py + versions/)
├── scripts/
│   ├── roi_picker.py           # dibujar la ROI de la calzada
│   ├── traffic_calibrate.py    # medir umbrales de nivel de tráfico
│   ├── seed_cameras.py         # cameras.yaml -> tabla cameras
│   └── test_*.py               # demos manuales del pipeline
├── services/vision_service/
│   ├── app/
│   │   ├── detection/ tracking/ motion/ events/ incidents/
│   │   ├── vision_engine.py    # orquestador
│   │   ├── db/                 # SQLAlchemy: modelos, sesión, escritor
│   │   └── api/                # capa FastAPI
│   │       ├── cameras.py      # registro de cámaras
│   │       ├── road_roi.py     # polígono de calzada y ocupación
│   │       ├── traffic_level.py
│   │       ├── protocol.py
│   │       └── routes/
│   └── requirements.txt
├── cameras.yaml                # registro de cámaras
├── alembic.ini
├── ARCHITECTURE.md
└── STACK.md
```

---

## Alcance

El servicio corre **una sesión de inferencia a la vez**: ByteTrack mantiene
estado sobre el modelo YOLO compartido, por lo que dos sesiones concurrentes se
corromperían los tracks. Cambiar de cámara reinicia el pipeline y una conexión
nueva cancela la anterior. Procesar varias cámaras en paralelo requiere separar
la inferencia del WebSocket con un worker por cámara, previsto en
`ARCHITECTURE.md`.

Los `scripts/test_*.py` son demos visuales del pipeline, no pruebas
automatizadas; el proyecto todavía no tiene suite de tests.

El pipeline está implementado end-to-end (detección, tracking, movimiento,
eventos, incidentes), expuesto por API para el frontend, con calibración por
cámara y persistencia en PostgreSQL. El desarrollo continúa hacia un sistema
completo de detección y gestión de incidentes: worker por cámara, gateway
NestJS, audio y fusión multimodal.
