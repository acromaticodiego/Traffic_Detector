# Traffic Detector

Plataforma de detección y gestión de incidentes de tráfico mediante visión
artificial. Procesa video de una o varias cámaras frame por frame para detectar
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
| Varias cámaras **en simultáneo** | ⛔ una sesión a la vez — ver [Limitaciones](#limitaciones-conocidas) |
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

Cada cámara es una entidad con su fuente y su calibración propia. Esto no es un
detalle de organización: la ROI, la perspectiva y los umbrales dependen del
ángulo y del encuadre, así que **los valores de una cámara dan niveles
equivocados en otra**.

El registro se resuelve en este orden, cayendo al siguiente si el anterior no
está disponible:

1. Tabla **`cameras`** de PostgreSQL — la fuente de verdad en producción.
2. **`cameras.yaml`** en la raíz — respaldo operativo y formato para versionar la
   calibración en Git.
3. Variables del **`.env`** — una sola cámara, comportamiento antiguo.

Una base caída no deja ciego al servicio: registra un aviso y sigue con el YAML.

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

Tras editar el YAML, aplicarlo a la base:

```powershell
.venv\Scripts\python scripts\seed_cameras.py --dry-run   # ver qué cambiaría
.venv\Scripts\python scripts\seed_cameras.py
```

---

## Nivel de tráfico

**No es un conteo de vehículos.** Contar no sabe qué tan grande es la vía: seis
carros en una avenida de seis carriles daban «alto» con la calzada prácticamente
vacía. El criterio actual combina las dos variables que sí separan los casos que
importan:

| Señal | Qué mide |
|---|---|
| **Ocupación** | Fracción del área de la **calzada** cubierta por vehículos, corregida por perspectiva. |
| **Velocidad** | Qué tan rápido avanzan respecto al flujo libre, en **anchos de frame por segundo** — no px/frame, que dependerían de la resolución y los fps. |

```
ocupación baja                      ->  bajo
ocupación alta + velocidad normal   ->  medio    (denso, pero fluye)
ocupación alta + velocidad baja     ->  alto     (congestión real)
ocupación media + mayoría detenida  ->  alto     (cola)
```

Ambas señales se suavizan con una EMA para que el nivel no parpadee cuando el
detector pierde una caja un par de frames.

### Calibrar una cámara

**1. Dibujar la ROI de la calzada.** Se encierra *todo el asfalto*, como si
estuviera vacío — los vehículos son lo que se mide *contra* esa área, no parte de
ella. Fuera quedan cielo, árboles, andenes y separadores.

```powershell
.venv\Scripts\python scripts\roi_picker.py --camera cra64c_cl78
.venv\Scripts\python scripts\roi_picker.py --camera cra55_cl37 --frame 900
```

Clic en cada vértice, `z` deshace, `r` reinicia, `ENTER` termina. Imprime el
bloque YAML listo para pegar. Las coordenadas van normalizadas (0..1) para que la
ROI siga sirviendo si cambia la resolución.

**2. Medir los umbrales** sobre material real de esa cámara:

```powershell
.venv\Scripts\python scripts\traffic_calibrate.py --camera cra64c_cl78 --frames 400
.venv\Scripts\python scripts\traffic_calibrate.py --camera cra55_cl37 --start 600 --frames 800
```

`--start` existe porque un clip puede cambiar de cámara a mitad. El script
imprime dos juegos de umbrales: uno si el material es de **flujo libre** (nunca
debería dar «alto») y otro si es **representativo** (incluye pico y valle).
Elegir según lo que se grabó.

---

## Incidentes

| Tipo | Cómo se detecta |
|---|---|
| `possible_collision` | Dos vehículos en **contacto** (cajas tocándose / IoU alto, relativo al tamaño del vehículo) junto con una **firma de frenazo** (cambio brusco reciente). Las detecciones solapadas en espacio y tiempo se **fusionan en un solo incidente** (`incident_id` estable, une los objetos involucrados y toma la confianza máxima). |
| `vehiculo_detenido` | Un vehículo que venía moviéndose y queda a velocidad ~0 (parada brusca, o parada muy prolongada). |

**Severidad** (según confianza): ≥ 80 % → *Confirmado*, 60–80 % → *Por confirmar*,
< 60 % → no se emite. En el frontend, además, el marcador sobre el video solo se
dibuja desde 90 % y se queda fijo desde 95 %.

### Persistencia

Los incidentes se guardan en la tabla `incidents`, **uno por evento y no uno por
frame**: el motor reporta el mismo choque en frames consecutivos y el escritor
deduplica por `incident_id`.

La escritura ocurre en un **hilo aparte con cola acotada**. El hilo de visión
nunca espera a PostgreSQL: si la base se pone lenta, se descartan filas de
histórico antes que frenar la inferencia en vivo. `/health` reporta cuántas van
escritas y cuántas descartadas.

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

Omitir `camera` toma la primera del registro.

La paginación de incidentes es por `before_id` y no por *offset* a propósito: con
inserciones llegando en vivo, un offset se salta filas o las repite entre
páginas.

### Protocolo del WebSocket

```
{"type": "meta",     "protocol", "camera", "fps", "frame_count", "width",
                     "height", "stride", "traffic_thresholds", "road_roi"}
{"type": "frame",    "frame_id", "t", "tracks", "incidents", "events", "traffic"}
{"type": "incident", "incident_type", "confidence", "track_ids", "bbox", "data", "t"}
{"type": "done",     "frames", "processed"}
{"type": "error",    "message"}
```

`protocol` (ver `api/protocol.py`) sube cuando cambia la **forma** de los
mensajes. El frontend compara contra la suya y avisa si no coinciden: dos
procesos, uno obsoleto, son indistinguibles desde afuera y ambos responden
alegremente. Junto con `config_fingerprint` en `/health`, es lo que delata un
servicio viejo que quedó vivo.

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

> Si el servicio se comporta como una versión anterior, revisar que no haya
> **dos** procesos en el puerto: Windows deja convivir uno atado a `127.0.0.1` y
> otro a `0.0.0.0`, y el específico gana, así que reiniciar puede estar matando
> siempre al equivocado.
>
> ```powershell
> Get-NetTCPConnection -LocalPort 8000 -State Listen | Select OwningProcess, LocalAddress
> ```

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
para que no existan dos fuentes de verdad ni se versione la contraseña.

```powershell
.venv\Scripts\python -m alembic upgrade head                        # aplicar
.venv\Scripts\python -m alembic revision --autogenerate -m "motivo" # generar
.venv\Scripts\python -m alembic downgrade -1                        # revertir
```

Revisar siempre la migración generada antes de aplicarla: el autogenerate no
detecta renombrados, los ve como borrar y crear una columna.

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
├── cameras.yaml                # registro de cámaras (respaldo del de la base)
├── alembic.ini
├── ARCHITECTURE.md
└── STACK.md
```

---

## Limitaciones conocidas

**Una sesión de inferencia a la vez.** ByteTrack guarda estado sobre el modelo
YOLO compartido, así que dos sesiones concurrentes se corromperían los tracks
mutuamente. Cambiar de cámara reinicia el pipeline, y una conexión nueva cancela
la anterior. Correr varias cámaras en paralelo exige separar la inferencia del
WebSocket (un worker por cámara), que es el siguiente paso de arquitectura.

**Sin tests automatizados.** Los `scripts/test_*.py` son demos visuales, no
pruebas: ninguno define funciones `test_` ni usa pytest. Es la deuda más
importante del repo, sobre todo en la aritmética de ocupación y umbrales, que se
cambia a mano y sin red.

**El frontend no lee el histórico.** `GET /api/incidents` funciona, pero la lista
de la interfaz sigue viviendo en memoria del navegador: al recargar se pierde.

---

## Estado

El pipeline de visión está implementado end-to-end (detección, tracking,
movimiento, eventos, incidentes), expuesto por API para el frontend, con
calibración por cámara y persistencia en PostgreSQL. El proyecto continúa hacia
un sistema completo de detección y gestión de incidentes (worker por cámara,
gateway NestJS, audio, fusión multimodal) — ver `ARCHITECTURE.md`.
