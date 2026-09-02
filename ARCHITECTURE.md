# Traffic Detector — Arquitectura

## Estado actual (MVP)

```
┌──────────────────────────┐        ┌─────────────────────────────┐
│  Frontend (React + Vite) │  HTTP  │  Vision Service (FastAPI)    │
│  traffic_detector_front  │ <────> │  services/vision_service     │
│                          │  WS    │                             │
│  - <video> del mp4 crudo │ <────> │  GET /api/video  (mp4+Range) │
│  - <canvas> overlay      │        │  GET /api/video/meta        │
│  - lista de incidentes   │        │  WS  /ws/inference          │
│  - mapa (stub)           │        │      └─ VisionEngine        │
└──────────────────────────┘        │         YOLO + ByteTrack    │
                                    │         TrackManager        │
             ngrok http 8000        │         MotionAnalyzer      │
        (un solo túnel gratis) ─────▶         EventEngine         │
                                    │         IncidentEngine      │
                                    └─────────────────────────────┘
```

El frontend corre local y apunta al backend (local o vía ngrok con
`VITE_API_BASE` / `VITE_WS_BASE`). La inferencia se ejecuta best-effort sobre el
video quemado y se transmite como JSON por WebSocket; el navegador reproduce el
mp4 y dibuja el overlay sincronizado por tiempo (`currentTime * fps`).

### Contrato del WebSocket `/ws/inference`

```
{"type":"meta","fps","frame_count","width","height","stride","traffic_thresholds":{"medium","high"}}
{"type":"frame","frame_id","t","tracks":[Track],"incidents":[Incident],"events":[Event],
                "traffic":{"level":"bajo|medio|alto","vehicles","people","score"}}
{"type":"incident", ...Incident, "t"}
{"type":"done","frames","processed"}
{"type":"error","message"}
```

`Track = {track_id, class_id, class_name, confidence, bbox:[x1,y1,x2,y2],
center:[x,y], trail:[[x,y]], speed, direction, moving, abrupt_change, acceleration}`
(coordenadas en el espacio de píxeles del video original).

Limitación: **una sola sesión de inferencia a la vez** (ByteTrack guarda estado
en el objeto YOLO compartido). Una nueva conexión cancela la anterior.

## Destino (microservicios)

```
                     ┌─────────────┐
                     │  Frontend   │  React
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ API Gateway │  NestJS   (pendiente)
                     └──────┬──────┘
          ┌─────────────────┼──────────────────┐
          ▼                 ▼                  ▼
   Vision Service     Incident Service   Transcription Service
   FastAPI  ✅         NestJS  (pend.)    Deepgram (pend.)
          │                 │
          ▼                 ▼
   YOLO/ByteTrack       PostgreSQL
          │                 │
          └────────┬────────┘
                   ▼
            Fusion / AI Service
              Gemini (pend.)

Infra (pend.): PostgreSQL · Redis · MinIO · NATS · Docker Compose
```

Ruta de evolución:

1. **Ahora:** `vision_service` (FastAPI) + frontend directo.
2. Añadir `api_gateway` (NestJS): el frontend deja de hablar directo con visión;
   el gateway agrega auth, un solo dominio y proxy de WS.
3. `incident_service` (NestJS + PostgreSQL): persiste incidentes y evidencias
   (hoy solo se emiten en vivo y se guardan a disco en `outputs/`).
4. `NATS`: el mensaje `incident` del WS pasa a ser un evento publicado en NATS
   que consumen incident/fusion services.
5. `fusion_service` (Gemini) y `transcription_service` (Deepgram).
6. `infra/docker-compose.yml` con Postgres/Redis/MinIO/NATS.

## Carpetas

```
traffic_detector/                 # backend / visión
  models/                         # pesos YOLO (.pt, git-ignored)
  videos/input/                   # video quemado (git-ignored)
  outputs/                        # video anotado + evidencias (git-ignored)
  scripts/                        # pruebas manuales del pipeline
  services/vision_service/
    app/
      detection/ tracking/ motion/ events/ incidents/   # pipeline (Python puro)
      vision_engine.py                                  # orquestador
      api/                                              # capa FastAPI (nueva)
    requirements.txt
  ARCHITECTURE.md

traffic_detector_front/           # frontend (repo separado)
  src/
    components/  hooks/  lib/  state/
```
