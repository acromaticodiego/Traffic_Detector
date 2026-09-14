# Traffic Detector

Detección temprana de accidentes de tránsito sobre las cámaras de Medellín.
Analiza video en tiempo real, reporta el choque en el momento en que ocurre e
identifica los vehículos involucrados.

---

## El problema

Hoy un accidente se reporta cuando alguien decide llamar. Y eso solo pasa si
alguien involucrado está en condiciones de hacerlo.

El número de vehículos en Medellín crece cada año, y los accidentes con él. El
problema no es solo que ocurran: es **cuánto tarda en enterarse quien puede
responder**. Ese retraso es el que este sistema elimina.

---

## Lo que hace

- **Detección y seguimiento en GPU.** YOLO + ByteTrack sobre CUDA, una sola
  inferencia por frame. Ocho clases: ambulancia, bus, carro, camión, moto,
  ciclista, patineta y peatón.
- **Homografía por cámara.** Convierte píxeles en **metros sobre el asfalto**.
  La distancia entre dos vehículos y el tiempo hasta el impacto se calculan en
  unidades físicas, no en lo cerca que se ven dos cajas en pantalla — que
  depende del ángulo de la cámara.
- **Confirmación por desenlace.** Un choque no se da por bueno porque dos cajas
  se toquen, sino porque **alguien queda inmovilizado** después. Es lo que
  elimina la mayoría de falsos positivos.
- **Multi-cámara concurrente.** Cada sesión tiene su propio tracker; una cámara
  se procesa una vez y se reparte a todos los operarios que la estén mirando.
- **Anonimización antes de escribir a disco.** Placas y rostros pixelados. El
  original identificable nunca se guarda.
- **Resumen del caso con IA.** Un modelo multimodal (Gemini) lee la evidencia y
  redacta qué se ve y qué tan grave es.
- **Bandeja de revisión.** Cada incidente se confirma o se descarta a mano.
- **Panel de gestión** con usuarios, tres roles y ocho permisos.

---

## Arquitectura

```
┌──────────────────────────────┐
│  FRONTEND (React + Vite)     │   repo aparte
│  video + overlay + bandeja   │
└──────┬────────────────┬──────┘
       │ HTTP           │ WebSocket
       │ (REST, video)  │ /ws/inference  ← un JSON por frame
┌──────▼────────────────▼──────────────────────────────────┐
│           VISION SERVICE  ·  FastAPI + Python 3.12        │
│                                                           │
│   ┌────────────────── VisionEngine ──────────────────┐    │
│   │  Detector      YOLO sobre CUDA                   │    │
│   │  TrackManager  ByteTrack · identidad por vehículo│    │
│   │  Homography    píxeles → metros                  │    │
│   │  MotionAnalyzer velocidad, aceleración, frenazos │    │
│   │  EventEngine   acercamiento, TTC, detención      │    │
│   │  IncidentEngine confirma el choque por desenlace │    │
│   └──────────────────────┬───────────────────────────┘    │
│                          │                                │
│   Evidencia ──► ANONIMIZAR ──► disco                      │
│   Gemini    ──► resumen del caso (bajo demanda)           │
│   Auth      ──► JWT · roles · permisos                    │
└──────────────────────────┬────────────────────────────────┘
                           ▼
                 ┌───────────────────────┐
                 │     PostgreSQL 18     │
                 │ incidents · cameras   │
                 │ users · roles         │
                 │ permissions · turnos  │
                 └───────────────────────┘
```

**Un solo servicio, no microservicios.** La inferencia y la API viven juntas a
propósito: el tracker guarda estado entre frames y partirlo en dos procesos
obligaría a mover ese estado por la red en cada frame.

**El video no pasa por el WebSocket.** El navegador reproduce el video por HTTP
y el backend le manda solo el JSON de cada frame —cajas, trayectorias,
incidentes—, que el frontend dibuja encima sincronizado por tiempo. Mandar
píxeles por el socket costaría dos órdenes de magnitud más de ancho de banda.

---

## Cómo se detecta un choque

```
frame
  → YOLO                 detección de 8 clases
  → ByteTrack            un identificador estable por vehículo
  → homografía           la posición en metros sobre la calzada
  → MotionAnalyzer       velocidad, aceleración, cambios bruscos
  → EventEngine          ¿se están acercando? ¿en cuánto tiempo chocan?
  → IncidentEngine       ¿alguien quedó detenido después?   → INCIDENTE
  → evidencia            recorte anonimizado + registro en base de datos
```

El paso que importa es el penúltimo. Dos cajas que se solapan en la imagen
pueden estar a diez metros de distancia si la cámara mira en diagonal: por eso
la geometría se resuelve en metros, y por eso el choque no se confirma hasta que
hay un **desenlace** —un vehículo que se detiene y no sigue.

---

## Tecnologías

| Capa | Stack |
|---|---|
| Backend | Python 3.12, FastAPI, WebSockets |
| Visión | PyTorch 2.11 + CUDA 12.8, Ultralytics YOLO, ByteTrack, OpenCV |
| Datos | PostgreSQL 18, SQLAlchemy, Alembic |
| IA | Gemini multimodal para el resumen del caso |
| Frontend | React, Vite, TypeScript ([repo aparte](../traffic_detector_front)) |
| Infraestructura | Docker Compose (perfil CPU y perfil GPU) |

---

## Números

| | |
|---|---|
| Rendimiento (RTX 3050) | 1 cámara **30 fps** · 2 cámaras **30 fps c/u** · 3 cámaras **28 fps c/u** |
| Latencia | tiempo real, sin saltar frames (stride 1) |
| Tests | **262**, corren en 2 s sin GPU y sin modelo |
| Esquema | 10 migraciones versionadas con Alembic |

Tres cámaras simultáneas en tiempo real sobre un portátil con RTX 3050.

---

## Privacidad

Las cámaras de tránsito graban vía pública, y eso incluye placas y caras.

**La anonimización ocurre antes de escribir.** Placas y rostros se pixelan en
memoria, y el recorte identificable original nunca llega al disco. El sistema
cumple la Ley 1581 de 2012 (Habeas Data). El detalle, en
[PRIVACY.md](PRIVACY.md).

---

## Cómo se levanta

```bash
docker compose up --build                                                  # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build  # GPU
```

Levanta PostgreSQL, aplica las migraciones y arranca el servicio. API en
`:8000`, interfaz en `:5173`.

---

## Más a fondo

- [INSTALL.md](INSTALL.md) — instalación manual paso a paso
- [PRIVACY.md](PRIVACY.md) — anonimización y retención
- [ARCHITECTURE.md](ARCHITECTURE.md) — pipeline en detalle
- [STACK.md](STACK.md) — versiones fijadas
