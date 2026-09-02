# Stack tecnológico — versiones fijadas

Documento de referencia para **todo el proyecto** (backend de visión + frontend +
servicios futuros). El objetivo es que cualquier persona/sesión que continúe el
trabajo use **exactamente las mismas versiones** y no introduzca incompatibilidades
(el caso típico: mezclar React 18 y 19, o `react-leaflet` v4 y v5).

> Regla general: **no subir de major version** sin actualizar este documento y
> probar `tsc` + arranque de ambos servicios.

---

## 1. Frontend — `traffic_detector_front/`

| Paquete | Versión fija | Notas / por qué |
|---|---|---|
| **React** | **18.3.1** | ⚠️ **NO subir a React 19.** `react-leaflet` v4 (el que usamos) requiere React 18. |
| **react-dom** | **18.3.1** | Siempre igual a `react`. |
| **@types/react** | **18.3.x** | Debe coincidir con el major de React. Nunca `@types/react@19`. |
| **@types/react-dom** | **18.3.x** | Igual que arriba. |
| **react-leaflet** | **4.2.1** | ⚠️ **NO subir a v5** (v5 exige React 19). Pareja: React 18 + react-leaflet 4. |
| **leaflet** | **1.9.4** | Peer de react-leaflet 4. |
| **@types/leaflet** | **1.9.x** | — |
| **lucide-react** | **0.460.x** | Librería de iconos (única fuente). Se importa siempre desde `src/components/icons.tsx` (re-export), no desde `lucide-react` directo. Tree-shakeable. |
| **zustand** | **4.5.x** | Estado global. Dos stores: `state/store.ts` (datos de visión) y `state/panels.ts` (posición/estado de los paneles flotantes, con `persist` a localStorage). v5 también sirve con React 18, pero nos quedamos en 4.x. API usada: `create`, `persist` (de `zustand/middleware`). |
| **vite** | **5.4.x** | ⚠️ No subir a Vite 6/7 sin revisar `@vitejs/plugin-react`. |
| **@vitejs/plugin-react** | **4.x** | Compatible con Vite 5. |
| **typescript** | **5.x** (5.6+) | `moduleResolution: "bundler"`. |
| **Node.js** | **20 LTS o 22 LTS** | Desarrollado y probado con Node 24 también OK. Mínimo 18. |

### Reglas del frontend

- **Gestor de paquetes:** `npm` (hay `package-lock.json`). No mezclar con pnpm/yarn.
- **Mapas:** Leaflet + OpenStreetMap con recoloreado CSS (`.leaflet-tile-pane` filter).
  No usar CARTO/Stadia/Mapbox como basemap: requieren API key. Si en el futuro se
  quiere un basemap vectorial, migrar a **MapLibre GL** (no `react-map-gl` de Mapbox).
- **Estilos:** CSS plano con custom properties en `src/styles.css`. **No** se añadió
  Tailwind ni librería de componentes a propósito (menos peso, menos versiones que
  alinear). Si se añade una librería UI, que sea headless (Radix) y compatible con React 18.
- **Charts (futuro):** si se necesitan, usar **Recharts** o **visx** (compatibles React 18).
- **Iconos:** `lucide-react`, re-exportados desde `src/components/icons.tsx` con
  nombres `IconXxx`. Para cambiar de librería de iconos en el futuro, solo se toca
  ese archivo.
- **Paneles:** la UI es un sistema de ventanas flotantes sobre el mapa
  (`FloatingPanel` + `Dock`). Estado y posiciones en `state/panels.ts` (persistido).

---

## 2. Backend — servicio de visión — `traffic_detector/`

| Paquete | Versión fija | Notas |
|---|---|---|
| **Python** | **3.12.0** | El `.venv` de la raíz. No usar 3.13 todavía (ruedas de torch/opencv). |
| **fastapi** | **0.141.1** | — |
| **uvicorn** | **0.52.4** | Servidor ASGI. |
| **websockets** | **>=13** (instalado 17.1) | Requerido por uvicorn para el WS. |
| **starlette** | **1.6.0** | Lo trae FastAPI; no instalar aparte. |
| **pydantic** | **2.13.4** | v2 (no v1). |
| **ultralytics** | **8.4.129** | YOLO + ByteTrack. ⚠️ Fijar: cambian la API de `solutions`/`track` entre minors. |
| **opencv-python** | **5.0.0.93** | — |
| **torch** | **2.11.0+cu128** | Compilado para CUDA 12.8. GPU probada: RTX 3050. Reinstalar con el índice `--index-url https://download.pytorch.org/whl/cu128`. |
| **torchvision** | **0.26.0+cu128** | Pareja de torch. Mismo major.minor que torch. |
| **numpy** | **2.5.2** | numpy 2.x (ojo con libs que aún piden <2). |
| **python-multipart** | **0.0.32** | Uploads/formularios en FastAPI. |

### Reglas del backend

- **torch + torchvision van juntos** y con el mismo build CUDA. Nunca actualizar uno solo.
- **ultralytics fijo**: si se sube, revisar `ByteTrackTracker` (usa `model.track(persist=...)`).
- El modelo (`models/detectorfinal.pt`) y los videos NO van a git (ver `.gitignore`).
- Instalación: `pip install -r services/vision_service/requirements.txt`.

---

## 3. Servicios futuros — versiones objetivo

Aún no implementados. Cuando se creen, usar estas versiones para mantener coherencia:

| Componente | Tecnología | Versión objetivo | Notas |
|---|---|---|---|
| API Gateway | **NestJS** | **11.x** | Node 20/22. TypeScript 5.x (igual que el front). |
| Incident Service | NestJS 11.x + **Prisma** 6.x o **TypeORM** 0.3.x | — | Elegir uno y documentarlo. |
| Base de datos | **PostgreSQL** | **16** | — |
| Cache / colas cortas | **Redis** | **7.x** | — |
| Object storage | **MinIO** | última estable | Compatible API S3. |
| Bus de eventos | **NATS** | **2.10.x** | JetStream para persistencia de eventos. |
| Transcripción | **Deepgram SDK** (`@deepgram/sdk`) | **3.x** | Node. |
| Fusión / LLM | **Gemini** (`@google/genai`) | SDK nuevo `@google/genai` (no el viejo `@google/generative-ai`) | — |
| Contenedores | **Docker** + Compose v2 | — | `infra/docker-compose.yml`. |

### Reglas de los servicios

- **Un solo `tsconfig` base compartido** para todos los servicios Node (mismo target,
  mismo TypeScript major que el frontend).
- Todo servicio Node usa **Node 20 o 22 LTS**, nunca versiones distintas entre servicios.
- Comunicación entre servicios: **NATS** (no HTTP directo servicio-a-servicio salvo el gateway).
- El contrato del WebSocket de visión (`ARCHITECTURE.md`) es el primer "evento de dominio";
  cuando entre NATS, ese payload se publica tal cual.

---

## 4. Checklist antes de actualizar cualquier dependencia

1. ¿Cambia un major? → actualizar este archivo primero.
2. Frontend: `npm run build` (corre `tsc -b` + `vite build`) sin errores.
3. Backend: `python -c "import services.vision_service.app.api.main"` + arrancar uvicorn + `/health`.
4. Prueba end-to-end: `scripts/ws_smoke.py` y abrir `http://localhost:5173`.
5. React/react-dom/@types/react/react-leaflet: **los cuatro** en el mismo tren de versiones.
