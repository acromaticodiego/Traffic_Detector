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

## Instalación y puesta en marcha

Guía completa para dejar el proyecto corriendo en una máquina nueva (Windows).
En Linux/macOS es lo mismo cambiando `.venv\Scripts\` por `.venv/bin/`.

### 1. Requisitos previos

| Software | Versión | Para qué | Dónde |
|---|---|---|---|
| **Python** | **3.12** | Este repo (servicio de visión) | [python.org](https://www.python.org/downloads/) — marcar *Add python.exe to PATH* |
| **PostgreSQL** | **16 o superior** | Cámaras e incidentes | [postgresql.org](https://www.postgresql.org/download/windows/) |
| **Node.js** | **20 LTS o 22** | Solo para el **frontend**, que vive en otro repo | [nodejs.org](https://nodejs.org/) |
| **npm** | el que trae Node (10.x) | Ídem | viene incluido con Node |
| **Git** | cualquiera reciente | Clonar el repo | [git-scm.com](https://git-scm.com/) |
| GPU NVIDIA + CUDA | opcional | Acelera la inferencia | sin GPU corre en CPU, más lento |

Sobre las versiones: **Python 3.12 no es negociable** — es la que fija
`pyproject.toml` (`target-version = "py312"`) y la que usa el CI; con 3.13 los
wheels de torch y ultralytics que están pinneados pueden no existir todavía.
Con Postgres en cambio hay margen: [`STACK.md`](STACK.md) fija **16**, el
entorno de desarrollo actual corre **18.6**, y el esquema no usa nada exclusivo
de ninguna de las dos (solo tipos estándar y `JSONB`), así que cualquiera
sirve. Lo que no conviene es mezclar: un `pg_dump` hecho en 18 no restaura en
16.

Verificar antes de seguir:

```powershell
python --version    # Python 3.12.x
node --version      # v20.x o v22.x  (solo si vas a levantar el frontend)
npm --version       # 10.x
psql --version      # psql (PostgreSQL) 16.x o 18.x
```

Si `psql` no aparece, está en `C:\Program Files\PostgreSQL\<versión>\bin`: hay
que agregar esa carpeta al PATH o llamarlo con la ruta completa.

### 2. Clonar el repo

```powershell
git clone <url-del-repo> traffic_detector
cd traffic_detector
```

### 3. Entorno de Python y dependencias

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Son unos 2 GB entre torch, ultralytics y opencv: la primera vez tarda varios
minutos.

Hay tres archivos de dependencias y conviene saber cuál es cuál:

| Archivo | Contenido | Cuándo se usa |
|---|---|---|
| `requirements.txt` | El de la raíz; delega en el del servicio | Para **ejecutar** el proyecto |
| `services/vision_service/requirements.txt` | Los pines reales (FastAPI, torch, ultralytics, SQLAlchemy, Alembic, psycopg) | Lo instala el anterior |
| `requirements-dev.txt` | Solo `ruff`, `pytest`, numpy y PyYAML | Para **lint y tests**; es lo único que instala el CI |

Los pines viven en un solo lugar a propósito, para que no existan dos listas
que se puedan desincronizar.

### 4. Lo que **no** viene en el clon

Dos cosas que el repo ignora por peso (ver `.gitignore`) y hay que copiar a
mano, o el servicio arranca pero no detecta nada:

| Qué | Dónde va | Cómo conseguirlo |
|---|---|---|
| **Pesos del modelo** `detectorfinal.pt` | `models/detectorfinal.pt` | Pedirlo a quien mantiene el proyecto (`*.pt` está en `.gitignore`) |
| **Video de entrada** | `videos/input/<archivo>.mp4` | Igual; los `.mp4` también están ignorados |

El nombre del video debe coincidir con el `source` de la cámara en
`cameras.yaml`, o se edita el YAML para que apunte al archivo que se tenga.

### 5. Base de datos PostgreSQL

**5.1. Crear la base.** Alembic crea las *tablas*, pero no la *base de datos*:

```powershell
psql -U postgres -c "CREATE DATABASE traffic_detector;"
```

Pide la contraseña que se definió al instalar Postgres.

**5.2. Configurar el `.env`.** Ese archivo está en `.gitignore` porque lleva la
contraseña, así que cada quien crea el suyo a partir del ejemplo versionado:

```powershell
copy .env.example .env
```

y edita la línea de la base de datos con sus credenciales:

```
VISION_DATABASE_URL=postgresql+psycopg://postgres:TU_CONTRASEÑA@localhost:5432/traffic_detector
```

**5.3. Aplicar las migraciones:**

```powershell
.venv\Scripts\alembic upgrade head
```

Esto crea `cameras`, `incidents` y sus índices. La URL **no** está escrita en
`alembic.ini`: `migrations/env.py` la lee de la misma configuración que usa el
servicio, para no tener dos fuentes de verdad ni versionar la contraseña. Por
eso el paso 5.2 tiene que estar hecho antes que este.

Comprobar que quedó en la última revisión — ambos comandos deben imprimir la
misma:

```powershell
.venv\Scripts\alembic current
.venv\Scripts\alembic heads
```

**5.4. Cargar las cámaras:**

```powershell
.venv\Scripts\python scripts\seed_cameras.py
```

Lee `cameras.yaml` y lo vuelca en la tabla `cameras`. Es idempotente: se puede
volver a correr cada vez que cambie la calibración y actualiza las filas
existentes en vez de duplicarlas. Con `--dry-run` muestra qué haría sin tocar
la base.

> **Migrar el esquema no es lo mismo que copiar los datos.**
> `alembic upgrade head` deja una base vacía con la estructura correcta, que es
> lo que necesita alguien que arranca de cero. Si además hacen falta los
> incidentes ya registrados en otra máquina, eso es un volcado aparte:
> `pg_dump -U postgres traffic_detector > dump.sql` en el origen y
> `psql -U postgres -d traffic_detector -f dump.sql` en el destino, con la
> misma versión mayor de Postgres a ambos lados.

### 6. Levantar el servicio de visión

```powershell
.venv\Scripts\python -m services.vision_service.app.api
```

o, equivalente y más explícito:

```powershell
.venv\Scripts\python -m uvicorn services.vision_service.app.api.main:app --host 0.0.0.0 --port 8000
```

Se ejecuta **desde la raíz del repo**, para que resuelva el paquete
`services.*`. Verificar en otra terminal:

```powershell
curl http://localhost:8000/health
```

Debe responder con el estado, el modelo cargado y el video detectado. Si dice
que no encuentra modelo o video, falta el paso 4.

La lista completa de variables de entorno (umbrales, ROI, CORS, puerto) está en
[`services/vision_service/README.md`](services/vision_service/README.md).

### 7. Frontend (repo aparte) — aquí entran Node y npm

La interfaz **no está en este repo**: vive en `../traffic_detector_front`
(React + Vite + TypeScript). Este repo es solo Python y no tiene `package.json`.

```powershell
cd ..\traffic_detector_front
npm install
copy .env.example .env
npm run dev
```

El `.env` del frontend apunta al backend; para local ya viene bien por defecto:

```
VITE_API_BASE=http://localhost:8000
VITE_WS_BASE=ws://localhost:8000
```

Vite sirve en `http://localhost:5173`. El backend tiene que estar corriendo
(paso 6) o la interfaz carga sin datos.

### 8. Verificar que todo quedó bien

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m pytest
```

La suite corre en menos de un segundo y no necesita ni el modelo ni el video.

### 9. Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ModuleNotFoundError: services...` | Se ejecutó desde otra carpeta | Correr siempre desde la raíz del repo |
| `connection refused` en el puerto 5432 | El servicio de Postgres no está arriba | Iniciar `postgresql-x64-<versión>` en *Servicios* de Windows |
| `password authentication failed` | Contraseña mal puesta | Revisar `VISION_DATABASE_URL` en `.env` |
| `database "traffic_detector" does not exist` | Falta el paso 5.1 | Crear la base antes de migrar |
| `/health` dice que no hay modelo | `models/detectorfinal.pt` no está | Ver paso 4 |
| Inferencia muy lenta | Está corriendo en CPU | Instalar torch con CUDA, o subir `VISION_FRAME_STRIDE` |
| El front carga pero sin video | El backend no está arriba, o CORS | Levantar el paso 6; revisar `VISION_CORS_ORIGINS` |

---

## Despliegue con Docker

```bash
docker compose up --build                                          # sin GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build   # con GPU
```

Levanta Postgres, aplica las migraciones y arranca el servicio en el 8000. Las
migraciones corren como un servicio de un solo uso del que depende el API, así
que este nunca arranca contra un esquema viejo.

### Antes de la primera vez

En el `.env` (ver [`.env.example`](.env.example)):

| Variable | |
|---|---|
| `POSTGRES_PASSWORD` | **Sin valor por defecto.** El compose se niega a arrancar sin ella, en vez de levantar una base con una contraseña conocida |
| `JWT_SECRET` | Sin él el servicio arranca y rechaza todo inicio de sesión |
| `GEMINI_API_KEY` | Opcional; sin ella no se ofrece "Analizar el caso" |

### Qué NO va dentro de la imagen

El modelo (`models/`) y los videos (`videos/`) se montan como volúmenes de
solo lectura: no están en git, pesan, y hornearlos obligaría a reconstruir
tres gigas para cambiar de modelo. El `.env` tampoco entra — los secretos se
pasan como variables de entorno.

La evidencia va en un **volumen con nombre**, no en un bind mount, para que el
dueño lo ponga Docker y el usuario no-root de la imagen pueda escribir sin
pelearse con los permisos del anfitrión. Para sacarla:

```bash
docker compose cp vision:/app/outputs/incidents ./outputs/
```

### GPU

Va en un archivo aparte porque `deploy.devices` **hace fallar el arranque** en
una máquina sin NVIDIA, y el compose base tiene que levantar en cualquier
parte. Requiere en el anfitrión el driver y `nvidia-container-toolkit`; en
Windows, Docker Desktop sobre WSL2 con el driver del anfitrión. Comprobar que
Docker ve la GPU antes de pelearse con lo demás:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

Para construir sin CUDA, `TORCH_INDEX=https://download.pytorch.org/whl/cpu`.
La imagen pasa de unos 7 GB a unos 2, y la inferencia va notablemente más
lenta.

### CORS

El compose fija `VISION_CORS_ORIGINS` explícitamente. Corriendo a mano el
valor por defecto sigue siendo `*`, que es cómodo en desarrollo, pero **desde
que hay login ya no es inocuo**: cualquier página que abra un operario puede
llamar a la API, y el token viaja en la query del WebSocket. El servicio lo
registra como aviso al arrancar.

---

## Datos personales y retención

El sistema graba vía pública, así que la evidencia que guarda contiene placas
y rostros. Eso es tratamiento de datos personales bajo la **Ley 1581 de 2012**
(Habeas Data) y su decreto reglamentario 1377 de 2013, y condiciona dos cosas
del diseño.

### Anonimización

Las imágenes se anonimizan **antes** de escribirse: el original identificable
no llega a existir en disco en ningún momento. Se pixelan —no se difuminan, un
gaussiano suave conserva demasiada señal— la franja inferior de cada vehículo,
donde va la placa, y la superior de todo lo que lleva a una persona.

No se detectan las placas, se tapa la zona donde una placa tiene que estar. Un
detector de placas falla abierto: la que no reconoce queda legible en disco y
nadie se entera. La franja geométrica falla cerrada. Por el mismo criterio,
una clase que el modelo no conozca se tapa por las dos puntas.

Alcance declarado: se anonimiza lo que **se conserva**. La vista en vivo
muestra la calle tal cual, porque es lo que el operario necesita para hacer su
trabajo, y va detrás del permiso `stream:view`.

| Variable | Qué hace |
|---|---|
| `VISION_ANONYMIZE` | Enciende la anonimización (por defecto sí) |
| `VISION_ANONYMIZE_PLATE_BAND` | Fracción inferior de la caja que se tapa |
| `VISION_ANONYMIZE_FACE_BAND` | Fracción superior |

Si con el ángulo de una cámara se ve asomar una placa, **subir** el valor.
Quedarse corto deja una placa legible guardada; pasarse solo cuesta imagen.

### Retención

La regla es: **el incidente se queda, la imagen caduca.**

El dato personal es la foto, no la fila. La bandeja de revisión conserva su
histórico entero y sus veredictos —de ahí sale la métrica de precisión— y a
los N días la imagen se borra y `evidence_path` queda en null. Esto convive
con el principio de que un incidente no se borra nunca.

| Variable | Qué hace |
|---|---|
| `VISION_EVIDENCE_RETENTION_DAYS` | Días que vive una imagen. **0 = para siempre** |
| `VISION_RETENTION_DRY_RUN` | Registra qué borraría, sin borrar |

El valor por defecto es 0 —conservar— a propósito: esta configuración borra
archivos, y una variable vacía tiene que dejar el disco intacto, no vaciarlo.
**Un despliegue real tiene que fijar el plazo explícitamente.**

La limpieza corre al arrancar y cada seis horas. Solo borra directorios bajo
la raíz de la evidencia cuyo contenido sean únicamente imágenes: cualquier
otra cosa se deja intacta.

### Lo que todavía falta para un contrato público

El código cubre la parte técnica. La Ley 1581 además exige, y esto **no es
código**:

- Una **política de tratamiento de la información** publicada, con la
  finalidad declarada (medición de conflictos viales) y el plazo de
  conservación, que debe coincidir con `VISION_EVIDENCE_RETENTION_DAYS`.
- Un **responsable del tratamiento** identificado — normalmente la entidad
  contratante, no el proveedor.
- Registro de la base de datos ante la SIC, si aplica según el volumen.
- Señalización en vía de que la zona está videovigilada.

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


