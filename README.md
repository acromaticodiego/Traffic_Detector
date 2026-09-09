# Traffic Detector

Plataforma de visión artificial que **mide conflictos viales** sobre cámaras de
tránsito de Medellín: detecta vehículos, los sigue, y reporta choques y
vehículos detenidos con evidencia auditable y revisión humana.

No es un "detector de choques": los choques reales son rarísimos. Lo que mide
son **conflictos** —near-miss, tiempo hasta el impacto, velocidad de cierre—
que son miles al mes y es lo que un ingeniero de tránsito usa para priorizar
intervenciones.

## Lo que hace

- **Detección y seguimiento en GPU** — YOLO + ByteTrack sobre CUDA, una sola
  inferencia por frame. 8 clases: ambulancia, bus, carro, ciclista, patineta,
  moto, peatón, camión.
- **Homografía por cámara** — convierte píxeles a **metros sobre el asfalto**,
  así el TTC y la separación se reportan en unidades físicas y no en píxeles,
  que dependen del ángulo.
- **ROI de calzada y nivel de tráfico** — polígono de la vía por cámara, con
  ocupación y velocidad corregidas por perspectiva.
- **Confirmación por desenlace** — un choque no se confirma por geometría, sino
  porque **inmoviliza a alguien**. Dos cajas que se tocan en la imagen pueden
  estar a metros de distancia: es el origen de casi todos los falsos positivos.
- **Multi-cámara concurrente** — cada sesión con su propio tracker; una cámara
  se procesa una vez y se reparte a todos los operarios que la miran.
- **Anonimización antes de escribir** — placas y rostros pixelados; el original
  identificable nunca toca el disco (Ley 1581, Habeas Data).
- **Bandeja de revisión** — cada incidente se confirma o descarta a mano. Nada
  se borra: los veredictos son el conjunto etiquetado con el que se mide la
  precisión real.
- **Resumen del caso con IA** — Gemini multimodal lee la evidencia y redacta
  qué se ve, bajo demanda y cacheado.
- **Login con roles y permisos** — JWT, tres roles, 8 permisos normalizados.

## Stack

`Python 3.12` · `FastAPI` · `PyTorch 2.11 + CUDA 12.8` · `Ultralytics YOLO` ·
`OpenCV` · `PostgreSQL 18` · `SQLAlchemy` + `Alembic` · `Docker Compose` ·
Frontend en `React` + `Vite` + `TypeScript` ([repo aparte](../traffic_detector_front))

## Números

| | |
|---|---|
| Rendimiento medido (RTX 3050) | 1 cámara **30 fps**, 2 cámaras **30 fps c/u**, 3 cámaras **28 fps c/u** |
| Tests | **262**, corren en 2 s sin GPU ni modelo |
| Migraciones | 10, esquema versionado con Alembic |
| Latencia del pipeline | tiempo real a 30 fps con stride 1 |

## Cómo se levanta

```bash
docker compose up --build                                                    # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build    # GPU
```

Postgres, migraciones y servicio. API en `:8000`, interfaz en `:5173`.

## Más a fondo

- [INSTALL.md](INSTALL.md) — instalación manual paso a paso
- [PRIVACY.md](PRIVACY.md) — anonimización y retención (Ley 1581 de 2012)
- [ARCHITECTURE.md](ARCHITECTURE.md) — pipeline y hoja de ruta
- [STACK.md](STACK.md) — versiones fijadas


