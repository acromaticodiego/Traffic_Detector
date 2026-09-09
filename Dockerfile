# Imagen del servicio de visión.
#
# Se construye desde la RAÍZ del repo, no desde services/vision_service: el
# servicio se ejecuta como `python -m services.vision_service.app.api` y
# necesita ver `cameras.yaml`, `migrations/` y `alembic.ini`, que viven arriba.
#
#   docker build -t traffic-detector .                      # con CUDA
#   docker build -t traffic-detector \
#          --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#
# Lo que NO va dentro de la imagen, a propósito:
#
#   models/   el .pt no está en git y pesa; se monta como volumen para poder
#             cambiar de modelo sin reconstruir tres gigas
#   videos/   ídem, y en producción la fuente será RTSP
#   .env      los secretos se pasan como variables de entorno, no se hornean
#
# La versión base se fija a bookworm y no a `slim` a secas porque el nombre de
# los paquetes de sistema cambia entre releases de Debian, y una imagen que se
# construye distinto según el día no es un despliegue reproducible.

FROM python:3.12-slim-bookworm

# opencv necesita libGL y glib aunque aquí nunca dibuje una ventana: los
# importa al cargarse. Sin esto el servicio arranca y muere en el primer
# `import cv2` con un error que no menciona a opencv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# CUDA 12.8 por defecto, igual que el entorno de desarrollo que fija STACK.md.
# Para una máquina sin GPU:
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# torch va en su propia capa y antes que el resto: es con diferencia lo más
# pesado que se descarga, y así cambiar cualquier otra dependencia no obliga a
# volver a bajarlo. La versión es la misma que pide requirements.txt, así que
# el paso siguiente lo da por satisfecho en vez de reinstalarlo.
RUN pip install --no-cache-dir \
        torch==2.11.0 \
        torchvision==0.26.0 \
        --index-url ${TORCH_INDEX}

COPY services/vision_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini cameras.yaml ./
COPY migrations/ ./migrations/
COPY services/ ./services/
COPY scripts/ ./scripts/

# Sin esto el proceso corre como root: un fallo en el servicio pasaría a ser
# root dentro del contenedor. La evidencia se escribe en un volumen con nombre
# —no en un bind mount— justamente para que el dueño lo ponga Docker y esto no
# choque con los permisos del anfitrión.
RUN useradd --create-home --uid 10001 vision \
    && mkdir -p /app/outputs/incidents \
    && chown -R vision:vision /app/outputs

USER vision

EXPOSE 8000

# El propio servicio ya sabe decir si está vivo, y de paso comprueba que el
# modelo cargó: un contenedor que responde pero sin modelo no sirve de nada.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request, json, sys; \
d = json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=4)); \
sys.exit(0 if d.get('model_loaded') else 1)"

CMD ["python", "-m", "services.vision_service.app.api"]
