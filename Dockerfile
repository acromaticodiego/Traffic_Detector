
FROM python:3.12-slim-bookworm

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
