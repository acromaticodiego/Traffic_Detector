Traffic Detector

Sistema de detección y análisis de tráfico mediante visión artificial, construido con Python, YOLO y ByteTrack.

El proyecto procesa videos de tráfico frame por frame para detectar vehículos y peatones, mantener su seguimiento y generar eventos e incidentes a partir del comportamiento observado.

Arquitectura actual
Video
  │
  ▼
YOLO Detection
  │
  ▼
ByteTrack
  │
  ▼
Track Manager
  │
  ▼
Motion Analyzer
  │
  ▼
Event Engine
  │
  ▼
Incident Engine
  │
  ▼
Evidence
🔎 Funcionalidades actuales
Detección de objetos con YOLO.
Clases actualmente soportadas:
Bus
Car
Ciclista
Motocicleta
Peatón
Camión
Umbral de confianza: 0.70
IoU: 0.60
Tamaño de entrada del modelo: 640
Tracking de objetos mediante ByteTrack.
Identificación persistente mediante track_id.
Análisis básico de movimiento.
Generación de eventos:
vehicle_detected
vehicle_proximity
Detección de incidentes mediante IncidentEngine.
Generación de evidencia cuando se detecta un incidente.
Generación de un video de salida con:
Bounding boxes
Clase detectada
track_id
Incidentes detectados
⚙️ Tecnologías
Python
OpenCV
PyTorch
Ultralytics YOLO
ByteTrack
NumPy

La inferencia está configurada para utilizar GPU NVIDIA mediante CUDA cuando está disponible.

📁 Estructura
traffic_detector/
│
├── models/
│   └── detectorfinal.pt
│
├── videos/
│   └── input/
│
├── outputs/
│   ├── incidents/
│   └── videos/
│
├── services/
│   └── vision_service/
│       └── app/
│           ├── detection/
│           ├── tracking/
│           ├── motion/
│           ├── events/
│           ├── incidents/
│           └── vision_engine.py
│
├── scripts/
│   └── test_vision_engine.py
│
├── .gitignore
└── README.md
▶️ Ejecución

Crear y activar el entorno virtual:

python -m venv .venv
.venv\Scripts\Activate.ps1

Instalar las dependencias:

pip install -r requirements.txt

Colocar el modelo YOLO en:

models/detectorfinal.pt

Colocar el video de prueba en:

videos/input/

Configurar el nombre del video en:

scripts/test_vision_engine.py

Ejecutar:

python -m scripts.test_vision_engine

El resultado se genera en:

outputs/videos/vision_result.mp4

Las evidencias de incidentes se almacenan en:

outputs/incidents/
📌 Estado actual

Actualmente se encuentra implementado el pipeline principal de visión artificial, desde la detección y tracking hasta el análisis de movimiento, eventos e incidentes.

El proyecto continúa en desarrollo hacia un sistema completo de detección y gestión inteligente de incidentes de tráfico.
