# Image to TikZ Engine

Este repositorio es el núcleo principal para la inferencia y generación de código TikZ a partir de imágenes, implementado bajo una Arquitectura Hexagonal estricta para garantizar la escalabilidad, inmutabilidad de estados y separación de responsabilidades.

## Estructura de Directorios

- `core/`: 🧠 Núcleo matemático inmutable (PyTorch, NumPy, Einops). Funciones vectorizadas, cero I/O.
- `ports/`: 🔌 Interfaces abstractas. Define contratos de entrada/salida para mantener el sistema desacoplado.
- `adapters/`: ⚙️ Adaptadores e infraestructura de red (FastAPI, Pydantic, aiohttp). Manejo de concurrencia y validaciones de frontera.
- `scripts/`: 🚀 Bucle de entrenamiento, evaluación y orquestación (Makefiles, Docker, automatización).
- `frontend/`: 🌐 Aplicación cliente externa construida con Astro + Tailwind CSS para una óptima hidratación y carga rápida.

*Nota:* Las herramientas de agentes u orquestación de inteligencia artificial residen externamente en el directorio `agent/` (fuera de este repositorio).
