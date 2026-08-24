# TikZfy — Multimodal Image-to-TikZ Neural Engine & Geometric Compiler

TikZfy is an end-to-end multimodal deep learning engine that translates raster geometric figures into native, compile-ready LaTeX/TikZ markup. Built under strict Hexagonal Architecture (Ports and Adapters) and SOLID principles to ensure mathematical purity, high performance, and continuous integration.

## Architectural Structure

- `core/`: Pure immutable mathematical and neural domain (PyTorch, NumPy, Einops). Vectorized operations, zero I/O.
- `ports/`: Abstract inbound and outbound port interfaces enforcing boundary contracts.
- `adapters/`: Infrastructure adapters (FastAPI REST API, TeX Live compiler, Ghostscript rasterizer, persistence).
- `scripts/`: Training orchestrators, multi-tier benchmark evaluations, and data ingestion pipelines.
- `frontend/`: Interactive client application built with Astro, Tailwind CSS, and anime.js for live geometric reconstruction comparison.

## Web Application & Live Demo

- **Interactive Client:** [https://antoniomachuca.github.io/tikzfy/](https://antoniomachuca.github.io/tikzfy/)
- **API Documentation:** `http://127.0.0.1:8000/docs` (when running the FastAPI backend locally).
