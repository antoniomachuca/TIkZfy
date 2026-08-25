# TikZfy — Multimodal Image-to-TikZ Neural Engine & Geometric Compiler

TikZfy is an end-to-end multimodal deep learning engine that translates raster geometric figures into native, compile-ready LaTeX/TikZ markup. Built under strict Hexagonal Architecture (Ports and Adapters) and SOLID principles to ensure mathematical purity, high performance, and continuous integration.

---

## 🏛️ Architectural Structure

- `core/`: Pure immutable mathematical and neural domain (PyTorch, NumPy, Einops). Vectorized operations, zero I/O, pure structured programming.
  - `core/ml/`: Multi-layer Vision Transformer Encoder with 2D CoordConv Cartesian injection, Autoregressive Decoder, and `SpatialAwareHybridLoss` (Cross-Entropy + Smooth L1 Huber coordinate loss).
  - `core/dataset/`: Procedural grammar generators across 8 canonical geometric families and hierarchical Stochastic Context-Free Grammars (SCFG).
  - `core/math/`: Vectorized coordinate quantization, tokenization, and metric computations (SSIM, Hungarian Graph Edit Distance).
- `ports/`: Abstract inbound and outbound port interfaces enforcing boundary contracts.
- `adapters/`: Infrastructure adapters (FastAPI REST API, TeX Live compiler, Ghostscript rasterizer, atomic checkpoint persistence).
- `scripts/`: Production training orchestrators, multi-tier benchmark evaluations, and showcase generators.
- `frontend/`: Interactive client application built with Astro, Tailwind CSS, and anime.js for live geometric reconstruction comparison.

---

## 🚀 Model Benchmarks & Spatial Alignment Results (Phase 3.5)

### Quantitative Performance Matrix (Google Cloud NVIDIA L4 GPU)

| Metric | Baseline Model (Cross-Entropy) | Spatial-Aligned Model (CoordConv + Huber) | Relative $\Delta$ |
| :--- | :---: | :---: | :---: |
| **TeX Live Compilation Rate ($\text{CR}$)** | `100.0%` | **`100.0%`** | $+0.0\%$ (Perfect syntax) |
| **Visual Similarity ($\text{SSIM}$)** | `0.342` | **`0.732`** | **$+114.0\%$** |
| **Mean Coordinate Error ($\text{MSE}_{x,y}$)** | `2.84` | **`0.41`** | **$-85.6\%$** |
| **Geometric Primitive Disambiguation** | Mode Collapse | **Distinct geometric forms** | Full variance |

### Multi-Level Showcase Visual

![TikZfy Multimodal Showcase](results/showcase/comparison_grid.png)

---

## 🧪 Testing & Code Quality Standards

- **Unit & Integration Tests:** 245/245 passing (100% green).
- **Type Checking:** `mypy --strict` passing across 90 Python source files.
- **Linter & Formatter:** Strict `ruff` compliance with zero warnings.

---

## 🌐 Web Application & Live Demo

- **Interactive Client:** [https://antoniomachuca.github.io/tikzfy/](https://antoniomachuca.github.io/tikzfy/)
- **API Documentation:** `http://127.0.0.1:8000/docs` (when running the FastAPI backend locally).
