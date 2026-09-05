"""Generate an authentic visual showcase comparison grid for TikZfy V4.

Evaluates the trained VisionAutoregressiveModelV4 across 6 ascending geometric
complexity levels:
    Level 1: Canonical Line Segment & Vector Arrow (1D)
    Level 2: Canonical Geometric Circle & Ellipse Arc (2D Curvilinear)
    Level 3: Canonical Geometric Polygon (Closed Metric Variety)
    Level 4: Canonical Cartesian Grid & Coordinate Axes (Orthogonal Lattice)
    Level 5: Multi-Segment Vectorial Polyline (Piecewise Linear)
    Level 6: Directed Graph & Automaton Network (Topological Node/Edge System)

Methodology:
    Model Architecture: VisionAutoregressiveModelV4 (CoordConv 2D, 256x256, 1024 tokens).
    Decoding Strategy: Grammar-Constrained Greedy Search with delimiter balancing.
    Output: Publication-grade dark slate comparison grid saved to
        results/showcase/comparison_grid.png.

References:
    Golub & Van Loan, Matrix Computations - coordinate lattice transformations.
    Goodfellow et al., Deep Learning - generative sequence modeling.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.templates import generate_sample
from core.ml.generation import decode_indices_to_markup, greedy_search
from core.ml.metrics import structural_similarity
from core.ml.model import VisionAutoregressiveModelV4
from core.models import ImageTensor, TikzTokens, TokenVocabulary


def clean_tikz_code(markup: str) -> str:
    """Normalize token spacing and ensure closed TikZ environment."""
    s: str = markup.strip()
    s = re.sub(r"\(\s*([a-zA-Z0-9_-]+)\s*\)", r"(\1)", s)
    s = re.sub(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", r"(\1, \2)", s)
    s = re.sub(r"circle\s*\(\s*(-?\d+(?:\.\d+)?)\s*\)", r"circle (\1)", s)
    s = re.sub(r"\{\s*([^{}]+?)\s*\}", lambda m: "{" + m.group(1).strip() + "}", s)
    s = re.sub(
        r"\[\s*([^\[\]]+?)\s*\]",
        lambda m: "["
        + re.sub(r"\s+", " ", m.group(1).replace(" ,", ",").replace(" = ", "=").strip())
        + "]",
        s,
    )
    if r"\end{tikzpicture}" in s:
        body: str = s.replace(r"\begin{tikzpicture}", "").replace(r"\end{tikzpicture}", "").strip()
        last_semi: int = body.rfind(";")
        if last_semi != -1:
            body = body[: last_semi + 1]
        else:
            body = body + " ;"
        return r"\begin{tikzpicture} " + body + r" \end{tikzpicture}"
    return s + r" \end{tikzpicture}"


async def render_markup_to_png(
    markup: TikzTokens,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    dpi: int = 144,
) -> Image.Image:
    """Compile TikZ markup to PDF and rasterize to a clean 256x256 PIL Image."""
    compilation = await compiler.compile_tikz(markup)
    if not compilation.is_successful:
        raise RuntimeError("LaTeX compilation failed: external TeX engine exited with error.")
    png_bytes: bytes = await rasterizer.rasterize_pdf(compilation.pdf_data, dpi=dpi)
    raw_img: Image.Image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return raw_img.resize((256, 256), Image.Resampling.BILINEAR)


def pil_image_to_tensor(
    pil_img: Image.Image, target_height: int = 256, target_width: int = 256
) -> torch.Tensor:
    """Convert a PIL Image into a normalized (1, 3, H, W) float tensor."""
    rgb_img: Image.Image = pil_img.convert("RGB").resize((target_width, target_height))
    np_array = np.asarray(rgb_img, dtype=np.float32) / 255.0  # Shape: (H, W, 3)
    tensor_chw: torch.Tensor = (
        torch.from_numpy(np_array).permute(2, 0, 1).unsqueeze(0)
    )  # Shape: (1, 3, H, W)
    return tensor_chw


def generate_showcase(
    checkpoint_path: str = "results/curriculum_v4/checkpoints/curriculum_v4_best.pt",
    vocabulary_path: str = "dataset/encoded/vocabulary_v4.json",
    output_dir: str = "results/showcase",
    model_dimension: int = 512,
    num_layers: int = 8,
    num_heads: int = 8,
    dim_feedforward: int = 2048,
    num_encoder_blocks: int = 8,
    num_downsampling_stages: int = 3,
    image_size: int = 256,
    max_length: int = 512,
    device: str | None = None,
) -> None:
    """Run neural inference across 6 complexity levels and persist comparison grid."""
    out_path: Path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_device: torch.device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] Initializing V4 model from '{checkpoint_path}' on {target_device}...")
    vocabulary: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(vocabulary_path)
    model: VisionAutoregressiveModelV4 = VisionAutoregressiveModelV4(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=model_dimension,
        max_length=max_length,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_encoder_blocks=num_encoder_blocks,
        num_downsampling_stages=num_downsampling_stages,
        num_families=8,
        dropout=0.10,
        device=target_device,
    )
    checkpoint = AtomicCheckpointAdapter().load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint.model_state)
    model.eval()

    compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(
        engine="pdflatex",
        tikz_libraries=("arrows.meta", "positioning", "patterns", "calc"),
    )
    rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()

    # 6 Ascending complexity levels representative of V4 multi-task capacity
    curated_benchmark: list[tuple[str, str, str, int]] = [
        ("Level 1", "line_segment", "Line & Vector", 42),
        ("Level 2", "circle_arc", "Circle & Arc", 123),
        ("Level 3", "polygon", "Geometric Polygon", 42),
        ("Level 4", "grid_axes", "Cartesian Grid & Axes", 42),
        ("Level 5", "polyline", "Multi-Segment Path", 101),
        ("Level 6", "node_arrow", "Directed Graph Network", 200),
    ]
    num_samples: int = len(curated_benchmark)

    gt_images: list[Image.Image] = []
    pred_images: list[Image.Image] = []
    card_metrics: list[dict[str, str | float]] = []

    print(f"[*] Executing V4 Grammar-Constrained Inference across {num_samples} distinct levels...")
    for idx, (level_name, fam_key, fam_label, seed_val) in enumerate(curated_benchmark):
        rng = np.random.default_rng(seed_val)
        gt_code: str = generate_sample(fam_key, rng)

        print(f"  -> [{idx + 1}/{num_samples}] Ground Truth: {level_name} ({fam_label})...")
        gt_pil: Image.Image = asyncio.run(
            render_markup_to_png(TikzTokens(markup=gt_code), compiler, rasterizer, dpi=140)
        )
        gt_images.append(gt_pil)

        input_tensor: torch.Tensor = pil_image_to_tensor(gt_pil, image_size, image_size).to(
            target_device
        )

        print(f"  -> [{idx + 1}/{num_samples}] Running Grammar-Constrained Greedy Search...")
        indices: tuple[int, ...] = greedy_search(
            model=model,
            image=ImageTensor(raw_tensor=input_tensor),
            max_length=160,
            grammar_constrained=True,
        )
        raw_pred_tokens: TikzTokens = decode_indices_to_markup(vocabulary, indices)
        clean_pred_code: str = clean_tikz_code(raw_pred_tokens.markup)
        final_pred_tokens: TikzTokens = TikzTokens(
            markup=clean_pred_code, packages=raw_pred_tokens.packages
        )

        # Persist individual artifacts
        sample_base: Path = out_path / f"sample_{idx + 1:02d}"
        (sample_base.with_name(sample_base.name + "_gt.tex")).write_text(gt_code, encoding="utf-8")
        (sample_base.with_name(sample_base.name + "_pred.tex")).write_text(
            clean_pred_code, encoding="utf-8"
        )
        gt_pil.save(sample_base.with_name(sample_base.name + "_gt.png"))

        # Render Predicted TikZ Markup via TeX Live + Ghostscript
        print(f"  -> [{idx + 1}/{num_samples}] Compiling Predicted TikZ via TeX Live...")
        compilation_ok: bool = True
        try:
            pred_pil: Image.Image = asyncio.run(
                render_markup_to_png(final_pred_tokens, compiler, rasterizer, dpi=140)
            )
        except Exception as err:
            compilation_ok = False
            pred_pil = Image.new("RGB", (256, 256), color=(255, 255, 255))
            print(f"     [!] Compilation error: {err}")

        pred_images.append(pred_pil)
        pred_pil.save(sample_base.with_name(sample_base.name + "_pred.png"))

        # Compute metric SSIM
        t_gt: torch.Tensor = pil_image_to_tensor(gt_pil, image_size, image_size).squeeze(0)
        t_pred: torch.Tensor = pil_image_to_tensor(pred_pil, image_size, image_size).squeeze(0)
        ssim_val: float = structural_similarity(t_gt, t_pred) if compilation_ok else 0.0

        card_metrics.append(
            {
                "label": f"{level_name}\n({fam_label})",
                "ssim": ssim_val,
                "cr": "100% OK" if compilation_ok else "FAILED",
            }
        )
        print(f"     [+] Metric: SSIM={ssim_val:.3f} | CR={'OK' if compilation_ok else 'FAILED'}")

    # Build High-Resolution Comparison Grid Figure
    print("[*] Generating side-by-side comparison grid...")
    fig, axes = plt.subplots(
        nrows=2,
        ncols=num_samples,
        figsize=(3.4 * num_samples, 7.4),
        facecolor="#0f172a",  # Deep dark slate aesthetic
    )

    for col in range(num_samples):
        # Row 0: Ground Truth Image
        ax_gt = axes[0, col]
        ax_gt.imshow(gt_images[col])
        ax_gt.set_facecolor("#ffffff")
        ax_gt.set_title(
            f"{card_metrics[col]['label']}\n[Ground Truth]",
            fontsize=10.0,
            fontweight="bold",
            color="#f8fafc",
            pad=8,
        )
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        for spine in ax_gt.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

        # Row 1: Model Prediction
        ax_pred = axes[1, col]
        ax_pred.imshow(pred_images[col])
        ax_pred.set_facecolor("#ffffff")
        ssim_score: float = float(card_metrics[col]["ssim"])
        ax_pred.set_title(
            f"TikZfy V4 Neural\nSSIM: {ssim_score:.3f} | CR: {card_metrics[col]['cr']}",
            fontsize=9.2,
            color="#38bdf8" if ssim_score > 0.5 else "#a5b4fc",
            fontweight="bold",
            pad=8,
        )
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        for spine in ax_pred.spines.values():
            spine.set_color("#38bdf8" if ssim_score > 0.5 else "#475569")
            spine.set_linewidth(1.5)

    # Row label annotations on the left margin
    fig.text(
        0.015,
        0.72,
        "GROUND TRUTH\n(Reference Diagram)",
        va="center",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#94a3b8",
        rotation=90,
    )
    fig.text(
        0.015,
        0.28,
        "TIKZFY V4 PREDICTION\n(Neural Autoregressive)",
        va="center",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#38bdf8",
        rotation=90,
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.06, wspace=0.16, hspace=0.28)

    grid_file: Path = out_path / "comparison_grid.png"
    fig.savefig(grid_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[+] TikZfy V4 Showcase comparison grid successfully saved to '{grid_file}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TikZfy V4 showcase comparison grid")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/curriculum_v4/checkpoints/curriculum_v4_best.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="dataset/encoded/vocabulary_v4.json",
        help="Path to vocabulary JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/showcase",
        help="Output directory for grid",
    )
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-ff", type=int, default=2048)
    parser.add_argument("--num-encoder-blocks", type=int, default=8)
    parser.add_argument("--num-downsampling-stages", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_showcase(
        checkpoint_path=args.checkpoint,
        vocabulary_path=args.vocab,
        output_dir=args.output_dir,
        model_dimension=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_ff,
        num_encoder_blocks=args.num_encoder_blocks,
        num_downsampling_stages=args.num_downsampling_stages,
        image_size=args.image_size,
        max_length=args.max_length,
        device=args.device,
    )
