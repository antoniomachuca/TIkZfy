"""Generate a 100% neural visual showcase comparison grid using Spatial-Aligned Model.

Uses CoordConv 2D Cartesian plane injection and Smooth L1 Huber Loss.

Produces an authentic, side-by-side visual comparison across 5 ascending complexity levels:
    Level 1: Canonical Line Segment & Vector Arrow
    Level 2: Canonical Geometric Circle & Ellipse Arc
    Level 3: Canonical Cartesian Grid & Axes
    Level 4: Canonical Directed Node Graph & Automaton
    Level 5: Compositional Hierarchical SCFG Architecture Diagram

Methodology:
    Model Architecture: VisionAutoregressiveModel with CoordConv 2D Cartesian plane injection.
    Trained Objective: SpatialAwareHybridLoss (CrossEntropy + smooth L1 Huber coordinate loss).
    Decoding Strategy: Visual Contrastive Decoding (CFG, gamma = 3.2, T = 0.7, top-p = 0.9).
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.compositional import generate_compositional_batch
from core.dataset.templates import generate_sample
from core.math.spatial import resize_spatial_dimensions
from core.ml.generation import decode_indices_to_markup
from core.ml.metrics import structural_similarity
from core.ml.model import VisionAutoregressiveModel
from core.models import ImageTensor, TikzTokens


def sanitize_raw_tikz(code: str) -> str:
    """Ensure complete TikZ environment and drop trailing unclosed commands."""
    s: str = code.strip()
    if r"\end{tikzpicture}" in s:
        body: str = s.replace(r"\begin{tikzpicture}", "").replace(r"\end{tikzpicture}", "").strip()
        last_semi: int = body.rfind(";")
        if last_semi != -1:
            body = body[: last_semi + 1]
        else:
            body = body + " ;"
        return r"\begin{tikzpicture} " + body + r" \end{tikzpicture}"
    return s + r" \end{tikzpicture}"


def ensure_white_background(pil_img: Image.Image) -> Image.Image:
    """Flatten RGBA image onto a clean white square background."""
    rgb_img: Image.Image
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        rgb_img = bg
    else:
        rgb_img = pil_img.convert("RGB")

    w, h = rgb_img.size
    max_dim = max(w, h, 64)
    square_bg = Image.new("RGB", (max_dim + 24, max_dim + 24), (255, 255, 255))
    offset = ((max_dim + 24 - w) // 2, (max_dim + 24 - h) // 2)
    square_bg.paste(rgb_img, offset)
    return square_bg


async def render_markup_to_png(markup: TikzTokens, dpi: int = 140) -> Image.Image:
    """Compile TikZ markup to PDF and rasterize to a clean white-background PIL Image."""
    compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(
        engine="pdflatex",
        tikz_libraries=("arrows.meta", "positioning", "patterns", "calc"),
    )
    rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()
    compilation = await compiler.compile_tikz(markup)
    if not compilation.is_successful:
        raise RuntimeError("LaTeX compilation failed.")
    png_bytes: bytes = await rasterizer.rasterize_pdf(compilation.pdf_data, dpi=dpi)
    raw_img: Image.Image = Image.open(io.BytesIO(png_bytes))
    return ensure_white_background(raw_img)


def pil_image_to_tensor(
    pil_img: Image.Image, target_height: int = 64, target_width: int = 64
) -> torch.Tensor:
    """Convert a PIL Image into a normalized (1, 3, H, W) float tensor."""
    rgb_img: Image.Image = pil_img.convert("RGB")
    np_array = np.asarray(rgb_img, dtype=np.float32) / 255.0  # Shape: (H, W, 3)
    tensor_chw: torch.Tensor = (
        torch.from_numpy(np_array).permute(2, 0, 1).unsqueeze(0)
    )  # Shape: (1, 3, H, W)
    image_obj: ImageTensor = ImageTensor(raw_tensor=tensor_chw)
    resized_obj: ImageTensor = resize_spatial_dimensions(image_obj, target_height, target_width)
    return resized_obj.raw_tensor  # Shape: (1, 3, target_height, target_width)


def contrastive_visual_decode(
    model: VisionAutoregressiveModel,
    image_tensor: torch.Tensor,
    gamma: float = 3.2,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 80,
) -> tuple[int, ...]:
    """Execute Classifier-Free Guidance (CFG) decoding conditioned on visual features."""
    uncond_img: torch.Tensor = torch.ones_like(image_tensor)
    v_cond: torch.Tensor = model.encoder.forward(image_tensor)
    v_uncond: torch.Tensor = model.encoder.forward(uncond_img)

    gen: torch.Tensor = torch.tensor([[1]], device=image_tensor.device, dtype=torch.long)  # BOS
    step: int = 0
    finished: bool = False

    while step < max_length and not finished:
        l_cond: torch.Tensor = model.decoder.forward(v_cond, gen)[:, -1, :]
        l_uncond: torch.Tensor = model.decoder.forward(v_uncond, gen)[:, -1, :]

        # Visual Contrastive Logit Guidance
        guided_logits: torch.Tensor = l_cond + gamma * (l_cond - l_uncond)
        scaled_logits: torch.Tensor = guided_logits / temperature
        probs: torch.Tensor = F.softmax(scaled_logits, dim=-1)

        # Nucleus Top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs: torch.Tensor = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove: torch.Tensor = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        next_token: torch.Tensor = sorted_indices[
            0, torch.multinomial(sorted_probs[0], 1)
        ].unsqueeze(0)
        gen = torch.cat([gen, next_token], dim=1)
        if next_token.item() == 2:  # EOS
            finished = True
        step += 1

    return tuple(gen[0].tolist())


def generate_showcase(
    checkpoint_path: str = "results/checkpoints/curriculum_v3_best.pt",
    vocabulary_path: str = "dataset/encoded/vocabulary_v3.json",
    output_dir: str = "results/showcase",
    model_dimension: int = 512,
    num_layers: int = 8,
    num_heads: int = 8,
    dim_feedforward: int = 2048,
    num_encoder_blocks: int = 8,
    num_downsampling_stages: int = 3,
    image_size: int = 128,
    device: str | None = None,
) -> None:
    """Run 100% authentic CFG neural inference with Spatial model and persist grid."""
    out_path: Path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_device: torch.device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] Initializing spatial model from '{checkpoint_path}' on {target_device}...")
    vocabulary = JsonVocabularyAdapter().load_vocabulary(vocabulary_path)
    model = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=model_dimension,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_encoder_blocks=num_encoder_blocks,
        use_coord_conv=True,
        num_downsampling_stages=num_downsampling_stages,
        max_length=512,
        device=target_device,
    )
    checkpoint = AtomicCheckpointAdapter().load_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint.model_state)
    model.eval()

    # 5 Ascending validation benchmarks
    curated_benchmark: list[tuple[str, str, str]] = [
        ("Level 1", "Line & Vector", generate_sample("line_segment", np.random.default_rng(42))),
        ("Level 2", "Circle & Arc", generate_sample("circle_arc", np.random.default_rng(42))),
        ("Level 3", "Grid & Axes", generate_sample("grid_axes", np.random.default_rng(42))),
        ("Level 4", "Node Network", generate_sample("node_arrow", np.random.default_rng(42))),
        ("Level 5", "Hierarchical SCFG", generate_compositional_batch(1, seed=42)[0]),
    ]
    num_samples: int = len(curated_benchmark)

    gt_images: list[Image.Image] = []
    pred_images: list[Image.Image] = []
    card_metrics: list[dict[str, str | float]] = []

    print(f"[*] Executing Spatial-Aligned CFG Inference across {num_samples} distinct levels...")
    for idx, (tier_name, fam_name, gt_code) in enumerate(curated_benchmark):
        print(f"  -> [{idx + 1}/{num_samples}] Compiling Ground Truth: {tier_name} ({fam_name})...")
        gt_pil: Image.Image = asyncio.run(render_markup_to_png(TikzTokens(markup=gt_code), dpi=140))
        gt_images.append(gt_pil)

        # Encode Ground Truth to tensor
        input_tensor: torch.Tensor = pil_image_to_tensor(gt_pil, image_size, image_size).to(
            target_device
        )

        # 100% Neural Inactive Logit CFG Decoding with Spatial-Aware model
        print(f"  -> [{idx + 1}/{num_samples}] Running Visual Contrastive Decoding (gamma=3.2)...")
        token_indices: tuple[int, ...] = contrastive_visual_decode(
            model=model,
            image_tensor=input_tensor,
            gamma=3.2,
            temperature=0.7,
            top_p=0.9,
            max_length=80,
        )
        raw_pred_tokens: TikzTokens = decode_indices_to_markup(vocabulary, token_indices)
        clean_pred_code: str = sanitize_raw_tikz(raw_pred_tokens.markup)
        final_pred_tokens: TikzTokens = TikzTokens(
            markup=clean_pred_code, packages=raw_pred_tokens.packages
        )

        # Save individual TeX and PNG artifacts
        sample_base: Path = out_path / f"sample_{idx + 1:02d}"
        (sample_base.with_name(sample_base.name + "_gt.tex")).write_text(gt_code, encoding="utf-8")
        (sample_base.with_name(sample_base.name + "_pred.tex")).write_text(
            clean_pred_code, encoding="utf-8"
        )
        gt_pil.save(sample_base.with_name(sample_base.name + "_gt.png"))

        # Render Predicted TikZ Markup via TeX Live + Ghostscript
        print(f"  -> [{idx + 1}/{num_samples}] Compiling Spatial Prediction via TeX Live...")
        compilation_ok: bool = True
        try:
            pred_pil: Image.Image = asyncio.run(render_markup_to_png(final_pred_tokens, dpi=140))
        except Exception as err:
            compilation_ok = False
            pred_pil = Image.new("RGB", (256, 256), color=(255, 255, 255))
            print(f"     [!] Compilation error: {err}")

        pred_images.append(pred_pil)
        pred_pil.save(sample_base.with_name(sample_base.name + "_pred.png"))

        # Compute SSIM metric
        t_gt: torch.Tensor = pil_image_to_tensor(gt_pil, image_size, image_size)
        t_pred: torch.Tensor = pil_image_to_tensor(pred_pil, image_size, image_size)
        ssim_val: float = (
            structural_similarity(t_gt.squeeze(0), t_pred.squeeze(0)) if compilation_ok else 0.0
        )

        card_metrics.append(
            {
                "label": f"{tier_name}\n({fam_name})",
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
        figsize=(3.6 * num_samples, 7.6),
        facecolor="#0f172a",  # Deep dark slate aesthetic
    )

    for col in range(num_samples):
        # Row 0: Ground Truth Image
        ax_gt = axes[0, col]
        ax_gt.imshow(gt_images[col])
        ax_gt.set_facecolor("#ffffff")
        ax_gt.set_title(
            f"{card_metrics[col]['label']}\n[Ground Truth]",
            fontsize=10.5,
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
            f"Spatial-Aligned Neural\nSSIM: {ssim_score:.3f} | CR: {card_metrics[col]['cr']}",
            fontsize=9.5,
            color="#38bdf8" if ssim_score > 0.4 else "#a5b4fc",
            fontweight="bold",
            pad=8,
        )
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        for spine in ax_pred.spines.values():
            spine.set_color("#38bdf8" if ssim_score > 0.4 else "#475569")
            spine.set_linewidth(1.5)

    # Row label annotations on the left margin
    fig.text(
        0.015,
        0.72,
        "GROUND TRUTH\n(Reference Input)",
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#94a3b8",
        rotation=90,
    )
    fig.text(
        0.015,
        0.28,
        "SPATIAL-ALIGNED TIKZ\n(CoordConv + Huber Decoded)",
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
        rotation=90,
    )

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.06, wspace=0.18, hspace=0.28)

    grid_file: Path = out_path / "comparison_grid.png"
    fig.savefig(grid_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[+] Spatial-Aligned Showcase comparison grid successfully saved to '{grid_file}'!")


if __name__ == "__main__":
    import argparse

    def _resolve_default_checkpoint() -> str:
        candidates = [
            "results/checkpoints/curriculum_v3_best.pt",
            "results/checkpoints/curriculum_v2_best.pt",
            "checkpoints/grounded_best_model.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def _resolve_default_vocab() -> str:
        candidates = [
            "dataset/encoded/vocabulary_v3.json",
            "dataset/encoded/vocabulary.json",
        ]
        for v in candidates:
            if os.path.exists(v):
                return v
        return candidates[0]

    parser = argparse.ArgumentParser(description="Generate showcase comparison grid")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=_resolve_default_checkpoint(),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=_resolve_default_vocab(),
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
    parser.add_argument("--image-size", type=int, default=128)
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
        device=args.device,
    )
