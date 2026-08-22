"""End-to-end smoke test: image -> model -> TikZ -> compile -> raster.

Selects a small set of validation samples, decodes them greedily, compiles each
prediction with TeX Live, rasterizes the PDF with Ghostscript, and renders a
side-by-side comparison grid.

References:
    Goodfellow et al., Deep Learning - end-to-end system integration testing.
"""

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from core.exceptions import DomainError
from core.ml.generation import decode_indices_to_markup, greedy_search
from core.models import ImageTensor, TikzTokens, TokenVocabulary
from scripts.evaluate_baseline import load_model


async def render_markup(markup: TikzTokens) -> bytes:
    """Compile and rasterize one markup into PNG bytes."""
    compiler: AsyncTexLiveAdapter = AsyncTexLiveAdapter(engine="pdflatex")
    rasterizer: GhostscriptRasterizer = GhostscriptRasterizer()
    compilation = await compiler.compile_tikz(markup)
    return await rasterizer.rasterize_pdf(compilation.pdf_data)


def save_input_image(tensor: torch.Tensor, path: Path) -> None:
    """Persist a ``(C, H, W)`` float image as a PNG via matplotlib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, tensor.permute(1, 2, 0).numpy())


def smoke_test(arguments: argparse.Namespace) -> None:
    """Run the end-to-end smoke test and persist showcase artifacts.

    Scans validation samples until ``num_samples`` predictions compile
    successfully, then renders the side-by-side comparison grid.
    """
    encoded_dir: Path = arguments.encoded_dir
    results_dir: Path = arguments.results_dir
    with (results_dir / "training_results.json").open("r", encoding="utf-8") as handle:
        config: dict[str, object] = json.load(handle)["config"]

    val_images: torch.Tensor = torch.load(encoded_dir / "val_images.pt", weights_only=True)
    model = load_model(encoded_dir, arguments.checkpoint, config)
    vocabulary: TokenVocabulary = model.vocabulary

    showcase_dir: Path = results_dir / "showcase"
    showcase_dir.mkdir(parents=True, exist_ok=True)

    input_paths: list[Path] = []
    output_paths: list[Path] = []
    tex_paths: list[Path] = []
    attempts: int = 0
    successes: int = 0
    index: int = 0
    total: int = int(val_images.shape[0])

    while successes < arguments.num_samples and index < total:
        image: ImageTensor = ImageTensor(raw_tensor=val_images[index].unsqueeze(0))
        token_indices: tuple[int, ...] = greedy_search(
            model, image, max_length=arguments.max_length
        )
        markup: TikzTokens = decode_indices_to_markup(vocabulary, token_indices)
        attempts += 1

        try:
            output_png: bytes = asyncio.run(render_markup(markup))
        except DomainError:
            index += 1
            continue

        base: Path = showcase_dir / f"sample_{successes:02d}"
        input_path: Path = base.with_name(base.name + "_input.png")
        output_path: Path = base.with_name(base.name + "_output.png")
        tex_path: Path = base.with_name(base.name + ".tex")
        save_input_image(val_images[index], input_path)
        tex_path.write_text(markup.markup, encoding="utf-8")
        output_path.write_bytes(output_png)
        input_paths.append(input_path)
        output_paths.append(output_path)
        tex_paths.append(tex_path)
        successes += 1
        print(f"[{successes}/{arguments.num_samples}] compiled sample #{index}: {tex_path}")
        index += 1

    print(f"Scanned {attempts} samples to collect {successes} compilable predictions.")

    if not input_paths:
        print("No compilable predictions found; skipping grid rendering.")
        return

    grid_path: Path = showcase_dir / "comparison_grid.png"
    figure, axes = plt.subplots(2, len(input_paths), figsize=(3 * len(input_paths), 6))
    for column in range(len(input_paths)):
        axes[0, column].imshow(mpimg.imread(input_paths[column]))
        axes[0, column].set_title(f"input {column}")
        axes[0, column].axis("off")
        axes[1, column].imshow(mpimg.imread(output_paths[column]))
        axes[1, column].set_title(f"generated {column}")
        axes[1, column].axis("off")
    figure.tight_layout()
    figure.savefig(grid_path, dpi=120)
    plt.close(figure)
    print(f"Showcase persisted to '{showcase_dir}'.")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI contract for the smoke test."""
    repo_root: Path = Path(__file__).resolve().parent.parent
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the image-to-TikZ end-to-end smoke test."
    )
    parser.add_argument(
        "--encoded-dir", type=Path, default=repo_root / "dataset" / "encoded"
    )
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_root / "results" / "checkpoints" / "checkpoint_epoch_020.pt",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=5)
    return parser


def main() -> None:
    """Run the smoke-test entrypoint."""
    smoke_test(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
