"""Smoke benchmark for Phase 1: Grammar-Constrained Decoding and Best-of-N Re-ranking.

Evaluates the V3 checkpoint on representative samples across 5 geometric families,
contrasting:
    1. Baseline Unconstrained Greedy
    2. Grammar-Constrained Greedy
    3. Execution-Guided Best-of-4 Re-ranking (SSIM feedback)

References:
    Goodfellow et al., Deep Learning - empirical decoding validation.
"""

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.orchestrator import ImageToTikzOrchestrator
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from core.ml.generation import (
    best_of_n_search,
    decode_indices_to_markup,
    greedy_search,
)
from core.ml.metrics import structural_similarity
from core.models import ImageTensor, TikzTokens


def load_png_to_tensor(png_path: Path, image_size: int = 128) -> torch.Tensor:
    """Load an RGB PNG from disk into a normalized float32 tensor of shape (3, H, W)."""
    img: Image.Image = Image.open(png_path).convert("RGB")
    arr: np.ndarray[Any, Any] = np.asarray(img, dtype=np.float32) / 255.0
    tensor: torch.Tensor = torch.from_numpy(arr).permute(2, 0, 1)
    if tensor.shape[1:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


async def compile_and_rasterize(
    markup: TikzTokens,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    image_size: int = 128,
) -> tuple[bool, torch.Tensor | None]:
    """Compile TikZ markup and rasterize to (3, H, W) float32 tensor."""
    try:
        res = await compiler.compile_tikz(markup)
        if not res.is_successful or not res.pdf_data:
            return False, None

        png_bytes = await rasterizer.rasterize_pdf(res.pdf_data, dpi=72)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        if t.shape[1:] != (image_size, image_size):
            t = F.interpolate(
                t.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return True, t
    except Exception:
        return False, None


async def run_smoke_benchmark() -> None:
    """Execute the comparative smoke benchmark across representative samples."""
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_path = repo_root / "results" / "checkpoints" / "curriculum_v3_best.pt"
    vocab_path = repo_root / "dataset" / "encoded" / "vocabulary_v3.json"
    diag_dir = repo_root / "results" / "diagnostics" / "v3_decode_comparison"
    out_file = repo_root / "results" / "diagnostics" / "smoke_phase1_results.json"

    print("[*] Initializing Phase 1 Smoke Benchmark...")
    print(f"[*] Loading checkpoint: {ckpt_path.name}")
    print(f"[*] Loading vocabulary: {vocab_path.name}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Device: {device}")

    orchestrator = ImageToTikzOrchestrator.from_checkpoint(
        checkpoint_path=ckpt_path,
        vocabulary_path=vocab_path,
        device=device,
        max_length=512,
    )
    model = orchestrator.model
    vocabulary = orchestrator.vocabulary

    compiler = AsyncTexLiveAdapter()
    rasterizer = GhostscriptRasterizer()

    families: list[str] = ["line_segment", "circle_arc", "grid_axes", "node_arrow", "composed"]
    samples_per_family: int = 2

    results: list[dict[str, Any]] = []

    print("\n" + "=" * 92)
    header = (
        f"{'Family':<14} {'Sample':<8} {'Greedy SSIM':<13} "
        f"{'Grammar SSIM':<14} {'Best-of-4 SSIM':<16} {'CR Status'}"
    )
    print(header)
    print("=" * 92)

    for fam in families:
        fam_dir = diag_dir / fam
        for s_idx in range(samples_per_family):
            sample_name = f"sample_{s_idx:02d}"
            sample_dir = fam_dir / sample_name
            ref_png_path = sample_dir / "reference.png"

            if not ref_png_path.exists():
                print(f"[!] Warning: {ref_png_path} not found, skipping.")
                continue

            ref_tensor = load_png_to_tensor(ref_png_path, image_size=128)
            image_obj = ImageTensor(raw_tensor=ref_tensor.unsqueeze(0).to(device))

            # Policy 1: Standard Greedy
            t0 = time.time()
            greedy_idx = greedy_search(model, image_obj, max_length=512, grammar_constrained=False)
            greedy_tokens = decode_indices_to_markup(vocabulary, greedy_idx)
            ok_greedy, t_greedy = await compile_and_rasterize(greedy_tokens, compiler, rasterizer)
            ssim_greedy = 0.0
            if ok_greedy and t_greedy is not None:
                ssim_greedy = structural_similarity(ref_tensor, t_greedy)
            time_greedy = time.time() - t0

            # Policy 2: Grammar-Constrained Greedy
            t0 = time.time()
            grammar_idx = greedy_search(model, image_obj, max_length=512, grammar_constrained=True)
            grammar_tokens = decode_indices_to_markup(vocabulary, grammar_idx)
            ok_grammar, t_grammar = await compile_and_rasterize(
                grammar_tokens, compiler, rasterizer
            )
            ssim_grammar = 0.0
            if ok_grammar and t_grammar is not None:
                ssim_grammar = structural_similarity(ref_tensor, t_grammar)
            time_grammar = time.time() - t0

            # Policy 3: Best-of-4 Re-ranked
            t0 = time.time()
            candidates = best_of_n_search(
                model,
                image_obj,
                n_hypotheses=4,
                max_length=512,
                temperature=0.6,
                top_p=0.9,
                grammar_constrained=True,
            )
            best_ssim = 0.0
            ok_best = False

            c_idx = 0
            while c_idx < len(candidates):
                cand_tok = decode_indices_to_markup(vocabulary, candidates[c_idx])
                cand_ok, cand_t = await compile_and_rasterize(cand_tok, compiler, rasterizer)
                if cand_ok and cand_t is not None:
                    score = structural_similarity(ref_tensor, cand_t)
                    if score > best_ssim:
                        best_ssim = score
                        ok_best = True
                c_idx += 1
            time_best = time.time() - t0

            status_str = (
                f"G:{'OK' if ok_greedy else 'FAIL'} "
                f"Gr:{'OK' if ok_grammar else 'FAIL'} "
                f"B4:{'OK' if ok_best else 'FAIL'}"
            )
            row = (
                f"{fam:<14} {sample_name:<8} {ssim_greedy:<13.4f} "
                f"{ssim_grammar:<14.4f} {best_ssim:<16.4f} {status_str}"
            )
            print(row)

            results.append({
                "family": fam,
                "sample": sample_name,
                "greedy": {
                    "ssim": ssim_greedy,
                    "compilation_ok": ok_greedy,
                    "time_sec": time_greedy,
                },
                "grammar_greedy": {
                    "ssim": ssim_grammar,
                    "compilation_ok": ok_grammar,
                    "time_sec": time_grammar,
                },
                "best_of_4": {
                    "ssim": best_ssim,
                    "compilation_ok": ok_best,
                    "time_sec": time_best,
                },
            })

    print("=" * 92)

    # Compute aggregate metrics
    greedy_ssims = [r["greedy"]["ssim"] for r in results]
    grammar_ssims = [r["grammar_greedy"]["ssim"] for r in results]
    best_ssims = [r["best_of_4"]["ssim"] for r in results]

    mean_greedy = float(np.mean(greedy_ssims))
    mean_grammar = float(np.mean(grammar_ssims))
    mean_best = float(np.mean(best_ssims))

    cr_greedy = sum(1 for r in results if r["greedy"]["compilation_ok"]) / len(results) * 100.0
    cr_grammar = (
        sum(1 for r in results if r["grammar_greedy"]["compilation_ok"]) / len(results) * 100.0
    )
    cr_best = sum(1 for r in results if r["best_of_4"]["compilation_ok"]) / len(results) * 100.0

    print("\n--- RESUMEN CONSOLIDADO DEL SMOKE BENCHMARK ---")
    print(f"Total Muestras Evaluadas: {len(results)}")
    print(f"Greedy Baseline:        SSIM = {mean_greedy:.4f} | CR = {cr_greedy:.1f}%")
    delta_gr = mean_grammar - mean_greedy
    delta_b4 = mean_best - mean_greedy
    print(f"Grammar-Constrained:    SSIM = {mean_grammar:.4f} | CR = {cr_grammar:.1f}%")
    print(f"                        (Delta SSIM: {delta_gr:+.4f})")
    print(f"Best-of-4 Re-ranking:   SSIM = {mean_best:.4f} | CR = {cr_best:.1f}%")
    print(f"                        (Delta SSIM: {delta_b4:+.4f})")
    print("-" * 50)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "samples_evaluated": len(results),
        "mean_ssim": {
            "greedy": mean_greedy,
            "grammar_greedy": mean_grammar,
            "best_of_4": mean_best,
        },
        "compilation_rate_percent": {
            "greedy": cr_greedy,
            "grammar_greedy": cr_grammar,
            "best_of_4": cr_best,
        },
        "detailed_results": results,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Resultados exportados a {out_file}")


def main() -> None:
    asyncio.run(run_smoke_benchmark())


if __name__ == "__main__":
    main()
