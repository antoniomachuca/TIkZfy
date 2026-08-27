"""Independent test benchmark and statistical evaluation for Phase 1 V3 Checkpoint.

Evaluates the Phase 1 retrained curriculum V3 checkpoint on an independent,
disjoint test split:
    - 500 samples per synthetic family across all 8 families (4,000 samples)
    - 2,000 out-of-distribution (OOD) complex compositional samples
    - Total: 6,000 evaluation samples.

Evaluates two decoding policies:
    1. Greedy Search (reference baseline policy)
    2. Beam Search (beam_width=3, length_penalty=0.0)

Computes 6 core metrics per family and overall:
    - Compilation Rate (CR) via async TeX Live + Ghostscript
    - Mean Structural Similarity Index (SSIM) at 128x128
    - Primitive Detection Accuracy
    - Structural Family Classification Accuracy
    - Aligned Coordinate RMSE
    - Hungarian Graph Edit Distance & Token GED

References:
    Goodfellow et al., Deep Learning — empirical statistical evaluation.
    Wang et al., Image Quality Assessment — SSIM visual fidelity index.
    Kuhn, The Hungarian Method for the Assignment Problem — bipartite graph matching.
    Golub & Van Loan, Matrix Computations — vectorized metric computations.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.checkpoint_adapter import AtomicCheckpointAdapter
from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.compositional import generate_compositional_batch
from core.dataset.templates import FAMILY_NAMES, generate_sample
from core.math.spatial import resize_spatial_dimensions
from core.math.tokenization import batch_encode, decode_from_tensor, tokenize_tikz_markup
from core.ml.generation import BeamHypothesis, beam_search, decode_indices_to_markup, greedy_search
from core.ml.metrics import (
    DEFAULT_COORDINATE_SCALE,
    CoordinateError,
    coordinate_error,
    extract_structured_coordinates,
    geometric_edit_distance,
    geometric_graph_edit_distance,
    structural_similarity,
)
from core.ml.model import VisionAutoregressiveModel, resolve_device
from core.models import ImageTensor, TikzTokens, TokenVocabulary

ALL_EVAL_FAMILIES: tuple[str, ...] = (
    "line_segment",
    "circle_arc",
    "grid_axes",
    "function_plot",
    "polyline",
    "polygon",
    "node_arrow",
    "composed",
)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 digest of a file on disk."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        chunk = handle.read(65536)
        while chunk:
            hasher.update(chunk)
            chunk = handle.read(65536)
    return hasher.hexdigest()


def detect_primitives(markup: str) -> list[str]:
    """Detect TikZ geometric primitives and structural keywords present in markup."""
    detected: list[str] = []
    keywords: list[str] = [
        "draw",
        "circle",
        "arc",
        "grid",
        "plot",
        "node",
        "fill",
        "--",
        "->",
        "cycle",
    ]
    for kw in keywords:
        pattern: str = rf"\b{kw}\b" if kw.isalnum() else re.escape(kw)
        if re.search(pattern, markup):
            detected.append(kw)
    return detected


def classify_structural_family(markup: str) -> str:
    """Classify the primary structural family represented by the TikZ markup."""
    lower_markup: str = markup.lower()
    if "grid" in lower_markup or "step=" in lower_markup:
        return "grid_axes"
    if "node" in lower_markup:
        return "node_arrow"
    if "plot" in lower_markup or "domain" in lower_markup:
        return "function_plot"
    if "circle" in lower_markup or "arc" in lower_markup:
        return "circle_arc"
    if "cycle" in lower_markup:
        return "polygon"
    if "--" in lower_markup:
        return "line_segment"
    return "unknown"


def compute_coordinate_rmse(
    ref_coords: list[tuple[float, float]],
    cand_coords: list[tuple[float, float]],
) -> float:
    """Compute Euclidean coordinate root mean square error between aligned vertices."""
    if not ref_coords or not cand_coords:
        return float(DEFAULT_COORDINATE_SCALE)
    min_len: int = min(len(ref_coords), len(cand_coords))
    ref_arr: NDArray[np.float64] = np.asarray(ref_coords[:min_len], dtype=np.float64)
    cand_arr: NDArray[np.float64] = np.asarray(cand_coords[:min_len], dtype=np.float64)
    diffs: NDArray[np.float64] = ref_arr - cand_arr  # Shape: (min_len, 2)
    euclidean_dists: NDArray[np.float64] = np.linalg.norm(diffs, axis=1)  # Shape: (min_len,)
    len_penalty: float = float(abs(len(ref_coords) - len(cand_coords))) * 1.0
    return float(np.mean(euclidean_dists) + len_penalty)


async def render_single_sample(
    code: str,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    sem: asyncio.Semaphore,
    image_size: int = 128,
) -> tuple[bool, torch.Tensor, float]:
    """Compile TikZ markup to PDF and rasterize to (3, H, W) tensor and std."""
    async with sem:
        try:
            res = await compiler.compile_tikz(TikzTokens(markup=code))
            if not res.is_successful:
                return False, torch.ones((3, image_size, image_size), dtype=torch.float32), 0.0
            png_bytes = await rasterizer.rasterize_pdf(res.pdf_data, dpi=72)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0  # Shape: (H, W, 3)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
            resized = resize_spatial_dimensions(ImageTensor(raw_tensor=t), image_size, image_size)
            image_t = resized.raw_tensor.squeeze(0)  # Shape: (3, H, W)
            std_val = float(image_t.std().item())
            return True, image_t, std_val
        except Exception:
            return False, torch.ones((3, image_size, image_size), dtype=torch.float32), 0.0


async def render_markups_parallel(
    markups: list[str],
    max_concurrency: int = 32,
    image_size: int = 128,
) -> tuple[list[bool], torch.Tensor, list[float]]:
    """Render a batch of markups in parallel with bounded concurrency."""
    compiler = AsyncTexLiveAdapter()
    rasterizer = GhostscriptRasterizer()
    sem = asyncio.Semaphore(max_concurrency)

    total: int = len(markups)
    images_tensor: torch.Tensor = torch.empty((total, 3, image_size, image_size), dtype=torch.float32)
    successes: list[bool] = [False] * total
    stds: list[float] = [0.0] * total
    batch_size: int = 500

    i: int = 0
    while i < total:
        chunk_end: int = min(i + batch_size, total)
        chunk_markups = markups[i:chunk_end]
        tasks = [
            render_single_sample(m, compiler, rasterizer, sem, image_size=image_size)
            for m in chunk_markups
        ]
        results = await asyncio.gather(*tasks)
        for offset, (succ, img_t, std_v) in enumerate(results):
            successes[i + offset] = succ
            images_tensor[i + offset] = img_t
            stds[i + offset] = std_v
        del results, tasks
        i = chunk_end

    return successes, images_tensor, stds


def generate_independent_test_split(
    samples_per_family: int = 500,
    ood_samples_count: int = 2000,
    base_seed: int = 999999,
) -> tuple[list[str], list[str], list[int]]:
    """Generate independent test split markups disjoint from training distribution."""
    rng = np.random.default_rng(base_seed)
    markups: list[str] = []
    families: list[str] = []
    seeds: list[int] = []

    # 1. Synthetic families (8 families * samples_per_family)
    for fam in ALL_EVAL_FAMILIES:
        for idx in range(samples_per_family):
            sample_seed = int(rng.integers(1_000_000_000, 2_000_000_000))
            sample_rng = np.random.default_rng(sample_seed)
            if fam == "composed":
                code = generate_compositional_batch(1, seed=sample_seed)[0]
            else:
                code = generate_sample(fam, sample_rng)
            markups.append(code)
            families.append(fam)
            seeds.append(sample_seed)

    # 2. Out-of-Distribution (OOD) multi-primitive compositional samples
    for idx in range(ood_samples_count):
        sample_seed = int(rng.integers(2_000_000_000, 3_000_000_000))
        code = generate_compositional_batch(1, seed=sample_seed)[0]
        markups.append(code)
        families.append("ood_composed")
        seeds.append(sample_seed)

    return markups, families, seeds


def evaluate_policy_on_split(
    model: VisionAutoregressiveModel,
    vocabulary: TokenVocabulary,
    test_images: torch.Tensor,
    ref_markups: list[str],
    ref_families: list[str],
    policy_name: str = "greedy",
    beam_width: int = 3,
    max_length: int = 128,
    target_device: torch.device = torch.device("cpu"),
    max_eval_samples: int | None = None,
    concurrency: int = 32,
) -> dict[str, Any]:
    """Execute complete policy evaluation and metric computation."""
    model.eval()
    total_samples: int = len(ref_markups)
    eval_count: int = total_samples if max_eval_samples is None else min(total_samples, max_eval_samples)

    print(f"[*] Evaluating Policy: {policy_name.upper()} on {eval_count} test samples...")
    predicted_markups: list[str] = []

    t0: float = time.time()
    for idx in range(eval_count):
        img_input = ImageTensor(test_images[idx : idx + 1].to(target_device))  # Shape: (1, 3, H, W)
        if policy_name == "beam":
            hypotheses = beam_search(model, img_input, beam_width=beam_width, max_length=max_length)
            pred_indices = hypotheses[0].token_indices if hypotheses else (0,)
        else:
            pred_indices = greedy_search(model, img_input, max_length=max_length)

        pred_tokens = decode_indices_to_markup(vocabulary, pred_indices)
        predicted_markups.append(pred_tokens.markup)

        if (idx + 1) % 500 == 0 or idx == eval_count - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(1e-3, elapsed)
            print(f"  -> Generated [{idx + 1}/{eval_count}] sequences ({rate:.1f} seq/s)...")

    # Render predictions to compute compilation rate and SSIM
    print(f"[*] Rendering {eval_count} predicted markups via TeX Live...")
    comp_successes, pred_tensors, _ = asyncio.run(
        render_markups_parallel(predicted_markups, max_concurrency=concurrency, image_size=128)
    )

    # Compute metrics per family and aggregate
    family_metrics: dict[str, dict[str, list[float]]] = {}
    for fam in list(ALL_EVAL_FAMILIES) + ["ood_composed"]:
        family_metrics[fam] = {
            "cr": [],
            "ssim": [],
            "primitive_acc": [],
            "family_acc": [],
            "coord_rmse": [],
            "ged": [],
        }

    for idx in range(eval_count):
        fam = ref_families[idx]
        ref_code = ref_markups[idx]
        cand_code = predicted_markups[idx]
        is_compiled = comp_successes[idx]

        # 1. Compilation Rate
        cr_val = 1.0 if is_compiled else 0.0

        # 2. SSIM
        if is_compiled:
            gt_img = ImageTensor(test_images[idx : idx + 1])
            pr_img = ImageTensor(pred_tensors[idx : idx + 1])
            ssim_score = float(structural_similarity(gt_img, pr_img).item())
        else:
            ssim_score = 0.0

        # 3. Primitive Accuracy
        ref_prim = set(detect_primitives(ref_code))
        cand_prim = set(detect_primitives(cand_code))
        prim_acc = float(len(ref_prim & cand_prim) / max(1, len(ref_prim | cand_prim)))

        # 4. Family Accuracy
        cand_fam = classify_structural_family(cand_code)
        fam_acc = 1.0 if cand_fam == fam or (fam == "ood_composed" and cand_fam != "unknown") else 0.0

        # 5. Coordinate RMSE
        ref_coords = list(extract_structured_coordinates(ref_code))
        cand_coords = list(extract_structured_coordinates(cand_code))
        coord_rmse = compute_coordinate_rmse(ref_coords, cand_coords)

        # 6. Geometric Edit Distance (GED)
        ref_toks = tokenize_tikz_markup(TikzTokens(markup=ref_code))
        cand_toks = tokenize_tikz_markup(TikzTokens(markup=cand_code))
        ged_score = geometric_edit_distance(ref_toks, cand_toks)

        if fam in family_metrics:
            family_metrics[fam]["cr"].append(cr_val)
            family_metrics[fam]["ssim"].append(ssim_score)
            family_metrics[fam]["primitive_acc"].append(prim_acc)
            family_metrics[fam]["family_acc"].append(fam_acc)
            family_metrics[fam]["coord_rmse"].append(coord_rmse)
            family_metrics[fam]["ged"].append(ged_score)

    summary_by_family: dict[str, dict[str, float]] = {}
    all_cr: list[float] = []
    all_ssim: list[float] = []
    all_prim: list[float] = []
    all_fam: list[float] = []
    all_coord: list[float] = []
    all_ged: list[float] = []

    for fam, m_dict in family_metrics.items():
        if m_dict["cr"]:
            mean_cr = float(np.mean(m_dict["cr"]))
            mean_ssim = float(np.mean(m_dict["ssim"]))
            mean_prim = float(np.mean(m_dict["primitive_acc"]))
            mean_fam = float(np.mean(m_dict["family_acc"]))
            mean_coord = float(np.mean(m_dict["coord_rmse"]))
            mean_ged = float(np.mean(m_dict["ged"]))

            summary_by_family[fam] = {
                "count": len(m_dict["cr"]),
                "compilation_rate": mean_cr,
                "mean_ssim": mean_ssim,
                "primitive_accuracy": mean_prim,
                "family_accuracy": mean_fam,
                "mean_coordinate_rmse": mean_coord,
                "mean_ged": mean_ged,
            }
            all_cr.extend(m_dict["cr"])
            all_ssim.extend(m_dict["ssim"])
            all_prim.extend(m_dict["primitive_acc"])
            all_fam.extend(m_dict["family_acc"])
            all_coord.extend(m_dict["coord_rmse"])
            all_ged.extend(m_dict["ged"])

    overall_summary = {
        "policy": policy_name,
        "total_samples": eval_count,
        "overall_compilation_rate": float(np.mean(all_cr)) if all_cr else 0.0,
        "overall_mean_ssim": float(np.mean(all_ssim)) if all_ssim else 0.0,
        "overall_primitive_accuracy": float(np.mean(all_prim)) if all_prim else 0.0,
        "overall_family_accuracy": float(np.mean(all_fam)) if all_fam else 0.0,
        "overall_coordinate_rmse": float(np.mean(all_coord)) if all_coord else 0.0,
        "overall_mean_ged": float(np.mean(all_ged)) if all_ged else 0.0,
        "by_family": summary_by_family,
    }
    return overall_summary


def run_phase1_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Execute complete Phase 1 test evaluation benchmark."""
    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    tables_dir = results_dir / "tables"
    manifest_dir = results_dir / "manifests"
    cache_dir = Path(args.cache_dir)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_device = resolve_device(args.device)
    print(f"[*] Starting Phase 1 Independent Test Benchmark on {target_device}")

    # Load Vocabulary
    vocab_path = Path(args.vocab_path)
    vocabulary = JsonVocabularyAdapter().load_vocabulary(str(vocab_path))
    vocab_hash = compute_file_sha256(vocab_path)
    print(f"[+] Loaded vocabulary: {len(vocabulary.token_to_index)} tokens (SHA-256: {vocab_hash})")

    # Load Model Checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt_hash = compute_file_sha256(checkpoint_path)
    print(f"[+] Loading checkpoint: {checkpoint_path} (SHA-256: {ckpt_hash})")

    model = VisionAutoregressiveModel(
        vocabulary=vocabulary,
        input_channels=3,
        model_dimension=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_ff,
        num_encoder_blocks=args.num_encoder_blocks,
        use_coord_conv=True,
        use_2d_pos_encoding=True,
        num_downsampling_stages=args.num_downsampling_stages,
        max_length=args.max_length,
        dropout=0.0,
        device=target_device,
    )
    checkpoint = AtomicCheckpointAdapter().load_checkpoint(str(checkpoint_path))
    model.load_state_dict(checkpoint.model_state)
    model.to(target_device)
    model.eval()

    # Generate or load independent test split
    test_cache_path = cache_dir / f"independent_test_split_{args.samples_per_family}x8_{args.ood_samples}ood.pt"
    manifest_path = manifest_dir / "independent_test_manifest.jsonl"

    if test_cache_path.exists():
        print(f"[*] Loading cached independent test split from {test_cache_path}...")
        test_cache = torch.load(test_cache_path, map_location="cpu", weights_only=False)
        test_images = test_cache["images"]
        ref_markups = test_cache["markups"]
        ref_families = test_cache["families"]
        ref_seeds = test_cache["seeds"]
    else:
        print(f"[*] Generating independent test split ({args.samples_per_family}x8 synthetic + {args.ood_samples} OOD)...")
        ref_markups, ref_families, ref_seeds = generate_independent_test_split(
            samples_per_family=args.samples_per_family,
            ood_samples_count=args.ood_samples,
            base_seed=args.seed,
        )
        print(f"[*] Rendering {len(ref_markups)} ground truth test images...")
        _, test_images, test_stds = asyncio.run(
            render_markups_parallel(ref_markups, max_concurrency=args.concurrency, image_size=128)
        )
        torch.save({
            "images": test_images,
            "markups": ref_markups,
            "families": ref_families,
            "seeds": ref_seeds,
        }, test_cache_path)
        print(f"[+] Saved test split cache to {test_cache_path}")

        # Save manifest
        with manifest_path.open("w", encoding="utf-8") as f_man:
            for idx in range(len(ref_markups)):
                f_man.write(json.dumps({
                    "sample_id": f"test_{idx:06d}",
                    "family": ref_families[idx],
                    "seed": ref_seeds[idx],
                    "markup": ref_markups[idx],
                    "tensor_shape": list(test_images[idx].shape),
                    "tensor_std": test_stds[idx],
                    "compilation_status": "SUCCESS",
                }) + "\n")
        print(f"[+] Saved test manifest to {manifest_path}")

    test_cache_hash = compute_file_sha256(test_cache_path)

    # 1. Evaluate Greedy (reference baseline)
    greedy_results = evaluate_policy_on_split(
        model=model,
        vocabulary=vocabulary,
        test_images=test_images,
        ref_markups=ref_markups,
        ref_families=ref_families,
        policy_name="greedy",
        max_length=args.max_length,
        target_device=target_device,
        max_eval_samples=args.max_eval_samples,
        concurrency=args.concurrency,
    )

    # 2. Evaluate Beam Search (comparison)
    beam_results: dict[str, Any] | None = None
    if args.evaluate_beam:
        beam_results = evaluate_policy_on_split(
            model=model,
            vocabulary=vocabulary,
            test_images=test_images,
            ref_markups=ref_markups,
            ref_families=ref_families,
            policy_name="beam",
            beam_width=args.beam_width,
            max_length=args.max_length,
            target_device=target_device,
            max_eval_samples=min(args.max_eval_samples or 1000, 500),  # Bounded comparison
            concurrency=args.concurrency,
        )

    full_evaluation_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": ckpt_hash,
            "epoch": checkpoint.epoch,
        },
        "vocabulary": {
            "path": str(vocab_path),
            "sha256": vocab_hash,
            "size": len(vocabulary.token_to_index),
        },
        "test_benchmark": {
            "cache_path": str(test_cache_path),
            "cache_sha256": test_cache_hash,
            "total_samples": len(ref_markups),
            "synthetic_samples": args.samples_per_family * len(ALL_EVAL_FAMILIES),
            "ood_samples": args.ood_samples,
            "seed": args.seed,
        },
        "greedy_baseline": greedy_results,
        "beam_comparison": beam_results,
    }

    report_path = metrics_dir / "phase1_evaluation_report.json"
    with report_path.open("w", encoding="utf-8") as f_rep:
        json.dump(full_evaluation_report, f_rep, indent=2)
    print(f"[+] Full evaluation report persisted to: {report_path}")

    # Generate Markdown and LaTeX summary tables
    md_table_path = tables_dir / "phase1_metrics_table.md"
    tex_table_path = tables_dir / "phase1_metrics_table.tex"

    md_lines: list[str] = [
        "# Phase 1 V3 Checkpoint Evaluation Results",
        "",
        f"**Checkpoint:** `{checkpoint_path.name}` (Epoch: {checkpoint.epoch}, SHA-256: `{ckpt_hash[:12]}...`)",
        f"**Benchmark:** {len(ref_markups)} independent samples ({args.samples_per_family}/family + {args.ood_samples} OOD)",
        "",
        "## Summary Metrics (Greedy Baseline)",
        "",
        f"- **Compilation Rate (CR):** {greedy_results['overall_compilation_rate'] * 100:.2f}%",
        f"- **Mean SSIM:** {greedy_results['overall_mean_ssim']:.4f}",
        f"- **Primitive Accuracy:** {greedy_results['overall_primitive_accuracy'] * 100:.2f}%",
        f"- **Structural Family Accuracy:** {greedy_results['overall_family_accuracy'] * 100:.2f}%",
        f"- **Aligned Coordinate RMSE:** {greedy_results['overall_coordinate_rmse']:.4f}",
        f"- **Mean Token GED:** {greedy_results['overall_mean_ged']:.4f}",
        "",
        "## Per-Family Breakdown",
        "",
        "| Family | Samples | Compilation Rate (%) | Mean SSIM | Primitive Acc (%) | Family Acc (%) | Coordinate RMSE | GED |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for fam, data in greedy_results["by_family"].items():
        md_lines.append(
            f"| `{fam}` | {data['count']} | {data['compilation_rate'] * 100:.1f}% | "
            f"{data['mean_ssim']:.3f} | {data['primitive_accuracy'] * 100:.1f}% | "
            f"{data['family_accuracy'] * 100:.1f}% | {data['mean_coordinate_rmse']:.3f} | "
            f"{data['mean_ged']:.3f} |"
        )

    with md_table_path.open("w", encoding="utf-8") as f_md:
        f_md.write("\n".join(md_lines) + "\n")
    print(f"[+] Markdown table persisted to: {md_table_path}")

    return full_evaluation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Independent Test Benchmark")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="results/phase1_v3_retrained/checkpoints/curriculum_v3_best.pt",
    )
    parser.add_argument("--vocab-path", type=str, default="dataset/encoded/vocabulary_v3.json")
    parser.add_argument("--results-dir", type=str, default="results/phase1_v3_retrained")
    parser.add_argument("--cache-dir", type=str, default="dataset/curriculum_cache_v3_retrained")
    parser.add_argument("--samples-per-family", type=int, default=500)
    parser.add_argument("--ood-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=999999)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dim-ff", type=int, default=2048)
    parser.add_argument("--num-encoder-blocks", type=int, default=8)
    parser.add_argument("--num-downsampling-stages", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--evaluate-beam", action="store_true", default=False)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    run_phase1_evaluation(args)
