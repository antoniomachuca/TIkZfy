"""Diagnostic benchmark and audit for V3 Image-to-TikZ checkpoint.

Evaluates the existing V3 checkpoint without retraining or weight modifications
across 5 representative geometric families (line_segment, circle_arc, grid_axes,
node_arrow, composed) with 20 deterministic samples each (100 total samples).

Executes and audits 4 decoding policies:
    1. Greedy Search (argmax)
    2. Beam Search (beam_width=3, length_penalty=0.0)
    3. Pure Top-p Sampling (gamma=0.0, T=0.7, top_p=0.9)
    4. Classifier-Free Guidance (gamma=3.2, T=0.7, top_p=0.9)

Calculates:
    - Compilation Rate (CR) via TeX Live + Ghostscript
    - Structural Similarity Index (SSIM) at 128x128
    - Geometric Edit Distance (GED) and Hungarian Graph Edit Distance
    - Coordinate error and Token Overlap
    - Primitive Accuracy, Structural Family Accuracy, and Mode Collapse Rate
    - Syntax audit: Delimiter balance, UNK/PAD counts, EOS presence/position

References:
    Goodfellow et al., Deep Learning — autoregressive sequence generation.
    Golub & Van Loan, Matrix Computations — vectorized metric evaluation.
    Tantau, The TikZ and PGF Packages Manual — geometric primitive grammar.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.ghostscript_rasterizer import GhostscriptRasterizer
from adapters.tex_live_adapter import AsyncTexLiveAdapter
from adapters.vocabulary_persistence import JsonVocabularyAdapter
from core.dataset.templates import generate_sample
from core.math.spatial import resize_spatial_dimensions
from core.ml.generation import (
    BeamHypothesis,
    beam_search,
    decode_indices_to_markup,
    greedy_search,
)
from core.ml.metrics import (
    DEFAULT_COORDINATE_SCALE,
    geometric_edit_distance,
    geometric_graph_edit_distance,
    structural_similarity,
)
from core.ml.model import VisionAutoregressiveModel, resolve_device
from core.models import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    UNK_INDEX,
    ImageTensor,
    TikzTokens,
    TokenVocabulary,
)

BENCHMARK_FAMILIES: tuple[str, ...] = (
    "line_segment",
    "circle_arc",
    "grid_axes",
    "node_arrow",
    "composed",
)
SAMPLES_PER_FAMILY: int = 20
BASE_SEED: int = 42000
MAX_LENGTH: int = 128
BEAM_WIDTH: int = 3
SAMPLING_TEMPERATURE: float = 0.7
SAMPLING_TOP_P: float = 0.9
CFG_GAMMA: float = 3.2
IMAGE_SIZE: int = 128


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hexadecimal digest of a file on disk."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        chunk = f.read(65536)
        while chunk:
            hasher.update(chunk)
            chunk = f.read(65536)
    return hasher.hexdigest()


def sanitize_raw_tikz(code: str) -> str:
    """Ensure complete TikZ environment and balanced semicolon termination."""
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


def count_unbalanced_delimiters(text: str) -> dict[str, int]:
    """Count unbalanced parentheses, brackets, and braces in a text string."""
    pairs: list[tuple[str, str]] = [("(", ")"), ("[", "]"), ("{", "}")]
    counts: dict[str, int] = {}
    for open_ch, close_ch in pairs:
        open_count: int = text.count(open_ch)
        close_count: int = text.count(close_ch)
        counts[f"{open_ch}{close_ch}"] = abs(open_count - close_count)
    return counts


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
        "<-",
        "<->",
        "cycle",
        "step",
        "domain",
    ]
    for kw in keywords:
        pattern: str = rf"\b{kw}\b" if kw.isalnum() else re.escape(kw)
        if re.search(pattern, markup):
            detected.append(kw)
    return detected


def classify_structural_family(markup: str) -> str:
    """Classify the most likely structural family represented by the TikZ markup."""
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


def extract_coordinates(tokens_or_markup: str | Sequence[str]) -> list[tuple[float, float]]:
    """Extract all (x, y) 2D Cartesian coordinate pairs from markup or token sequence."""
    text: str = (
        tokens_or_markup if isinstance(tokens_or_markup, str) else " ".join(tokens_or_markup)
    )
    matches = re.finditer(r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)", text)
    return [(float(m.group(1)), float(m.group(2))) for m in matches]


def compute_coordinate_rmse(
    ref_coords: list[tuple[float, float]],
    cand_coords: list[tuple[float, float]],
) -> float:
    """Compute Euclidean coordinate root mean square error between aligned vertices."""
    if not ref_coords or not cand_coords:
        return float(DEFAULT_COORDINATE_SCALE)
    min_len: int = min(len(ref_coords), len(cand_coords))
    ref_arr: np.ndarray = np.asarray(ref_coords[:min_len], dtype=np.float64)
    cand_arr: np.ndarray = np.asarray(cand_coords[:min_len], dtype=np.float64)
    diffs: np.ndarray = ref_arr - cand_arr  # Shape: (min_len, 2)
    euclidean_dists: np.ndarray = np.linalg.norm(diffs, axis=1)  # Shape: (min_len,)
    len_penalty: float = float(abs(len(ref_coords) - len(cand_coords))) * 1.0
    return float(np.mean(euclidean_dists) + len_penalty)


def compute_token_overlap(ref_tokens: list[str], cand_tokens: list[str]) -> float:
    """Compute token-level Jaccard intersection over union."""
    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_set: set[str] = set(ref_tokens)
    cand_set: set[str] = set(cand_tokens)
    intersection_size: int = len(ref_set & cand_set)
    union_size: int = len(ref_set | cand_set)
    return float(intersection_size / max(union_size, 1))


async def render_tikz_to_tensor_and_png(
    code: str,
    compiler: AsyncTexLiveAdapter,
    rasterizer: GhostscriptRasterizer,
    image_size: int = 128,
) -> tuple[bool, torch.Tensor, bytes | None, str | None]:
    """Compile TikZ markup to PDF and rasterize to (3, H, W) tensor and PNG bytes.

    Returns:
        (success, image_tensor, png_bytes, error_message).
    """
    try:
        res = await compiler.compile_tikz(TikzTokens(markup=code))
        if not res.is_successful:
            error_msg = "LaTeX compilation returned non-zero code."
            return (
                False,
                torch.ones((3, image_size, image_size), dtype=torch.float32),
                None,
                error_msg,
            )
        png_bytes = await rasterizer.rasterize_pdf(res.pdf_data, dpi=72)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # Shape: (H, W, 3)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
        resized = resize_spatial_dimensions(ImageTensor(raw_tensor=t), image_size, image_size)
        return True, resized.raw_tensor.squeeze(0), png_bytes, None
    except Exception as exc:
        return False, torch.ones((3, image_size, image_size), dtype=torch.float32), None, str(exc)


def sample_decode_tokens(
    model: VisionAutoregressiveModel,
    image_tensor: torch.Tensor,
    gamma: float = 0.0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_length: int = 128,
    seed: int = 42,
) -> tuple[int, ...]:
    """Execute stochastic or Classifier-Free Guidance (CFG) visual decoding."""
    torch.manual_seed(seed)
    v_cond: torch.Tensor = model.encoder(image_tensor)
    v_uncond: torch.Tensor | None = (
        model.encoder(torch.ones_like(image_tensor)) if gamma > 0.0 else None
    )

    gen: torch.Tensor = torch.tensor([[BOS_INDEX]], dtype=torch.long, device=image_tensor.device)
    finished: bool = False
    step: int = 0

    while step < max_length - 1 and not finished:
        l_cond: torch.Tensor = model.decoder(v_cond, gen)[:, -1, :]
        if gamma > 0.0 and v_uncond is not None:
            l_uncond: torch.Tensor = model.decoder(v_uncond, gen)[:, -1, :]
            guided: torch.Tensor = l_cond + gamma * (l_cond - l_uncond)
        else:
            guided = l_cond
        scaled: torch.Tensor = guided / max(temperature, 1e-4)
        probs: torch.Tensor = F.softmax(scaled, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs: torch.Tensor = torch.cumsum(sorted_probs, dim=-1)
        mask: torch.Tensor = cum_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_rel_idx: torch.Tensor = torch.multinomial(sorted_probs[0], 1)
        next_token: torch.Tensor = sorted_indices[0, sampled_rel_idx].unsqueeze(0)
        gen = torch.cat([gen, next_token], dim=1)
        if next_token.item() == EOS_INDEX:
            finished = True
        step += 1

    return tuple(gen[0].tolist())


@dataclass
class SamplePredictionRecord:
    family: str
    sample_index: int
    seed: int
    policy: str
    raw_indices: list[int]
    raw_emitted_tokens: list[str]
    decoded_markup: str
    sanitized_markup: str
    total_length: int
    length_to_eos: int
    has_bos: bool
    has_eos: bool
    eos_position: int | None
    unk_count: int
    pad_count: int
    unbalanced_delimiters_raw: dict[str, int]
    unbalanced_delimiters_sanitized: dict[str, int]
    detected_primitives: list[str]
    predicted_coordinates: list[tuple[float, float]]
    predicted_family: str
    compilation_successful: bool
    tex_error: str | None
    ssim: float
    coordinate_error: float
    token_ged: float
    graph_ged: float
    token_overlap: float
    is_line_segment_collapsed: bool


async def run_full_diagnostic(
    checkpoint_path: Path,
    vocabulary_path: Path,
    output_dir: Path,
    device_name: str | None = None,
) -> None:
    """Execute complete 4-policy diagnostic across 100 samples and persist all artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device: torch.device = resolve_device(device_name)
    print(f"[*] Initializing Diagnostic Engine on device: {device}...")

    # Load Vocabulary
    vocab: TokenVocabulary = JsonVocabularyAdapter().load_vocabulary(str(vocabulary_path))
    print(f"[+] Loaded vocabulary with {len(vocab.token_to_index)} tokens.")

    # Instantiate Model
    model = VisionAutoregressiveModel(
        vocabulary=vocab,
        model_dimension=512,
        num_layers=8,
        num_heads=8,
        dim_feedforward=2048,
        num_encoder_blocks=8,
        num_downsampling_stages=3,
        max_length=512,
        device=device,
    )
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_data["model_state"])
    model.to(device)
    model.eval()
    print(
        f"[+] Loaded V3 Checkpoint from {checkpoint_path} (epoch {checkpoint_data.get('epoch')})."
    )

    compiler = AsyncTexLiveAdapter()
    rasterizer = GhostscriptRasterizer()

    # Hashes & Config
    ckpt_hash: str = compute_file_sha256(checkpoint_path)
    vocab_hash: str = compute_file_sha256(vocabulary_path)
    config_record: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": ckpt_hash,
        "vocabulary_path": str(vocabulary_path.resolve()),
        "vocabulary_sha256": vocab_hash,
        "model_architecture": {
            "model_dimension": 512,
            "num_layers": 8,
            "num_heads": 8,
            "dim_feedforward": 2048,
            "num_encoder_blocks": 8,
            "num_downsampling_stages": 3,
            "max_length": 512,
            "coordinate_step": 0.05,
            "image_size": IMAGE_SIZE,
        },
        "benchmark_families": list(BENCHMARK_FAMILIES),
        "samples_per_family": SAMPLES_PER_FAMILY,
        "total_samples": len(BENCHMARK_FAMILIES) * SAMPLES_PER_FAMILY,
        "base_seed": BASE_SEED,
        "decoding_policies": {
            "greedy": {"type": "greedy_search", "max_length": MAX_LENGTH},
            "beam_search": {
                "type": "beam_search",
                "beam_width": BEAM_WIDTH,
                "length_penalty": 0.0,
                "max_length": MAX_LENGTH,
            },
            "sampling_gamma0": {
                "type": "nucleus_sampling",
                "gamma": 0.0,
                "temperature": SAMPLING_TEMPERATURE,
                "top_p": SAMPLING_TOP_P,
                "max_length": MAX_LENGTH,
            },
            "cfg_gamma32": {
                "type": "contrastive_sampling",
                "gamma": CFG_GAMMA,
                "temperature": SAMPLING_TEMPERATURE,
                "top_p": SAMPLING_TOP_P,
                "max_length": MAX_LENGTH,
            },
        },
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    # 1. Generate and Render Benchmark References
    print("[*] Step 1: Generating and rendering 100 benchmark reference samples...")
    sample_records: list[dict[str, Any]] = []

    for fam in BENCHMARK_FAMILIES:
        for idx in range(SAMPLES_PER_FAMILY):
            seed = BASE_SEED + idx
            rng = np.random.default_rng(seed)
            gt_code = generate_sample(fam, rng)
            sample_dir = output_dir / fam / f"sample_{idx:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            (sample_dir / "reference.tex").write_text(gt_code, encoding="utf-8")
            ok, ref_img_tensor, png_bytes, err = await render_tikz_to_tensor_and_png(
                gt_code, compiler, rasterizer, image_size=IMAGE_SIZE
            )
            if not ok or png_bytes is None:
                raise RuntimeError(f"Reference rendering failed for {fam} sample {idx}: {err}")

            # Check for non-blank image
            if float(ref_img_tensor.std().item()) < 1e-4:
                raise ValueError(f"Reference image is blank for {fam} sample {idx}!")

            (sample_dir / "reference.png").write_bytes(png_bytes)
            ref_tokens: list[str] = [t for t in gt_code.split() if t]
            ref_coords: list[tuple[float, float]] = extract_coordinates(gt_code)

            sample_records.append(
                {
                    "family": fam,
                    "sample_index": idx,
                    "seed": seed,
                    "gt_code": gt_code,
                    "ref_tokens": ref_tokens,
                    "ref_coords": ref_coords,
                    "ref_img_tensor": ref_img_tensor,
                    "sample_dir": sample_dir,
                }
            )

    print("[+] All 100 reference images generated, validated (non-empty), and rendered.")

    # 2. Run the 4 Decoding Policies
    print("[*] Step 2: Executing 4 decoding policies on 100 samples (400 total inferences)...")
    all_prediction_records: list[SamplePredictionRecord] = []
    showcase_samples: list[dict[str, Any]] = []

    policies = ["greedy", "beam_search", "sampling_gamma0", "cfg_gamma32"]
    start_time = time.time()

    for s_idx, s in enumerate(sample_records):
        fam = s["family"]
        idx = s["sample_index"]
        seed = s["seed"]
        gt_code = s["gt_code"]
        ref_tokens = s["ref_tokens"]
        ref_coords = s["ref_coords"]
        ref_img = s["ref_img_tensor"].unsqueeze(0).to(device)  # Shape: (1, 3, H, W)
        curr_sample_dir = cast(Path, s["sample_dir"])

        is_showcase_candidate = idx == 0
        showcase_entry: dict[str, Any] = {"family": fam, "gt_code": gt_code, "predictions": {}}

        for policy in policies:
            # Generate Indices
            if policy == "greedy":
                raw_indices = list(
                    greedy_search(model, ImageTensor(raw_tensor=ref_img), max_length=MAX_LENGTH)
                )
            elif policy == "beam_search":
                beam_results: list[BeamHypothesis] = beam_search(
                    model,
                    ImageTensor(raw_tensor=ref_img),
                    beam_width=BEAM_WIDTH,
                    max_length=MAX_LENGTH,
                )
                raw_indices = list(beam_results[0].tokens)
            elif policy == "sampling_gamma0":
                raw_indices = list(
                    sample_decode_tokens(
                        model,
                        ref_img,
                        gamma=0.0,
                        temperature=SAMPLING_TEMPERATURE,
                        top_p=SAMPLING_TOP_P,
                        max_length=MAX_LENGTH,
                        seed=seed,
                    )
                )
            else:  # cfg_gamma32
                raw_indices = list(
                    sample_decode_tokens(
                        model,
                        ref_img,
                        gamma=CFG_GAMMA,
                        temperature=SAMPLING_TEMPERATURE,
                        top_p=SAMPLING_TOP_P,
                        max_length=MAX_LENGTH,
                        seed=seed,
                    )
                )

            # Audit Token Level
            raw_emitted_tokens = [
                vocab.index_to_token[tok_idx]
                for tok_idx in raw_indices
                if tok_idx in vocab.index_to_token
            ]
            has_bos = bool(BOS_INDEX in raw_indices)
            has_eos = bool(EOS_INDEX in raw_indices)
            eos_position = raw_indices.index(EOS_INDEX) if has_eos else None
            unk_count = raw_indices.count(UNK_INDEX)
            pad_count = raw_indices.count(PAD_INDEX)
            total_length = len(raw_indices)
            length_to_eos = (eos_position + 1) if eos_position is not None else total_length

            # Decode to TikZ Markup Object
            decoded_obj: TikzTokens = decode_indices_to_markup(vocab, tuple(raw_indices))
            decoded_markup = decoded_obj.markup
            sanitized_markup = sanitize_raw_tikz(decoded_markup)

            # Delimiter and Primitive Audit
            unbal_raw = count_unbalanced_delimiters(decoded_markup)
            unbal_san = count_unbalanced_delimiters(sanitized_markup)
            primitives = detect_primitives(sanitized_markup)
            pred_coords = extract_coordinates(sanitized_markup)
            pred_fam = classify_structural_family(sanitized_markup)
            is_collapsed = pred_fam == "line_segment" and fam != "line_segment"

            # Render Prediction
            comp_ok, pred_img_tensor, pred_png_bytes, tex_err = await render_tikz_to_tensor_and_png(
                sanitized_markup, compiler, rasterizer, image_size=IMAGE_SIZE
            )

            # Compute SSIM
            ssim_score = 0.0
            if comp_ok and pred_img_tensor is not None:
                ssim_score = structural_similarity(s["ref_img_tensor"], pred_img_tensor)

            # Compute Distances
            cand_tokens_for_ged = [t for t in sanitized_markup.split() if t]
            token_ged_val = geometric_edit_distance(ref_tokens, cand_tokens_for_ged)
            graph_ged_val = geometric_graph_edit_distance(gt_code, sanitized_markup)
            coord_err_val = compute_coordinate_rmse(ref_coords, pred_coords)
            overlap_val = compute_token_overlap(ref_tokens, cand_tokens_for_ged)

            # Persist per-policy artifacts
            (curr_sample_dir / f"{policy}_prediction.tex").write_text(
                sanitized_markup, encoding="utf-8"
            )
            if pred_png_bytes is not None:
                (curr_sample_dir / f"{policy}_prediction.png").write_bytes(pred_png_bytes)

            meta_data = {
                "policy": policy,
                "raw_indices": raw_indices,
                "raw_emitted_tokens": raw_emitted_tokens,
                "decoded_markup": decoded_markup,
                "sanitized_markup": sanitized_markup,
                "total_length": total_length,
                "length_to_eos": length_to_eos,
                "has_bos": has_bos,
                "has_eos": has_eos,
                "eos_position": eos_position,
                "unk_count": unk_count,
                "pad_count": pad_count,
                "unbalanced_delimiters_raw": unbal_raw,
                "unbalanced_delimiters_sanitized": unbal_san,
                "detected_primitives": primitives,
                "predicted_coordinates": pred_coords,
                "predicted_family": pred_fam,
                "compilation_successful": comp_ok,
                "tex_error": tex_err,
                "ssim": ssim_score,
                "coordinate_error": coord_err_val,
                "token_ged": token_ged_val,
                "graph_ged": graph_ged_val,
                "token_overlap": overlap_val,
                "is_line_segment_collapsed": is_collapsed,
            }
            with (curr_sample_dir / f"{policy}_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

            record = SamplePredictionRecord(
                family=fam,
                sample_index=idx,
                seed=seed,
                policy=policy,
                raw_indices=raw_indices,
                raw_emitted_tokens=raw_emitted_tokens,
                decoded_markup=decoded_markup,
                sanitized_markup=sanitized_markup,
                total_length=total_length,
                length_to_eos=length_to_eos,
                has_bos=has_bos,
                has_eos=has_eos,
                eos_position=eos_position,
                unk_count=unk_count,
                pad_count=pad_count,
                unbalanced_delimiters_raw=unbal_raw,
                unbalanced_delimiters_sanitized=unbal_san,
                detected_primitives=primitives,
                predicted_coordinates=pred_coords,
                predicted_family=pred_fam,
                compilation_successful=comp_ok,
                tex_error=tex_err,
                ssim=ssim_score,
                coordinate_error=coord_err_val,
                token_ged=token_ged_val,
                graph_ged=graph_ged_val,
                token_overlap=overlap_val,
                is_line_segment_collapsed=is_collapsed,
            )
            all_prediction_records.append(record)

            if is_showcase_candidate:
                showcase_entry["predictions"][policy] = {
                    "markup": sanitized_markup,
                    "comp_ok": comp_ok,
                    "ssim": ssim_score,
                    "img_tensor": pred_img_tensor if comp_ok else None,
                }

        if is_showcase_candidate:
            showcase_entry["ref_img_tensor"] = s["ref_img_tensor"]
            showcase_samples.append(showcase_entry)

        if (s_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            done_inf = (s_idx + 1) * 4
            print(f"  -> Evaluated [{s_idx + 1}/100] samples ({done_inf} inf) in {elapsed:.1f}s...")

    # 3. Export Detailed CSV
    detailed_csv_path = output_dir / "detailed_results.csv"
    with detailed_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "family",
                "sample_index",
                "seed",
                "policy",
                "compilation_successful",
                "ssim",
                "coordinate_error",
                "token_ged",
                "graph_ged",
                "token_overlap",
                "predicted_family",
                "is_line_segment_collapsed",
                "total_length",
                "length_to_eos",
                "has_eos",
                "unk_count",
                "raw_unbalanced_parens",
                "san_unbalanced_parens",
                "tex_error",
            ]
        )
        for r in all_prediction_records:
            writer.writerow(
                [
                    r.family,
                    r.sample_index,
                    r.seed,
                    r.policy,
                    r.compilation_successful,
                    f"{r.ssim:.4f}",
                    f"{r.coordinate_error:.4f}",
                    f"{r.token_ged:.4f}",
                    f"{r.graph_ged:.4f}",
                    f"{r.token_overlap:.4f}",
                    r.predicted_family,
                    r.is_line_segment_collapsed,
                    r.total_length,
                    r.length_to_eos,
                    r.has_eos,
                    r.unk_count,
                    r.unbalanced_delimiters_raw.get("()", 0),
                    r.unbalanced_delimiters_sanitized.get("()", 0),
                    (r.tex_error or "").replace("\n", " ")[:100],
                ]
            )
    print(f"[+] Exported detailed CSV to {detailed_csv_path}")

    # 4. Aggregated Statistics & Summary Table
    summary_data: dict[str, Any] = {"overall": {}, "by_family": {}}
    family_csv_path = output_dir / "family_summary.csv"

    with family_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "family",
                "policy",
                "samples",
                "compilation_rate_mean",
                "ssim_mean",
                "ssim_std",
                "coord_error_mean",
                "coord_error_std",
                "token_ged_mean",
                "token_ged_std",
                "graph_ged_mean",
                "graph_ged_std",
                "token_overlap_mean",
                "family_accuracy",
                "collapse_rate",
                "eos_rate",
                "unk_rate",
                "mean_length",
            ]
        )

        for policy in policies:
            policy_records = [r for r in all_prediction_records if r.policy == policy]
            n_pol = len(policy_records)
            cr_pol = sum(1 for r in policy_records if r.compilation_successful) / n_pol
            ssim_vals = np.array([r.ssim for r in policy_records])
            coord_vals = np.array([r.coordinate_error for r in policy_records])
            token_ged_vals = np.array([r.token_ged for r in policy_records])
            graph_ged_vals = np.array([r.graph_ged for r in policy_records])
            overlap_vals = np.array([r.token_overlap for r in policy_records])
            fam_acc = sum(1 for r in policy_records if r.predicted_family == r.family) / n_pol
            coll_rate = sum(1 for r in policy_records if r.is_line_segment_collapsed) / max(
                1, sum(1 for r in policy_records if r.family != "line_segment")
            )
            eos_rate = sum(1 for r in policy_records if r.has_eos) / n_pol
            unk_rate = sum(r.unk_count for r in policy_records) / sum(
                r.total_length for r in policy_records
            )
            mean_len = float(np.mean([r.length_to_eos for r in policy_records]))

            summary_data["overall"][policy] = {
                "sample_count": n_pol,
                "compilation_rate": cr_pol,
                "ssim": {"mean": float(ssim_vals.mean()), "std": float(ssim_vals.std())},
                "coordinate_error": {
                    "mean": float(coord_vals.mean()),
                    "std": float(coord_vals.std()),
                },
                "token_ged": {
                    "mean": float(token_ged_vals.mean()),
                    "std": float(token_ged_vals.std()),
                },
                "graph_ged": {
                    "mean": float(graph_ged_vals.mean()),
                    "std": float(graph_ged_vals.std()),
                },
                "token_overlap": {
                    "mean": float(overlap_vals.mean()),
                    "std": float(overlap_vals.std()),
                },
                "family_accuracy": fam_acc,
                "collapse_to_segment_rate": coll_rate,
                "eos_rate": eos_rate,
                "unk_rate": unk_rate,
                "mean_length_to_eos": mean_len,
            }

            for fam in BENCHMARK_FAMILIES:
                fam_records = [r for r in policy_records if r.family == fam]
                n_fam = len(fam_records)
                cr_fam = sum(1 for r in fam_records if r.compilation_successful) / n_fam
                ssim_fam = np.array([r.ssim for r in fam_records])
                coord_fam = np.array([r.coordinate_error for r in fam_records])
                t_ged_fam = np.array([r.token_ged for r in fam_records])
                g_ged_fam = np.array([r.graph_ged for r in fam_records])
                ov_fam = np.array([r.token_overlap for r in fam_records])
                f_acc_fam = sum(1 for r in fam_records if r.predicted_family == fam) / n_fam
                coll_fam = (
                    sum(1 for r in fam_records if r.is_line_segment_collapsed) / n_fam
                    if fam != "line_segment"
                    else 0.0
                )
                eos_fam = sum(1 for r in fam_records if r.has_eos) / n_fam
                unk_fam = sum(r.unk_count for r in fam_records) / sum(
                    r.total_length for r in fam_records
                )
                len_fam = float(np.mean([r.length_to_eos for r in fam_records]))

                if fam not in summary_data["by_family"]:
                    summary_data["by_family"][fam] = {}
                summary_data["by_family"][fam][policy] = {
                    "sample_count": n_fam,
                    "compilation_rate": cr_fam,
                    "ssim": {"mean": float(ssim_fam.mean()), "std": float(ssim_fam.std())},
                    "coordinate_error": {
                        "mean": float(coord_fam.mean()),
                        "std": float(coord_fam.std()),
                    },
                    "token_ged": {"mean": float(t_ged_fam.mean()), "std": float(t_ged_fam.std())},
                    "graph_ged": {"mean": float(g_ged_fam.mean()), "std": float(g_ged_fam.std())},
                    "token_overlap": {"mean": float(ov_fam.mean()), "std": float(ov_fam.std())},
                    "family_accuracy": f_acc_fam,
                    "collapse_to_segment_rate": coll_fam,
                    "eos_rate": eos_fam,
                    "unk_rate": unk_fam,
                    "mean_length_to_eos": len_fam,
                }

                writer.writerow(
                    [
                        fam,
                        policy,
                        n_fam,
                        f"{cr_fam:.4f}",
                        f"{ssim_fam.mean():.4f}",
                        f"{ssim_fam.std():.4f}",
                        f"{coord_fam.mean():.4f}",
                        f"{coord_fam.std():.4f}",
                        f"{t_ged_fam.mean():.4f}",
                        f"{t_ged_fam.std():.4f}",
                        f"{g_ged_fam.mean():.4f}",
                        f"{g_ged_fam.std():.4f}",
                        f"{ov_fam.mean():.4f}",
                        f"{f_acc_fam:.4f}",
                        f"{coll_fam:.4f}",
                        f"{eos_fam:.4f}",
                        f"{unk_fam:.4f}",
                        f"{len_fam:.2f}",
                    ]
                )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Summary saved to {output_dir / 'summary.json'}")

    # 5. Build Qualitative Visual Comparison Grid
    print("[*] Step 5: Rendering qualitative comparison grid across 5 families...")
    fig, axes = plt.subplots(len(showcase_samples), 5, figsize=(18, 3.6 * len(showcase_samples)))

    col_titles = ["Reference (GT)", "Greedy", "Beam (B=3)", "Sampling (γ=0)", "CFG (γ=3.2)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i, s_entry in enumerate(showcase_samples):
        fam_name = s_entry["family"]
        # Ground Truth
        ref_arr = s_entry["ref_img_tensor"].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(ref_arr)
        axes[i, 0].set_ylabel(fam_name, fontsize=11, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        for j, pol in enumerate(["greedy", "beam_search", "sampling_gamma0", "cfg_gamma32"]):
            ax = axes[i, j + 1]
            p_data = s_entry["predictions"][pol]
            if p_data["comp_ok"] and p_data["img_tensor"] is not None:
                p_arr = p_data["img_tensor"].permute(1, 2, 0).cpu().numpy()
                ax.imshow(p_arr)
                ax.set_xlabel(f"SSIM: {p_data['ssim']:.3f}", fontsize=10)
            else:
                ax.imshow(np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32))
                ax.text(
                    IMAGE_SIZE // 2,
                    IMAGE_SIZE // 2,
                    "COMPILE FAIL",
                    color="red",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=10,
                )
                ax.set_xlabel("SSIM: 0.000", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    grid_path = output_dir / "comparison_grid.png"
    plt.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Qualitative comparison grid saved to {grid_path}")


def main() -> None:
    """CLI entrypoint for running diagnostic benchmark."""
    repo_root = Path(__file__).resolve().parent.parent
    ckpt_path = repo_root / "results" / "checkpoints" / "curriculum_v3_best.pt"
    vocab_path = repo_root / "dataset" / "encoded" / "vocabulary_v3.json"
    out_dir = repo_root / "results" / "diagnostics" / "v3_decode_comparison"

    asyncio.run(
        run_full_diagnostic(
            checkpoint_path=ckpt_path,
            vocabulary_path=vocab_path,
            output_dir=out_dir,
            device_name="mps" if torch.backends.mps.is_available() else "cpu",
        )
    )


if __name__ == "__main__":
    main()
