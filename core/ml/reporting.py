"""Scientific LaTeX reporting and statistical aggregation utilities.

Generates formal publication-ready LaTeX tables with sample mean and standard
deviation (\\mu \\pm \\sigma) across experimental runs and ablation studies.

References:
    Goodfellow et al., Deep Learning — empirical benchmarking and confidence reporting.
    Papineni et al., BLEU — precision metrics in natural and formal language.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path


def compute_mean_and_std(values: list[float]) -> tuple[float, float]:
    """Compute empirical sample mean and sample standard deviation in O(N).

    Args:
        values: Sequence of scalar observations across random seeds.

    Returns:
        tuple[float, float]: (mean, sample_std_dev). When N < 2, std is 0.0.
    """
    if not values:
        return 0.0, 0.0
    count: int = len(values)
    mean_val: float = sum(values) / count
    variance: float = sum((x - mean_val) ** 2 for x in values) / (count - 1) if count > 1 else 0.0
    return mean_val, math.sqrt(max(0.0, variance))


def format_mean_std(mean_val: float, std_val: float, precision: int = 3) -> str:
    """Format scalar statistics into academic LaTeX ``\\mu \\pm \\sigma`` notation."""
    return f"{mean_val:.{precision}f} \\pm {std_val:.{precision}f}"


def generate_multitier_latex_table(
    aggregated_metrics: Mapping[str, Mapping[str, Mapping[str, tuple[float, float]]]],
) -> str:
    """Generate LaTeX table string for multi-tier evaluation across models.

    Args:
        aggregated_metrics: Nested mapping of
            ``{model_name: {tier_name: {metric_name: (mean, std)}}}``.

    Returns:
        str: Clean LaTeX table markup ready for scientific inclusion.
    """
    headers: str = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Multi-Tier Experimental Evaluation across 3 Random Seeds "
        "($\\mu \\pm \\sigma$).}\n"
        "\\label{tab:multitier_evaluation}\n"
        "\\begin{tabular}{llccccc}\n"
        "\\toprule\n"
        "Model & Evaluation Tier & BLEU $\\uparrow$ & Token GED $\\downarrow$ & "
        "Hungarian GED $\\downarrow$ & CR (\\%) $\\uparrow$ & SSIM $\\uparrow$ \\\\\n"
        "\\midrule\n"
    )

    rows: list[str] = []
    models: list[str] = sorted(aggregated_metrics.keys())
    for model_name in models:
        tier_data = aggregated_metrics[model_name]
        tiers: list[str] = sorted(tier_data.keys())
        for tier_idx, tier_name in enumerate(tiers):
            metrics = tier_data[tier_name]
            bleu_str: str = format_mean_std(*metrics.get("corpus_bleu", (0.0, 0.0)))
            ged_str: str = format_mean_std(*metrics.get("mean_geometric_edit_distance", (0.0, 0.0)))
            h_ged_str: str = format_mean_std(*metrics.get("mean_graph_edit_distance", (0.0, 0.0)))
            cr_mean, cr_std = metrics.get("compilation_rate", (0.0, 0.0))
            cr_str: str = f"{cr_mean * 100.0:.1f} \\pm {cr_std * 100.0:.1f}"
            ssim_str: str = format_mean_std(*metrics.get("mean_ssim", (0.0, 0.0)))

            model_label: str = f"\\textbf{{{model_name.capitalize()}}}" if tier_idx == 0 else ""
            rows.append(
                f"{model_label} & {tier_name.upper()} & {bleu_str} & {ged_str} & "
                f"{h_ged_str} & {cr_str} & {ssim_str} \\\\"
            )
        rows.append("\\midrule")

    # Remove trailing midrule if present
    if rows and rows[-1] == "\\midrule":
        rows.pop()

    footer: str = "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    return headers + "\n".join(rows) + footer


def generate_ablation_latex_table(
    ablation_metrics: Mapping[str, Mapping[str, float]],
) -> str:
    """Generate LaTeX table string for architectural and data ablation studies.

    Args:
        ablation_metrics: Mapping of ``{variant_name: {metric_name: score}}``.

    Returns:
        str: Formatted LaTeX table showing absolute scores and deltas on Tier 3.
    """
    full_scores: Mapping[str, float] = ablation_metrics.get("Full", {})
    full_bleu: float = float(full_scores.get("corpus_bleu", 0.0))
    full_hged: float = float(full_scores.get("mean_graph_edit_distance", 0.0))
    full_cr: float = float(full_scores.get("compilation_rate", 0.0))
    full_ssim: float = float(full_scores.get("mean_ssim", 0.0))

    headers: str = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Ablation Study on Out-Of-Distribution Tier 3 Test Benchmark.}\n"
        "\\label{tab:ablation_study}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Ablation Variant & BLEU ($\\Delta$) & Hungarian GED ($\\Delta$) & "
        "CR \\% ($\\Delta$) & SSIM ($\\Delta$) \\\\\n"
        "\\midrule\n"
    )

    rows: list[str] = []
    for variant, scores in ablation_metrics.items():
        bleu: float = float(scores.get("corpus_bleu", 0.0))
        hged: float = float(scores.get("mean_graph_edit_distance", 0.0))
        cr: float = float(scores.get("compilation_rate", 0.0))
        ssim: float = float(scores.get("mean_ssim", 0.0))

        delta_bleu: float = bleu - full_bleu
        delta_hged: float = hged - full_hged
        delta_cr: float = (cr - full_cr) * 100.0
        delta_ssim: float = ssim - full_ssim

        delta_bleu_str: str = f" ({delta_bleu:+.3f})" if variant != "Full" else ""
        delta_hged_str: str = f" ({delta_hged:+.3f})" if variant != "Full" else ""
        delta_cr_str: str = f" ({delta_cr:+.1f})" if variant != "Full" else ""
        delta_ssim_str: str = f" ({delta_ssim:+.3f})" if variant != "Full" else ""

        rows.append(
            f"\\textbf{{{variant}}} & {bleu:.3f}{delta_bleu_str} & "
            f"{hged:.3f}{delta_hged_str} & {cr * 100.0:.1f}\\%{delta_cr_str} & "
            f"{ssim:.3f}{delta_ssim_str} \\\\"
        )

    footer: str = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return headers + "\n".join(rows) + footer


def save_latex_tables(
    aggregated_metrics: Mapping[str, Mapping[str, Mapping[str, tuple[float, float]]]],
    ablation_metrics: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist generated LaTeX tables to target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    multitier_path: Path = output_dir / "multitier_evaluation.tex"
    ablation_path: Path = output_dir / "ablation_study.tex"

    multitier_latex: str = generate_multitier_latex_table(aggregated_metrics)
    ablation_latex: str = generate_ablation_latex_table(ablation_metrics)

    multitier_path.write_text(multitier_latex, encoding="utf-8")
    ablation_path.write_text(ablation_latex, encoding="utf-8")
    return multitier_path, ablation_path
