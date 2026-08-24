"""Unit tests for the scientific LaTeX reporting and statistical aggregation module."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.ml.reporting import (
    compute_mean_and_std,
    format_mean_std,
    generate_ablation_latex_table,
    generate_multitier_latex_table,
    save_latex_tables,
)


def test_compute_mean_and_std_empty() -> None:
    mean_val, std_val = compute_mean_and_std([])
    assert mean_val == 0.0
    assert std_val == 0.0


def test_compute_mean_and_std_single() -> None:
    mean_val, std_val = compute_mean_and_std([4.2])
    assert mean_val == 4.2
    assert std_val == 0.0


def test_compute_mean_and_std_multiple() -> None:
    values: list[float] = [1.0, 2.0, 3.0]
    mean_val, std_val = compute_mean_and_std(values)
    assert mean_val == pytest.approx(2.0)
    assert std_val == pytest.approx(1.0)


def test_format_mean_std() -> None:
    formatted: str = format_mean_std(0.85432, 0.01234, precision=3)
    assert formatted == "0.854 \\pm 0.012"


def test_generate_multitier_latex_table() -> None:
    aggregated: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
        "baseline": {
            "tier1": {
                "corpus_bleu": (0.85, 0.02),
                "mean_geometric_edit_distance": (0.12, 0.01),
                "mean_graph_edit_distance": (0.08, 0.01),
                "compilation_rate": (0.95, 0.02),
                "mean_ssim": (0.89, 0.01),
            },
            "tier3": {
                "corpus_bleu": (0.45, 0.04),
                "mean_geometric_edit_distance": (0.42, 0.03),
                "mean_graph_edit_distance": (0.38, 0.02),
                "compilation_rate": (0.65, 0.05),
                "mean_ssim": (0.52, 0.03),
            },
        },
        "mixed": {
            "tier1": {
                "corpus_bleu": (0.88, 0.01),
                "mean_geometric_edit_distance": (0.09, 0.01),
                "mean_graph_edit_distance": (0.06, 0.01),
                "compilation_rate": (0.98, 0.01),
                "mean_ssim": (0.92, 0.01),
            },
        },
    }
    latex_output: str = generate_multitier_latex_table(aggregated)
    assert "\\begin{table*}" in latex_output
    assert "\\end{table*}" in latex_output
    assert "Baseline" in latex_output
    assert "TIER1" in latex_output
    assert "0.850 \\pm 0.020" in latex_output
    assert "95.0 \\pm 2.0" in latex_output


def test_generate_ablation_latex_table() -> None:
    ablations: dict[str, dict[str, float]] = {
        "Full": {
            "corpus_bleu": 0.62,
            "mean_graph_edit_distance": 0.22,
            "compilation_rate": 0.88,
            "mean_ssim": 0.74,
        },
        "No-Aug": {
            "corpus_bleu": 0.55,
            "mean_graph_edit_distance": 0.28,
            "compilation_rate": 0.81,
            "mean_ssim": 0.66,
        },
    }
    latex_output: str = generate_ablation_latex_table(ablations)
    assert "\\begin{table}" in latex_output
    assert "\\end{table}" in latex_output
    assert "Full" in latex_output
    assert "No-Aug" in latex_output
    assert "(-0.070)" in latex_output


def test_save_latex_tables(tmp_path: Path) -> None:
    aggregated: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
        "mixed": {
            "tier1": {
                "corpus_bleu": (0.88, 0.01),
                "mean_geometric_edit_distance": (0.09, 0.01),
                "mean_graph_edit_distance": (0.06, 0.01),
                "compilation_rate": (0.98, 0.01),
                "mean_ssim": (0.92, 0.01),
            }
        }
    }
    ablations: dict[str, dict[str, float]] = {
        "Full": {
            "corpus_bleu": 0.62,
            "mean_graph_edit_distance": 0.22,
            "compilation_rate": 0.88,
            "mean_ssim": 0.74,
        }
    }
    multi_path, ab_path = save_latex_tables(aggregated, ablations, tmp_path)
    assert multi_path.exists()
    assert ab_path.exists()
    assert "\\begin{table*}" in multi_path.read_text(encoding="utf-8")
    assert "\\begin{table}" in ab_path.read_text(encoding="utf-8")
