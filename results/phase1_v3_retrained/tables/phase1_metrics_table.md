# Phase 1 V3 Checkpoint Evaluation Results

**Checkpoint:** `curriculum_v3_best.pt` (Epoch: 10, SHA-256: `37b627c6a8f9...`)
**Benchmark:** 6000 independent samples (500/family + 2000 OOD)

## Summary Metrics (Greedy Baseline)

- **Compilation Rate (CR):** 44.52%
- **Mean SSIM:** 0.3645
- **Primitive Accuracy:** 69.65%
- **Structural Family Accuracy:** 79.73%
- **Aligned Coordinate RMSE:** 20.3725
- **Mean Token GED:** 0.4984

## Per-Family Breakdown

| Family | Samples | Compilation Rate (%) | Mean SSIM | Primitive Acc (%) | Family Acc (%) | Coordinate RMSE | GED |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `line_segment` | 500 | 90.8% | 0.743 | 94.6% | 91.6% | 6.368 | 0.189 |
| `circle_arc` | 500 | 91.6% | 0.854 | 92.9% | 72.6% | 7.770 | 0.160 |
| `grid_axes` | 500 | 100.0% | 0.744 | 100.0% | 100.0% | 0.011 | 0.004 |
| `function_plot` | 500 | 97.2% | 0.871 | 96.9% | 95.6% | 14.142 | 0.114 |
| `polyline` | 500 | 70.4% | 0.504 | 83.0% | 0.0% | 8.001 | 0.399 |
| `polygon` | 500 | 71.6% | 0.590 | 98.2% | 97.2% | 9.017 | 0.335 |
| `node_arrow` | 500 | 0.2% | 0.002 | 99.5% | 99.8% | 5.510 | 0.318 |
| `composed` | 500 | 2.6% | 0.013 | 33.9% | 0.0% | 38.279 | 0.889 |
| `ood_composed` | 2000 | 2.5% | 0.014 | 34.2% | 100.0% | 38.843 | 0.893 |
