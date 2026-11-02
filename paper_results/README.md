# Curated results for the JAG manuscript

This directory contains small, machine-readable outputs that support the
tables and claims in **Learned Optimization Costs for Tail-Robust InSAR Phase
Unwrapping**. Large per-scene files, model checkpoints, source rasters, and
generated figures remain outside Git.

## Contents

- `fair_benchmark/`: the 1,000-scene simulated summary, the independent SNAPHU
  calibration grid, the TanDEM-X/SRTM SNAPHU sensitivity grid, the
  same-predictor solver ablation, and the real-reference learned-baseline table.
- `lodo/`: leave-one-deformation-mechanism-out protocol and summaries.
- `uncertainty/`: scene-disjoint synthetic calibration/coverage results and
  real-event solver-disagreement/GPS ranking results.
- `reference_audit/`: TanDEM-X/SRTM provenance audit and training-distribution
  summary.

The full raw runs are written under `results/`, which is intentionally ignored
because it contains large per-scene tables and duplicate working copies. The
curated files here contain no local absolute paths.

## Interpretation constraints

- The simulated split supplies dense absolute reference phase; the
  TanDEM-X/SRTM targets are **DEM-derived reference solutions**, not ground
  truth.
- The primary SNAPHU configuration was selected on a separate, unaugmented
  calibration subset and frozen before the test summary was interpreted.
- Real-event entropy is evaluated as a ranking of solver disagreement and sparse
  GPS residuals; it is not presented as a calibrated real-domain error
  probability.
- Runtime values are meaningful only under the hardware and timing definition
  recorded by the evaluation scripts.
