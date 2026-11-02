# Auditable experiment rebuild

The new evaluation path is intentionally manifest-driven. Each claim can be
traced to per-scene rows and an explicit information contract.

## Main benchmark

`scripts/eval_fair_benchmark.py` evaluates the signed learned cost, the same
predictor collapsed to an unsigned cost, unit-cost MCF, optional DENet with the
same MCF backend, and a SNAPHU sensitivity grid. It records aligned RMSE,
cycle-slip rate, slip-free RMSE, residue-balance violations, runtime, P90/P95/
P99, CVaR95, maximum error, checkpoint hashes, data configuration, coherence
source, cost mode, and number of looks.

The data loader now marks whether coherence was actually supplied. The fair
benchmark refuses SNAPHU's `supplied` information mode on InSAR-DLPU, whose
second channel is only a constant-one placeholder.

## Strict domain generalization

`scripts/run_lodo.py` trains four models from scratch, holding out one of fault,
volcanic, subsidence, or atmospheric fringes. Atmospheric phase is absent from
the other three primary domains. `scripts/summarize_lodo.py` consolidates the
four result manifests into the manuscript table.

## Uncertainty

`scripts/uncertainty_calibration.py` splits complete scenes into calibration and
evaluation halves. It reports AUROC and average precision against a conventional
quality proxy, isotonic Brier score and ECE, reliability bins, and fixed retained
coverage. The current outputs are under `results/uncertainty`.

## Data and reference audits

`scripts/plot_training_distribution.py` audits class balance and the fraction
outside the three-class ambiguity head. `scripts/audit_tandemx_provenance.py`
inventories all 100 TanDEM-X pairs and records what metadata are absent. The
paper consequently uses “DEM-derived reference” rather than “ground truth.”

## Figures

`scripts/plot_representative_methods.py` selects a median case, the largest
tail rescue, and the largest counterexample directly from paired benchmark
rows, then renders unwrapped phases and integer-cycle error maps. The largest
gap is labelled “extreme-tail attenuation” rather than “rescue” because every
method can still fail on that scene. This prevents visual cherry-picking while
keeping the failure severity explicit.

## SNAPHU calibration and frozen test selection

The primary SNAPHU configuration must not be selected on the 1,000-scene test
split. Use the unaugmented `train_eval` view and a deterministic random subset
of the training files. This calibration tunes only the classical baseline; it
does not update the learned model.

```bash
python -m scripts.eval_fair_benchmark \
  eval.split=train_eval \
  data.limit=1000 data.sample_strategy=random data.sample_seed=20261030 \
  ext.fair_methods='[snaphu]' \
  ext.fair_snaphu_sources='[uniform,phase_proxy]' \
  ext.fair_snaphu_costs='[defo,smooth]' \
  ext.fair_snaphu_nlooks='[1,4,8,16]' \
  ext.out_dir=results/fair_benchmark/snaphu_calibration

python -m scripts.select_snaphu_config \
  results/fair_benchmark/snaphu_calibration/fair_benchmark_train_eval.csv \
  results/fair_benchmark/test/fair_benchmark_test.csv \
  --metric mean \
  --out results/fair_benchmark/snaphu_selection.json
```

The JSON freezes the calibration-selected method and copies its already-run
test statistics without reselecting on the test set. The complete test sweep
remains a parameter-sensitivity analysis. The historical minimum-mean test row
may be reported only as an optimistic “test-grid-best” sensitivity bound.

`scripts/summarize_fair_benchmark.py` can regenerate an aggregate CSV from
existing per-scene rows without rerunning any unwrap. The summary includes the
two declared catastrophic-scene frequencies: RMSE at least one full cycle and
pixel slip rate at least 5%.
