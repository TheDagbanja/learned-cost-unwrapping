# Tail-robust InSAR phase unwrapping with learned optimization costs

Research code for **Learned Optimization Costs for Tail-Robust InSAR Phase
Unwrapping**. The method predicts signed horizontal and vertical ambiguity
probabilities, converts them to optimization costs, and uses a
residue-constrained minimum-cost-flow (MCF) solver to recover a globally
feasible unwrapped phase.

The current repository supports the reconstructed evaluation used for the JAG
manuscript: fair SNAPHU calibration and sensitivity analysis, complete
cycle-slip decomposition, tail-risk metrics, same-predictor signed/unsigned
ablations, strict leave-one-deformation-mechanism-out (LODO) training, reference
provenance audits, uncertainty calibration, real-event disagreement ranking,
and independent GPS validation.

## Main result

The evidence supports a failure-tail claim rather than universal superiority:
learned costs reduce catastrophic whole-cycle failures under global constraints,
but the benefit depends on the relationship between training and deployment
distributions.

On the 1,000-scene simulated test split:

| Method | Mean RMSE | Median | P95 | CVaR95 | Scenes ≥ 2π | Pixel slips |
|---|---:|---:|---:|---:|---:|---:|
| Learned signed cost + MCF | 0.895 | 0.798 | 1.248 | 2.985 | 0.2% | 1.0% |
| Learned unsigned cost + MCF | 0.916 | 0.798 | 1.262 | 3.366 | 0.3% | 1.0% |
| DENet cost + shared MCF | 0.931 | 0.802 | 1.329 | 3.540 | 0.3% | 0.9% |
| Calibration-selected SNAPHU | 1.475 | 0.822 | 3.715 | 12.839 | 2.9% | 4.4% |
| Unit-cost MCF | 2.315 | 0.863 | 10.413 | 22.201 | 8.0% | 9.1% |

The selected SNAPHU setting is phase-derived proxy information, smooth costs,
and 16 looks. It was selected on a separate unaugmented calibration subset and
then frozen for the test comparison. Strict LODO evaluation gives the learned
signed-cost method the lowest reported mean and tail error in every fold, while
the fault holdout remains strongly heavy-tailed. Synthetic entropy ranking
achieves AUROC 0.918 and average precision 0.171; a 20% review budget captures
86.9% of cycle-error pixels. On the real earthquake cases, the solver-based
methods are tied at the reported GPS precision, and entropy is interpreted as a
review-ranking signal rather than calibrated real-domain error probability.

Small machine-readable summaries supporting these statements are committed in
[`paper_results`](paper_results/README.md). Large per-scene outputs are ignored.

## Implemented components

- Swin-based signed edge-ambiguity prediction (`src/`).
- Directed learned costs with exact MCF inference.
- Three-class signed edge-ambiguity prediction and common training infrastructure.
- Same-predictor signed, unsigned, unit-cost, and local-integration ablations.
- Classical SNAPHU controls with explicit information source, cost mode, and
  number-of-looks settings.
- DENet cost prediction with the shared MCF backend.
- Direct and hybrid baselines in `ext_dl`: DLPU-CNN, PhaseNet-2, GradientNet,
  Attention U-Net, DeepLabV3+, U-Net++, DENet, MoDL-PU, Unwrap-Net,
  DMP-PUNet, and WaveUNet.
- Coherence-weighted least squares for evaluation.
- Strict fault/volcanic/subsidence/atmosphere LODO training.
- TanDEM-X/SRTM provenance auditing.
- Synthetic precision-recall, reliability, Brier score, expected calibration
  error, risk-coverage, and fixed-review-coverage analysis.
- Zero-shot Sentinel-1 unwrapping, real-event solver-disagreement ranking, and
  GPS line-of-sight validation.

The five-class capacity ablation reported in the paper was produced from a
separate experimental checkpoint run and is not required for the core fair
benchmark. The current public MCF path uses the
three-class head. The repository does not redistribute third-party datasets,
large rasters, or model weights.

## Repository layout

```text
configs/          Hydra data, model, and real-event manifests
src/              training, learned costs, MCF, metrics, and data loaders
ext_dl/           direct-learning and hybrid baseline implementations
scripts/          audits, evaluation, LODO, uncertainty, GPS, and figures
audit/            lightweight protocol checks
paper_results/    curated manuscript summaries and audit outputs
```

`data/`, `checkpoints/`, `outputs/`, `results/`, `logs/`, and `manuscript/` are
local working directories and are intentionally excluded from Git.

## Environment

Python 3.10 or 3.11 and PyTorch 2.x are recommended. Install a CUDA-compatible
PyTorch build first on GPU systems, then install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ext.txt
```

If Hydra fails with `Could not deserialize ATN with version 3`, reinstall the
compatible parser stack:

```bash
python -m pip uninstall -y antlr4-python3-runtime
python -m pip install --force-reinstall \
  'hydra-core==1.3.2' 'omegaconf==2.3.1' 'antlr4-python3-runtime==4.9.3'
```

## Data and checkpoints

The code expects the public InSAR-DLPU release beneath `data/`; the loader
resolves the release's nested directory layout automatically. Real-event
products are declared in `configs/real_uncertainty_events.json`.

The paper runs used these checkpoint identities:

| Checkpoint | Size | SHA-256 |
|---|---:|---|
| Edge predictor | 383,521,311 bytes | `c09dbc6c4669ad04b8135253f4dcd02bd91f8e13970d17cf23111348fa7316bc` |
| DENet | 1,283,001 bytes | `8ccd2d07e5dac56b39e0b7242eef44e2711e70160d66b0396d393a37ce4dbd72` |

Set shell variables to the corresponding local files before running the paper
experiments:

```bash
EDGE_CKPT=/absolute/path/to/edge_dlpu_clean.pt
DENET_CKPT=/absolute/path/to/denet_last.pt
```

## Reproduction workflow

Run all commands from the repository root.

### 1. Protocol smoke checks

```bash
python -m scripts.test_evaluation_rebuild
python -m scripts.test_mismatch_protocol
python -m scripts.run_lodo --dry-run
python -m compileall -q src ext_dl scripts audit
```

### 2. Train the edge predictor

The main learned-cost model uses the signed three-class edge head.

```bash
python -m src.train \
  regime=pretrain device=cuda \
  model.decoder.head_type=edge model.decoder.out_channels=6 \
  train.epochs=50 checkpoint.dir=checkpoints/edge_main
```

### 3. Calibrate SNAPHU independently

This step tunes only the classical baseline. It does not update the learned
model and does not inspect the test split.

```bash
python -m scripts.eval_fair_benchmark \
  eval.split=train_eval \
  data.limit=1000 data.sample_strategy=random data.sample_seed=20261030 \
  'ext.fair_methods=[snaphu]' \
  'ext.fair_snaphu_sources=[uniform,phase_proxy]' \
  'ext.fair_snaphu_costs=[defo,smooth]' \
  'ext.fair_snaphu_nlooks=[1,4,8,16]' \
  ext.out_dir=results/fair_benchmark/snaphu_calibration
```

### 4. Run the fair simulated benchmark

```bash
python -m scripts.eval_fair_benchmark \
  checkpoint.resume="$EDGE_CKPT" \
  ext.denet_checkpoint="$DENET_CKPT" \
  eval.split=test \
  'ext.fair_methods=[learned_signed,learned_unsigned,unit_mcf,denet,snaphu]' \
  'ext.fair_snaphu_sources=[uniform,phase_proxy]' \
  'ext.fair_snaphu_costs=[defo,smooth]' \
  'ext.fair_snaphu_nlooks=[1,4,8,16]' \
  ext.out_dir=results/fair_benchmark/test

python -m scripts.select_snaphu_config \
  results/fair_benchmark/snaphu_calibration/fair_benchmark_train_eval.csv \
  results/fair_benchmark/test/fair_benchmark_test.csv \
  --metric mean \
  --out results/fair_benchmark/snaphu_selection.json
```

The evaluator writes per-scene rows, a summary, and a protocol manifest. The
summary includes mean, median, P90/P95/P99, CVaR95, maximum, cycle-slip rate,
slip-free error, two catastrophic-scene frequencies, residue violations, and
runtime statistics.

### 5. Run same-predictor solver ablations

```bash
python -m scripts.eval_solver_ablations \
  checkpoint.resume="$EDGE_CKPT" eval.split=test \
  ext.out_dir=results/extension
```

This compares signed and unsigned learned costs, unit costs, and local argmax
integration while keeping the predictor and evaluation scenes fixed wherever
the formulation permits.

### 6. Run strict leave-one-domain-out experiments

```bash
python -m scripts.run_lodo --mode all --device cuda --skip-existing
python -m scripts.summarize_lodo results/lodo
```

Each fold trains from scratch on three primary deformation mechanisms and tests
only the held-out fourth mechanism.

### 7. Audit the training and reference distributions

```bash
python -m scripts.plot_training_distribution \
  --root data --out results/training_distribution

python -m scripts.audit_tandemx_provenance \
  --root data --out results/tandemx_provenance
```

The TanDEM-X targets are reported as DEM-derived reference solutions, not ground
truth, because the public release does not contain the geolocation, sensor, or
per-scene construction metadata needed for independent physical verification.

### 8. Evaluate uncertainty

```bash
python -m scripts.dump_fig_arrays \
  checkpoint.resume="$EDGE_CKPT" eval.split=test \
  +out=results/uncertainty_input +n_scenes=500

python -m scripts.uncertainty_calibration \
  --input results/uncertainty_input/calib.npz \
  --out results/uncertainty

python -m scripts.eval_real_uncertainty \
  --manifest configs/real_uncertainty_events.json \
  --out results/real_uncertainty
```

Calibration and evaluation use disjoint complete scenes. Real-event analysis
uses entropy only as a ranking score against solver disagreement and sparse GPS
residuals.

### 9. Reproduce the GPS tables

```bash
python -m scripts.reproduce_gps_tables \
  outputs/gps_ridgecrest_v2_gated_stations.csv \
  outputs/gps_amatrice_gated_stations.csv \
  outputs/gps_kumamoto_big_stations.csv \
  --min-coherence 0.5
```

### 10. Rebuild paper figures from completed outputs

```bash
python -m scripts.plot_paper_figures
python -m scripts.plot_representative_methods --help
```

Use `--help` on any script for its current file and output arguments. Figure
selection is driven by paired per-scene tables rather than manual cherry-picking.

## Evaluation conventions

- Remove one global phase offset before aligned RMSE.
- Define a cycle slip from the nearest nonzero integer multiple of `2π` after
  alignment.
- Report slip-free residual separately from integer-cycle failure.
- Treat MCF as guaranteeing feasibility and optimality for the supplied costs,
  not physical correctness of the unknown absolute phase.
- Compare methods only on shared scene IDs and preserve calibration/test
  separation.
- Call real solver differences disagreement unless an independent observation
  supplies the endpoint.

## License and third-party data

Follow the licenses and citation requirements of InSAR-DLPU, Sentinel-1/HyP3,
Nevada Geodetic Laboratory, SNAPHU, and the baseline implementations. Their data
and weights are not included in this repository.
