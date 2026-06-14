# Learned Statistical Costs for Network-Flow InSAR Phase Unwrapping

A deep-learning-augmented phase unwrapping framework that preserves the global optimality of minimum-cost-flow solvers while learning scene-adaptive cost functions from data. Instead of parametric models, a vision transformer estimates per-edge ambiguity distributions whose negative log-likelihoods serve as solver inputs. This decoupled design ensures zero-residue outputs regardless of the learned prior's accuracy.

This is the reference implementation for the work
*"Learned Statistical Costs for Network-Flow InSAR Phase Unwrapping"*
(Simon Dagbanja).

## Results

All numbers are aligned RMSE in radians, scored on identical inputs. Every
network-flow result is residue-free.

| Benchmark | Least squares | SNAPHU | **Learned cost (ours)** | Oracle | Floor |
|---|---:|---:|---:|---:|---:|
| InSAR-DLPU simulated (1000 scenes) | 4.33 | 1.71 | **0.90** | 0.84 | 0.76 |
| InSAR-DLPU real TanDEM-X (100 scenes) | 7.42 | 5.30 | **3.02** | 3.71 | 1.55 |

- On the simulated benchmark the learned cost beats SNAPHU by **48 %** and least
  squares by **79 %**, reaching the noise floor.
- On the real subset it beats SNAPHU by **43 %**, the largest margin of any
  network-flow method there.
- On four Sentinel-1 coseismic interferograms (Ridgecrest, Amatrice, Gorkha,
  Kumamoto) the method is residue-free, agrees with SNAPHU to a fraction of a
  fringe where coherence is good, and matches GPS to about 2 cm, all zero-shot.

Inference on a 1024×1024 scene takes ≈0.7 s for the edge head and ≈11.8 s for the
MCF solve on an RTX 3090.

## Method

```
wrapped phase ψ
      │
      ▼
Swin-Tiny + LoRA edge head      → per-edge class probabilities p(kₑ), kₑ ∈ {−1,0,+1}
      │
      ▼
per-edge cost  cₑ = −log p(kₑ)   (optionally + coherence cost)
      │
      ▼
minimum-cost flow on the dual grid (Costantini)   → residue-free integer field
      │
      ▼
DCT least-squares integration    → unwrapped phase  +  per-pixel uncertainty map
```

- **Solver core** ([`src/mcf.py`](src/mcf.py)): Costantini min-cost flow on the dual
  grid with asymmetric directional costs, residue-free by construction. OR-Tools
  network simplex backend with a networkx fallback.
- **Learned cost** ([`src/mcf_infer.py`](src/mcf_infer.py)): the edge head turns
  wrapped phase into directional negative-log-likelihood costs for the solver.
- **Combined cost**: fuses the learned cost with a coherence-based statistical
  cost, strongest where coherence is informative.
- **End-to-end differentiability** ([`src/mcf_diff.py`](src/mcf_diff.py)): a
  blackbox-solver gradient lets the unwrapping RMSE backpropagate through the
  combinatorial solver.
- **Uncertainty**: per-pixel entropy of the edge ambiguity distributions.

## Repository layout

```
configs/      Hydra YAML configs (default / data / model)
src/          Core package
  mcf.py            min-cost-flow solver + learned/combined costs
  mcf_infer.py      torch bridge: edge head -> costs -> solver
  mcf_diff.py       differentiable solver (blackbox gradient)
  baselines.py      least squares, quality-guided, Goldstein, SNAPHU wrappers
  models/           Swin + LoRA edge head, decoder, integrator
  dataset.py        InSAR-DLPU (.mat) and GeoTIFF loaders
  synthetic_engine.py  physics-based interferogram generator
scripts/      Entry points (eval, real-scene unwrapping, GPS, figures)
run_train.sh  Remote GPU launcher (forwards Hydra overrides)
```

## Setup

```bash
# Python 3.10 / 3.11. On a CUDA box install the matching torch build first
# (see pytorch.org), then:
pip install -r requirements.txt

# Solver backend (fast, exact) and an optional baseline:
pip install ortools          # min-cost-flow network simplex
pip install snaphu           # only needed to run SNAPHU as a baseline
```

Data is **not** included in the repository. Place the InSAR-DLPU release under
`data/` (the loader resolves its nested split folders automatically) and any
real Sentinel-1 HyP3 products under `data/S1_data/`. See
[`src/dataset.py`](src/dataset.py) for the expected layout.

## Reproduce the results

Benchmark the learned cost against every baseline on the same scenes (add
`eval.snaphu=true` to include SNAPHU):

```bash
# Simulated benchmark (Table I)
python -m scripts.eval_learned_mcf checkpoint.resume=checkpoints/edge_dlpu_clean.pt \
    data.source=dlpu eval.split=test eval.snaphu=true logging.backend=none

# Real TanDEM-X subset (Table II)
python -m scripts.eval_learned_mcf checkpoint.resume=checkpoints/edge_dlpu_clean.pt \
    data.source=dlpu eval.split=test_real eval.snaphu=true logging.backend=none
```

Unwrap a real Sentinel-1 scene and compare to the bundled SNAPHU result:

```bash
python -m scripts.unwrap_geotiff \
    checkpoint.resume=checkpoints/edge_dlpu_clean.pt \
    geotiff.dir=data/S1_data/<product folder> \
    geotiff.row0=300 geotiff.col0=750 geotiff.size=1536 \
    geotiff.coh_weight=3.0 geotiff.out=outputs/ridgecrest logging.backend=none
```

Train the edge head (the released model used these settings):

```bash
python -m src.train regime=pretrain data.source=dlpu data.batch_size=8 \
    data.epoch_size=30000 train.epochs=25 train.lr=2e-4 \
    model.decoder.head_type=edge logging.backend=none
```

Regenerate the summary figures:

```bash
python -m scripts.make_figures        # bar charts into outputs/
```

## Synthetic engine

[`src/synthetic_engine.py`](src/synthetic_engine.py) generates interferograms on
the fly from physical sources (Okada faults, Mogi sources, subsidence bowls) plus
fractal topography, a turbulent atmosphere, and a spatially varying coherence
field with coherence-driven noise. Unlike InSAR-DLPU it emits a coherence band,
so the coherence-weighted path can be exercised.

```bash
python -m src.synthetic_engine        # physics self-test
```

## Configuration

Everything is managed with Hydra; any field in `configs/` can be overridden on the
command line. Logging supports Weights & Biases or TensorBoard, or `none` for a
quiet run.

## Citation

If you use this code, please cite the manuscript:

```bibtex
@article{dagbanja_learned_mcf,
  title  = {Learned Statistical Costs for Network-Flow InSAR Phase Unwrapping},
  author = {Dagbanja, Simon},
  year   = {2026}
}
```
