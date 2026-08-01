# Learned Statistical Costs for Network-Flow InSAR Phase Unwrapping

A phase unwrapper that keeps the rigorous minimum-cost-flow solver of SNAPHU and
**learns its cost function** instead of deriving it by hand. A Swin Transformer
with low-rank adaptation predicts, for every edge of the interferogram grid, a
probability over the signed integer ambiguity. The negative log of that
probability becomes the per-edge cost of a min-cost-flow solver, so the unwrapped
field is **residue-free by construction** while the statistical model that drives
it is learned from data.

Reference implementation for *"Learned Statistical Costs for Network-Flow InSAR
Phase Unwrapping"* — Simon Dagbanja and Jiang Qian (University of Electronic
Science and Technology of China), Lv Haitao (Aerospace Information Technology
University).

## Results

> **Note on numbers.** The figures below reflect corrections to the evaluation
> harness made *after* the conference submission, so they differ from the
> submitted manuscript. Three changes: the error floor is now optimised jointly
> with the reference offset (the earlier per-pixel definition is not a bound for
> the aligned metric); the residue counter no longer takes the solution's own
> gradient curl, which is identically zero for any congruent field; and SNAPHU is
> run in its strongest configuration rather than with a pseudo-coherence quality
> map. The corrected margins over SNAPHU are **41 %** (simulated) and **25 %**
> (real), against 48 % and 43 % before. The row previously labelled *Oracle* is a
> reference cost, not an upper bound, and is renamed accordingly. Every
> derivation, including the tests that established each defect, is in
> [`audit/DIAGNOSIS.md`](audit/DIAGNOSIS.md).

Aligned RMSE in radians, all methods scored on identical scenes. SNAPHU is run
with ENL-matched looks and the deformation cost mode. Every flow result is
residue-free.

| Benchmark | Least squares | SNAPHU | **Learned cost (ours)** | GT-gradient cost | Floor |
|---|---:|---:|---:|---:|---:|
| InSAR-DLPU simulated (1000 scenes) | 4.33 | 1.53 | **0.90** | 0.84 | 0.76 |
| InSAR-DLPU real TanDEM-X (100 scenes) | 7.42 | 4.01 | **3.02** | 3.71 | 1.27 |

- **41 %** below SNAPHU on the simulated benchmark and **79 %** below least
  squares, within 0.14 rad of the achievable noise floor.
- **25 %** below SNAPHU on the real subset.
- On four Sentinel-1 coseismic interferograms (Ridgecrest, Amatrice, Gorkha,
  Kumamoto) the method is residue-free, agrees with SNAPHU to a fraction of a
  fringe where coherence is good, and matches GPS to about 2 cm, all without
  retraining on real data.

Two caveats we state explicitly rather than bury. The **GT-gradient-cost** row is
a reference cost built from the true per-edge cuts, **not an upper bound** —
matching the true gradient field is not RMSE-optimal under noise, so it can be,
and is, beaten. The **floor** is the only bound in the table. On the real subset
reliability-guided unwrapping reaches 2.38 rad and beats every global solver,
including the ground-truth-derived cost; that subset is noise-dominated (only
0.66 % of edges carry a genuine discontinuity) and cannot adjudicate unwrapper
accuracy. Both points are derived in [`audit/DIAGNOSIS.md`](audit/DIAGNOSIS.md).

Inference on a 1024×1024 scene takes about 0.7 s for the edge head and 11.8 s for
the MCF solve on an RTX 3090.

## Method

```
wrapped phase psi
      |
      v
Swin-Tiny + LoRA edge head      -> per-edge class probabilities p(k_e), k_e in {-1,0,+1}
      |
      v
per-edge cost  c_e^+- = -log p(+-1)   (optionally + lambda * coherence)
      |
      v
minimum-cost flow on the dual grid (Costantini)  -> residue-free integer field
      |
      v
DCT least-squares integration   -> unwrapped phase  +  reliability map
```

- **Solver** ([`src/mcf.py`](src/mcf.py)): Costantini min-cost flow with
  asymmetric directional arc costs, residue-free by construction. OR-Tools
  network simplex, networkx fallback.
- **Learned cost** ([`src/mcf_infer.py`](src/mcf_infer.py)): edge head to
  directional negative-log-likelihood costs.
- **Combined cost**: fuses the learned cost with a coherence term. Helps only
  where genuine coherence exists; see the ablation notes below.
- **Differentiability** ([`src/mcf_diff.py`](src/mcf_diff.py)): blackbox-solver
  gradient through the combinatorial step.
- **Reliability map**: per-pixel entropy of the edge ambiguity distributions. It
  predicts cycle errors with AUROC 0.911, against 0.836 for a conventional
  wrapped-gradient quality proxy.

## Evaluation methodology

This repository implements three corrections to common practice, derived and
validated in [`audit/DIAGNOSIS.md`](audit/DIAGNOSIS.md):

1. **The residue count must not be the solution's own gradient curl.** That
   telescopes to zero for any single-valued field, so it reads zero for every
   congruent method regardless of quality. `_residues_pred` now uses the per-edge
   implied ambiguity checked against the observed wrapped phase, which correctly
   reports thousands of inconsistent loops for least squares and zero for the
   flow methods.
2. **Aggregate RMSE conflates whole-cycle slips with sub-cycle noise.** The
   harness reports `slip%` and `RMSE_slipfree` alongside RMSE. On both splits the
   slip-free error is nearly constant across methods and close to the floor, so
   the ranking is a cycle-slip ranking rather than an accuracy ranking.
3. **The floor must be optimized jointly with the reference offset**
   (`mcf.aligned_gtk_floor`). The usual per-pixel nearest-integer definition
   minimizes the *unaligned* residual and is not a bound for the aligned metric;
   methods beat it on roughly one scene in four of the real subset.

## Repository layout

```
configs/      Hydra YAML configs (default / data / model)
src/          Core package
  mcf.py            min-cost-flow solver, learned/combined costs, aligned floor
  mcf_infer.py      torch bridge: edge head -> costs -> solver
  mcf_diff.py       differentiable solver (blackbox gradient)
  baselines.py      least squares, quality-guided, Goldstein, SNAPHU wrappers
  models/           Swin + LoRA edge head, decoder, integrator
  dataset.py        InSAR-DLPU (.mat) and GeoTIFF loaders
  synthetic_engine.py  physics-based interferogram generator
scripts/      Entry points (eval, real-scene unwrapping, GPS, figures)
audit/        Forensic audit of the evaluation harness + diagnostic scripts
run_train.sh  Remote GPU launcher (forwards Hydra overrides)
```

## Setup

```bash
# Python 3.10 / 3.11. On a CUDA box install the matching torch build first
# (see pytorch.org), then:
pip install -r requirements.txt

pip install ortools          # fast, exact min-cost-flow backend
pip install snaphu           # only needed to run SNAPHU as a baseline
```

Data is **not** included. Place the InSAR-DLPU release under `data/` (the loader
resolves its nested split folders automatically) and any Sentinel-1 HyP3 products
under `data/S1_data/`. See [`src/dataset.py`](src/dataset.py).

## Reproduce

Benchmark against every baseline on identical scenes. The harness prints RMSE,
cycle-slip rate, slip-free RMSE and the corrected residue count per method:

```bash
python -m scripts.eval_learned_mcf checkpoint.resume=checkpoints/edge_dlpu_clean.pt \
    data.source=dlpu eval.split=test eval.snaphu=true logging.backend=none

python -m scripts.eval_learned_mcf checkpoint.resume=checkpoints/edge_dlpu_clean.pt \
    data.source=dlpu eval.split=test_real eval.snaphu=true logging.backend=none
```

Unwrap a real Sentinel-1 scene and compare against the bundled SNAPHU product:

```bash
python -m scripts.unwrap_geotiff \
    checkpoint.resume=checkpoints/edge_dlpu_clean.pt \
    geotiff.dir=data/S1_data/<product folder> \
    geotiff.row0=300 geotiff.col0=750 geotiff.size=1536 \
    geotiff.coh_weight=3.0 geotiff.out=outputs/ridgecrest logging.backend=none
```

Validate against GPS. Any number of methods can be scored at the same stations
with repeated `--method NAME=PATH`:

```bash
python -m scripts.gps_validate --product data/S1_data/<product folder> \
    --ours outputs/ridgecrest_learned_unw.tif \
    --gps data/gps_m64_*.txt data/gps_m71_*.txt --out outputs/gps
```

Train the edge head (settings used for the released model):

```bash
python -m src.train regime=pretrain data.source=dlpu data.batch_size=8 \
    data.epoch_size=30000 train.epochs=25 train.lr=2e-4 \
    model.decoder.head_type=edge logging.backend=none
```

Figures:

```bash
python -m scripts.make_figures          # summary bar charts into outputs/
python -m scripts.make_pipeline_figure  # method diagram (PDF / PNG / SVG)
```

## Notes from the ablation

A one-factor-at-a-time sweep, with three seeds on the decisive groups, found:

- The method is insensitive to capacity: LoRA rank 1 to 32 spans 1.1 % of error,
  and the tiny backbone is no worse than the small one, so the useful operating
  point trains about 1.4e5 parameters.
- Inputs and data dominate: the wrapped-gradient input channels are worth 4.6 %,
  and 5000 training pairs already beat SNAPHU by 43 %.
- Two components are **harmful** in this setting. The congruence loss degrades
  error by 57 %, and coherence fusion degrades it monotonically when only a
  pseudo-coherence proxy is available, so the combined cost is justified only
  where genuine coherence accompanies the interferogram.

## Citation

```bibtex
@inproceedings{dagbanja2026learnedcost,
  title     = {Learned Statistical Costs for Network-Flow InSAR Phase Unwrapping},
  author    = {Dagbanja, Simon and Qian, Jiang and Lv, Haitao},
  booktitle = {Proc. IEEE Asia-Pacific Conf. Geoscience, Electronics and Remote
               Sensing Technology (AP-GARSS)},
  year      = {2026}
}
```
