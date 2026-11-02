# ext_dl — deep-learning baselines for the journal extension

Eleven reproduced deep unwrappers, trained and scored on the **identical
InSAR-DLPU data and protocol** as the learned-cost method, so the manuscript's
Table I/II comparison is like-for-like. Reuses the repo's `src` (dataset,
utils, integrator, MCF solver) and the exact slip-decomposition scorer from
`scripts/eval_learned_mcf.py`.

## The eleven baselines
| name | type | scored on |
|---|---|---|
| `dlpu_cnn` | per-pixel wrap-count regression (U-Net) | `phi_hat` |
| `phasenet2` | per-pixel wrap-count classification | `phi_hat` |
| `attention_unet` | per-pixel regression (attention U-Net) | `phi_hat` |
| `deeplabv3plus` | per-pixel classification (ASPP) | `phi_hat` |
| `unetpp` | per-pixel regression (UNet++) | `phi_hat` |
| `gradient_net` | edge-gradient regression + LS integration | `phi_hat` (not residue-free) |
| `denet` | **hybrid**: learned per-edge cost → **same MCF solver** | cost → solve → `phi_hat` |
| `modl_pu` | two-stage model-based deep unrolling adapter | `phi_hat` |
| `unwrap_net` | direct phase-regression adapter | `phi_hat` |
| `dmp_punet` | direct phase-regression adapter | `phi_hat` |
| `waveunet` | wavelet-domain phase-regression adapter | `phi_hat` |

`denet` is the nearest neighbour of our method (keeps the solver, learns the
cost); it is scored *through* the min-cost-flow solver, isolating the cost.

## Train (one baseline at a time), from the repo root
```bash
python -m ext_dl.train_dl baseline=dlpu_cnn       train.epochs=25
python -m ext_dl.train_dl baseline=phasenet2      train.epochs=25 model.decoder.k_max=16
python -m ext_dl.train_dl baseline=gradient_net   train.epochs=25
python -m ext_dl.train_dl baseline=attention_unet train.epochs=25
python -m ext_dl.train_dl baseline=deeplabv3plus  train.epochs=25 model.decoder.k_max=16
python -m ext_dl.train_dl baseline=unetpp         train.epochs=25
python -m ext_dl.train_dl baseline=denet          train.epochs=25
python -m ext_dl.train_dl baseline=modl_pu        train.epochs=25
python -m ext_dl.train_dl baseline=unwrap_net     train.epochs=25
python -m ext_dl.train_dl baseline=dmp_punet      train.epochs=25
python -m ext_dl.train_dl baseline=waveunet       train.epochs=25
```
Same optimizer/schedule/tiles as the main model (AdamW 2e-4, wd 0.05, batch 8,
warmup + cosine). Checkpoints → `<checkpoint.dir>/{baseline}_last.pt`.

Loop all eleven:
```bash
for b in dlpu_cnn phasenet2 gradient_net attention_unet deeplabv3plus unetpp denet modl_pu unwrap_net dmp_punet waveunet; do
  python -m ext_dl.train_dl baseline=$b train.epochs=25 checkpoint.dir=checkpoints/dlpu
done
```

## Score (all trained baselines at once)
```bash
# simulated benchmark (1000 scenes) -> Table I DL rows
python -m ext_dl.eval_dl eval.split=test checkpoint.dir=checkpoints/dlpu
# real TanDEM-X subset (100 scenes) -> Table II DL rows
python -m ext_dl.eval_dl eval.split=test_real checkpoint.dir=checkpoints/dlpu
```
Prints `RMSE / slip% / slip-free RMSE / residues` per method (the paper's
decomposition, identical scorer to the learned-cost rows) and writes
`checkpoints/dlpu/dl_leaderboard_<split>.csv`.

## Feed the manuscript
The CSV has columns `split,name,rmse,slip,eps,residues,family` — drop the DL
rows into Tables I/II (replacing the `\TODO` cells) and, combined with the
learned-cost/classical rows, pass the merged CSV to the figure generator:
```bash
python -m scripts.paper_figures --figure slip \
    --leaderboard <merged_leaderboard>.csv \
    --out manuscript/Extension/figs
```

## Notes
- `k_max` for the per-pixel classifiers (`phasenet2`, `deeplabv3plus`) is the
  wrap-count classification range (`model.decoder.k_max`, default 16); for the
  edge-based `denet`/`gradient_net` labels it is the C-band ambiguity range
  (`data.k_max`, default 1).
- All eleven were CPU-smoke-tested end-to-end on the common protocol
  (forward/loss/backward, and both eval paths) before hand-off; only GPU
  training remains, which produces the genuine numbers.
