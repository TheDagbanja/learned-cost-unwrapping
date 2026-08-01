"""Stage-2 evaluation: learned-cost MCF unwrapping with a trained edge head.

Loads an edge-head checkpoint, then for each scene compares (aligned RMSE vs GT):
  * MCF + unit cost          (solver only, no learning)
  * MCF + LEARNED cost       (edge head -> costs -> solver)   <- the method
  * MCF + GT-GRADIENT cost   (cost prior from the true per-edge cuts)
  * GT-K floor (offset-aligned)                               <- the CEILING
All MCF outputs are residue-free by construction.

NOTE (audit/DIAGNOSIS.md sec.1): the GT-gradient-cost row is a *reference cost*
built from the ground-truth gradients, NOT an upper bound. Matching the true
per-edge field k* does not minimise RMSE under noise -- the GT-K floor
does -- so the GT-gradient row can legitimately be beaten. The floor is the only
genuine ceiling.

Run:
    python -m scripts.eval_learned_mcf checkpoint.resume=checkpoints/edge_last.pt \\
        data.source=dlpu eval.split=test logging.backend=none
    python -m scripts.eval_learned_mcf checkpoint.resume=checkpoints/edge_last.pt \\
        data.source=dlpu eval.split=test_real logging.backend=none
"""

from __future__ import annotations

import os

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src import dataset as dataset_module
from src import mcf
from src import mcf_infer
from src import baselines
from src.models.unwrapper import PhaseUnwrapper


def _aligned_rmse(a, b):
    d = a - b
    d = d - d.mean()
    return float(np.sqrt((d ** 2).mean()))


def _residues_pred(phi, psi, R_psi=None):
    """Loops where ``phi`` is NOT integrable-consistent with ``psi``.

    Uses the per-edge implied ambiguity ``n = round((d_phi - W(d_psi)) / 2pi)`` and
    checks ``curl(n) + R_psi == 0``. The previous version took the curl of the
    solution's OWN raw gradient, which telescopes to zero for any single-valued
    field and so returned 0 for every method including least squares -- it could
    not discriminate anything (audit/DIAGNOSIS.md sec.0, validated sec.3-I2).
    """
    if R_psi is None:
        R_psi = mcf.residues(psi)
    nx = np.round(((phi[:, 1:] - phi[:, :-1])
                   - mcf.wrap(psi[:, 1:] - psi[:, :-1])) / mcf.TWO_PI)
    ny = np.round(((phi[1:, :] - phi[:-1, :])
                   - mcf.wrap(psi[1:, :] - psi[:-1, :])) / mcf.TWO_PI)
    defect = nx[:-1, :] + ny[:, 1:] - nx[1:, :] - ny[:, :-1] + R_psi
    return int(np.count_nonzero(defect))


def _metrics(pred, gt, psi, R_psi=None):
    """(rmse, cycle-slip rate, slip-free rmse, residues) for one scene.

    Headline RMSE conflates two distinct failures: whole-cycle slips and sub-cycle
    noise. ``slip`` is the fraction of pixels whose mean-aligned residual rounds to
    a nonzero multiple of 2pi; ``rmse_slipfree`` removes those integer offsets and
    so measures the noise alone (audit/DIAGNOSIS.md sec.2).
    """
    d = pred - gt
    d = d - d.mean()
    r = np.round(d / mcf.TWO_PI)
    dsf = d - mcf.TWO_PI * r
    return (float(np.sqrt((d ** 2).mean())), float((r != 0).mean()),
            float(np.sqrt((dsf ** 2).mean())), _residues_pred(pred, psi, R_psi))


def _gt_gradient_costs(psi, phi):
    """Cut-location prior from the ground-truth per-edge cuts k*.

    NOT an oracle/ceiling: symmetric (sign of k* is discarded at mcf.py:323) and
    soft (0.01 vs 1.0). Even a hard directional version gives the identical field
    (audit/phase1_oracle.py), because k*-matching is not the RMSE-optimal
    residue-free unwrapping under noise.
    """
    dxw = mcf.wrap(psi[:, 1:] - psi[:, :-1])
    dyw = mcf.wrap(psi[1:, :] - psi[:-1, :])
    kx = np.round(((phi[:, 1:] - phi[:, :-1]) - dxw) / mcf.TWO_PI)
    ky = np.round(((phi[1:, :] - phi[:-1, :]) - dyw) / mcf.TWO_PI)
    return np.where(kx != 0, 0.01, 1.0), np.where(ky != 0, 0.01, 1.0)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not cfg.checkpoint.resume or not os.path.exists(cfg.checkpoint.resume):
        raise SystemExit(f"Need a trained edge-head checkpoint at checkpoint.resume "
                         f"(got {cfg.checkpoint.resume!r}).")
    ckpt = torch.load(cfg.checkpoint.resume, map_location=device)
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if saved and "model" in saved:
        cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(saved["model"]))
        if "data" in saved and "input_grad" in saved["data"]:
            cfg.data.input_grad = saved["data"]["input_grad"]
    cfg.model.backbone.in_chans = dataset_module.input_channels(cfg.data)
    model = PhaseUnwrapper(model_cfg=cfg.model, img_size=cfg.data.get("img_size", 256)).to(device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise SystemExit(f"Checkpoint/arch mismatch (missing={len(missing)}, "
                         f"unexpected={len(unexpected)}). Wrong head_type?")
    if model.head_type != "edge":
        raise SystemExit(f"Learned-cost MCF needs head_type=edge; checkpoint is "
                         f"{model.head_type!r}.")
    model.eval()
    print(f"Loaded edge head: {cfg.model.backbone.name}")

    split = cfg.eval.get("split", "test")
    loader = dataset_module.build_dataloader(cfg.data, split=split, shuffle=False, device=device)
    max_batches = cfg.eval.get("max_batches", None)
    do_snaphu = bool(cfg.eval.get("snaphu", False))
    if do_snaphu and not baselines.snaphu_available():
        raise SystemExit("eval.snaphu=true but the 'snaphu' package is not installed "
                         "(pip install snaphu).")

    do_gtgrad = bool(cfg.eval.get("oracle_cost", True))      # config key kept for compat
    do_pathfollow = bool(cfg.eval.get("pathfollow", True))   # LS / quality-guided / Goldstein
    from src import utils
    M: dict = {}                      # method -> list of (rmse, slip, slipfree, residues)
    floor, res_in = [], []

    def add(name, pred, gt_, psi_, R_psi):
        M.setdefault(name, []).append(_metrics(pred, gt_, psi_, R_psi))
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))
    bar = tqdm(loader, total=n_batches, desc=f"Learned-cost MCF [{split}]", dynamic_ncols=True)
    for i, batch in enumerate(bar):
        if max_batches is not None and i >= max_batches:
            break
        batch = utils.move_batch_to_device(batch, device)
        phi_l, phi_u = mcf_infer.unwrap_batch(model, batch, also_unit=True)
        psi = batch["wrapped"][:, 0].detach().cpu().numpy()
        gt = batch["absolute"][:, 0].detach().cpu().numpy()
        for b in range(gt.shape[0]):
            R_psi = mcf.residues(psi[b])
            res_in.append(int(np.abs(R_psi).sum()))
            add("learned-cost MCF (ours)", phi_l[b], gt[b], psi[b], R_psi)
            add("unit-cost MCF", phi_u[b], gt[b], psi[b], R_psi)
            if do_gtgrad:
                wx, wy = _gt_gradient_costs(psi[b], gt[b])
                add("GT-gradient-cost MCF", mcf.unwrap_mcf(psi[b], wx=wx, wy=wy),
                    gt[b], psi[b], R_psi)
            floor.append(mcf.aligned_gtk_floor(psi[b], gt[b]))
            if do_snaphu:
                add("SNAPHU", baselines.snaphu_unwrap(psi[b]), gt[b], psi[b], R_psi)
            if do_pathfollow:                                  # same scenes -> consistent Table I
                add("least squares", baselines.ls_unwrap(psi[b]), gt[b], psi[b], R_psi)
                add("quality-guided", baselines.quality_guided_unwrap(psi[b]),
                    gt[b], psi[b], R_psi)
                add("Goldstein", baselines.goldstein_unwrap(psi[b]), gt[b], psi[b], R_psi)
        ours = M["learned-cost MCF (ours)"]
        bar.set_postfix(learned=f"{np.mean([x[0] for x in ours]):.3f}", n=len(ours))
    bar.close()

    n_sc = len(floor)
    mean_of = lambda name, j: float(np.mean([x[j] for x in M[name]]))
    print(f"\n=== Learned-cost MCF on '{split}'  ({n_sc} scenes) ===")
    print(f"  input residues / image : {np.mean(res_in):.1f}")
    print(f"\n{'method':26s}{'RMSE':>8}{'slip%':>8}{'RMSE_slipfree':>15}{'residues':>10}")
    for name in sorted(M, key=lambda k: mean_of(k, 0)):
        print(f"{name:26s}{mean_of(name,0):>8.3f}{100*mean_of(name,1):>7.1f}%"
              f"{mean_of(name,2):>15.3f}{mean_of(name,3):>10.1f}")
    print(f"{'GT-K floor (the CEILING)':26s}{np.mean(floor):>8.3f}")
    print("\nRMSE conflates whole-cycle slips with sub-cycle noise: `slip%` is how often a"
          "\nmethod is off by a whole cycle, `RMSE_slipfree` is the noise once those integer"
          "\noffsets are removed. `residues` counts loops inconsistent with the wrapped input"
          "\n(0 for congruent methods, >0 only for gradient-domain/LS).")
    ours = mean_of("learned-cost MCF (ours)", 0)
    label = "SNAPHU" if do_snaphu else "unit-cost MCF"
    if label in M and mean_of(label, 0) > ours:
        print(f"\n  --> learned-cost MCF beats {label} by "
              f"{100 * (1 - ours / mean_of(label, 0)):.0f}%")


if __name__ == "__main__":
    main()
