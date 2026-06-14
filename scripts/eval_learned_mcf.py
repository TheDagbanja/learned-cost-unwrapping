"""Stage-2 evaluation: learned-cost MCF unwrapping with a trained edge head.

Loads an edge-head checkpoint, then for each scene compares (aligned RMSE vs GT):
  * MCF + unit cost          (solver only, no learning)
  * MCF + LEARNED cost       (edge head -> costs -> solver)   <- the method
  * MCF + ORACLE cost        (GT cuts -> costs)               <- ceiling
  * oracle floor (per-pixel GT-K)                             <- noise floor
All MCF outputs are residue-free by construction.

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


def _residues_pred(phi):
    """Goldstein residues of the SOLUTION's gradient field (raw diffs, no wrap),
    matching evaluate.count_residues. Zero for any consistent unwrap (the MCF
    guarantee); contrast with the re-wrapped input residues."""
    gx = phi[:, 1:] - phi[:, :-1]
    gy = phi[1:, :] - phi[:-1, :]
    curl = gx[:-1, :] + gy[:, 1:] - gx[1:, :] - gy[:, :-1]
    return int((np.round(curl / mcf.TWO_PI) != 0).sum())


def _oracle_costs(psi, phi):
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

    do_oracle = bool(cfg.eval.get("oracle_cost", True))
    do_pathfollow = bool(cfg.eval.get("pathfollow", True))   # LS / quality-guided / Goldstein
    from src import utils
    unit, learned, oracle, floor, res_in, snaphu_r = [], [], [], [], [], []
    res_learned = []
    ls_r, qg_r, gold_r = [], [], []
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
            res_in.append(int(np.abs(mcf.residues(psi[b])).sum()))
            res_learned.append(_residues_pred(phi_l[b]))      # solution gradient curl == 0
            learned.append(_aligned_rmse(phi_l[b], gt[b]))
            unit.append(_aligned_rmse(phi_u[b], gt[b]))
            if do_oracle:
                wx, wy = _oracle_costs(psi[b], gt[b])
                oracle.append(_aligned_rmse(mcf.unwrap_mcf(psi[b], wx=wx, wy=wy), gt[b]))
            k = np.round((gt[b] - psi[b]) / mcf.TWO_PI)
            floor.append(_aligned_rmse(psi[b] + mcf.TWO_PI * k, gt[b]))
            if do_snaphu:
                snaphu_r.append(_aligned_rmse(baselines.snaphu_unwrap(psi[b]), gt[b]))
            if do_pathfollow:                                  # same scenes -> consistent Table I
                ls_r.append(_aligned_rmse(baselines.ls_unwrap(psi[b]), gt[b]))
                qg_r.append(_aligned_rmse(baselines.quality_guided_unwrap(psi[b]), gt[b]))
                gold_r.append(_aligned_rmse(baselines.goldstein_unwrap(psi[b]), gt[b]))
        bar.set_postfix(learned=f"{np.mean(learned):.3f}", unit=f"{np.mean(unit):.3f}",
                        n=len(learned))
    bar.close()

    print(f"\n=== Learned-cost MCF on '{split}'  ({len(learned)} scenes) ===")
    print(f"  input residues / image : {np.mean(res_in):.1f}")
    if do_pathfollow:                                          # Table I, top block
        print(f"  Least squares (Ghiglia): {np.mean(ls_r):.3f}")
        print(f"  Quality-guided         : {np.mean(qg_r):.3f}")
        print(f"  Goldstein branch-cut   : {np.mean(gold_r):.3f}")
    if do_snaphu:
        print(f"  SNAPHU (Chen&Zebker)   : {np.mean(snaphu_r):.3f}   <- the baseline to beat")
    print(f"  MCF unit cost     rmse : {np.mean(unit):.3f}")
    print(f"  MCF LEARNED cost  rmse : {np.mean(learned):.3f}   <- the method "
          f"(residues mean={np.mean(res_learned):.1f})")
    if do_oracle:
        print(f"  MCF oracle cost   rmse : {np.mean(oracle):.3f}")
    print(f"  oracle floor (GT-K)    : {np.mean(floor):.3f}")
    base = np.mean(snaphu_r) if do_snaphu else np.mean(unit)
    label = "SNAPHU" if do_snaphu else "unit-MCF"
    if base > np.mean(learned):
        print(f"  --> learned-cost MCF beats {label} by "
              f"{100 * (1 - np.mean(learned) / base):.0f}%")


if __name__ == "__main__":
    main()
