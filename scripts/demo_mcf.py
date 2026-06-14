"""Stage-1 proof: minimum-cost-flow (MCF) unwrapping vs the least-squares (LS)
integrator, on real ground-truth data.

Shows the thesis of the pivot: the rigorous integer solver (SNAPHU's core) is
residue-free by construction and beats the LS relaxation that smears 2pi errors.
The per-edge COSTS are the hook where learning plugs in next (Stage 2).

Run on the remote box:
    python -m scripts.demo_mcf --source synthetic --n 30
    python -m scripts.demo_mcf --source dlpu --split test --n 50
    python -m scripts.demo_mcf --source dlpu --split test_real --n 50
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src import dataset as dataset_module
from src import mcf
from src import baselines
from src.integrator import poisson_dct_solve


def aligned_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    d = d - d.mean()
    return float(np.sqrt((d ** 2).mean()))


def ls_unwrap(psi: np.ndarray) -> np.ndarray:
    """Classic least-squares (Ghiglia) unwrap: integrate the wrapped gradients."""
    p = torch.from_numpy(psi).double().unsqueeze(0)
    gx = mcf.wrap((p[..., :, 1:] - p[..., :, :-1]).numpy())
    gy = mcf.wrap((p[..., 1:, :] - p[..., :-1, :]).numpy())
    phi = poisson_dct_solve(torch.from_numpy(gx), torch.from_numpy(gy))
    return phi.squeeze(0).numpy()


def oracle_floor(psi: np.ndarray, phi: np.ndarray) -> float:
    """RMSE of the GT-integer reconstruction = irreducible noise floor."""
    k = np.round((phi - psi) / mcf.TWO_PI)
    return aligned_rmse(psi + mcf.TWO_PI * k, phi)


def oracle_costs(psi: np.ndarray, phi: np.ndarray):
    """Per-edge costs from GROUND-TRUTH cuts: cheap on true-cut edges, dear else.

    Upper bound on what any cost-learning method can achieve with this solver:
    if oracle-cost MCF approaches the floor, learned costs have a high ceiling.
    """
    dx_w = mcf.wrap(psi[:, 1:] - psi[:, :-1])
    dy_w = mcf.wrap(psi[1:, :] - psi[:-1, :])
    kx_true = np.round(((phi[:, 1:] - phi[:, :-1]) - dx_w) / mcf.TWO_PI)
    ky_true = np.round(((phi[1:, :] - phi[:-1, :]) - dy_w) / mcf.TWO_PI)
    wx = np.where(kx_true != 0, 0.01, 1.0)
    wy = np.where(ky_true != 0, 0.01, 1.0)
    return wx, wy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "dlpu"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--snaphu", action="store_true", help="add the real SNAPHU baseline")
    args = ap.parse_args()
    if args.snaphu and not baselines.snaphu_available():
        raise SystemExit("--snaphu given but 'snaphu' is not installed (pip install snaphu).")

    cfg = {"source": args.source, "root": args.root, "input_grad": False,
           "return_labels": True, "val_size": max(args.n, 50),
           "synthetic": {"img_size": 128, "looks": 8, "coherence_mean": [0.4, 0.95],
                         "difficulty": 0.6}}
    ds = dataset_module.build_dataset(cfg, args.split)
    n = min(args.n, len(ds))

    rows = {"LS": [], "QG": [], "Goldstein": [], "SNAPHU": [], "MCF": [], "MCF+statcost": [], "MCF+oracle": []}
    res_in, floor = [], []
    for i in range(n):
        s = ds[i]
        if "absolute" not in s:
            continue
        psi = s["wrapped"][0].numpy().astype(np.float64)
        phi = s["absolute"][0].numpy().astype(np.float64)

        res_in.append(int(np.abs(mcf.residues(psi)).sum()))
        floor.append(oracle_floor(psi, phi))
        rows["LS"].append(aligned_rmse(ls_unwrap(psi), phi))
        rows["QG"].append(aligned_rmse(baselines.quality_guided_unwrap(psi), phi))
        rows["Goldstein"].append(aligned_rmse(baselines.goldstein_unwrap(psi), phi))
        if args.snaphu:
            rows["SNAPHU"].append(aligned_rmse(baselines.snaphu_unwrap(psi), phi))
        rows["MCF"].append(aligned_rmse(mcf.unwrap_mcf(psi), phi))
        # statistical cost from a phase-derivative-variance quality map (no GT,
        # no coherence band needed) -- the hand-designed baseline for learned costs
        q = mcf.quality_map(psi)
        rows["MCF+statcost"].append(aligned_rmse(mcf.unwrap_mcf(psi, coherence=q), phi))
        # oracle cost ceiling: best achievable with this solver given perfect costs
        wx, wy = oracle_costs(psi, phi)
        rows["MCF+oracle"].append(aligned_rmse(mcf.unwrap_mcf(psi, wx=wx, wy=wy), phi))

    print(f"\n=== MCF vs LS on '{args.source}/{args.split}'  ({n} samples) ===")
    print(f"  input residues / image : {np.mean(res_in):.1f}")
    print(f"  oracle floor (GT-K)    : {np.nanmean(floor):.3f}")
    print(f"  LS integrator      rmse: {np.nanmean(rows['LS']):.3f}")
    print(f"  quality-guided     rmse: {np.nanmean(rows['QG']):.3f}")
    print(f"  Goldstein branchcut rmse: {np.nanmean(rows['Goldstein']):.3f}")
    if args.snaphu:
        print(f"  SNAPHU (Chen&Zebker) rmse: {np.nanmean(rows['SNAPHU']):.3f}")
    print(f"  MCF (unit cost)    rmse: {np.nanmean(rows['MCF']):.3f}   residues_pred=0 (by construction)")
    print(f"  MCF (stat cost)    rmse: {np.nanmean(rows['MCF+statcost']):.3f}   <- hand-designed cost")
    print(f"  MCF (ORACLE cost)  rmse: {np.nanmean(rows['MCF+oracle']):.3f}   <- CEILING for learned costs")
    imp = 100 * (1 - np.nanmean(rows["MCF"]) / max(np.nanmean(rows["LS"]), 1e-9))
    print(f"  --> MCF beats LS by {imp:.0f}%.  Learned-cost target: drive {np.nanmean(rows['MCF']):.2f} "
          f"toward oracle {np.nanmean(rows['MCF+oracle']):.2f} (floor {np.nanmean(floor):.2f}).")


if __name__ == "__main__":
    main()
