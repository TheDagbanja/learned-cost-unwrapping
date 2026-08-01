"""Phase 3 -- recompute the table and assert invariants, failing loudly.

Invariants under test:
  I1  floor_rmse <= every method's rmse   (per scene and in the mean)
  I2  residues(output) == 0 for congruent methods (MCF family, branch-cut,
      quality-guided); nonzero only for gradient-domain / LS methods
  I3  hard-oracle rmse <= learned rmse    (as originally proposed)

I2 uses the CORRECT per-edge implied-ambiguity counter (the one adopted in
Lband_ext/eval/metrics.py), not the degenerate always-zero counter in
scripts/eval_learned_mcf.py:40-47 -- the latter telescopes to 0 for any
single-valued field and so cannot test anything.

    python -m audit.phase3_invariants [--checkpoint PATH]
"""
import argparse
import glob
import os

import numpy as np
from scipy.io import loadmat

from src import baselines, mcf

TWO_PI = 2.0 * np.pi
TOL = 0.05
ROOT = "data/InSAR-DPLU/InSAR-DLPU/InSAR-DLPU"
WDIR = os.path.join(ROOT, "test_wrapped_real")
ADIR = os.path.join(ROOT, "test_absolute_real")

CONGRUENT = {"floor", "quality_guided", "goldstein", "unit_mcf", "gt_grad_cost",
             "hard_oracle", "ours_learned"}
GRADIENT_DOMAIN = {"weighted_ls"}


def load(path, key):
    m = loadmat(path)
    ks = [k for k in m if not k.startswith("__")]
    return np.asarray(m[key if key in m else ks[0]], dtype=np.float64)


def aligned_rmse(a, b):
    d = a - b
    d = d - d.mean()
    return float(np.sqrt((d ** 2).mean()))


def implied_edge_k(phi_hat, psi):
    nx = np.round(((phi_hat[:, 1:] - phi_hat[:, :-1])
                   - mcf.wrap(psi[:, 1:] - psi[:, :-1])) / TWO_PI)
    ny = np.round(((phi_hat[1:, :] - phi_hat[:-1, :])
                   - mcf.wrap(psi[1:, :] - psi[:-1, :])) / TWO_PI)
    return nx, ny


def residue_count(phi_hat, psi, R_psi=None):
    """Loops where phi_hat is NOT integrable-consistent with psi (correct counter)."""
    if R_psi is None:
        R_psi = mcf.residues(psi)
    nx, ny = implied_edge_k(phi_hat, psi)
    defect = nx[:-1, :] + ny[:, 1:] - nx[1:, :] - ny[:, :-1] + R_psi
    return int(np.count_nonzero(defect))


def gt_gradient_costs(psi, phi):
    kx, ky = implied_edge_k(phi, psi)
    return np.where(kx != 0, 0.01, 1.0), np.where(ky != 0, 0.01, 1.0)


def hard_oracle_costs(psi, phi, large=1.0e3):
    kx, ky = implied_edge_k(phi, psi)
    return (np.where(kx > 0, 0.0, large), np.where(kx < 0, 0.0, large),
            np.where(ky > 0, 0.0, large), np.where(ky < 0, 0.0, large))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    model = device = None
    if args.checkpoint:
        import torch
        from omegaconf import OmegaConf
        from src import dataset as dm
        from src.models.unwrapper import PhaseUnwrapper
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.checkpoint, map_location=device)
        cfg = OmegaConf.create(ck["config"])
        cfg.model.backbone.in_chans = dm.input_channels(cfg.data)
        model = PhaseUnwrapper(model_cfg=cfg.model, img_size=256).to(device)
        model.load_state_dict(ck.get("model", ck), strict=False)
        model.eval()

    rmse = {}
    resid = {}
    v1, v2, v3 = [], [], []

    for wp in sorted(glob.glob(os.path.join(WDIR, "*.mat")))[:args.n]:
        sid = os.path.basename(wp)
        psi = load(wp, "input")
        phi = load(os.path.join(ADIR, sid), "output")
        if psi.shape != phi.shape:
            continue
        R_psi = mcf.residues(psi)

        preds = {
            "floor": psi + TWO_PI * np.round((phi - psi) / TWO_PI),
            "quality_guided": baselines.quality_guided_unwrap(psi),
            "weighted_ls": baselines.ls_unwrap(psi),
            "goldstein": baselines.goldstein_unwrap(psi),
            "unit_mcf": mcf.unwrap_mcf(psi),
            "gt_grad_cost": mcf.unwrap_mcf(psi, *gt_gradient_costs(psi, phi)),
            "hard_oracle": mcf.unwrap_mcf(psi, edge_costs=hard_oracle_costs(psi, phi)),
        }
        if model is not None:
            import torch
            from src import utils
            with torch.no_grad():
                t = torch.from_numpy(psi).float()
                b = {"input": utils.encode_input(t, include_grad=True).unsqueeze(0),
                     "wrapped": t.unsqueeze(0).unsqueeze(0)}
                lo, _ = model.encode(utils.move_batch_to_device(b, device))
            preds["ours_learned"] = mcf.unwrap_mcf(
                psi, edge_costs=mcf.edge_costs_from_logits(lo.detach().float().cpu().numpy()[0]))

        f = aligned_rmse(preds["floor"], phi)
        for k, p in preds.items():
            r = aligned_rmse(p, phi)
            rmse.setdefault(k, []).append(r)
            rc = residue_count(p, psi, R_psi)
            resid.setdefault(k, []).append(rc)
            if k != "floor" and r < f - TOL:                       # I1
                v1.append((sid, k, round(r, 3), round(f, 3)))
            if k in CONGRUENT and rc != 0:                         # I2
                v2.append((sid, k, rc))
            if k in GRADIENT_DOMAIN and rc == 0:
                v2.append((sid, k, "expected nonzero, got 0"))
        if "ours_learned" in preds:                                # I3
            if rmse["hard_oracle"][-1] > rmse["ours_learned"][-1] + TOL:
                v3.append((sid, round(rmse["hard_oracle"][-1], 3),
                           round(rmse["ours_learned"][-1], 3)))

    n = len(rmse["floor"])
    print(f"\n=== Phase 3: invariants, test_real ({n} scenes, tol={TOL}) ===\n")
    print(f"{'method':16s}{'RMSE':>9}{'residues':>11}")
    for k in rmse:
        print(f"{k:16s}{np.mean(rmse[k]):>9.3f}{np.mean(resid[k]):>11.1f}")

    fm = np.mean(rmse["floor"])
    print(f"\nI1  floor <= every method")
    bad_mean = [k for k in rmse if k != "floor" and np.mean(rmse[k]) < fm - TOL]
    print(f"    mean:      {'PASS' if not bad_mean else 'FAIL ' + str(bad_mean)}")
    print(f"    per-scene: {'PASS' if not v1 else f'FAIL ({len(v1)}) {v1[:5]}'}")

    print(f"\nI2  residues==0 for congruent, >0 for gradient-domain")
    print(f"    {'PASS' if not v2 else f'FAIL ({len(v2)}) {v2[:5]}'}")

    print(f"\nI3  hard-oracle <= learned")
    if model is None:
        ho, qg = np.mean(rmse["hard_oracle"]), np.mean(rmse["quality_guided"])
        print(f"    learned row needs --checkpoint; proxy: hard_oracle {ho:.3f} vs "
              f"quality_guided {qg:.3f} -> {'PASS' if ho <= qg + TOL else 'FAILS PREMISE'}")
    else:
        print(f"    mean:      hard_oracle {np.mean(rmse['hard_oracle']):.3f} vs "
              f"learned {np.mean(rmse['ours_learned']):.3f}")
        print(f"    per-scene: {'PASS' if not v3 else f'FAILS PREMISE ({len(v3)}) {v3[:5]}'}")
    print("    NOTE: Phase 1 established this is NOT a valid invariant -- k*-matching\n"
          "          is not the RMSE-optimal residue-free unwrapping under noise.")


if __name__ == "__main__":
    main()
