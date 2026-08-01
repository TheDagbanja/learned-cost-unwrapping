"""Phase 1 -- oracle killer test (ADDITIVE; mutates nothing in the pipeline).

Compares, on a fixed subset of InSAR-DLPU test_real scenes:
  * SOFT oracle  = the harness's current _oracle_costs: symmetric wx/wy in {0.01,1.0}
                   marking |k*|!=0 (location only, no sign) -> unwrap_mcf(wx,wy).
  * HARD oracle  = directional edge_costs: cost 0 for the TRUE class per edge,
                   a large constant otherwise -> unwrap_mcf(edge_costs=...), the
                   SAME solver path the learned method uses.
  * FLOOR        = integrate the per-pixel GT-K field directly.

Reports (A) frac of edges where solver k != k*, (B) integrated RMSE, (C) floor
RMSE -- all with the EXACT alignment/RMSE from scripts/eval_learned_mcf.py.
"""
import glob
import os

import numpy as np
from scipy.io import loadmat

from src import mcf

TWO_PI = 2.0 * np.pi
LARGE = 1.0e3
N = 40


def wrap(x):
    return (x + np.pi) % TWO_PI - np.pi


def aligned_rmse(a, b):                      # exact copy of eval_learned_mcf._aligned_rmse
    d = a - b
    d = d - d.mean()
    return float(np.sqrt((d ** 2).mean()))


def edge_k(psi, phi):                        # implied per-edge integer field
    kx = np.round(((phi[:, 1:] - phi[:, :-1]) - wrap(psi[:, 1:] - psi[:, :-1])) / TWO_PI)
    ky = np.round(((phi[1:, :] - phi[:-1, :]) - wrap(psi[1:, :] - psi[:-1, :])) / TWO_PI)
    return kx, ky


ROOT = "data/InSAR-DPLU/InSAR-DLPU/InSAR-DLPU"
WDIR = os.path.join(ROOT, "test_wrapped_real")
ADIR = os.path.join(ROOT, "test_absolute_real")


def load(path, key):
    m = loadmat(path)
    ks = [k for k in m if not k.startswith("__")]
    return np.asarray(m[key if key in m else ks[0]], dtype=np.float64)


def main():
    wfiles = sorted(glob.glob(os.path.join(WDIR, "*.mat")))[:N]
    A_soft, A_hard, B_soft, B_hard, C_floor = [], [], [], [], []
    viol = []
    for wp in wfiles:
        base = os.path.basename(wp)
        psi = load(wp, "input")
        phi = load(os.path.join(ADIR, base), "output")
        if psi.shape != phi.shape:
            continue
        kxs, kys = edge_k(psi, phi)                     # k* (true per-edge integers)

        k = np.round((phi - psi) / TWO_PI)
        C = aligned_rmse(psi + TWO_PI * k, phi)
        C_floor.append(C)

        # SOFT oracle (current harness): symmetric, location-only, 0.01 vs 1.0
        wx = np.where(kxs != 0, 0.01, 1.0)
        wy = np.where(kys != 0, 0.01, 1.0)
        phi_s = mcf.unwrap_mcf(psi, wx=wx, wy=wy)
        kxo, kyo = edge_k(psi, phi_s)
        A_soft.append(0.5 * (np.mean(kxo != kxs) + np.mean(kyo != kys)))
        B_soft.append(aligned_rmse(phi_s, phi))

        # HARD oracle: directional, cost 0 for true class, LARGE otherwise
        cpx = np.where(kxs > 0, 0.0, LARGE); cnx = np.where(kxs < 0, 0.0, LARGE)
        cpy = np.where(kys > 0, 0.0, LARGE); cny = np.where(kys < 0, 0.0, LARGE)
        phi_h = mcf.unwrap_mcf(psi, edge_costs=(cpx, cnx, cpy, cny))
        kxh, kyh = edge_k(psi, phi_h)
        A_hard.append(0.5 * (np.mean(kxh != kxs) + np.mean(kyh != kys)))
        Bh = aligned_rmse(phi_h, phi)
        B_hard.append(Bh)
        if Bh > C + 0.05:
            viol.append((base, round(Bh, 3), round(C, 3)))

    print(f"scenes: {len(C_floor)}   (fixed: first {N} sorted test_real)")
    print(f"(A) edge k != k*   SOFT oracle: {100*np.mean(A_soft):6.2f}%   "
          f"HARD oracle: {100*np.mean(A_hard):.4f}%")
    print(f"(B) integrated RMSE  SOFT: {np.mean(B_soft):.3f}   HARD: {np.mean(B_hard):.3f}")
    print(f"(C) floor RMSE (GT-K)     : {np.mean(C_floor):.3f}")
    print(f"|B_hard - C_floor|        : {abs(np.mean(B_hard) - np.mean(C_floor)):.4f}")
    print(f"scenes hard-oracle > floor+0.05: {len(viol)}  {viol[:5]}")


if __name__ == "__main__":
    main()
