"""Does pre-filtering reduce the forced cut budget on the noise-dominated subset?

Residue-free solvers must neutralise every residue, and on `test_real` most
residues are noise-induced: the ground truth cuts 0.45 % of edges while even a
cost-free unit-cost MCF cuts 0.96 %. Cost-side interventions cannot change that
budget (gating, entropy-TTA and temperature scaling all failed). Goldstein
adaptive filtering attacks the cause instead, by suppressing the noise that
creates the spurious residues.

Applies the same filter to every method, so any gain is from unwrapping and not
from preprocessing one method's input only. Model-free: unit-cost MCF and
quality-guided need no checkpoint, so this establishes whether the idea works at
all before spending GPU time on the learned cost.

    python -m audit.test_filtering [--n 30] [--alpha 0.5]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.io import loadmat

from src import baselines, mcf

TWO_PI = 2.0 * np.pi
ROOT = "data/InSAR-DPLU/InSAR-DLPU/InSAR-DLPU"


def load(path, key):
    m = loadmat(path)
    ks = [k for k in m if not k.startswith("__")]
    return np.asarray(m[key if key in m else ks[0]], dtype=np.float64)


def goldstein_filter(psi, alpha=0.5, patch=32, step=16):
    """Goldstein adaptive spectral filter on the unit-modulus interferogram.

    Overlap-add: each patch is transformed, its spectrum weighted by
    ``|S|**alpha``, inverted, and accumulated under a Hann window. ``alpha=0``
    is a no-op, larger ``alpha`` filters harder.
    """
    z = np.exp(1j * psi)
    H, W = psi.shape
    acc = np.zeros((H, W), np.complex128)
    wsum = np.zeros((H, W))
    w1 = np.hanning(patch + 2)[1:-1]
    win = np.outer(w1, w1) + 1e-6

    def starts(n):
        s = list(range(0, max(n - patch, 0) + 1, step))
        if s and s[-1] != n - patch:
            s.append(max(n - patch, 0))
        return s or [0]

    for r in starts(H):
        for c in starts(W):
            sl = (slice(r, r + patch), slice(c, c + patch))
            blk = z[sl]
            if blk.shape != (patch, patch):
                continue
            S = np.fft.fft2(blk)
            S *= np.abs(S) ** alpha
            acc[sl] += np.fft.ifft2(S) * win
            wsum[sl] += win
    out = acc / np.maximum(wsum, 1e-9)
    return np.angle(out)


def aligned_rmse(a, b):
    d = a - b
    d = d - d.mean()
    return float(np.sqrt((d ** 2).mean()))


def cut_fraction(phi, psi):
    kx = np.round(((phi[:, 1:] - phi[:, :-1]) - mcf.wrap(psi[:, 1:] - psi[:, :-1])) / TWO_PI)
    ky = np.round(((phi[1:, :] - phi[:-1, :]) - mcf.wrap(psi[1:, :] - psi[:-1, :])) / TWO_PI)
    return (int((kx != 0).sum()) + int((ky != 0).sum())) / (kx.size + ky.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--alphas", default="0.0,0.3,0.5,0.8,1.0")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    files = sorted(glob.glob(os.path.join(ROOT, "test_wrapped_real", "*.mat")))[:args.n]
    rows = {a: dict(res=[], unit=[], qg=[], cut=[], floor=[]) for a in alphas}
    true_cut = []

    for wp in files:
        base = os.path.basename(wp)
        psi0 = load(wp, "input")
        phi = load(os.path.join(ROOT, "test_absolute_real", base), "output")
        if psi0.shape != phi.shape:
            continue
        true_cut.append(cut_fraction(phi, psi0))
        for a in alphas:
            psi = psi0 if a == 0.0 else goldstein_filter(psi0, alpha=a)
            r = rows[a]
            r["res"].append(int(np.abs(mcf.residues(psi)).sum()))
            u = mcf.unwrap_mcf(psi)
            r["unit"].append(aligned_rmse(u, phi))
            r["cut"].append(cut_fraction(u, psi))
            r["qg"].append(aligned_rmse(baselines.quality_guided_unwrap(psi), phi))
            r["floor"].append(mcf.aligned_gtk_floor(psi, phi))

    n = len(true_cut)
    print(f"\n=== Goldstein pre-filtering, test_real ({n} scenes) ===")
    print(f"ground-truth cut fraction: {100*np.mean(true_cut):.2f}%   "
          f"(targets: QG 2.42, DENet 2.98, ours 3.02)")
    print(f"\n{'alpha':>6}{'residues':>10}{'unit-MCF':>10}{'cut%':>8}"
          f"{'quality-gd':>12}{'floor':>8}")
    for a in alphas:
        r = rows[a]
        print(f"{a:>6.1f}{np.mean(r['res']):>10.0f}{np.mean(r['unit']):>10.3f}"
              f"{100*np.mean(r['cut']):>7.2f}%{np.mean(r['qg']):>12.3f}"
              f"{np.mean(r['floor']):>8.3f}")
    print("\nThe filter changes the input, so the floor moves too: a lower RMSE is\n"
          "only meaningful if it falls faster than the floor it is measured against.")


if __name__ == "__main__":
    main()
