"""Weighted least-squares (WLS) phase-unwrapping baseline.

Solves ``min_phi sum_e w_e (D_e phi - wrap(D_e psi))^2`` with a matrix-free
conjugate-gradient solve with an unweighted Poisson/DCT preconditioner.
Coherence-derived edge weights are used when
available; otherwise the method falls back to unit weights (ordinary LS) and
reports that choice through the caller.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


TWO_PI = 2.0 * np.pi


def wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % TWO_PI - np.pi


def coherence_edge_weights(coherence: np.ndarray, floor: float = 1e-3,
                           power: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    gamma = np.clip(np.asarray(coherence, dtype=np.float64), 0.0, 1.0)
    # Floor the final edge weight, not gamma before exponentiation.  The latter
    # accidentally produced 1e-6 weights with floor=1e-3 and made the system
    # needlessly ill-conditioned in decorrelated regions.
    wx = np.maximum(floor, (0.5 * (gamma[:, :-1] + gamma[:, 1:])) ** power)
    wy = np.maximum(floor, (0.5 * (gamma[:-1, :] + gamma[1:, :])) ** power)
    return wx, wy


def integrate_gradients(gx: np.ndarray, gy: np.ndarray,
                        wx: Optional[np.ndarray] = None, wy: Optional[np.ndarray] = None,
                        rtol: float = 1e-4, maxiter: int = 300,
                        weight_floor: float = 1e-3) -> np.ndarray:
    """Integrate horizontal/vertical edge gradients by weighted least squares."""
    try:
        from scipy.fft import dctn, idctn
        from scipy.sparse.linalg import LinearOperator, cg
    except ImportError as exc:
        raise ImportError("WLS requires scipy (pip install scipy>=1.9).") from exc
    gx, gy = np.asarray(gx, np.float64), np.asarray(gy, np.float64)
    h, wm1 = gx.shape; hm1, w = gy.shape
    if (hm1 + 1, wm1 + 1) != (h, w):
        raise ValueError(f"incompatible gradient shapes gx={gx.shape}, gy={gy.shape}")
    unweighted = wx is None and wy is None
    wx = np.ones_like(gx) if wx is None else np.maximum(np.asarray(wx, np.float64), weight_floor)
    wy = np.ones_like(gy) if wy is None else np.maximum(np.asarray(wy, np.float64), weight_floor)

    def apply(v):
        x = v.reshape(h, w); y = np.zeros_like(x)
        dx = x[:, 1:] - x[:, :-1]; dy = x[1:, :] - x[:-1, :]
        y[:, :-1] -= wx * dx; y[:, 1:] += wx * dx
        y[:-1, :] -= wy * dy; y[1:, :] += wy * dy
        # Resolve the constant nullspace without privileging one pixel.  This
        # adds eigenvalue one only to the spatially constant mode.
        y += x.mean()
        return y.ravel()

    b = np.zeros((h, w), np.float64)
    b[:, :-1] -= wx * gx; b[:, 1:] += wx * gx
    b[:-1, :] -= wy * gy; b[1:, :] += wy * gy
    op = LinearOperator((h*w, h*w), matvec=apply, dtype=np.float64)

    # DCT-II diagonalizes the unit-weight Neumann Laplacian.  Using its exact
    # inverse as M^{-1} is a much stronger preconditioner than Jacobi for WLS.
    ey = 2.0 - 2.0 * np.cos(np.pi * np.arange(h) / h)
    ex = 2.0 - 2.0 * np.cos(np.pi * np.arange(w) / w)
    eigenvalues = ey[:, None] + ex[None, :]
    eigenvalues[0, 0] = 1.0

    def poisson_inverse(v):
        coeff = dctn(v.reshape(h, w), type=2, norm="ortho")
        return idctn(coeff / eigenvalues, type=2, norm="ortho").ravel()

    # Ordinary LS is solved directly; no iterative work is needed.
    if unweighted:
        result = poisson_inverse(b.ravel()).reshape(h, w)
        return result - result.mean()

    preconditioner = LinearOperator((h*w, h*w), matvec=poisson_inverse, dtype=np.float64)
    try:
        solution, info = cg(op, b.ravel(), rtol=rtol, atol=0.0, maxiter=maxiter, M=preconditioner)
    except TypeError:  # scipy < 1.12
        solution, info = cg(op, b.ravel(), tol=rtol, maxiter=maxiter, M=preconditioner)
    if info < 0:
        raise RuntimeError(f"WLS conjugate-gradient solver failed (info={info}).")
    if info > 0:
        import warnings
        warnings.warn(f"WLS reached maxiter={maxiter}; using the current PCG iterate.", RuntimeWarning)
    result = solution.reshape(h, w)
    return result - result.mean()


def unwrap_wls(psi: np.ndarray, coherence: Optional[np.ndarray] = None,
               weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
               rtol: float = 1e-4, maxiter: int = 300) -> np.ndarray:
    psi = np.asarray(psi, np.float64)
    gx, gy = wrap(psi[:, 1:] - psi[:, :-1]), wrap(psi[1:, :] - psi[:-1, :])
    if weights is not None:
        wx, wy = weights
    elif coherence is not None:
        wx, wy = coherence_edge_weights(coherence)
    else:
        wx = wy = None
    return integrate_gradients(gx, gy, wx, wy, rtol=rtol, maxiter=maxiter)


def batch_coherence(batch, index: int):
    """Extract a 2-D coherence map from common dataset field names."""
    for key in ("coherence", "coh", "gamma"):
        if key in batch:
            value = batch[key][index].detach().cpu().numpy()
            return value[0] if value.ndim == 3 else value
    return None


def batch_edge_weights(batch, index: int, shape):
    """Extract repository-provided WLS weights and crop padded boundaries."""
    if "weights_x" not in batch or "weights_y" not in batch:
        return None
    h, w = shape
    wx = batch["weights_x"][index].detach().cpu().numpy()
    wy = batch["weights_y"][index].detach().cpu().numpy()
    if wx.ndim == 3: wx = wx[0]
    if wy.ndim == 3: wy = wy[0]
    return wx[:h, :w-1], wy[:h-1, :w]
