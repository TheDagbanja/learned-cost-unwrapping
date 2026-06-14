r"""Differentiable weighted least-squares (Poisson) integration (proposal Eq. 4).

Given corrected per-edge gradients ``g_e`` and coherence-derived edge weights
``w_e``, recover the absolute phase as

    phi_hat = argmin_phi  sum_e  w_e * ( (Delta phi)_e - g_e )^2 .

The minimiser solves the weighted-Poisson normal equations ``A_w phi = b`` with

    A_w phi = Dx^T (w_x . Dx phi) + Dy^T (w_y . Dy phi)
    b       = Dx^T (w_x . g_x)    + Dy^T (w_y . g_y)

where ``Dx, Dy`` are forward-difference operators and ``Dx^T, Dy^T`` their
adjoints (negative divergence). ``A_w`` is symmetric positive semidefinite with
a one-dimensional null space (the additive reference constant), which we fix by
zero-meaning the solution.

Two solvers are provided, both fully differentiable:

* **DCT** -- exact, closed-form solution of the *unweighted* problem via the
  cosine transform that diagonalises the Neumann Laplacian. Used when the edge
  weights are uniform (e.g. InSAR-DLPU, which has no coherence band).
* **PCG** -- Jacobi-preconditioned conjugate gradients for the general
  *weighted* problem (real Sentinel-1, Stream C). Differentiated by implicit
  differentiation of the linear system (the adjoint solve), which is exact and
  memory-cheap — no unrolling of the iterations.

All inputs are full-resolution, padded edge maps of shape ``(..., H, W)`` (the
invalid last column / last row are ignored), matching
:func:`src.utils.wrapped_gradients` and the decoder output convention. Run this
module directly (``python -m src.integrator``) for a self-test that verifies
exact recovery on a noise-free, Itoh-satisfying field (proposal Week-1 task).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REDUCE_DIMS = (-2, -1)


# ---------------------------------------------------------------------------
# Difference operators and their adjoints (mirrors the validated NumPy ops)
# ---------------------------------------------------------------------------
def diff_x(p: torch.Tensor) -> torch.Tensor:
    """Forward difference along x: ``(..., H, W) -> (..., H, W-1)``."""
    return p[..., :, 1:] - p[..., :, :-1]


def diff_y(p: torch.Tensor) -> torch.Tensor:
    """Forward difference along y: ``(..., H, W) -> (..., H-1, W)``."""
    return p[..., 1:, :] - p[..., :-1, :]


def diff_xT(e: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`diff_x`: ``(..., H, W-1) -> (..., H, W)``."""
    z = e[..., :1] * 0
    ep = torch.cat((z, e, z), dim=-1)
    return ep[..., :-1] - ep[..., 1:]


def diff_yT(e: torch.Tensor) -> torch.Tensor:
    """Adjoint of :func:`diff_y`: ``(..., H-1, W) -> (..., H, W)``."""
    z = e[..., :1, :] * 0
    ep = torch.cat((z, e, z), dim=-2)
    return ep[..., :-1, :] - ep[..., 1:, :]


def _zero_mean(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(dim=_REDUCE_DIMS, keepdim=True)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample inner product, keeping batch dims for broadcasting."""
    return (a * b).sum(dim=_REDUCE_DIMS, keepdim=True)


# ---------------------------------------------------------------------------
# Weighted Laplacian, diagonal preconditioner
# ---------------------------------------------------------------------------
def apply_weighted_laplacian(
    phi: torch.Tensor, w_x: torch.Tensor, w_y: torch.Tensor
) -> torch.Tensor:
    """Apply ``A_w phi = Dx^T(w_x . Dx phi) + Dy^T(w_y . Dy phi)``.

    ``w_x`` has shape ``(..., H, W-1)`` and ``w_y`` shape ``(..., H-1, W)``
    (the *valid* edge weights, already de-padded).
    """
    return diff_xT(w_x * diff_x(phi)) + diff_yT(w_y * diff_y(phi))


def _weighted_diagonal(w_x: torch.Tensor, w_y: torch.Tensor) -> torch.Tensor:
    """Diagonal of ``A_w`` (for Jacobi preconditioning), shape ``(..., H, W)``."""
    d_x = F.pad(w_x, (1, 0)) + F.pad(w_x, (0, 1))      # left + right pixels
    d_y = F.pad(w_y, (0, 0, 1, 0)) + F.pad(w_y, (0, 0, 0, 1))  # top + bottom
    return d_x + d_y


# ---------------------------------------------------------------------------
# Preconditioned conjugate gradients (batched, run without autograd)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _pcg(
    b: torch.Tensor,
    w_x: torch.Tensor,
    w_y: torch.Tensor,
    iters: int,
    tol: float,
) -> torch.Tensor:
    """Solve ``A_w x = b`` (mean-zero b) by **DCT-preconditioned** CG.

    The preconditioner ``M^{-1}`` is the exact unweighted-Poisson solve
    (:func:`dct_poisson_from_rhs`), which captures the bulk of the weighted
    operator and yields convergence in ~tens of iterations (a diagonal/Jacobi
    preconditioner needs thousands here and was the cause of garbage RMSE on
    coherence-weighted data). Returns a mean-zero solution.
    """
    x = torch.zeros_like(b)
    r = _zero_mean(b)
    z = dct_poisson_from_rhs(r)
    p = z.clone()
    rz = _dot(r, z)
    eps = torch.finfo(b.dtype).eps
    for _ in range(iters):
        ap = apply_weighted_laplacian(p, w_x, w_y)
        alpha = rz / _dot(p, ap).clamp_min(eps)
        x = x + alpha * p
        r = r - alpha * ap
        if torch.sqrt(_dot(r, r)).max() < tol:
            break
        z = dct_poisson_from_rhs(r)
        rz_new = _dot(r, z)
        beta = rz_new / rz.clamp_min(eps)
        p = z + beta * p
        rz = rz_new
    return _zero_mean(x)


class _WeightedLSSolve(torch.autograd.Function):
    """Autograd wrapper around the PCG solve with implicit differentiation.

    Forward solves ``A_w phi = b``. Because ``A_w`` is symmetric, the gradient of
    a downstream loss w.r.t. ``b`` is ``A_w^+ grad_phi`` (projected to mean-zero),
    obtained by a second PCG solve. Weights are treated as constants (coherence),
    so no gradient flows to them.
    """

    @staticmethod
    def forward(ctx, b, w_x, w_y, iters, tol):  # type: ignore[override]
        phi = _pcg(b, w_x, w_y, iters, tol)
        ctx.save_for_backward(w_x, w_y)
        ctx.iters = iters
        ctx.tol = tol
        return phi

    @staticmethod
    def backward(ctx, grad_phi):  # type: ignore[override]
        w_x, w_y = ctx.saved_tensors
        grad_b = _pcg(_zero_mean(grad_phi), w_x, w_y, ctx.iters, ctx.tol)
        return grad_b, None, None, None, None


# ---------------------------------------------------------------------------
# Exact DCT solver for the unweighted problem
# ---------------------------------------------------------------------------
_DCT_CACHE: Dict[Tuple, torch.Tensor] = {}
_LAM_CACHE: Dict[Tuple, torch.Tensor] = {}


def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``C`` with ``dct(x) = C x`` (cached)."""
    key = (n, device, dtype)
    C = _DCT_CACHE.get(key)
    if C is None:
        k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        m = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
        C = torch.cos(torch.pi * (2 * m + 1) * k / (2 * n)) * (2.0 / n) ** 0.5
        C[0, :] *= 0.5 ** 0.5
        _DCT_CACHE[key] = C
    return C


def _laplacian_eigenvalues(
    h: int, w: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Neumann-Laplacian eigenvalues for the DCT solve (cached), shape ``(H, W)``."""
    key = (h, w, device, dtype)
    lam = _LAM_CACHE.get(key)
    if lam is None:
        i = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)
        j = torch.arange(w, device=device, dtype=dtype).unsqueeze(0)
        lam = (2 - 2 * torch.cos(torch.pi * i / h)) + (2 - 2 * torch.cos(torch.pi * j / w))
        _LAM_CACHE[key] = lam
    return lam


def _dct2(x: torch.Tensor, ch: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(ch, x), cw.transpose(-1, -2))


def _idct2(x: torch.Tensor, ch: torch.Tensor, cw: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(ch.transpose(-1, -2), x), cw)


def dct_poisson_from_rhs(b: torch.Tensor) -> torch.Tensor:
    """Solve the **unweighted** Neumann-Poisson system ``L z = b`` via the DCT.

    ``b`` is a pixel-space right-hand side ``(..., H, W)``; returns the mean-zero
    solution. This is exact for the unweighted problem and is reused as the CG
    preconditioner for the weighted problem (Ghiglia-Romero), where it converges
    in ~tens of iterations instead of the ~thousands a diagonal preconditioner
    needs.
    """
    h, w = b.shape[-2], b.shape[-1]
    ch = _dct_matrix(h, b.device, b.dtype)
    cw = _dct_matrix(w, b.device, b.dtype)
    lam = _laplacian_eigenvalues(h, w, b.device, b.dtype).clone()
    lam[0, 0] = 1.0                                   # avoid 0/0 at the DC term
    bhat = _dct2(b, ch, cw)
    phihat = bhat / lam
    phihat[..., 0, 0] = 0.0                           # mean (reference) undetermined
    return _idct2(phihat, ch, cw)


def poisson_dct_solve(g_x: torch.Tensor, g_y: torch.Tensor) -> torch.Tensor:
    """Exact unweighted least-squares phase via the DCT (differentiable).

    ``g_x`` shape ``(..., H, W-1)``, ``g_y`` shape ``(..., H-1, W)`` (valid edges).
    Returns a mean-zero field ``(..., H, W)``.
    """
    return dct_poisson_from_rhs(diff_xT(g_x) + diff_yT(g_y))


# ---------------------------------------------------------------------------
# Public module
# ---------------------------------------------------------------------------
def _is_uniform(w: torch.Tensor, eps: float = 1e-6) -> bool:
    return bool((w.amax() - w.amin()).abs().item() < eps and w.amin().item() > 0)


class WeightedLeastSquaresIntegrator(nn.Module):
    """Differentiable weighted least-squares phase integrator (Eq. 4).

    Parameters
    ----------
    solver:
        ``"auto"`` (default) uses the exact DCT solve when the edge weights are
        uniform and PCG otherwise; ``"dct"`` and ``"cg"`` force a solver.
    cg_iters, cg_tol:
        Conjugate-gradient budget for the weighted solver.

    The forward call takes full-resolution padded edge maps ``(..., H, W)`` and
    returns a mean-zero unwrapped phase with the same leading dims ``(..., H, W)``.
    All operators reduce over the last two dims and broadcast over any leading
    (batch/channel) dims. The integration always runs in float32 with autocast
    disabled for numerical stability under mixed precision.
    """

    def __init__(self, solver: str = "auto", cg_iters: int = 100, cg_tol: float = 1e-6) -> None:
        super().__init__()
        if solver not in ("auto", "dct", "cg"):
            raise ValueError(f"solver must be auto|dct|cg, got {solver!r}.")
        self.solver = solver
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol

    def forward(
        self,
        g_x: torch.Tensor,
        g_y: torch.Tensor,
        w_x: torch.Tensor | None = None,
        w_y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = g_x.dtype
        with torch.autocast(device_type=g_x.device.type, enabled=False):
            gx = g_x.float()
            gy = g_y.float()
            # De-pad: drop the invalid last column (x) / last row (y).
            gxv = gx[..., :, :-1]
            gyv = gy[..., :-1, :]
            if w_x is None or w_y is None:
                wxv = torch.ones_like(gxv)
                wyv = torch.ones_like(gyv)
            else:
                wxv = w_x.float()[..., :, :-1]
                wyv = w_y.float()[..., :-1, :]

            use_dct = self.solver == "dct" or (
                self.solver == "auto" and _is_uniform(wxv) and _is_uniform(wyv)
            )
            if use_dct:
                phi = poisson_dct_solve(gxv, gyv)
            else:
                b = diff_xT(wxv * gxv) + diff_yT(wyv * gyv)
                phi = _WeightedLSSolve.apply(b, wxv, wyv, self.cg_iters, self.cg_tol)
            phi = _zero_mean(phi)

        return phi.to(out_dtype)


# ---------------------------------------------------------------------------
# Self-test (proposal Week-1: exact recovery on noise-free Itoh patches)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    B, H, W = 2, 64, 72
    yy, xx = torch.meshgrid(
        torch.linspace(0, 3, H), torch.linspace(0, 3, W), indexing="ij"
    )
    # Smooth field with |gradient| < pi everywhere (Itoh-satisfying), batched.
    phi = (0.4 * xx + 0.3 * yy + 0.2 * torch.sin(xx) * torch.cos(yy)).expand(B, H, W).clone()
    phi = phi - phi.mean(dim=(-2, -1), keepdim=True)

    # Corrected gradients g_e equal the true gradients (k_e = 0), padded full-res.
    gx = F.pad(diff_x(phi), (0, 1))                   # (B, H, W)
    gy = F.pad(diff_y(phi), (0, 0, 0, 1))             # (B, H, W)

    # DCT is exact; DCT-preconditioned CG converges to near-exact on the unweighted
    # (uniform-weight) test in a handful of iterations.
    tolerances = {"dct": 1e-4, "cg": 1e-3}
    for solver in ("dct", "cg"):
        integ = WeightedLeastSquaresIntegrator(solver=solver, cg_iters=100, cg_tol=1e-8)
        rec = integ(gx, gy)
        err = (rec - phi).abs().max().item()
        print(f"[self-test] solver={solver:3s}  max|rec - phi| = {err:.3e}")
        assert err < tolerances[solver], f"{solver} recovery error too large: {err}"

    # Gradient flows to g (differentiability check).
    g = gx.clone().requires_grad_(True)
    out = WeightedLeastSquaresIntegrator(solver="cg", cg_iters=200)(g, gy)
    out.pow(2).sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all(), "bad gradient"
    print("[self-test] gradient through PCG solver: OK")
    print("[self-test] PASSED")


if __name__ == "__main__":
    _self_test()
