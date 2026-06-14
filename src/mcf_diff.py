"""End-to-end differentiable learned-cost MCF unwrapping (Stage-2 v2).

Wraps the (non-differentiable) min-cost-flow solver in a torch autograd Function
using the blackbox-solver gradient of Vlastelica & Pogancic (2020): forward solves
the flow; backward perturbs the per-edge costs by ``lam * dL/dk``, re-solves, and
takes the finite difference. The integer flow then feeds a differentiable
least-squares integration (the corrected gradients are curl-free, so LS = exact),
so the loss is the actual unwrapping RMSE -- and gradients flow all the way back to
the edge head. This trains the *cost function* against final unwrapping accuracy,
not just supervised edge labels.

The blackbox gradient direction is finite-difference-validated (see project notes).
``lam`` trades gradient fidelity (small) for informativeness (large); ~10-30 works.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from . import mcf
from . import utils
from .integrator import poisson_dct_solve

_EPS = 1e-3


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


class MCFFlow(torch.autograd.Function):
    """k = argmin_flow <cost, flow> s.t. residue constraints; blackbox gradient."""

    @staticmethod
    def forward(ctx, cpx, cnx, cpy, cny, psi_np, R, lam):
        M, N = psi_np.shape
        if R.any():
            kx, ky = mcf._solve_min_cost_flow(M, N, R, _np(cpx), _np(cnx), _np(cpy), _np(cny))
        else:
            kx, ky = np.zeros((M, N - 1)), np.zeros((M - 1, N))
        ctx.save_for_backward(cpx, cnx, cpy, cny)
        ctx.M, ctx.N, ctx.R, ctx.lam = M, N, R, float(lam)
        ctx.kx, ctx.ky = kx, ky
        dev = cpx.device
        return (torch.from_numpy(kx).to(dev, torch.float32),
                torch.from_numpy(ky).to(dev, torch.float32))

    @staticmethod
    def backward(ctx, gkx, gky):
        cpx, cnx, cpy, cny = ctx.saved_tensors
        M, N, R, lam = ctx.M, ctx.N, ctx.R, ctx.lam
        gx, gy = _np(gkx), _np(gky)
        # perturb directional costs by lam * dL/dk (arc space: +flow cost up, -flow down)
        cpx2 = np.maximum(_np(cpx) + lam * gx, _EPS); cnx2 = np.maximum(_np(cnx) - lam * gx, _EPS)
        cpy2 = np.maximum(_np(cpy) + lam * gy, _EPS); cny2 = np.maximum(_np(cny) - lam * gy, _EPS)
        if R.any():
            kx2, ky2 = mcf._solve_min_cost_flow(M, N, R, cpx2, cnx2, cpy2, cny2)
        else:
            kx2, ky2 = np.zeros((M, N - 1)), np.zeros((M - 1, N))
        kx, ky = ctx.kx, ctx.ky
        relu = lambda a: np.maximum(a, 0.0)
        g_cpx = -(relu(kx) - relu(kx2)) / lam
        g_cnx = -(relu(-kx) - relu(-kx2)) / lam
        g_cpy = -(relu(ky) - relu(ky2)) / lam
        g_cny = -(relu(-ky) - relu(-ky2)) / lam
        dev = cpx.device
        f = lambda a: torch.from_numpy(a).to(dev, torch.float32)
        return f(g_cpx), f(g_cnx), f(g_cpy), f(g_cny), None, None, None


def edge_costs_torch(logits6: torch.Tensor, eps: float = _EPS, clip: float = 8.0):
    """Differentiable per-edge directional costs from (6,H,W) edge logits."""
    ph = torch.softmax(logits6[0:3], dim=0)   # (3,H,W) horizontal
    pv = torch.softmax(logits6[3:6], dim=0)   # vertical

    def cost(p):
        return torch.clamp(-torch.log(p + eps), 0.0, clip)

    return (cost(ph[2, :, :-1]), cost(ph[0, :, :-1]),   # cpx, cnx  (H, W-1)
            cost(pv[2, :-1, :]), cost(pv[0, :-1, :]))    # cpy, cny  (H-1, W)


def _aligned_rmse(phi: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    d = phi - gt
    d = d - d.mean()
    return torch.sqrt((d ** 2).mean())


def differentiable_unwrap_loss(model, batch, lam: float = 20.0) -> torch.Tensor:
    """Edge head -> costs -> MCF flow -> LS integrate -> aligned RMSE vs GT.

    Differentiable end-to-end (blackbox gradient through the solver). Requires
    ground-truth ``absolute``. Solves MCF twice per image (fwd + bwd), so use on a
    modest number of steps as a fine-tune of the supervised edge head.
    """
    logits, _ = model.encode(batch)                    # (B, 6, H, W)
    psi = batch["wrapped"][:, 0]                        # (B, H, W)
    gt = batch["absolute"][:, 0]
    losses = []
    for b in range(logits.shape[0]):
        psi_np = _np(psi[b])
        R = mcf.residues(psi_np)
        cpx, cnx, cpy, cny = edge_costs_torch(logits[b])
        kx, ky = MCFFlow.apply(cpx, cnx, cpy, cny, psi_np, R, lam)
        dx = utils.wrap_to_pi(psi[b][:, 1:] - psi[b][:, :-1])    # (H, W-1)
        dy = utils.wrap_to_pi(psi[b][1:, :] - psi[b][:-1, :])    # (H-1, W)
        gx = dx + mcf.TWO_PI * kx
        gy = dy + mcf.TWO_PI * ky
        phi = poisson_dct_solve(gx.unsqueeze(0), gy.unsqueeze(0))[0]
        losses.append(_aligned_rmse(phi, gt[b]))
    return torch.stack(losses).mean()
