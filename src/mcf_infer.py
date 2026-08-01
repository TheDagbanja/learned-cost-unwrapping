"""Bridge: Swin+LoRA edge head -> learned MCF costs -> residue-free unwrap.

This is the Stage-2 inference path. The edge head predicts per-edge ambiguity
logits ``(B, 6, H, W)``; we turn them into directional min-cost-flow costs
(:func:`mcf.edge_costs_from_logits`) and solve the integer unwrapping
(:func:`mcf.unwrap_mcf`). The network supplies the *prior* (where cuts are
likely); the solver supplies the *guarantee* (global consistency, zero residues).

Kept separate from ``src/mcf.py`` so the solver stays torch-free. Inference only
(non-differentiable) -- end-to-end cost learning through the solver (blackbox
gradient) is Stage-2 v2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import mcf
from . import utils


@torch.no_grad()
def unwrap_batch(model, batch: Dict[str, torch.Tensor], backend: str = "auto",
                 also_unit: bool = False):
    """Unwrap a batch with learned MCF costs from the edge head.

    Returns ``phi_learned`` ``(B, H, W)`` numpy. If ``also_unit`` also returns the
    unit-cost MCF unwrap (same solver, flat costs) for a controlled comparison.
    Requires ``model.head_type == 'edge'`` (the 6-channel per-edge head).
    """
    if getattr(model, "head_type", None) != "edge":
        raise ValueError(
            f"learned-cost MCF needs the edge head (6-ch logits); got "
            f"head_type={getattr(model, 'head_type', None)!r}."
        )
    model.eval()
    logits, _ = model.encode(batch)                # (B, 6, H, W)
    logits_np = logits.detach().float().cpu().numpy()

    psi = batch["wrapped"]
    psi = psi[:, 0] if psi.dim() == 4 else psi
    psi_np = psi.detach().float().cpu().numpy()

    B = psi_np.shape[0]
    phi_learned = np.empty_like(psi_np, dtype=np.float64)
    phi_unit = np.empty_like(psi_np, dtype=np.float64) if also_unit else None
    for b in range(B):
        ec = mcf.edge_costs_from_logits(logits_np[b])
        phi_learned[b] = mcf.unwrap_mcf(psi_np[b], edge_costs=ec, backend=backend)
        if also_unit:
            phi_unit[b] = mcf.unwrap_mcf(psi_np[b], backend=backend)
    return (phi_learned, phi_unit) if also_unit else phi_learned


def _aligned_rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    d = d - d.mean()
    return float(np.sqrt((d ** 2).mean()))


@torch.no_grad()
def evaluate_learned_cost(model, loader, device, max_batches: Optional[int] = None,
                          backend: str = "auto") -> Dict[str, float]:
    """Compare learned-cost MCF vs unit-cost MCF vs the oracle floor over a loader.

    Reports mean aligned RMSE for each, plus the input-residue count (the unwrap is
    residue-free by construction). Ground truth (``absolute``) must be present.
    """
    learned: List[float] = []
    unit: List[float] = []
    floor: List[float] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = utils.move_batch_to_device(batch, device)
        if "absolute" not in batch:
            raise ValueError("evaluate_learned_cost needs ground-truth 'absolute'.")
        phi_l, phi_u = unwrap_batch(model, batch, backend=backend, also_unit=True)
        psi = batch["wrapped"][:, 0].detach().cpu().numpy()
        gt = batch["absolute"][:, 0].detach().cpu().numpy()
        for b in range(gt.shape[0]):
            learned.append(_aligned_rmse_np(phi_l[b], gt[b]))
            unit.append(_aligned_rmse_np(phi_u[b], gt[b]))
            floor.append(mcf.aligned_gtk_floor(psi[b], gt[b]))
    return {
        "rmse_learned": float(np.mean(learned)),
        "rmse_unit": float(np.mean(unit)),
        "rmse_floor": float(np.mean(floor)),
        "n": float(len(learned)),
    }
