"""Shared phase-regression helpers for the external PU baselines."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from src import utils


def phase_outputs(phi_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Convert a ``(B,1,H,W)`` phase prediction to the ext_dl output schema."""
    if phi_hat.dim() == 3:
        phi_hat = phi_hat.unsqueeze(1)
    phase = phi_hat[:, 0]
    return {
        "phi_hat": phi_hat,
        "g_x": utils.pad_edge_x(utils.edge_diff_x(phase), 0.0),
        "g_y": utils.pad_edge_y(utils.edge_diff_y(phase), 0.0),
    }


def aligned_residual(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Offset-invariant residual; absolute phase has an arbitrary constant."""
    if pred.dim() == 4:
        pred = pred[:, 0]
    if target.dim() == 4:
        target = target[:, 0]
    residual = pred - target
    return residual - residual.mean(dim=(-2, -1), keepdim=True)


def regression_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    lambda_grad: float = 0.2,
    lambda_congruence: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Offset-invariant L1 + gradient + rewrapping consistency loss."""
    if "absolute" not in batch:
        raise ValueError("Direct phase regression requires batch['absolute'] ground truth.")
    pred = outputs["phi_hat"]
    gt = batch["absolute"]
    residual = aligned_residual(pred, gt)
    l1 = residual.abs().mean()
    grad = (
        F.l1_loss(utils.edge_diff_x(pred[:, 0]), utils.edge_diff_x(gt[:, 0]))
        + F.l1_loss(utils.edge_diff_y(pred[:, 0]), utils.edge_diff_y(gt[:, 0]))
    ) * 0.5
    psi = batch["wrapped"][:, 0]
    congruence = (1.0 - torch.cos(pred[:, 0] - psi)).mean()
    total = l1 + float(lambda_grad) * grad + float(lambda_congruence) * congruence
    return {"l1": l1, "grad": grad, "congruence": congruence, "total": total}


class ParameterCountMixin:
    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "lora": 0}
