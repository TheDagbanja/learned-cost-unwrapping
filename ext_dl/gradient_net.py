"""Gradient-estimation network baseline (NEW, this work) -- proposal Table 3.

Reproduces the class of deep-learning unwrappers that predict phase
*gradients* (rather than the phase itself) and integrate them with a plain
least-squares solve -- no minimum-cost-flow, no discrete loop-closure
constraint. This is the most direct architectural sibling of the main
method's edge head (predicting a correction *per edge*), which is precisely
why it is the sharpest illustration of proposal §5's point: "a low-RMSE
deep-learning output that still contains residues is not usable in a
geodetic pipeline" -- the MCF is what turns a per-edge *prior* into a
*guarantee*, and this baseline has the prior without the guarantee.

Architecture: a plain CNN regresses a **continuous** per-edge correction
directly (not a classification over discrete levels), added to the observed
wrapped gradient and passed through the reused (non-MCF) DCT/PCG
least-squares integrator (:mod:`src.integrator`) -- the same integrator the
main method uses for its *weights*, but here fed uncorrected, unconstrained
gradients instead of a residue-free integer flow.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src import utils
from src.integrator import WeightedLeastSquaresIntegrator
from .common import PlainUNet


class GradientNetBaseline(nn.Module):
    """Plain-CNN per-edge gradient-correction regressor + LS integration."""

    def __init__(self, in_chans: int = 4, widths=(32, 64, 128, 256), max_correction: float = 3.0) -> None:
        super().__init__()
        # 2 output channels: continuous correction to the horizontal / vertical
        # wrapped gradient, in units of 2*pi (i.e. directly comparable to k_e).
        self.net = PlainUNet(in_chans=in_chans, widths=widths, out_channels=2)
        self.max_correction = max_correction
        self.integrator = WeightedLeastSquaresIntegrator(solver="dct")

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False) -> Dict[str, torch.Tensor]:
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        corr = self.net(batch["input"])                    # (B,2,H,W) continuous
        corr = self.max_correction * torch.tanh(corr / self.max_correction)  # bounded, stable
        amb_h, amb_v = corr[:, 0], corr[:, 1]
        if hard:
            amb_h, amb_v = torch.round(amb_h), torch.round(amb_v)

        wgx, wgy = utils.wrapped_gradients(psi, pad=True)
        g_x = wgx + utils.TWO_PI * amb_h
        g_y = wgy + utils.TWO_PI * amb_v
        # Plain (non-MCF) least-squares integration: no loop-closure constraint,
        # so an inconsistent correction field is silently smoothed, not resolved.
        phi_hat = self.integrator(g_x, g_y, batch.get("weights_x"), batch.get("weights_y"))
        return {
            "amb_h": amb_h, "amb_v": amb_v,
            "phi_hat": phi_hat.unsqueeze(1), "g_x": g_x, "g_y": g_y,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "lora": 0}


def loss_from_k_max(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any],
                    k_max: int, lambda_cong: float = 0.1) -> Dict[str, torch.Tensor]:
    """Supervised smooth-L1 on per-edge ambiguity + a light congruence term.

    ``k_max`` is the ambiguity half-range the labels were clamped to
    (``utils.compute_ambiguity_labels``'s ``class = k + k_max``).
    """
    valid_h, valid_v = batch.get("valid_x"), batch.get("valid_y")
    target_h = batch["k_x"].float() - k_max
    target_v = batch["k_y"].float() - k_max

    def masked_l1(pred, target, mask):
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        if mask is None:
            return loss.mean()
        m = mask.to(loss.dtype)
        return (loss * m).sum() / m.sum().clamp_min(1.0)

    l_h = masked_l1(outputs["amb_h"], target_h, valid_h)
    l_v = masked_l1(outputs["amb_v"], target_v, valid_v)
    reg = 0.5 * (l_h + l_v)
    cong = (1.0 - torch.cos(utils.align_reference(outputs["phi_hat"].squeeze(1), batch["wrapped"].squeeze(1))
                            - batch["wrapped"].squeeze(1))).mean()
    return {"reg": reg, "cong": cong, "total": reg + lambda_cong * cong}


def _smoke_test() -> None:
    torch.manual_seed(0)
    b, h, w = 2, 256, 256
    psi = (torch.rand(b, h, w) * 2 - 1) * utils.PI
    phi = psi + utils.TWO_PI * torch.randint(-1, 2, (b, h, w)).float()
    k_max = 1
    cls_x, cls_y = utils.compute_ambiguity_labels(psi, phi, pad=True, clamp=True, k_max=k_max)
    valid_x, valid_y = utils.edge_valid_masks(h, w)
    batch = {
        "input": utils.encode_input(psi, include_grad=True),
        "wrapped": psi.unsqueeze(1),
        "k_x": cls_x, "k_y": cls_y, "valid_x": valid_x, "valid_y": valid_y,
    }
    model = GradientNetBaseline(in_chans=4)
    out = model(batch)
    terms = loss_from_k_max(out, batch, k_max=k_max)
    terms["total"].backward()
    print(f"[smoke] GradientNetBaseline params={model.count_parameters()['total']:,} "
          f"phi_hat {tuple(out['phi_hat'].shape)} loss={terms['total'].item():.4f}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    _smoke_test()
