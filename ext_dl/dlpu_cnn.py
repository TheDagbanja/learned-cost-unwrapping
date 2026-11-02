""""DLPU" reference CNN baseline (NEW, this work) -- proposal Table 3.

Reproduces the class of direct-regression deep-learning phase unwrappers
(Zhou et al.-style "DLPU": a residual/U-Net CNN mapping wrapped phase to
unwrapped phase) for a fair RMSE + residue comparison against the
learned-cost MCF. A plain :class:`~src.baselines_dl.common.PlainUNet`
regresses the per-pixel wrap count ``K`` from ``(cos psi, sin psi[, grad])``;
the unwrapped phase is ``phi_hat = psi + 2*pi*K``. Unlike the MCF-based
methods, nothing here enforces the discrete loop-closure constraint as an
explicit combinatorial guarantee -- the network is simply trained to make
``phi_hat`` match the reference field, so residual errors are whatever a
smooth-L1 regression leaves behind (no structural residue-free proof).

Trained with the same physics-guided regularisers available to the main
model (congruence + wrap-count total variation) so the comparison is not
handicapped by an unfair loss, only by architecture/method.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src import utils
from .common import PlainUNet


class DLPUCNNBaseline(nn.Module):
    """Residual-U-Net wrap-count regression baseline."""

    def __init__(self, in_chans: int = 4, widths=(32, 64, 128, 256)) -> None:
        super().__init__()
        self.net = PlainUNet(in_chans=in_chans, widths=widths, out_channels=1)

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False) -> Dict[str, torch.Tensor]:
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        k_reg = self.net(batch["input"])                  # (B,1,H,W) continuous K
        k = torch.round(k_reg.squeeze(1)) if hard else k_reg.squeeze(1)
        phi_hat = psi + utils.TWO_PI * k
        g_x = utils.pad_edge_x(utils.edge_diff_x(phi_hat), 0.0)
        g_y = utils.pad_edge_y(utils.edge_diff_y(phi_hat), 0.0)
        return {
            "k_reg": k_reg, "k_pred": torch.round(k_reg),
            "phi_hat": phi_hat.unsqueeze(1), "g_x": g_x, "g_y": g_y,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "lora": 0}


def dlpu_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any],
             lambda_tv: float = 0.01) -> Dict[str, torch.Tensor]:
    """Supervised smooth-L1 on K plus a light total-variation smoothness prior."""
    if "k_pixel" not in batch:
        raise ValueError("DLPU-CNN baseline needs 'k_pixel' labels (ground-truth wrap count).")
    k = outputs["k_reg"]
    kreg = torch.nn.functional.smooth_l1_loss(k, batch["k_pixel"])
    tv = (k[..., :, 1:] - k[..., :, :-1]).abs().mean() + (k[..., 1:, :] - k[..., :-1, :]).abs().mean()
    return {"kreg": kreg, "ktv": tv, "total": kreg + lambda_tv * tv}


def _smoke_test() -> None:
    torch.manual_seed(0)
    b, h, w = 2, 256, 256
    psi = (torch.rand(b, h, w) * 2 - 1) * utils.PI
    phi = psi + utils.TWO_PI * torch.randint(-3, 4, (b, h, w)).float()
    batch = {
        "input": utils.encode_input(psi, include_grad=True),
        "wrapped": psi.unsqueeze(1),
        "k_pixel": ((phi - psi) / utils.TWO_PI).round().unsqueeze(1),
    }
    model = DLPUCNNBaseline(in_chans=4)
    out = model(batch)
    terms = dlpu_loss(out, batch)
    terms["total"].backward()
    print(f"[smoke] DLPUCNNBaseline params={model.count_parameters()['total']:,} "
          f"phi_hat {tuple(out['phi_hat'].shape)} loss={terms['total'].item():.4f}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    _smoke_test()
