"""UNet++ baseline (NEW) -- Zhou et al. 2018, applied to PU.

A nested U-Net with dense skip pathways: intermediate decoder nodes ``X[i][j]``
are built from all shallower same-level nodes plus the up-sampled deeper node,
re-using features at many semantic scales before the final 1x1 head. Framed as
per-pixel wrap-count *regression* (like :mod:`dlpu_cnn`) but with the nested
dense-skip topology instead of a single decoder path.

Output schema matches the other DL baselines so the harness and
:func:`src.evaluate.evaluate_outputs` treat it identically.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from .common import ConvBlock
from .dlpu_cnn import dlpu_loss   # reuse smooth-L1 + TV wrap-count loss


def _up(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class UNetPPBaseline(nn.Module):
    """4-level UNet++ (nested dense skips), per-pixel wrap-count regression."""

    def __init__(self, in_chans: int = 4, widths: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        f = list(widths)
        self.pool = nn.MaxPool2d(2)
        self.c00 = ConvBlock(in_chans, f[0])
        self.c10 = ConvBlock(f[0], f[1]); self.c20 = ConvBlock(f[1], f[2]); self.c30 = ConvBlock(f[2], f[3])
        self.c01 = ConvBlock(f[0] + f[1], f[0])
        self.c11 = ConvBlock(f[1] + f[2], f[1]); self.c21 = ConvBlock(f[2] + f[3], f[2])
        self.c02 = ConvBlock(f[0] * 2 + f[1], f[0])
        self.c12 = ConvBlock(f[1] * 2 + f[2], f[1])
        self.c03 = ConvBlock(f[0] * 3 + f[1], f[0])
        self.head = nn.Conv2d(f[0], 1, 1)

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False) -> Dict[str, torch.Tensor]:
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        x00 = self.c00(batch["input"])
        x10 = self.c10(self.pool(x00))
        x01 = self.c01(torch.cat([x00, _up(x10, x00)], 1))
        x20 = self.c20(self.pool(x10))
        x11 = self.c11(torch.cat([x10, _up(x20, x10)], 1))
        x02 = self.c02(torch.cat([x00, x01, _up(x11, x00)], 1))
        x30 = self.c30(self.pool(x20))
        x21 = self.c21(torch.cat([x20, _up(x30, x20)], 1))
        x12 = self.c12(torch.cat([x10, x11, _up(x21, x10)], 1))
        x03 = self.c03(torch.cat([x00, x01, x02, _up(x12, x00)], 1))
        k_reg = self.head(x03)                                 # (B,1,H,W)
        k = torch.round(k_reg.squeeze(1)) if hard else k_reg.squeeze(1)
        phi_hat = psi + utils.TWO_PI * k
        g_x = utils.pad_edge_x(utils.edge_diff_x(phi_hat), 0.0)
        g_y = utils.pad_edge_y(utils.edge_diff_y(phi_hat), 0.0)
        return {"k_reg": k_reg, "k_pred": torch.round(k_reg),
                "phi_hat": phi_hat.unsqueeze(1), "g_x": g_x, "g_y": g_y}

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "lora": 0}


def unetpp_loss(outputs, batch, lambda_tv: float = 0.01):
    return dlpu_loss(outputs, batch, lambda_tv=lambda_tv)


def _smoke_test() -> None:
    torch.manual_seed(0)
    b, h, w = 2, 128, 128
    psi = (torch.rand(b, h, w) * 2 - 1) * utils.PI
    phi = psi + utils.TWO_PI * torch.randint(-3, 4, (b, h, w)).float()
    batch = {"input": utils.encode_input(psi, include_grad=True), "wrapped": psi.unsqueeze(1),
             "k_pixel": ((phi - psi) / utils.TWO_PI).round().unsqueeze(1)}
    model = UNetPPBaseline(in_chans=4)
    out = model(batch)
    terms = unetpp_loss(out, batch)
    terms["total"].backward()
    print(f"[smoke] UNetPP params={model.count_parameters()['total']:,} "
          f"phi_hat {tuple(out['phi_hat'].shape)} loss={terms['total'].item():.4f}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    _smoke_test()
