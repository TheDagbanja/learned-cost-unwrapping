"""Attention U-Net baseline (NEW) -- Oktay et al. 2018, applied to PU.

A U-Net whose skip connections are gated by additive attention (attention gates
learn to suppress irrelevant background and focus on fringe/cut regions). Frames
unwrapping as per-pixel wrap-count *regression* (like :mod:`dlpu_cnn`) but with
attention-gated skips instead of plain concatenation -- included to test whether
attention over the skip features helps a direct regressor on this benchmark.

Output schema matches the other DL baselines (``phi_hat``, ``k_reg``, ``k_pred``,
``g_x``, ``g_y``) so the evaluation harness and :func:`src.evaluate.evaluate_outputs`
treat it identically.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from .common import ConvBlock
from .dlpu_cnn import dlpu_loss   # reuse the smooth-L1 + TV wrap-count loss


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay 2018): gate skip features ``x`` by ``g``."""

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        a = F.relu(self.w_g(g) + self.w_x(x), inplace=True)
        return x * self.psi(a)


class AttentionUNetBaseline(nn.Module):
    """4-level Attention U-Net, per-pixel wrap-count regression."""

    def __init__(self, in_chans: int = 4, widths: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList()
        prev = in_chans
        for c in w:
            self.enc.append(ConvBlock(prev, c)); prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(w[-1], w[-1] * 2)
        self.up = nn.ModuleList(); self.gates = nn.ModuleList(); self.dec = nn.ModuleList()
        prev = w[-1] * 2
        for c in reversed(w):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.gates.append(AttentionGate(c, c, max(c // 2, 8)))
            self.dec.append(ConvBlock(c * 2, c))
            prev = c
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False) -> Dict[str, torch.Tensor]:
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        skips = []
        h = batch["input"]
        for block in self.enc:
            h = block(h); skips.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        for up, gate, dec, skip in zip(self.up, self.gates, self.dec, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = gate(h, skip)
            h = dec(torch.cat((h, skip), dim=1))
        k_reg = self.head(h)                                   # (B,1,H,W)
        k = torch.round(k_reg.squeeze(1)) if hard else k_reg.squeeze(1)
        phi_hat = psi + utils.TWO_PI * k
        g_x = utils.pad_edge_x(utils.edge_diff_x(phi_hat), 0.0)
        g_y = utils.pad_edge_y(utils.edge_diff_y(phi_hat), 0.0)
        return {"k_reg": k_reg, "k_pred": torch.round(k_reg),
                "phi_hat": phi_hat.unsqueeze(1), "g_x": g_x, "g_y": g_y}

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "lora": 0}


def attention_unet_loss(outputs, batch, lambda_tv: float = 0.01):
    return dlpu_loss(outputs, batch, lambda_tv=lambda_tv)


def _smoke_test() -> None:
    torch.manual_seed(0)
    b, h, w = 2, 128, 128
    psi = (torch.rand(b, h, w) * 2 - 1) * utils.PI
    phi = psi + utils.TWO_PI * torch.randint(-3, 4, (b, h, w)).float()
    batch = {"input": utils.encode_input(psi, include_grad=True), "wrapped": psi.unsqueeze(1),
             "k_pixel": ((phi - psi) / utils.TWO_PI).round().unsqueeze(1)}
    model = AttentionUNetBaseline(in_chans=4)
    out = model(batch)
    terms = attention_unet_loss(out, batch)
    terms["total"].backward()
    print(f"[smoke] AttentionUNet params={model.count_parameters()['total']:,} "
          f"phi_hat {tuple(out['phi_hat'].shape)} loss={terms['total'].item():.4f}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    _smoke_test()
