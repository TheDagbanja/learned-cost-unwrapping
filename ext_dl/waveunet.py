"""Adapter of the official WaveUNet architecture to the ext_dl API.

Official source: https://github.com/xiangHQ/WaveUNet (accessed 2026-09-22).
Mao et al., ISPRS JPRS 239 (2026), DOI 10.1016/j.isprsjprs.2026.06.025.

The layer graph follows the official ``builders/model/WaveUNet.py``.  A native
Haar DWT is included so the extension has no mandatory ``pytorch_wavelets``
dependency; it produces the same four decimated subbands (up to conventional
detail-band signs/order, which are learned away when training from scratch).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase_common import ParameterCountMixin, phase_outputs


class ConvOne(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x): return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch))
        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x): return F.relu(self.body(x) + self.skip(x), inplace=True)


class HaarDWT(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), mode="replicate")
        a, b, c, d = x[..., 0::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        return (a + b + c + d) * 0.5, torch.stack(((a - b + c - d) * 0.5,
                                                   (a + b - c - d) * 0.5,
                                                   (a - b - c + d) * 0.5), dim=2)


class PyramidPooling(nn.Module):
    def __init__(self, channels: int, sizes: Sequence[int]):
        super().__init__(); n = len(sizes)
        self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(channels, channels // n, 1, bias=False),
                                                    nn.BatchNorm2d(channels // n), nn.ReLU(inplace=True)) for s in sizes])
        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
    def forward(self, x):
        h, w = x.shape[-2:]
        return self.fuse(torch.cat([x] + [F.interpolate(s(x), (h, w), mode="bilinear", align_corners=False) for s in self.stages], 1))


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.q = nn.Conv2d(channels, channels // 8, 1); self.k = nn.Conv2d(channels, channels // 8, 1)
        self.v = nn.Conv2d(channels, channels, 1); self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b, c, h, w = x.shape; q = self.q(x).flatten(2).transpose(1, 2); k = self.k(x).flatten(2)
        a = torch.softmax(torch.bmm(q, k), dim=-1)
        return self.gamma * torch.bmm(self.v(x).flatten(2), a.transpose(1, 2)).view(b, c, h, w) + x


class WaveUNetBaseline(ParameterCountMixin, nn.Module):
    def __init__(self, in_chans: int = 4, base: int = 16):
        super().__init__()
        # Official model consumes the scalar wrapped phase, not encoded channels.
        self.pre = ConvOne(1, base); self.grad = nn.Conv2d(2, base, 3, padding=1); self.dwt = HaarDWT()
        self.enc1 = ResidualBlock(4*base, 2*base); self.enc2 = ResidualBlock(8*base, 4*base); self.enc3 = ResidualBlock(16*base, 8*base)
        self.skip1 = nn.Conv2d(base, 2*base, 3, 2, 1); self.skip2 = nn.Conv2d(2*base, 4*base, 3, 2, 1); self.skip3 = nn.Conv2d(4*base, 8*base, 3, 2, 1)
        self.ppm1 = PyramidPooling(2*base, (8,16,32,64)); self.ppm2 = PyramidPooling(4*base, (4,8,16,32)); self.ppm3 = PyramidPooling(8*base, (2,4,8,16))
        self.attn = SelfAttention(8*base); self.mid = ConvOne(8*base, 8*base); self.dec3 = ResidualBlock(16*base, 16*base)
        self.up1 = nn.ConvTranspose2d(16*base, 4*base, 4, 2, 1); self.dec2 = ResidualBlock(8*base, 8*base)
        self.up2 = nn.ConvTranspose2d(8*base, 2*base, 4, 2, 1); self.dec1 = ResidualBlock(4*base, 4*base)
        self.up3 = nn.ConvTranspose2d(4*base, base, 4, 2, 1); self.dec0 = ResidualBlock(2*base, 2*base)
        self.head = nn.Sequential(ResidualBlock(2*base, 2*base), ResidualBlock(2*base, 2*base), nn.Conv2d(2*base, 1, 1))

    def merge(self, x):
        lo, hi = self.dwt(x); return torch.cat([lo, hi[:, :, 0], hi[:, :, 1], hi[:, :, 2]], 1)
    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False):
        psi = batch["wrapped"] if batch["wrapped"].dim() == 4 else batch["wrapped"].unsqueeze(1)
        gx = F.pad(psi[..., 1:] - psi[..., :-1], (0,1)); gy = F.pad(psi[..., 1:, :] - psi[..., :-1, :], (0,0,0,1))
        grad = self.grad(torch.cat([gx, gy], 1)); x = self.pre(psi)
        x1 = self.enc1(self.merge(x)) + self.skip1(grad); p1 = self.ppm1(x1)
        x2 = self.enc2(self.merge(p1)) + self.skip2(p1); p2 = self.ppm2(x2)
        x3 = self.enc3(self.merge(p2)) + self.skip3(p2); x4 = self.ppm3(x3)
        x = self.dec3(torch.cat([self.mid(self.attn(x4)), x3], 1))
        x = self.dec2(torch.cat([self.up1(x), x2], 1)); x = self.dec1(torch.cat([self.up2(x), x1], 1))
        x = self.dec0(torch.cat([self.up3(x), grad], 1))
        return phase_outputs(self.head(x))


def waveunet_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]):
    # Official repository uses L1Loss for InSAR-DLPU.
    pred, gt = outputs["phi_hat"], batch["absolute"]
    residual = pred - gt; residual = residual - residual.mean(dim=(-2,-1), keepdim=True)
    l1 = residual.abs().mean(); return {"l1": l1, "total": l1}
