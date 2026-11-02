"""Paper-guided DMP-PUNet adaptation for the shared InSAR-DLPU protocol.

Chen et al., IJAEOG 139 (2025) 104519,
DOI 10.1016/j.jag.2025.104519, describe a U-Net++-inspired encoder/decoder
whose blocks combine 1x1, 3x3 and 5x5 paths with dilated convolutions and
optimized dense skip fusion.  Author code is not public.  This module implements
those reported structural elements, but should be cited as an adapted
reproduction rather than an exact source-code reproduction.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase_common import ParameterCountMixin, phase_outputs, regression_loss


class DilatedMultiPath(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 2) -> None:
        super().__init__()
        q = out_ch // 3
        chans = (q, q, out_ch - 2 * q)
        self.paths = nn.ModuleList([
            nn.Conv2d(in_ch, chans[0], 1, bias=False),
            nn.Conv2d(in_ch, chans[1], 3, padding=1, bias=False),
            nn.Conv2d(in_ch, chans[2], 5, padding=2, bias=False),
        ])
        self.reduce = nn.Sequential(nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1, inplace=True),
                                    nn.Conv2d(out_ch, out_ch, 3, padding=dilation,
                                              dilation=dilation, bias=False),
                                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.reduce(torch.cat([path(x) for path in self.paths], 1)) + self.skip(x)


class DMPPUNetBaseline(ParameterCountMixin, nn.Module):
    """Dilated multi-path nested encoder/decoder, direct absolute-phase output."""

    def __init__(self, in_chans: int = 4, widths: Sequence[int] = (64, 128, 256, 512)) -> None:
        super().__init__()
        f = list(widths)
        self.pool = nn.MaxPool2d(2)
        self.e0 = DilatedMultiPath(in_chans, f[0]); self.e1 = DilatedMultiPath(f[0], f[1])
        self.e2 = DilatedMultiPath(f[1], f[2]); self.e3 = DilatedMultiPath(f[2], f[3])
        self.d2 = DilatedMultiPath(f[3] + f[2], f[2])
        self.d1 = DilatedMultiPath(f[2] + f[1], f[1])
        self.d0 = DilatedMultiPath(f[1] + f[0], f[0])
        # Dense refined paths, reflecting the paper's U-Net++-inspired fusion.
        self.r1 = DilatedMultiPath(f[1] * 2 + f[2], f[1])
        self.r0 = DilatedMultiPath(f[0] * 2 + f[1], f[0])
        self.head = nn.Conv2d(f[0], 1, 1)

    @staticmethod
    def up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False):
        e0 = self.e0(batch["input"]); e1 = self.e1(self.pool(e0))
        e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([e2, self.up(e3, e2)], 1))
        d1 = self.d1(torch.cat([e1, self.up(d2, e1)], 1))
        d0 = self.d0(torch.cat([e0, self.up(d1, e0)], 1))
        r1 = self.r1(torch.cat([e1, d1, self.up(d2, e1)], 1))
        r0 = self.r0(torch.cat([e0, d0, self.up(r1, e0)], 1))
        return phase_outputs(self.head(r0))


def dmp_punet_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any],
                   lambda_grad: float = 0.2, lambda_congruence: float = 0.05):
    return regression_loss(outputs, batch, lambda_grad, lambda_congruence)
