"""Paper-guided single-baseline MoDL-PU adaptation.

Zhou and Yu, TGRS 2025, DOI 10.1109/TGRS.2025.3549607 publicly specify a
Res-UNet-Inception backbone and two-stage training: data-supervised low-frequency
learning followed by task-specific model-knowledge refinement.  No author code
or complete layer/loss specification was publicly available when this package
was prepared.  Consequently this is an *adapted reproduction*, not an exact
reimplementation.  It implements the published backbone family and the SB-PU
two-stage idea while retaining this repository's data and training protocol.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase_common import ParameterCountMixin, aligned_residual, phase_outputs, regression_loss


class InceptionResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        q = out_ch // 4
        sizes = (q, q, q, out_ch - 3 * q)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, sizes[0], 1, bias=False), nn.BatchNorm2d(sizes[0]), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_ch, sizes[1], 1, bias=False), nn.ReLU(),
                          nn.Conv2d(sizes[1], sizes[1], 3, padding=1, bias=False), nn.BatchNorm2d(sizes[1]), nn.ReLU()),
            nn.Sequential(nn.Conv2d(in_ch, sizes[2], 1, bias=False), nn.ReLU(),
                          nn.Conv2d(sizes[2], sizes[2], 5, padding=2, bias=False), nn.BatchNorm2d(sizes[2]), nn.ReLU()),
            nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_ch, sizes[3], 1, bias=False),
                          nn.BatchNorm2d(sizes[3]), nn.ReLU()),
        ])
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return F.relu(torch.cat([b(x) for b in self.branches], 1) + self.skip(x), inplace=True)


class MoDLPUBaseline(ParameterCountMixin, nn.Module):
    def __init__(self, in_chans: int = 4, widths=(32, 64, 128, 256)) -> None:
        super().__init__()
        f = widths
        self.enc = nn.ModuleList([InceptionResidual(in_chans, f[0]), InceptionResidual(f[0], f[1]),
                                  InceptionResidual(f[1], f[2]), InceptionResidual(f[2], f[3])])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(f[3], f[2], 2, 2),
                                 nn.ConvTranspose2d(f[2], f[1], 2, 2),
                                 nn.ConvTranspose2d(f[1], f[0], 2, 2)])
        self.dec = nn.ModuleList([InceptionResidual(f[2] * 2, f[2]),
                                  InceptionResidual(f[1] * 2, f[1]),
                                  InceptionResidual(f[0] * 2, f[0])])
        self.head = nn.Conv2d(f[0], 1, 1)

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False):
        skips = []
        x = batch["input"]
        for i, block in enumerate(self.enc):
            x = block(x); skips.append(x)
            if i < len(self.enc) - 1:
                x = self.pool(x)
        for up, block, skip in zip(self.up, self.dec, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return phase_outputs(self.head(x))


def modl_pu_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any], stage: int = 2,
                 lambda_grad: float = 0.3, lambda_congruence: float = 0.1):
    if int(stage) == 1:
        pred, gt = outputs["phi_hat"], batch["absolute"]
        # Published stage 1 emphasizes the low-frequency phase component.
        pred_l = F.avg_pool2d(pred, 9, stride=1, padding=4)
        gt_l = F.avg_pool2d(gt, 9, stride=1, padding=4)
        low = aligned_residual(pred_l, gt_l).abs().mean()
        return {"low_frequency": low, "total": low}
    return regression_loss(outputs, batch, lambda_grad, lambda_congruence)
