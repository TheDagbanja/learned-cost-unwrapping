"""PyTorch port of the public Unwrap-Net architecture.

Architecture source: https://github.com/yangwangyangzi48/UNWRAPNETV1/blob/main/ModelS.py

The public code defines a two-input SERB/ASPP U-Net (interferogram and
coherence), but does not publish ``hfloss.py`` or the data-building code.  This
port preserves the layer topology and uses the common ext_dl phase loss.  It
uses real coherence when the batch provides it; otherwise it derives a clearly
identified local-resultant coherence proxy from the wrapped phase so the model
can still be trained on the unchanged InSAR-DLPU protocol.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase_common import ParameterCountMixin, phase_outputs, regression_loss


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(-2, -1))
        return x * self.fc(s).unsqueeze(-1).unsqueeze(-1)


class SERB(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), SEBlock(out_channels),
        )
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class ASPP(nn.Module):
    def __init__(self, channels: int, rates: Sequence[int] = (1, 6, 12)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=r, dilation=r, bias=False),
                          nn.BatchNorm2d(channels), nn.ReLU(inplace=True)) for r in rates
        ])
        self.fuse = nn.Sequential(nn.Conv2d(channels * len(rates), channels, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


def _local_coherence(psi: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    c = F.avg_pool2d(torch.cos(psi), kernel, stride=1, padding=kernel // 2)
    s = F.avg_pool2d(torch.sin(psi), kernel, stride=1, padding=kernel // 2)
    return torch.sqrt(c.square() + s.square() + 1e-8)


def _coherence_from_batch(batch: Dict[str, torch.Tensor], psi: torch.Tensor) -> torch.Tensor:
    for key in ("coherence", "coh", "gamma"):
        if key in batch:
            value = batch[key]
            return value if value.dim() == 4 else value.unsqueeze(1)
    return _local_coherence(psi)


class UnwrapNetBaseline(ParameterCountMixin, nn.Module):
    """Official SERB/ASPP topology, adapted to the ext_dl batch/output API."""

    def __init__(self, in_chans: int = 4, widths=(64, 128, 256, 512, 512)) -> None:
        super().__init__()
        # The public implementation consumes exactly two scalar inputs.
        f1, f2, f3, f4, f5 = widths
        self.c1 = SERB(2, f1); self.c2 = SERB(f1, f2); self.c3 = SERB(f2, f3)
        self.c4 = SERB(f3, f4); self.c5 = SERB(f4, f5); self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(f5)
        self.c6 = SERB(f5 + f4, 256); self.c7 = SERB(256 + f3, 128)
        self.c8 = SERB(128 + f2, 64); self.c9 = SERB(64 + f1, 64)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False):
        psi = batch["wrapped"]
        if psi.dim() == 3:
            psi = psi.unsqueeze(1)
        coherence = _coherence_from_batch(batch, psi)
        x = torch.cat([psi, coherence.to(dtype=psi.dtype)], dim=1)
        c1 = self.c1(x); c2 = self.c2(self.pool(c1)); c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3)); c5 = self.aspp(self.c5(self.pool(c4)))
        c6 = self.c6(torch.cat([F.interpolate(c5, size=c4.shape[-2:], mode="nearest"), c4], 1))
        c7 = self.c7(torch.cat([F.interpolate(c6, size=c3.shape[-2:], mode="nearest"), c3], 1))
        c8 = self.c8(torch.cat([F.interpolate(c7, size=c2.shape[-2:], mode="nearest"), c2], 1))
        c9 = self.c9(torch.cat([F.interpolate(c8, size=c1.shape[-2:], mode="nearest"), c1], 1))
        return phase_outputs(self.head(c9))


def unwrap_net_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any],
                    lambda_grad: float = 0.2, lambda_congruence: float = 0.05):
    return regression_loss(outputs, batch, lambda_grad, lambda_congruence)
