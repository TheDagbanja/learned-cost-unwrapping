"""Shared plain-CNN encoder-decoder trunk for the DL baselines (NEW, this work).

Deliberately independent of :mod:`src.models.backbone` / :mod:`src.models.decoder`
(no Swin, no LoRA, no attention) -- a standard convolutional U-Net, matching
the class of architecture the cited prior deep-learning unwrappers use, so
the comparison in proposal Table 3 is between *methods* (learned-cost MCF vs.
plain-CNN direct prediction) rather than an artefact of a shared backbone.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """Two conv(3x3)+BN+ReLU layers (the classic U-Net double-conv)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DilatedBlock(nn.Sequential):
    """Two conv(3x3, dilation=d)+BN+ReLU layers (wider receptive field, no downsampling)."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class PlainUNet(nn.Module):
    """A standard 4-level convolutional U-Net (encoder/decoder + skips).

    Parameters
    ----------
    in_chans:
        Network input channels (2 for (cos,sin), 4 with wrapped-gradient
        channels -- matches :func:`src.utils.encode_input`).
    widths:
        Channel widths per encoder level, shallow -> deep.
    out_channels:
        Output channel count (head-specific; set by the subclass).
    """

    def __init__(
        self, in_chans: int = 4, widths: Sequence[int] = (32, 64, 128, 256), out_channels: int = 1,
    ) -> None:
        super().__init__()
        w = list(widths)
        self.enc_blocks = nn.ModuleList()
        prev = in_chans
        for c in w:
            self.enc_blocks.append(ConvBlock(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(w[-1], w[-1] * 2)

        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev = w[-1] * 2
        for c in reversed(w):
            self.up_convs.append(nn.ConvTranspose2d(prev, c, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock(c * 2, c))
            prev = c

        self.head = nn.Conv2d(w[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)
        for up, block, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(torch.cat((h, skip), dim=1))
        return self.head(h)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: parallel dilated convs + global context.

    The multi-scale-context module that distinguishes "PhaseNet 2.0"-style
    architectures from a plain U-Net (proposal Table 3): wide fringe patterns
    at one scale, sharp cut boundaries at another, are captured by the same
    layer instead of relying purely on encoder depth.
    """

    def __init__(self, in_ch: int, out_ch: int, rates: Sequence[int] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False) if r > 1
                else nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.pool_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        pooled = self.pool_branch(x)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))
