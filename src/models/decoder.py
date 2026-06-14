"""Lightweight U-Net decoder over Swin-T multi-scale features (proposal §3.3).

The decoder fuses the four hierarchical Swin feature maps (strides 4/8/16/32)
**and a full-resolution convolutional stem on the raw input** back to full
resolution, producing per-edge ambiguity logits.

Output contract (matches :mod:`src.utils` / :mod:`src.dataset` conventions)::

    logits: (B, 6, H, W)
        channels 0:3 -> horizontal edge ambiguity classes {-1, 0, +1}
        channels 3:6 -> vertical   edge ambiguity classes {-1, 0, +1}

Why the stem matters: Swin's 4x4 patch embedding makes the encoder's finest
feature map stride-4, so a decoder built only on Swin features can predict at
~4x4-block resolution at best. Per-edge cuts are pixel-sharp and concentrated in
dense-fringe bands, so block-level prediction caps the achievable RMSE (~7.9 vs
~0.56 full-resolution on synthetic data). The stem provides a stride-1 and
stride-2 skip path carrying full-resolution detail, restoring pixel-precise
prediction. The penultimate full-resolution feature map is exposed as the
"edge-feature space" ``h_e`` (proposal §3.6) for K-shot prototypes.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    """conv(k) + BatchNorm + GELU (the decoder's basic building block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class DecoderBlock(nn.Module):
    """Upsample, concatenate a skip connection, then two conv layers.

    The spatial size is taken from the skip tensor, so the block is robust to the
    odd feature sizes that Swin's window padding can produce.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip), dim=1)
        return self.conv2(self.conv1(x))


class UNetDecoder(nn.Module):
    """U-Net decoder fusing Swin features + a full-resolution input stem.

    Parameters
    ----------
    encoder_channels:
        Channel counts of the Swin feature maps, shallow->deep
        (e.g. ``[96, 192, 384, 768]``).
    decoder_channels:
        Output widths for the five decoder blocks: three fuse the Swin skips
        (strides 16/8/4) and two fuse the stride-2 / stride-1 stem skips.
    in_chans:
        Number of input channels (2 for the ``(cos, sin)`` encoding).
    stem_channels:
        Channels of the (stride-2, stride-1) stem skip features.
    out_channels:
        Number of output logit channels (6 = 2 directions x 3 classes).
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (96, 192, 384, 768),
        decoder_channels: Sequence[int] = (256, 128, 96, 64, 32),
        in_chans: int = 2,
        stem_channels: Sequence[int] = (48, 32),
        out_channels: int = 6,
    ) -> None:
        super().__init__()
        enc = list(encoder_channels)
        dec = list(decoder_channels)
        if len(enc) != 4:
            raise ValueError("Expected 4 Swin encoder feature maps.")
        if len(dec) != 5:
            raise ValueError("decoder_channels must have 5 entries (3 Swin + 2 stem skips).")

        # Full-resolution stem on the raw input: stride-1 and stride-2 skips.
        self.stem1 = ConvBNAct(in_chans, stem_channels[1])                    # stride 1
        self.stem2 = nn.Sequential(
            ConvBNAct(in_chans, stem_channels[0], stride=2),                  # stride 2
            ConvBNAct(stem_channels[0], stem_channels[0]),
        )

        # Skip channels in decoder order (deep->shallow):
        #   f2 (s16), f1 (s8), f0 (s4), stem2 (s2), stem1 (s1).
        skip_ch = [enc[2], enc[1], enc[0], stem_channels[0], stem_channels[1]]
        self.blocks = nn.ModuleList()
        in_ch = enc[3]                                                        # f3 (s32) start
        for sc, oc in zip(skip_ch, dec):
            self.blocks.append(DecoderBlock(in_ch, sc, oc))
            in_ch = oc

        self.edge_feature_channels = dec[-1]
        self.head = nn.Conv2d(dec[-1], out_channels, kernel_size=1)

    def forward(
        self, features: List[torch.Tensor], input_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, edge_features)`` at the input resolution.

        ``features`` is the shallow->deep Swin list; ``input_image`` is the raw
        ``(B, in_chans, H, W)`` network input feeding the full-resolution stem.
        """
        s1 = self.stem1(input_image)                                         # (B, c, H, W)
        s2 = self.stem2(input_image)                                         # (B, c, H/2, W/2)
        skips = [features[2], features[1], features[0], s2, s1]              # deep->shallow

        x = features[3]
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        edge_features = x
        logits = self.head(edge_features)
        return logits, edge_features
