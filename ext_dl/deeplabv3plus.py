"""DeepLabV3+ baseline (NEW) -- Chen et al. 2018, applied to PU.

A dilated-convolution encoder + Atrous Spatial Pyramid Pooling (ASPP) + a shallow
decoder that fuses a low-level (1/4-resolution) feature map -- the segmentation
architecture, here framed as per-pixel wrap-count *classification* over
``[-k_max, k_max]`` (like :mod:`phasenet2`, but with DeepLab's encoder/decoder
instead of a symmetric U-Net). Reuses the shared :class:`ASPP` module.

Output schema matches the other DL baselines so the harness and
:func:`src.evaluate.evaluate_outputs` treat it identically.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from .common import ASPP, ConvBlock
from .phasenet2 import phasenet2_loss   # reuse the wrap-count cross-entropy loss


class DeepLabV3PlusBaseline(nn.Module):
    """Dilated encoder + ASPP + low-level-fusion decoder, wrap-count classification."""

    def __init__(self, in_chans: int = 4, width: int = 64, k_max: int = 16,
                 aspp_rates: Sequence[int] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.k_max = k_max
        self.stem = ConvBlock(in_chans, width)                     # full res
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(width, width * 2))       # 1/2 (low-level)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(width * 2, width * 4))   # 1/4
        self.aspp = ASPP(width * 4, width * 2, rates=aspp_rates)
        self.low_proj = nn.Sequential(nn.Conv2d(width * 2, 48, 1, bias=False),
                                      nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            ConvBlock(width * 2 + 48, width * 2), nn.Conv2d(width * 2, 2 * k_max + 1, 1))

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False) -> Dict[str, torch.Tensor]:
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        s0 = self.stem(batch["input"])            # full res
        low = self.down1(s0)                       # 1/2 low-level features (width*2)
        d2 = self.down2(low)                        # 1/4
        x = self.aspp(d2)                          # 1/4
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, self.low_proj(low)), dim=1)
        x = self.decoder(x)                        # 1/2
        logits = F.interpolate(x, size=psi.shape[-2:], mode="bilinear", align_corners=False)

        probs = torch.softmax(logits, dim=1)
        kvals = torch.arange(-self.k_max, self.k_max + 1, device=logits.device,
                             dtype=logits.dtype).view(1, -1, 1, 1)
        ek = (probs * kvals).sum(dim=1)
        k = torch.round(ek) if hard else ek
        phi_hat = psi + utils.TWO_PI * k
        g_x = utils.pad_edge_x(utils.edge_diff_x(phi_hat), 0.0)
        g_y = utils.pad_edge_y(utils.edge_diff_y(phi_hat), 0.0)
        return {"k_logits": logits, "k_reg": ek.unsqueeze(1), "k_pred": torch.round(ek).unsqueeze(1),
                "phi_hat": phi_hat.unsqueeze(1), "g_x": g_x, "g_y": g_y}

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "lora": 0}


def deeplabv3plus_loss(outputs, batch, k_max: int = 16):
    return phasenet2_loss(outputs, batch, k_max=k_max)


def _smoke_test() -> None:
    torch.manual_seed(0)
    b, h, w = 2, 128, 128
    psi = (torch.rand(b, h, w) * 2 - 1) * utils.PI
    phi = psi + utils.TWO_PI * torch.randint(-3, 4, (b, h, w)).float()
    batch = {"input": utils.encode_input(psi, include_grad=True), "wrapped": psi.unsqueeze(1),
             "k_pixel": ((phi - psi) / utils.TWO_PI).round().unsqueeze(1)}
    model = DeepLabV3PlusBaseline(in_chans=4, k_max=16)
    out = model(batch)
    terms = deeplabv3plus_loss(out, batch, k_max=16)
    terms["total"].backward()
    print(f"[smoke] DeepLabV3Plus params={model.count_parameters()['total']:,} "
          f"phi_hat {tuple(out['phi_hat'].shape)} loss={terms['total'].item():.4f}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    _smoke_test()
