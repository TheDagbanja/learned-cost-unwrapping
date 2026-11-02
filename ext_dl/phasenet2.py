"""PhaseNet 2.0 baseline (NEW, this work) -- proposal Table 3.

Reproduces the class of per-pixel wrap-count *classification* deep-learning
unwrappers (PhaseNet-style: Zhang et al. 2019 and successors), with the
wider multi-scale receptive field ("2.0") an ASPP module adds over a plain
encoder-decoder -- dense fringe bands need a small receptive field, broad
smooth regions need a large one, and no single dilation rate serves both.

Frames unwrapping as classification of the per-pixel wrap count over
``[-k_max, k_max]`` (distinct from :mod:`src.baselines_dl.dlpu_cnn`'s direct
regression), trained with plain cross-entropy against the ground-truth wrap
count -- no physics-guided losses, matching how PhaseNet-style methods are
normally trained end-to-end as a segmentation problem.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from .common import ASPP, ConvBlock


class PhaseNet2Baseline(nn.Module):
    """Dilated-conv / ASPP encoder + light decoder, wrap-count classification head."""

    def __init__(self, in_chans: int = 4, width: int = 64, k_max: int = 16,
                aspp_rates: Sequence[int] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.k_max = k_max
        self.stem = ConvBlock(in_chans, width)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(width, width * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(width * 2, width * 4))
        self.aspp = ASPP(width * 4, width * 4, rates=aspp_rates)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = ConvBlock(width * 2, width)
        self.head = nn.Conv2d(width, 2 * k_max + 1, kernel_size=1)

    def forward(self, batch: Dict[str, torch.Tensor], hard: bool = False) -> Dict[str, torch.Tensor]:
        psi = batch["wrapped"]
        if psi.dim() == 4:
            psi = psi.squeeze(1)
        s0 = self.stem(batch["input"])
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s2 = self.aspp(s2)
        u1 = self.up2(s2)
        if u1.shape[-2:] != s1.shape[-2:]:
            u1 = F.interpolate(u1, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec2(torch.cat((u1, s1), dim=1))
        u0 = self.up1(d1)
        if u0.shape[-2:] != s0.shape[-2:]:
            u0 = F.interpolate(u0, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        d0 = self.dec1(torch.cat((u0, s0), dim=1))
        logits = self.head(d0)                            # (B, 2k+1, H, W)

        probs = torch.softmax(logits, dim=1)
        kvals = torch.arange(-self.k_max, self.k_max + 1, device=logits.device,
                             dtype=logits.dtype).view(1, -1, 1, 1)
        ek = (probs * kvals).sum(dim=1)                    # soft expected K
        k = torch.round(ek) if hard else ek
        phi_hat = psi + utils.TWO_PI * k
        g_x = utils.pad_edge_x(utils.edge_diff_x(phi_hat), 0.0)
        g_y = utils.pad_edge_y(utils.edge_diff_y(phi_hat), 0.0)
        return {
            "k_logits": logits, "k_reg": ek.unsqueeze(1), "k_pred": torch.round(ek).unsqueeze(1),
            "phi_hat": phi_hat.unsqueeze(1), "g_x": g_x, "g_y": g_y,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "lora": 0}


def phasenet2_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any],
                   k_max: int = 16) -> Dict[str, torch.Tensor]:
    """Cross-entropy on the wrap-count class (centred so the unidentifiable
    global offset doesn't blow the label range)."""
    if "k_pixel" not in batch:
        raise ValueError("PhaseNet2 baseline needs 'k_pixel' labels (ground-truth wrap count).")
    logits = outputs["k_logits"]
    kp = batch["k_pixel"]
    center = torch.round(kp.mean(dim=(-2, -1), keepdim=True))
    cls = (kp - center + k_max).clamp(0, 2 * k_max).long().squeeze(1)
    ce = F.cross_entropy(logits, cls)
    return {"ce": ce, "total": ce}


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
    model = PhaseNet2Baseline(in_chans=4, k_max=16)
    out = model(batch)
    terms = phasenet2_loss(out, batch, k_max=16)
    terms["total"].backward()
    print(f"[smoke] PhaseNet2Baseline params={model.count_parameters()['total']:,} "
          f"phi_hat {tuple(out['phi_hat'].shape)} loss={terms['total'].item():.4f}")
    print("[smoke] PASSED")


if __name__ == "__main__":
    _smoke_test()
