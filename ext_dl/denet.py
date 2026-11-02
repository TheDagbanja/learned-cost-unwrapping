"""DENet -- learned phase-discontinuity costs for a min-cost-flow solver.

Reimplementation of

    Z. Wu, T. Wang, Y. Wang, R. Wang and D. Ge, "Deep-learning based phase
    discontinuity prediction for two-dimensional phase unwrapping of SAR
    interferograms", IEEE TGRS, 2021, doi:10.1109/TGRS.2021.3121906

This is the *nearest neighbour* of our method: it also keeps the combinatorial
solver and learns its cost, so it is the baseline that actually tests the
"keep the solver, learn the cost" thesis (the per-pixel networks in this package
replace the solver instead).

Faithful to the published design:
  * multi-channel input -- interferogram, range/azimuth phase gradients, residues;
  * a *branching* trunk, one branch preserving detail at full resolution and one
    enlarging the receptive field for context, fused before the head;
  * a single network predicting discontinuities in BOTH directions at once, as
    two sigmoid maps (range, azimuth).

The one structural difference from our method is the quantity predicted, and it
is the point of the comparison. DENet predicts an UNSIGNED probability that an
edge carries a discontinuity, which yields a symmetric cost (a cut-*location*
prior). We predict a distribution over SIGNED ambiguity classes {-1,0,+1} and
use the per-class negative log-likelihood, which yields a directional cost
(c+ != c-, a cut-*value* prior). See ``mcf.unwrap_mcf``: symmetric costs set
cpx=cnx and therefore discard the sign of the cut.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ASPP, ConvBlock

TWO_PI = 2.0 * torch.pi


def denet_inputs(psi: torch.Tensor) -> torch.Tensor:
    """(B,H,W) wrapped phase -> (B,5,H,W): cos, sin, d_range, d_azimuth, residues.

    Matches the paper's multi-channel input (interferogram, range/azimuthal phase
    gradients, residues map). Gradients are wrapped differences, padded to full
    size; the residue map is the rounded loop curl.
    """
    if psi.dim() == 4:
        psi = psi[:, 0]
    wrap = lambda x: (x + torch.pi) % TWO_PI - torch.pi
    dx = wrap(psi[:, :, 1:] - psi[:, :, :-1])          # range      (B,H,W-1)
    dy = wrap(psi[:, 1:, :] - psi[:, :-1, :])          # azimuth    (B,H-1,W)
    dxp = F.pad(dx, (0, 1))
    dyp = F.pad(dy, (0, 0, 0, 1))
    curl = dx[:, :-1, :] + dy[:, :, 1:] - dx[:, 1:, :] - dy[:, :, :-1]
    res = F.pad(torch.round(curl / TWO_PI), (0, 1, 0, 1))
    return torch.stack([torch.cos(psi), torch.sin(psi), dxp / torch.pi,
                        dyp / torch.pi, res], dim=1)


class DENetBaseline(nn.Module):
    """Branching discontinuity predictor; returns two sigmoid maps (range, azimuth)."""

    def __init__(self, in_chans: int = 5, width: int = 48) -> None:
        super().__init__()
        self.stem = ConvBlock(in_chans, width)
        # MAIN network: no down-sampling anywhere, every feature map at full
        # resolution -- this is what preserves the thin, pixel-wide cut structures.
        self.detail = nn.Sequential(ConvBlock(width, width), ConvBlock(width, width))
        # AUXILIARY network: dilated-convolution cascade + spatial pyramid pooling,
        # which enlarges the field of view and fuses multi-scale context WITHOUT
        # down-sampling (the paper cites atrous convolution / DeepLab for this).
        self.cascade = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.aspp = ASPP(width, width)
        self.fuse = ConvBlock(2 * width, width)
        self.head = nn.Conv2d(width, 2, kernel_size=1)     # logits: range, azimuth

    def forward(self, batch: Dict[str, Any], hard: bool = False) -> Dict[str, torch.Tensor]:
        psi = batch["wrapped"]
        psi2 = psi[:, 0] if psi.dim() == 4 else psi
        x = denet_inputs(psi2)
        f = self.stem(x)
        d = self.detail(f)                     # main: detail, full resolution
        c = self.aspp(self.cascade(f))         # auxiliary: dilated cascade + pyramid
        logits = self.head(self.fuse(torch.cat([d, c], dim=1)))     # (B,2,H,W)
        return {"disc_logits": logits, "p_disc": torch.sigmoid(logits)}

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": total, "lora": 0}


def denet_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, Any],
               k_max: int = 1) -> Dict[str, torch.Tensor]:
    """Binary cross-entropy on |k_e| > 0, per direction.

    Labels ``k_x``/``k_y`` are stored as class indices (``k + k_max``), so an edge
    is a discontinuity when the index differs from the centre class. Positives are
    rare (~1 % of edges), so the loss is pos-weighted.
    """
    logits = outputs["disc_logits"]
    H, W = logits.shape[-2:]
    kx = batch["k_x"].long()
    ky = batch["k_y"].long()
    tx = (kx != k_max).float()
    ty = (ky != k_max).float()
    if tx.shape[-2:] != (H, W):
        tx = F.pad(tx, (0, W - tx.shape[-1], 0, H - tx.shape[-2]))
    if ty.shape[-2:] != (H, W):
        ty = F.pad(ty, (0, W - ty.shape[-1], 0, H - ty.shape[-2]))
    target = torch.stack([tx, ty], dim=1)
    pos = target.mean().clamp_min(1e-6)
    w = ((1.0 - pos) / pos).clamp(1.0, 100.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=w)
    return {"total": bce, "bce": bce.detach()}


def denet_edge_costs(p_disc, scale: float = 1000.0):
    """Discontinuity probability -> SYMMETRIC per-edge MCF costs.

    The published transform, verbatim: ``c_e = (1 - output) * 1000`` (Wu et al.,
    IGARSS 2021, Sec. 2.3), applied identically to both flow directions because
    the prediction is unsigned. A likely discontinuity is cheap to cut; a
    confident continuity is expensive. In the original this is handed to SNAPHU's
    MCF via ``-w``; here it goes to the same solver our method uses, so the
    comparison isolates the cost and holds the solver fixed.

    Note this is LINEAR in the probability, unlike our own negative-log-likelihood
    cost -- that difference in cost shaping is part of what is being compared, so
    we keep their formula rather than substituting ours.
    """
    import numpy as np
    p = np.asarray(p_disc, dtype=np.float64)          # (2,H,W)
    c = (1.0 - p) * float(scale)
    return c[0][:, :-1], c[1][:-1, :]


if __name__ == "__main__":                            # smoke test
    m = DENetBaseline()
    b = {"wrapped": torch.randn(2, 1, 64, 64),
         "k_x": torch.randint(0, 3, (2, 64, 63)), "k_y": torch.randint(0, 3, (2, 63, 64))}
    o = m(b)
    l = denet_loss(o, b)
    l["total"].backward()
    n = sum(p.numel() for p in m.parameters())
    print(f"DENet ok: p_disc {tuple(o['p_disc'].shape)}, loss {l['total'].item():.4f}, "
          f"params {n:,}")
