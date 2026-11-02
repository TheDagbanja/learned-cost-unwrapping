"""Unwrap a real Sentinel-1 ROI with the DENet baseline, for the GPS comparison.

Companion to ``scripts/unwrap_geotiff.py`` (our learned-cost method): same HyP3
product, same ROI, same output convention, so the two GeoTIFFs are pixel-
aligned and both readable by ``scripts/gps_validate.py --method NAME=PATH``.

DENet is a plain fully-convolutional net (no fixed input size like the Swin
edge head), but its ASPP module concatenates several full-resolution branches,
so a single forward pass over a large ROI (e.g. Ridgecrest's 1536x1536) is
memory-heavy -- comparable in spirit to the edge head, though DENet's
receptive field (a dilation-2/4/8 cascade + ASPP, no attention) is much
smaller, so a modest overlap is enough to blend tile seams away. We therefore
tile it the same way ``unwrap_geotiff.py`` tiles the Swin edge head: run
overlapping ``geotiff.tile``-sized windows, Hann-window-average the raw
discontinuity LOGITS (matching how the edge head averages logits, not
probabilities), then take a single sigmoid over the assembled canvas. This
caps peak VRAM at one tile's worth regardless of ROI size.

The discontinuity probability becomes a symmetric MCF cost
(``ext_dl.denet.denet_edge_costs``, the paper's own ``(1-p)*1000`` transform)
fed to the identical min-cost-flow solver our method uses, so the comparison
isolates the cost model with the solver held fixed.

    python -m scripts.unwrap_geotiff_denet \\
        +denet_checkpoint=checkpoints/dlpu/denet_last.pt \\
        geotiff.dir=data/S1_data/S1AA_20190704T135158_20190716T135159_VVP012_INT80_G_ueF_1138 \\
        geotiff.row0=300 geotiff.col0=750 geotiff.size=1536 \\
        geotiff.out=./outputs/ridgecrest_denet logging.backend=none

geotiff.tile/geotiff.stride (default 256/128, same keys the edge head uses)
control the tiling; lower geotiff.tile further if VRAM is still tight.

Then feed both GeoTIFFs to the GPS scatter:
    python -m scripts.gps_validate \\
        --product data/S1_data/S1AA_..._1138 \\
        --ours outputs/ridgecrest_learned_unw.tif \\
        --method denet=outputs/ridgecrest_denet_unw.tif \\
        --gps ridgecrest_gps_M6.4.txt ridgecrest_gps_M7.1.txt \\
        --out outputs/gps_with_denet
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from src import mcf
from scripts.unwrap_geotiff import load_product, _residues, _aligned_rmse, _hann2d


@torch.no_grad()
def denet_logits_over_roi(model, psi: np.ndarray, device, tile: int, stride: int,
                          batch: int = 8) -> np.ndarray:
    """Run DENet over the ROI in overlapping tiles; return (2, H, W) raw logits.

    Mirrors ``unwrap_geotiff.edge_logits_over_roi``: Hann-window-averaged
    logits (not probabilities), one sigmoid taken after assembly.
    """
    H, W = psi.shape
    canvas = np.zeros((2, H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)
    win = _hann2d(tile)

    def starts(extent):
        ss = list(range(0, max(extent - tile, 0) + 1, stride))
        if ss[-1] != extent - tile:
            ss.append(max(extent - tile, 0))
        return ss

    coords = [(r, c) for r in starts(H) for c in starts(W)]
    psi_t = torch.from_numpy(psi).float().to(device)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = torch.stack([psi_t[r:r + tile, c:c + tile] for r, c in chunk]).unsqueeze(1)
        out = model({"wrapped": tiles})
        logits = out["disc_logits"].detach().float().cpu().numpy()   # (B,2,tile,tile)
        for j, (r, c) in enumerate(chunk):
            canvas[:, r:r + tile, c:c + tile] += logits[j] * win[None]
            weight[r:r + tile, c:c + tile] += win
    return canvas / np.maximum(weight, 1e-6)[None]


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg.get("denet_checkpoint", None)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise SystemExit(f"Need +denet_checkpoint=<path to denet_last.pt> (got {ckpt_path!r}).")

    from ext_dl.denet import DENetBaseline, denet_edge_costs
    model = DENetBaseline().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded DENet: {ckpt_path}  params={model.count_parameters()['total']:,}")

    data = load_product(cfg)
    psi, coh, unw = data["psi"], data["coh"], data["unw"]
    mask = coh > 0
    print(f"ROI {psi.shape}  valid={mask.mean():.2%}  mean coh={coh[mask].mean():.2f}")

    tile = int(cfg.geotiff.get("tile", 256))
    stride = int(cfg.geotiff.get("stride", 128))
    print(f"  tiling DENet over ROI (tile={tile}, stride={stride})...")
    t0 = time.perf_counter()
    logits = denet_logits_over_roi(model, psi, device, tile, stride)
    t_fwd = time.perf_counter() - t0
    p_disc = 1.0 / (1.0 + np.exp(-logits))                      # sigmoid, (2,H,W)
    wx, wy = denet_edge_costs(p_disc)
    t0 = time.perf_counter()
    denet_unw = mcf.unwrap_mcf(psi, wx=wx, wy=wy)
    t_mcf = time.perf_counter() - t0
    print(f"  TIMING ({psi.shape[0]}x{psi.shape[1]}): forward = {t_fwd:.2f} s | MCF solve = {t_mcf:.2f} s")
    print(f"  residues_pred (denet) : {_residues(denet_unw)}   (0 = residue-free)")

    if unw is not None:
        m = mask & (unw != 0)
        hi = m & (coh > 0.7)
        print(f"  agreement vs SNAPHU (aligned rmse) | high-coh(>0.7, {hi.mean():.0%}) | overall:")
        print(f"    {'DENet':<20s} | {_aligned_rmse(denet_unw, unw, hi):9.3f} | "
              f"{_aligned_rmse(denet_unw, unw, m):7.3f}")

    import rasterio
    os.makedirs(os.path.dirname(cfg.geotiff.out) or ".", exist_ok=True)
    out_path = cfg.geotiff.out + "_unw.tif"
    with rasterio.open(out_path, "w", **data["profile"]) as dst:
        dst.write(np.where(mask, denet_unw, 0).astype("float32"), 1)
    print(f"  geotiff: {out_path}")


if __name__ == "__main__":
    main()
