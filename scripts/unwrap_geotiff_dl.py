"""Unwrap a real Sentinel-1 ROI with a per-pixel DL baseline, for Table IV.

Companion to ``unwrap_geotiff.py`` (ours) / ``unwrap_geotiff_denet.py`` (the
hybrid): same product, same ROI convention, same output convention. Unlike
DENet -- a cost-predicting hybrid that plugs into the SAME global MCF solver
and is therefore naturally globally consistent -- the six per-pixel baselines
(dlpu_cnn, phasenet2, gradient_net, attention_unet, deeplabv3plus, unetpp)
predict ``phi_hat`` directly and have NO mechanism to reconcile predictions
across tiles. Each is trained at, and here run at, its native 256x256
resolution; tiles are placed side by side with NO post-hoc cycle-matching.

This is deliberate, not a shortcut: it is the exact structural limitation
Related Work Sec. II-B names ("large scenes must be tiled after prediction
and the tiles reconciled by a separate cycle-matching step, whereas a
solver-based method tiles the cost before a single global solve"). Table IV's
``DL (range)``/``DL slip`` columns are meant to measure that cost honestly,
not paper over it with per-tile realignment against SNAPHU (which would
silently erase the very whole-cycle disagreement being measured).

Aligned RMSE and slip (Eq. 7 of the manuscript) are computed against SNAPHU
(the only available reference on real interferograms) with ONE global
constant removed over the whole stitched canvas -- the same convention used
for every other real-data number in this paper ("aligned by removing the
global reference offset", Sec. V-A) and for "ours" in ``unwrap_geotiff.py``.

    python -m scripts.unwrap_geotiff_dl \\
        +dl_checkpoint=checkpoints/dlpu/phasenet2_last.pt +dl_name=phasenet2 \\
        geotiff.dir=data/S1_data/<PRODUCT> \\
        geotiff.row0=300 geotiff.col0=750 geotiff.size=1536 \\
        geotiff.out=./outputs/ridgecrest_phasenet2 logging.backend=none \\
        +event=Ridgecrest +summary_csv=outputs/dl_events_summary.csv

Run once per (event, baseline) pair; each run APPENDS one row to
``+summary_csv`` (default ``outputs/dl_events_summary.csv``). After the full
4-event x 6-baseline grid, ``scripts/summarize_dl_events.py`` turns that CSV
into Table IV's ``DL (range)``/``DL slip`` cells directly.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from src import dataset as dataset_module
from src import mcf
from src import utils
from scripts.unwrap_geotiff import load_product, _residues, _aligned_rmse


@torch.no_grad()
def dl_phi_over_roi(model, psi: np.ndarray, device, tile: int, include_grad: bool) -> np.ndarray:
    """Tile the ROI into non-overlapping ``tile``x``tile`` blocks (the baseline's
    native training resolution), run each independently, and place the results
    side by side -- NO cross-tile reconciliation (see module docstring)."""
    H, W = psi.shape
    canvas = np.zeros((H, W), dtype=np.float64)

    def starts(extent):
        ss = list(range(0, max(extent - tile, 0) + 1, tile))
        if not ss or ss[-1] != extent - tile:
            ss.append(max(extent - tile, 0))
        return ss

    coords = [(r, c) for r in starts(H) for c in starts(W)]
    psi_t = torch.from_numpy(psi).float().to(device)
    for r, c in coords:
        chunk = psi_t[r:r + tile, c:c + tile].unsqueeze(0)          # (1,tile,tile)
        inp = utils.encode_input(chunk, include_grad=include_grad)   # (1,C,tile,tile)
        batch = {"input": inp, "wrapped": chunk.unsqueeze(1)}
        out = model(batch, hard=True)
        canvas[r:r + tile, c:c + tile] = out["phi_hat"][0, 0].detach().float().cpu().numpy()
    return canvas


def _slip(pred, ref, mask):
    """Eq. 7 cycle-slip rate of ``pred`` against ``ref`` (SNAPHU here, no GT on
    real events), one global constant removed over ``mask``."""
    d = (pred - ref)[mask]
    d = d - d.mean()
    r = np.round(d / mcf.TWO_PI)
    return float((r != 0).mean())


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = cfg.get("dl_checkpoint", None)
    name = str(cfg.get("dl_name", ""))
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise SystemExit(f"Need +dl_checkpoint=<path> (got {ckpt_path!r}).")
    from ext_dl.train_dl import BUILDERS
    if name not in BUILDERS or name == "denet":
        raise SystemExit(f"+dl_name={name!r} must be one of the 6 per-pixel baselines "
                         f"{[n for n in BUILDERS if n != 'denet']} (use unwrap_geotiff_denet.py "
                         f"for denet).")
    in_chans = dataset_module.input_channels(cfg.data)
    model = BUILDERS[name](cfg, in_chans).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded {name}: {ckpt_path}")

    data = load_product(cfg)
    psi, coh, unw = data["psi"], data["coh"], data["unw"]
    mask = coh > 0
    print(f"ROI {psi.shape}  valid={mask.mean():.2%}  mean coh={coh[mask].mean():.2f}")
    if unw is None:
        raise SystemExit("Need SNAPHU (*_unw_phase.tif) in the product as the real-data reference.")

    tile = int(cfg.geotiff.get("tile", 256))
    include_grad = dataset_module.input_channels(cfg.data) == 4
    t0 = time.perf_counter()
    phi_dl = dl_phi_over_roi(model, psi, device, tile, include_grad)
    t_fwd = time.perf_counter() - t0
    print(f"  TIMING ({psi.shape[0]}x{psi.shape[1]}, tile={tile}): forward = {t_fwd:.2f} s")
    print(f"  residues_pred ({name}) : {_residues(phi_dl)}   (0 = residue-free per tile; "
          f"tile SEAMS are expected to carry residues -- that IS the phenomenon)")

    m = mask & (unw != 0)
    hi = m & (coh > 0.7)
    agree_hi = _aligned_rmse(phi_dl, unw, hi)
    agree_all = _aligned_rmse(phi_dl, unw, m)
    slip_hi = _slip(phi_dl, unw, hi)
    slip_all = _slip(phi_dl, unw, m)
    print(f"  agreement vs SNAPHU (aligned rmse) | high-coh(>0.7, {hi.mean():.0%}) | overall:")
    print(f"    {name:<20s} | {agree_hi:9.3f} | {agree_all:7.3f}")
    print(f"  slip vs SNAPHU (Eq.7)              | high-coh                | overall:")
    print(f"    {name:<20s} | {slip_hi*100:8.1f}% | {slip_all*100:6.1f}%")

    import rasterio
    os.makedirs(os.path.dirname(cfg.geotiff.out) or ".", exist_ok=True)
    out_path = cfg.geotiff.out + "_unw.tif"
    with rasterio.open(out_path, "w", **data["profile"]) as dst:
        dst.write(np.where(mask, phi_dl, 0).astype("float32"), 1)
    print(f"  geotiff: {out_path}")

    event = str(cfg.get("event", os.path.basename(cfg.geotiff.dir.rstrip("/"))))
    csv_path = str(cfg.get("summary_csv", "outputs/dl_events_summary.csv"))
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["event", "method", "agree_hi_rad", "agree_all_rad",
                       "slip_hi_pct", "slip_all_pct", "residues"])
        w.writerow([event, name, f"{agree_hi:.3f}", f"{agree_all:.3f}",
                   f"{slip_hi*100:.1f}", f"{slip_all*100:.1f}", _residues(phi_dl)])
    print(f"  appended to {csv_path}")


if __name__ == "__main__":
    main()
