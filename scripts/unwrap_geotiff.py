"""Stage-3: unwrap a real Sentinel-1 interferogram with the learned-cost MCF method.

Reads a HyP3 product (wrapped phase + coherence + SNAPHU unwrapped), crops a ROI
over the rupture, tiles it through the Swin+LoRA edge head to build a per-edge
cost map, solves the integer unwrapping with min-cost flow (residue-free), and
compares to the bundled SNAPHU result.

    python -m scripts.unwrap_geotiff \\
        checkpoint.resume=checkpoints/pretrain_last.pt \\
        geotiff.dir=data/S1_data/S1AA_20190704T135158_20190716T135159_VVP012_INT80_G_ueF_1138 \\
        geotiff.row0=512 geotiff.col0=1152 geotiff.size=768 logging.backend=none

Edge head is 256-trained, so the ROI is tiled (overlap-averaged logits); edge
prediction is local, so tiling is clean. MCF solves the FULL ROI for global
consistency. Use the OR-Tools backend (pip install ortools) for speed at scale.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src import dataset as dataset_module
from src import mcf
from src import utils
from src.models.unwrapper import PhaseUnwrapper


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _find(product_dir: str, tag: str) -> str:
    hits = glob.glob(os.path.join(product_dir, f"*{tag}.tif"))
    if not hits:
        raise FileNotFoundError(f"no *{tag}.tif in {product_dir!r}")
    return hits[0]


def find_roi(cfg) -> tuple:
    """Auto-locate the best size x size window: maximal deformation gradient subject
    to validity >= min_valid. Lets a new scene (e.g. Amatrice) run without hand-
    picking pixel coordinates."""
    import rasterio

    g = cfg.geotiff
    cf = g.get("corr") or _find(g.dir, "_corr")
    sf = g.get("unw") or (_find(g.dir, "_unw_phase") if g.dir else None) \
        or g.get("wrapped") or _find(g.dir, "_wrapped_phase")
    with rasterio.open(cf) as s:
        coh = s.read(1).astype(np.float64)
    with rasterio.open(sf) as s:
        score = s.read(1).astype(np.float64)
    size = int(g.size)
    stride = max(size // 2, 1)
    min_valid = float(g.get("min_valid", 0.98))
    # COHERENCE-WEIGHTED deformation: |grad(unw)| * endpoint coherence, so noisy
    # low-coherence gradients (decorrelation, not signal) don't dominate the search.
    gx = np.abs(np.diff(score, axis=0)) * np.minimum(coh[:-1, :], coh[1:, :])
    gy = np.abs(np.diff(score, axis=1)) * np.minimum(coh[:, :-1], coh[:, 1:])
    H, W = coh.shape
    best = None
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            cc = coh[r:r + size, c:c + size]
            if (cc > 0).mean() < min_valid:
                continue
            s = gx[r:r + size - 1, c:c + size].mean() + gy[r:r + size, c:c + size - 1].mean()
            if best is None or s > best[0]:
                best = (s, r, c)
    if best is None:
        raise SystemExit(f"No {size}x{size} window with validity>={min_valid}; "
                         f"lower geotiff.min_valid or geotiff.size.")
    return int(best[1]), int(best[2])


def load_product(cfg) -> dict:
    import rasterio

    g = cfg.geotiff
    wf = g.get("wrapped") or _find(g.dir, "_wrapped_phase")
    cf = g.get("corr") or _find(g.dir, "_corr")
    uf = g.get("unw") or (_find(g.dir, "_unw_phase") if g.dir else None)

    r0, c0, sz = int(g.row0), int(g.col0), int(g.size)
    win = rasterio.windows.Window(c0, r0, sz, sz)
    with rasterio.open(wf) as s:
        psi = s.read(1, window=win).astype(np.float64)
        prof = s.profile.copy()
        transform = s.window_transform(win)
    with rasterio.open(cf) as s:
        coh = s.read(1, window=win).astype(np.float64)
    unw = None
    if uf:
        with rasterio.open(uf) as s:
            unw = s.read(1, window=win).astype(np.float64)
    prof.update(height=sz, width=sz, transform=transform, count=1, dtype="float32",
                compress="deflate", tiled=False)
    for k in ("blockxsize", "blockysize"):           # avoid CPLE_IllegalArg on untiled out
        prof.pop(k, None)
    return {"psi": psi, "coh": coh, "unw": unw, "profile": prof}


# ---------------------------------------------------------------------------
# Tiled edge-head inference -> stitched per-edge logits
# ---------------------------------------------------------------------------
def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]            # avoid zeros at the ends
    return np.outer(w, w) + 1e-3


@torch.no_grad()
def edge_logits_over_roi(model, psi: np.ndarray, device, tile: int, stride: int,
                         include_grad: bool, batch: int = 8) -> np.ndarray:
    """Run the edge head over the ROI in overlapping tiles; return (6, H, W) logits."""
    H, W = psi.shape
    canvas = np.zeros((6, H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)
    win = _hann2d(tile)

    def starts(extent):
        ss = list(range(0, max(extent - tile, 0) + 1, stride))
        if ss[-1] != extent - tile:
            ss.append(max(extent - tile, 0))
        return ss

    coords = [(r, c) for r in starts(H) for c in starts(W)]
    psi_t = torch.from_numpy(psi).to(device)
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = torch.stack([psi_t[r:r + tile, c:c + tile] for r, c in chunk]).float()
        inp = utils.encode_input(tiles, include_grad=include_grad)   # (B,C,tile,tile)
        logits, _ = model.encode({"input": inp, "wrapped": tiles.unsqueeze(1)})
        logits = logits.detach().float().cpu().numpy()               # (B,6,tile,tile)
        for j, (r, c) in enumerate(chunk):
            canvas[:, r:r + tile, c:c + tile] += logits[j] * win[None]
            weight[r:r + tile, c:c + tile] += win
    return canvas / np.maximum(weight, 1e-6)[None]


# ---------------------------------------------------------------------------
# Metrics & figures
# ---------------------------------------------------------------------------
def _aligned_rmse(a, b, mask):
    d = (a - b)[mask]
    if d.size == 0:
        return float("nan")
    d = d - d.mean()
    return float(np.sqrt((d ** 2).mean()))


def _residues(phi):
    gx = phi[:, 1:] - phi[:, :-1]
    gy = phi[1:, :] - phi[:-1, :]
    curl = gx[:-1, :] + gy[:, 1:] - gx[1:, :] - gy[:, :-1]
    return int((np.round(curl / mcf.TWO_PI) != 0).sum())


def save_figure(out, psi, coh, ours, snaphu, mask, uncert=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    panels = [
        ("Wrapped phase", np.where(mask, psi, np.nan), "twilight"),
        ("Coherence", coh, "gray"),
        ("Learned-cost (ours)", np.where(mask, ours, np.nan), "jet"),
        ("SNAPHU", np.where(mask, snaphu, np.nan) if snaphu is not None else None, "jet"),
        ("Ours - SNAPHU", np.where(mask, ours - snaphu, np.nan) if snaphu is not None else None, "RdBu"),
        ("Uncertainty (nats)", np.where(mask, uncert, np.nan) if uncert is not None else None, "magma"),
    ]
    # Identical layout to the original (same figsize, colorbars, tight_layout, dpi).
    # The ONLY change is the title text size and weight. Titles stay ABOVE the
    # panels and the image panels keep their size: the panels are limited by column
    # width here, so the larger title uses the spare vertical room above each panel.
    for a, (title, img, cmap) in zip(ax.ravel(), panels):
        if img is None:
            a.axis("off"); continue
        im = a.imshow(img, cmap=cmap)
        a.set_title(title, fontsize=24, fontweight="bold"); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out + "_compare.png", dpi=130)
    print(f"  figure: {out}_compare.png")


# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not cfg.checkpoint.resume or not os.path.exists(cfg.checkpoint.resume):
        raise SystemExit(f"Need a trained edge-head checkpoint (got {cfg.checkpoint.resume!r}).")
    ckpt = torch.load(cfg.checkpoint.resume, map_location=device)
    saved = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    if saved and "model" in saved:
        cfg.model = OmegaConf.merge(cfg.model, OmegaConf.create(saved["model"]))
        if "data" in saved and "input_grad" in saved["data"]:
            cfg.data.input_grad = saved["data"]["input_grad"]
    cfg.model.backbone.in_chans = dataset_module.input_channels(cfg.data)
    model = PhaseUnwrapper(model_cfg=cfg.model, img_size=int(cfg.geotiff.tile)).to(device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 5 or len(unexpected) > 5:
        raise SystemExit(f"Checkpoint/arch mismatch (missing={len(missing)}, unexpected={len(unexpected)}).")
    if model.head_type != "edge":
        raise SystemExit(f"Stage-3 needs head_type=edge; checkpoint is {model.head_type!r}.")
    model.eval()

    clon, clat = cfg.geotiff.get("center_lon", None), cfg.geotiff.get("center_lat", None)
    if clon is not None and clat is not None:
        # center the ROI on a known epicenter (robust on low-coherence scenes where
        # gradient auto-ROI latches onto decorrelation noise).
        import rasterio
        from rasterio.warp import transform as warp_transform
        wf = cfg.geotiff.get("wrapped") or _find(cfg.geotiff.dir, "_wrapped_phase")
        with rasterio.open(wf) as s:
            xs, ys = warp_transform("EPSG:4326", s.crs, [float(clon)], [float(clat)])
            row, col = s.index(xs[0], ys[0])
        sz = int(cfg.geotiff.size)
        cfg.geotiff.row0 = int(min(max(row - sz // 2, 0), s.height - sz))
        cfg.geotiff.col0 = int(min(max(col - sz // 2, 0), s.width - sz))
        print(f"center ({clon},{clat}) -> row0={cfg.geotiff.row0} col0={cfg.geotiff.col0} size={sz}")
    elif bool(cfg.geotiff.get("auto_roi", False)):
        r0, c0 = find_roi(cfg)
        cfg.geotiff.row0, cfg.geotiff.col0 = r0, c0
        print(f"auto-ROI: row0={r0} col0={c0} size={int(cfg.geotiff.size)}")

    data = load_product(cfg)
    psi, coh, unw = data["psi"], data["coh"], data["unw"]
    mask = coh > 0                                  # nodata = coherence 0
    valid = float(mask.mean())
    print(f"ROI {psi.shape}  valid={valid:.2%}  mean coh={coh[mask].mean():.2f}")
    if valid < 0.98:
        print(f"  WARNING: {100*(1-valid):.0f}% nodata in ROI; pick a fully-valid window "
              f"for a clean result (geotiff.row0/col0/size).")

    include_grad = dataset_module.input_channels(cfg.data) == 4
    print("  tiling edge head over ROI...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = edge_logits_over_roi(model, psi, device, int(cfg.geotiff.tile),
                                  int(cfg.geotiff.stride), include_grad)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_edge = time.perf_counter() - t0
    ec = mcf.edge_costs_from_logits(logits)

    print("  solving MCF (learned / combined / unit / coherence costs)...")
    t0 = time.perf_counter()
    ours = mcf.unwrap_mcf(psi, edge_costs=ec)
    t_mcf = time.perf_counter() - t0
    print(f"  TIMING ({psi.shape[0]}x{psi.shape[1]}): edge head = {t_edge:.2f} s (Y) | "
          f"MCF solve = {t_mcf:.2f} s (Z)")
    cw = float(cfg.geotiff.get("coh_weight", 0.0))
    comb = mcf.unwrap_mcf(psi, edge_costs=ec, coherence=coh, coh_weight=cw) if cw > 0 else None
    unit = mcf.unwrap_mcf(psi)
    cohc = mcf.unwrap_mcf(psi, coherence=coh)
    uncert = mcf.uncertainty_from_logits(logits)        # per-pixel reliability (nats)
    # established baselines
    from src import baselines
    ls = baselines.ls_unwrap(psi)
    qg = baselines.quality_guided_unwrap(psi)
    res_in = int(np.abs(mcf.residues(psi)).sum())
    gold = baselines.goldstein_unwrap(psi) if res_in < 25000 else None  # greedy; impractical at high residue density

    label = os.path.basename(cfg.geotiff.dir.rstrip("/")) if cfg.geotiff.dir else "ROI"
    print(f"\n=== {label}  ROI {psi.shape[0]}x{psi.shape[1]} ===")
    print(f"  input residues          : {res_in}")
    print(f"  residues_pred (ours)     : {_residues(ours)}   (0 = residue-free)")
    if unw is not None:
        m = mask & (unw != 0)
        hi, lo = m & (coh > 0.7), m & (coh < 0.5)
        methods = [("LS (Ghiglia)", ls), ("quality-guided", qg), ("Goldstein", gold),
                   ("unit-cost MCF", unit), ("coherence-cost MCF", cohc),
                   ("LEARNED-cost MCF", ours), ("LEARNED+coh MCF", comb)]
        # The overall RMSE on high-deformation scenes is dominated by the tiny
        # low-coh tail (unrecoverable for ALL methods), so the HIGH-COH column --
        # agreement with SNAPHU where the data is reliable -- is the meaningful one.
        print(f"  agreement vs SNAPHU (aligned rmse) | high-coh(>0.7, {hi.mean():.0%}) | overall:")
        for nm, arr in methods:
            if arr is None:
                why = ("residues > 25k; unit-MCF = optimal branch-cut"
                       if nm == "Goldstein" else "coh_weight=0")
                print(f"    {nm:<20s} | {'skipped':>9s} | ({why})")
                continue
            tag = "   <- ours" if nm.startswith("LEARNED") else ""
            print(f"    {nm:<20s} | {_aligned_rmse(arr, unw, hi):9.3f} | {_aligned_rmse(arr, unw, m):7.3f}{tag}")
        d = (ours - unw)[m]; d = d - d.mean()
        print(f"  ours: low-coh(<0.5,{lo.mean():.0%})={_aligned_rmse(ours, unw, lo):.2f} | "
              f"median|ours-SNAPHU|={np.median(np.abs(d)):.3f} | within 1 fringe={np.mean(np.abs(d) < np.pi):.1%}")

    # save GeoTIFFs (unwrap + uncertainty) + figure
    import rasterio
    os.makedirs(os.path.dirname(cfg.geotiff.out) or ".", exist_ok=True)
    for suffix, arr in [("_learned_unw", ours), ("_uncertainty", uncert),
                        ("_combined_unw", comb)]:
        if arr is None:
            continue
        with rasterio.open(cfg.geotiff.out + suffix + ".tif", "w", **data["profile"]) as dst:
            dst.write(np.where(mask, arr, 0).astype("float32"), 1)
    print(f"  geotiff: {cfg.geotiff.out}_learned_unw.tif (+ _uncertainty"
          f"{', _combined_unw' if comb is not None else ''})")
    try:
        save_figure(cfg.geotiff.out, psi, coh, ours, unw, mask, uncert=uncert)
    except Exception as e:
        print(f"  (figure skipped: {e})")


if __name__ == "__main__":
    main()
