"""Phase 2 -- cycle-slip vs alignment audit (ADDITIVE; mutates nothing).

Hypothesis under test: a cost-MCF can place a full-cycle slip (a connected region
off by an integer number of 2*pi) that is residue-free but wrong; quality-guided's
greedy path avoids this on smooth data; and because the harness aligns by
subtracting the MEAN residual, one slipped patch drags the mean and inflates RMSE
disproportionately.

Separates "did it slip" from "how noisy is the non-slipped field":
  slip_rate  = fraction of pixels whose round(residual / 2pi) != 0
  n_regions  = connected components of the slipped mask
  rmse_mean  = the harness's current metric (mean-aligned)
  med|res|   = median absolute residual
  rmse_slipfree = RMSE after removing the integer 2*pi offsets (noise only)
plus alignment sensitivity: mean vs median vs single-reference-pixel.

    python -m audit.phase2_slips                       # model-free rows (local)
    python -m audit.phase2_slips --checkpoint PATH     # adds the learned row (server)
"""
import argparse
import glob
import os

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import label as cc_label

from src import baselines, mcf

TWO_PI = 2.0 * np.pi
ROOT = "data/InSAR-DPLU/InSAR-DLPU/InSAR-DLPU"
WDIR = os.path.join(ROOT, "test_wrapped_real")
ADIR = os.path.join(ROOT, "test_absolute_real")


def load(path, key):
    m = loadmat(path)
    ks = [k for k in m if not k.startswith("__")]
    return np.asarray(m[key if key in m else ks[0]], dtype=np.float64)


def gt_gradient_costs(psi, phi):                 # same prior as the harness
    dxw = mcf.wrap(psi[:, 1:] - psi[:, :-1])
    dyw = mcf.wrap(psi[1:, :] - psi[:-1, :])
    kx = np.round(((phi[:, 1:] - phi[:, :-1]) - dxw) / TWO_PI)
    ky = np.round(((phi[1:, :] - phi[:-1, :]) - dyw) / TWO_PI)
    return np.where(kx != 0, 0.01, 1.0), np.where(ky != 0, 0.01, 1.0)


def analyse(pred, gt):
    """All Phase-2 quantities for one (pred, gt) pair."""
    raw = pred - gt
    d = raw - raw.mean()                          # the harness's alignment
    r = np.round(d / TWO_PI)
    slipped = r != 0
    _, n_reg = cc_label(slipped)
    d_slipfree = d - TWO_PI * r                   # integer-corrected residual
    return {
        "rmse_mean": float(np.sqrt((d ** 2).mean())),
        "rmse_median": float(np.sqrt(((raw - np.median(raw)) ** 2).mean())),
        "rmse_refpix": float(np.sqrt(((raw - raw[raw.shape[0] // 2, raw.shape[1] // 2]) ** 2).mean())),
        "med_abs": float(np.median(np.abs(d))),
        "slip_rate": float(slipped.mean()),
        "n_regions": int(n_reg),
        "rmse_slipfree": float(np.sqrt((d_slipfree ** 2).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--checkpoint", default=None, help="edge head -> adds the learned row")
    args = ap.parse_args()

    model = device = None
    if args.checkpoint:
        import torch
        from omegaconf import OmegaConf
        from src import dataset as dm
        from src.models.unwrapper import PhaseUnwrapper
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.checkpoint, map_location=device)
        cfg = OmegaConf.create(ck["config"])
        cfg.model.backbone.in_chans = dm.input_channels(cfg.data)
        model = PhaseUnwrapper(model_cfg=cfg.model, img_size=256).to(device)
        model.load_state_dict(ck.get("model", ck), strict=False)
        model.eval()

    names = ["floor", "quality_guided", "weighted_ls", "goldstein", "unit_mcf", "gt_grad_cost"]
    if model is not None:
        names.insert(1, "ours_learned")
    acc = {k: [] for k in names}

    for wp in sorted(glob.glob(os.path.join(WDIR, "*.mat")))[:args.n]:
        base = os.path.basename(wp)
        psi = load(wp, "input")
        phi = load(os.path.join(ADIR, base), "output")
        if psi.shape != phi.shape:
            continue
        preds = {
            "floor": psi + TWO_PI * np.round((phi - psi) / TWO_PI),
            "quality_guided": baselines.quality_guided_unwrap(psi),
            "weighted_ls": baselines.ls_unwrap(psi),
            "goldstein": baselines.goldstein_unwrap(psi),
            "unit_mcf": mcf.unwrap_mcf(psi),
            "gt_grad_cost": mcf.unwrap_mcf(psi, *gt_gradient_costs(psi, phi)),
        }
        if model is not None:
            import torch
            from src import utils
            with torch.no_grad():
                psi_t = torch.from_numpy(psi).float()
                batch = {"input": utils.encode_input(psi_t, include_grad=True).unsqueeze(0),
                         "wrapped": psi_t.unsqueeze(0).unsqueeze(0)}
                batch = utils.move_batch_to_device(batch, device)
                lo, _ = model.encode(batch)
                lo = lo.detach().float().cpu().numpy()[0]
            preds["ours_learned"] = mcf.unwrap_mcf(psi, edge_costs=mcf.edge_costs_from_logits(lo))
        for k in names:
            acc[k].append(analyse(preds[k], phi))

    n = len(acc["floor"])
    m = lambda k, f: np.mean([r[f] for r in acc[k]])
    print(f"\n=== Phase 2: cycle-slips vs alignment, test_real ({n} scenes) ===\n")
    print(f"{'method':16s}{'RMSE(mean)':>11}{'med|res|':>10}{'slip%':>8}{'#regions':>10}{'RMSE_slipfree':>15}")
    for k in names:
        print(f"{k:16s}{m(k,'rmse_mean'):>11.3f}{m(k,'med_abs'):>10.3f}"
              f"{100*m(k,'slip_rate'):>7.1f}%{m(k,'n_regions'):>10.1f}{m(k,'rmse_slipfree'):>15.3f}")
    print(f"\n--- alignment sensitivity (same residuals, 3 offset rules) ---")
    print(f"{'method':16s}{'mean-sub':>11}{'median-sub':>12}{'ref-pixel':>11}")
    for k in names:
        print(f"{k:16s}{m(k,'rmse_mean'):>11.3f}{m(k,'rmse_median'):>12.3f}{m(k,'rmse_refpix'):>11.3f}")


if __name__ == "__main__":
    main()
