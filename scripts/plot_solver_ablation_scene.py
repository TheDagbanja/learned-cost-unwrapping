"""One-scene qualitative companion to the aggregate solver-ablation CSV."""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from src import dataset as dataset_module, mcf, mcf_infer, utils
from ext_dl.wls import integrate_gradients
from scripts.eval_learned_mcf import _metrics
from scripts.ext_experiment_common import load_edge_model, probabilities


def _interior(x, horizontal):
    return x[:, :-1] if horizontal else x[:-1, :]


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ext = cfg.get("ext", {})
    index = int(ext.get("scene_index", -1))
    if index < 0:
        raise ValueError("Set +ext.scene_index to a predeclared test scene")
    device = utils.get_device(cfg.device)
    model = load_edge_model(cfg, device)
    loader = dataset_module.build_dataloader(cfg.data, split=str(cfg.eval.split),
                                               shuffle=False, device=device)
    offset = 0
    chosen = None
    for batch in loader:
        size = len(batch["wrapped"])
        if offset <= index < offset + size:
            chosen = (batch, index - offset)
            break
        offset += size
    if chosen is None:
        raise IndexError(f"scene_index={index} exceeds split={cfg.eval.split}")
    batch, b = chosen
    batch = utils.move_batch_to_device(batch, device)
    with torch.no_grad():
        signed, unit = mcf_infer.unwrap_batch(model, batch, also_unit=True)
        ph, pv = probabilities(model, batch)
    ph = ph[b].detach().cpu().numpy()
    pv = pv[b].detach().cpu().numpy()
    psi = batch["wrapped"][b, 0].detach().cpu().numpy()
    gt = batch["absolute"][b, 0].detach().cpu().numpy()
    wx = -np.log(np.clip(_interior(ph[0] + ph[-1], True), 1e-8, 1))
    wy = -np.log(np.clip(_interior(pv[0] + pv[-1], False), 1e-8, 1))
    unsigned = mcf.unwrap_mcf(psi.astype(np.float64), wx=wx, wy=wy)
    kx = _interior(ph.argmax(0) - ph.shape[0] // 2, True)
    ky = _interior(pv.argmax(0) - pv.shape[0] // 2, False)
    gx = mcf.wrap(psi[:, 1:] - psi[:, :-1]) + mcf.TWO_PI * kx
    gy = mcf.wrap(psi[1:, :] - psi[:-1, :]) + mcf.TWO_PI * ky
    local = integrate_gradients(gx, gy)
    names = ("Signed learned + MCF", "Unsigned learned + MCF",
             "Unit + MCF", "Local argmax + LS")
    preds = (signed[b], unsigned, unit[b], local)
    errors = []
    for pred in preds:
        diff = pred - gt
        errors.append(diff - diff.mean())
    cycles = [np.rint(error / mcf.TWO_PI) for error in errors]
    phase_limits = np.percentile(gt, [1, 99])
    error_limit = max(1.0, float(np.percentile(np.abs(np.stack(errors)), 99)))
    cycle_limit = max(1.0, float(np.percentile(np.abs(np.stack(cycles)), 99)))
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(3, 5, figsize=(16, 9), layout="constrained")
    reference = (
        (gt, "GT absolute phase", "viridis", phase_limits[0], phase_limits[1]),
        (psi, "Wrapped observation", "twilight", -np.pi, np.pi),
        (np.rint(((gt[:, 1:] - gt[:, :-1])
                  - mcf.wrap(psi[:, 1:] - psi[:, :-1])) / mcf.TWO_PI),
         "GT horizontal ambiguity", "coolwarm", -1, 1),
    )
    for row, (array, title, cmap, lo, hi) in enumerate(reference):
        ax = axes[row, 0]
        im = ax.imshow(array, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.015)
    for col, (name, pred, error, cycle) in enumerate(zip(names, preds, errors, cycles), 1):
        aligned = gt + error
        rmse, slip, _, _ = _metrics(pred, gt, psi)
        for row, (array, cmap, lo, hi, title) in enumerate((
            (aligned, "viridis", phase_limits[0], phase_limits[1], f"{name}\nRMSE={rmse:.2f} rad"),
            (error, "coolwarm", -error_limit, error_limit, "Aligned phase error"),
            (cycle, "coolwarm", -cycle_limit, cycle_limit, f"Cycle error; {100*slip:.1f}% slips"),
        )):
            ax = axes[row, col]
            im = ax.imshow(array, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.015)
    out_dir = Path(str(ext.get("out_dir", "results/paper_figures")))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fig_solver_ablation_scene_{cfg.eval.split}_{index:04d}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
