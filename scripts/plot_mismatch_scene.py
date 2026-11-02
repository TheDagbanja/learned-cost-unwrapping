"""Reconstruct one protocol-v3 scene and export mechanism/audit vector PDFs.

This runs only one scene and one condition. It checks its four RMSEs against
the frozen per-scene CSV so an altered checkpoint or observation is not silently
presented as a figure from the completed sweep.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from src import baselines, dataset as dataset_module, mcf, mcf_infer, utils
from scripts.eval_cost_mismatch import METHODS, _subset_batch, reencode
from scripts.eval_learned_mcf import _metrics
from scripts.ext_experiment_common import load_edge_model, probabilities
from scripts.mismatch_protocol import coherence_field, make_scene_latents, noisy_wrapped


def _find_scene(cfg, index, device):
    loader = dataset_module.build_dataloader(cfg.data, split=str(cfg.eval.split),
                                               shuffle=False, device=device)
    offset = 0
    for batch in loader:
        size = len(batch["wrapped"])
        if offset <= index < offset + size:
            return utils.move_batch_to_device(_subset_batch(batch, [index - offset]), device)
        offset += size
    raise IndexError(f"scene_index={index} exceeds the {cfg.eval.split} split")


def _verify_metrics(csv_path, index, regime, severity, preds, gt, psi):
    expected = {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (int(row["scene_index"]) == index and row["regime"] == regime
                    and float(row["severity"]) == severity):
                expected[row["method"]] = float(row["rmse"])
    if set(expected) != {METHODS[key] for key in preds}:
        raise ValueError("The requested scene/condition lacks four paired methods in the frozen CSV")
    metrics = {}
    for key, pred in preds.items():
        result = _metrics(pred, gt, psi)
        difference = abs(result[0] - expected[METHODS[key]])
        if difference > 1e-3:
            raise ValueError(f"{METHODS[key]} RMSE differs from frozen CSV by {difference:.5g} rad; "
                             "check checkpoint, dataset, seed, and SNAPHU version")
        metrics[key] = result
    return metrics


def _pad(x, horizontal):
    return np.pad(x, ((0, 0), (0, 1))) if horizontal else np.pad(x, ((0, 1), (0, 0)))


def _show(ax, array, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, shrink=0.7, pad=0.015)


def _save_mechanism(path, psi, gamma, gt, probs_h, probs_v, preds, metrics):
    eps = 1e-8
    kh = np.rint(((gt[:, 1:] - gt[:, :-1]) - mcf.wrap(psi[:, 1:] - psi[:, :-1])) / mcf.TWO_PI)
    kv = np.rint(((gt[1:, :] - gt[:-1, :]) - mcf.wrap(psi[1:, :] - psi[:-1, :])) / mcf.TWO_PI)
    ph_pos, ph_neg = probs_h[-1], probs_h[0]
    pv_pos, pv_neg = probs_v[-1], probs_v[0]
    pref_h = np.log(np.clip(ph_pos, eps, 1) / np.clip(ph_neg, eps, 1))
    pref_v = np.log(np.clip(pv_pos, eps, 1) / np.clip(pv_neg, eps, 1))
    unsigned_h = -np.log(np.clip(ph_pos + ph_neg, eps, 1))
    classical_h = 0.5 * (gamma[:, :-1] + gamma[:, 1:])
    cycle = {}
    for key, pred in preds.items():
        residual = pred - gt
        residual -= residual.mean()
        cycle[key] = np.rint(residual / mcf.TWO_PI)
    vmax_pref = max(float(np.percentile(np.abs(pref_h), 99)),
                    float(np.percentile(np.abs(pref_v), 99)), 0.1)
    vmax_cycle = max(1.0, float(np.percentile(np.abs(np.stack(list(cycle.values()))), 99)))
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), layout="constrained")
    panels = [
        (psi, "Stressed wrapped phase", "twilight", -np.pi, np.pi),
        (gamma, "True simulated coherence", "viridis", 0, 1),
        (_pad(kh, True), "GT ambiguity: horizontal", "coolwarm", -1, 1),
        (_pad(kv, False), "GT ambiguity: vertical", "coolwarm", -1, 1),
        (ph_pos, "Learned p(+1): horizontal", "viridis", 0, 1),
        (ph_neg, "Learned p(-1): horizontal", "viridis", 0, 1),
        (pv_pos, "Learned p(+1): vertical", "viridis", 0, 1),
        (pv_neg, "Learned p(-1): vertical", "viridis", 0, 1),
        (pref_h, "Signed log-odds: horizontal", "coolwarm", -vmax_pref, vmax_pref),
        (pref_v, "Signed log-odds: vertical", "coolwarm", -vmax_pref, vmax_pref),
        (unsigned_h, "Learned unsigned cost: horizontal", "magma", 0, None),
        (_pad(classical_h, True), "Coherence edge weight (not SNAPHU cost)", "viridis", 0, 1),
    ]
    for ax, (array, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        _show(ax, array, title, cmap, vmin, vmax)
    for ax, key, title in zip(axes[3], preds, ("Learned MCF", "Unit MCF", "SNAPHU known", "SNAPHU uniform")):
        _show(ax, cycle[key], f"{title} cycle error; RMSE={metrics[key][0]:.2f}",
              "coolwarm", -vmax_cycle, vmax_cycle)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _save_audit(path, psi, gamma, gt, preds, metrics):
    order = ("learned_mcf", "snaphu_true", "snaphu_uniform", "unit_mcf")
    aligned, errors, cycles = {}, {}, {}
    for key in order:
        error = preds[key] - gt
        error = error - error.mean()
        aligned[key] = gt + error
        errors[key] = error
        cycles[key] = np.rint(error / mcf.TWO_PI)
    phase_lim = (float(np.percentile(gt, 1)), float(np.percentile(gt, 99)))
    error_lim = max(1.0, float(np.percentile(np.abs(np.stack(list(errors.values()))), 99)))
    cycle_lim = max(1.0, float(np.percentile(np.abs(np.stack(list(cycles.values()))), 99)))
    titles = ("Learned MCF", "SNAPHU known", "SNAPHU uniform", "Unit MCF")
    fig, axes = plt.subplots(4, 4, figsize=(14, 12), layout="constrained")
    _show(axes[0, 0], gt, "Clean reference phase", "viridis", *phase_lim)
    _show(axes[0, 1], psi, "Stressed wrapped observation", "twilight", -np.pi, np.pi)
    _show(axes[0, 2], gamma, "True simulated coherence", "viridis", 0, 1)
    _show(axes[0, 3], np.abs(mcf.residues(psi)), "Input residue magnitude", "magma", 0, None)
    for j, (key, title) in enumerate(zip(order, titles)):
        _show(axes[1, j], aligned[key], f"{title} aligned output", "viridis", *phase_lim)
        _show(axes[2, j], errors[key], f"{title} aligned error", "coolwarm", -error_lim, error_lim)
        _show(axes[3, j], cycles[key], f"{title} cycle error; {100*metrics[key][1]:.1f}% slips",
              "coolwarm", -cycle_lim, cycle_lim)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ext = cfg.get("ext", {})
    index = int(ext.get("scene_index", -1))
    regime = str(ext.get("regime", "mixed"))
    severity = float(ext.get("severity", 1.0))
    mismatch_dir = Path(str(ext.get("mismatch_dir", "results/mismatch_v3_full")))
    with (mismatch_dir / "cost_mismatch_test_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (manifest.get("protocol") != 3 or manifest.get("split") != str(cfg.eval.split)
            or regime not in manifest["regimes"] or severity not in manifest["severities"]):
        raise ValueError("Requested scene/condition is not in the frozen protocol-v3 manifest")
    if index < 0:
        raise ValueError("Set +ext.scene_index to a scene ID from the frozen sweep")
    device = utils.get_device(cfg.device)
    model = load_edge_model(cfg, device)
    batch = _find_scene(cfg, index, device)
    gt = batch["absolute"][0, 0].detach().cpu().numpy()
    latents = make_scene_latents(gt.shape, int(manifest["seed"]), index)
    gamma = coherence_field(gt.shape, regime, severity, latents)
    psi = noisy_wrapped(gt, gamma, latents, int(manifest["noise_looks"]))
    stressed = reencode(batch, torch.from_numpy(psi[None]).to(device),
                        torch.from_numpy(gamma[None]).to(device), cfg)
    with torch.no_grad():
        learned, unit = mcf_infer.unwrap_batch(model, stressed, also_unit=True)
        ph, pv = probabilities(model, stressed)
    snaphu_true = baselines.snaphu_unwrap(
        psi, coherence=gamma, nlooks=float(manifest["snaphu_nlooks"]),
        cost=str(manifest["snaphu_cost"]))
    # Match eval_cost_mismatch's exact homogeneous-coherence shortcut.
    snaphu_uniform = (snaphu_true if np.ptp(gamma) < 1e-7 else
                      baselines.snaphu_unwrap(
                          psi, coherence=np.full_like(gamma, float(gamma.mean())),
                          nlooks=float(manifest["snaphu_nlooks"]),
                          cost=str(manifest["snaphu_cost"])))
    preds = {
        "learned_mcf": learned[0], "unit_mcf": unit[0],
        "snaphu_true": snaphu_true,
        "snaphu_uniform": snaphu_uniform,
    }
    metrics = _verify_metrics(mismatch_dir / "cost_mismatch_test_per_scene.csv",
                              index, regime, severity, preds, gt, psi)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    out_dir = Path(str(ext.get("out_dir", "results/paper_figures")))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{index:04d}_{regime}_{severity:.2f}".replace(".", "p")
    _save_mechanism(out_dir / f"fig_mechanism_{tag}.pdf", psi, gamma, gt,
                    ph[0].detach().cpu().numpy(), pv[0].detach().cpu().numpy(), preds, metrics)
    _save_audit(out_dir / f"fig_audit_{tag}.pdf", psi, gamma, gt, preds, metrics)


if __name__ == "__main__":
    main()
