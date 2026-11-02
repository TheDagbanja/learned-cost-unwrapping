"""Plot median, largest-failure-reduction, and SNAPHU-favored scenes.

Scene selection is determined entirely by the per-scene CSV: the scene nearest
the learned method's median RMSE, the largest SNAPHU-minus-learned RMSE gain,
and the largest learned-minus-SNAPHU gap.  The script then re-runs only those
three scenes with the checkpoint and data configuration recorded by Hydra.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from scripts.eval_fair_benchmark import _unsigned_costs
from scripts.ext_experiment_common import load_edge_model
from src import baselines, dataset as dataset_module, mcf, utils
from src.metrics import aligned_residual, decomposition_metrics


LEARNED = "learned signed cost + MCF"


def _read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _choose_snaphu(rows, requested: str) -> str:
    names = sorted({row["method"] for row in rows if row["method"].startswith("SNAPHU")})
    if requested:
        if requested not in names:
            raise ValueError(f"Requested SNAPHU method not present. Available: {names}")
        return requested
    means = {
        name: np.mean([float(row["rmse"]) for row in rows if row["method"] == name])
        for name in names
    }
    if not means:
        raise ValueError("The benchmark CSV has no SNAPHU rows")
    return min(means, key=means.get)


def _select(rows, snaphu_name: str):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[int(row["scene_index"])][row["method"]] = row
    paired = {
        index: values for index, values in grouped.items()
        if LEARNED in values and snaphu_name in values
    }
    if len(paired) < 3:
        raise ValueError("At least three scenes with paired learned/SNAPHU rows are required")
    learned_values = np.array([float(values[LEARNED]["rmse"]) for values in paired.values()])
    median = float(np.median(learned_values))
    typical = min(paired, key=lambda i: abs(float(paired[i][LEARNED]["rmse"]) - median))
    attenuation = max(
        paired,
        key=lambda i: float(paired[i][snaphu_name]["rmse"]) - float(paired[i][LEARNED]["rmse"]),
    )
    counter = max(
        paired,
        key=lambda i: float(paired[i][LEARNED]["rmse"]) - float(paired[i][snaphu_name]["rmse"]),
    )
    selected = [
        ("Median-RMSE case", typical),
        ("Largest failure reduction", attenuation),
        ("SNAPHU-favored case", counter),
    ]
    # Degenerate small test sets can choose the same scene twice.  Preserve the
    # selection rule but substitute the next-ranked unused scene.
    used = set()
    fixed = []
    rankings = {
        "Median-RMSE case": sorted(paired, key=lambda i: abs(float(paired[i][LEARNED]["rmse"]) - median)),
        "Largest failure reduction": sorted(
            paired,
            key=lambda i: float(paired[i][snaphu_name]["rmse"])
            - float(paired[i][LEARNED]["rmse"]),
            reverse=True,
        ),
        "SNAPHU-favored case": sorted(paired, key=lambda i: float(paired[i][LEARNED]["rmse"]) - float(paired[i][snaphu_name]["rmse"]), reverse=True),
    }
    for label, preferred in selected:
        index = next(candidate for candidate in rankings[label] if candidate not in used)
        fixed.append((label, index)); used.add(index)
    return fixed, paired


def _batch(sample, device):
    out = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.unsqueeze(0).to(device)
        else:
            out[key] = [value]
    return out


def _predict(model, sample, snaphu_row, device):
    batch = _batch(sample, device)
    psi = sample["wrapped"][0].numpy()
    gt = sample["absolute"][0].numpy()
    with torch.no_grad():
        logits, _ = model.encode(batch)
    logits = logits[0].detach().float().cpu().numpy()
    signed = mcf.unwrap_mcf(psi, edge_costs=mcf.edge_costs_from_logits(logits))
    wx, wy = _unsigned_costs(logits)
    unsigned = mcf.unwrap_mcf(psi, wx=wx, wy=wy)
    unit = mcf.unwrap_mcf(psi)
    information = snaphu_row["information"]
    if information == "uniform":
        coherence = np.ones_like(psi, dtype=np.float32)
    elif information == "phase_proxy":
        coherence = mcf.quality_map(psi).astype(np.float32)
    elif information == "supplied":
        available = sample.get("coherence_available", torch.tensor(False))
        if not bool(available):
            raise ValueError("Selected SNAPHU row requires supplied coherence, but it is unavailable")
        coherence = sample["coherence"][0].numpy()
    else:
        raise ValueError(f"Unknown SNAPHU information mode: {information}")
    snaphu = baselines.snaphu_unwrap(
        psi, coherence=coherence, nlooks=float(snaphu_row["nlooks"]), cost=snaphu_row["snaphu_cost"]
    )
    return psi, gt, {"Ours (signed)": signed, "Unsigned": unsigned, "Unit MCF": unit, "SNAPHU": snaphu}


def _aligned(pred, gt):
    residual = aligned_residual(pred, gt)
    return gt + residual


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ext = cfg.get("ext", {})
    split = str(cfg.eval.get("split", "test"))
    out_dir = str(ext.get("out_dir", "results/fair_benchmark"))
    csv_path = str(ext.get(
        "representative_csv", os.path.join(out_dir, f"fair_benchmark_{split}_per_scene.csv")
    ))
    rows = _read_rows(csv_path)
    snaphu_name = _choose_snaphu(rows, str(ext.get("representative_snaphu_method", "") or ""))
    selected, paired = _select(rows, snaphu_name)

    device = utils.get_device(cfg.device)
    model = load_edge_model(cfg, device)
    dataset = dataset_module.build_dataset(cfg.data, split=split)
    cases = []
    for label, index in selected:
        sample = dataset[index]
        snaphu_row = paired[index][snaphu_name]
        psi, gt, predictions = _predict(model, sample, snaphu_row, device)
        cases.append((label, index, str(sample.get("name", index)), psi, gt, predictions))

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9})
    fig, axes = plt.subplots(2 * len(cases), 6, figsize=(13.2, 10.0), constrained_layout=True)
    for block, (label, index, name, psi, gt, predictions) in enumerate(cases):
        top, bottom = axes[2 * block], axes[2 * block + 1]
        aligned = {method: _aligned(pred, gt) for method, pred in predictions.items()}
        lo = min(float(gt.min()), *(float(array.min()) for array in aligned.values()))
        hi = max(float(gt.max()), *(float(array.max()) for array in aligned.values()))
        top[0].imshow(psi, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        top[0].set_title("Wrapped phase")
        top[1].imshow(gt, cmap="viridis", vmin=lo, vmax=hi)
        top[1].set_title("Reference")
        for axis, (method, array) in zip(top[2:], aligned.items()):
            axis.imshow(array, cmap="viridis", vmin=lo, vmax=hi)
            metrics = decomposition_metrics(predictions[method], gt, psi)
            axis.set_title(f"{method}\nRMSE {metrics['rmse']:.2f}, slip {100*metrics['slip']:.1f}%")

        ambiguity = np.round((gt - psi) / mcf.TWO_PI)
        bottom[0].imshow(ambiguity, cmap="coolwarm")
        bottom[0].set_title("Reference ambiguity (cycles)")
        bottom[1].axis("off")
        bottom[1].text(
            0.02, 0.95,
            f"{label}\nscene {index}: {name}\nselection applied to paired per-scene RMSE\nSNAPHU: {snaphu_name}",
            ha="left", va="top", transform=bottom[1].transAxes,
        )
        cycle_errors = [np.round(aligned_residual(pred, gt) / mcf.TWO_PI) for pred in predictions.values()]
        bound = max(1.0, max(float(np.max(np.abs(error))) for error in cycle_errors))
        for axis, (method, error) in zip(bottom[2:], zip(predictions, cycle_errors)):
            image = axis.imshow(error, cmap="coolwarm", vmin=-bound, vmax=bound)
            axis.set_title(f"{method} cycle error")
        colorbar = fig.colorbar(image, ax=bottom[2:].tolist(), fraction=0.018, pad=0.01)
        colorbar.set_label("Aligned cycle error")
        for axis in (top.tolist() + bottom.tolist()):
            axis.set_xticks([]); axis.set_yticks([])

    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, "fig_representative_methods.pdf")
    png = os.path.join(out_dir, "fig_representative_methods.png")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
