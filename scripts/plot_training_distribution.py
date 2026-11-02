"""Audit and visualize the InSAR-DLPU training distribution.

The public training pairs contain wrapped and absolute phase but no coherence.
The resulting figure states this explicitly and never plots the loader's all-one
placeholder as measured coherence.  A second figure shows automatically selected
low/median/high ambiguity-density examples from the audited sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import dataset as dataset_module, mcf


def _edge_statistics(wrapped: np.ndarray, absolute: np.ndarray):
    gx = np.diff(absolute, axis=1)
    gy = np.diff(absolute, axis=0)
    wx = mcf.wrap(np.diff(wrapped, axis=1))
    wy = mcf.wrap(np.diff(wrapped, axis=0))
    kx = np.round((gx - wx) / mcf.TWO_PI).astype(np.int16)
    ky = np.round((gy - wy) / mcf.TWO_PI).astype(np.int16)
    return gx, gy, kx, ky


def _sample_flat(array: np.ndarray, count: int, rng: np.random.Generator):
    flat = np.asarray(array).ravel()
    if len(flat) <= count:
        return flat
    return flat[rng.choice(len(flat), size=count, replace=False)]


def _save(figure, out_dir: str, stem: str):
    for extension in ("pdf", "png"):
        figure.savefig(os.path.join(out_dir, f"{stem}.{extension}"), dpi=250,
                       bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--out", default="results/training_distribution")
    parser.add_argument("--scenes", type=int, default=1000)
    parser.add_argument("--edges-per-scene", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root = dataset_module.resolve_dlpu_root(args.root)
    wrapped_dir = os.path.join(root, "train_wrapped")
    absolute_dir = os.path.join(root, "train_absolute")
    files = sorted(name for name in os.listdir(wrapped_dir) if name.endswith(".mat"))
    if not files:
        raise FileNotFoundError(wrapped_dir)
    n = min(args.scenes, len(files))
    selected = np.linspace(0, len(files) - 1, n, dtype=int)
    rng = np.random.default_rng(args.seed)

    gradients, ambiguities, rows = [], [], []
    for dataset_index in selected:
        name = files[dataset_index]
        wrapped = dataset_module._load_mat_array(os.path.join(wrapped_dir, name), "input")
        absolute = dataset_module._load_mat_array(os.path.join(absolute_dir, name), "output")
        gx, gy, kx, ky = _edge_statistics(wrapped, absolute)
        gradient = np.concatenate((_sample_flat(np.abs(gx), args.edges_per_scene // 2, rng),
                                   _sample_flat(np.abs(gy), args.edges_per_scene // 2, rng)))
        ambiguity = np.concatenate((_sample_flat(kx, args.edges_per_scene // 2, rng),
                                    _sample_flat(ky, args.edges_per_scene // 2, rng)))
        gradients.append(gradient); ambiguities.append(ambiguity)
        all_k = np.concatenate((kx.ravel(), ky.ravel()))
        rows.append({
            "dataset_index": int(dataset_index), "name": name,
            "phase_ptp_rad": float(np.ptp(absolute)),
            "phase_p01_p99_rad": float(np.quantile(absolute, 0.99) - np.quantile(absolute, 0.01)),
            "nonzero_ambiguity_fraction": float(np.mean(all_k != 0)),
            "outside_three_class_fraction": float(np.mean(np.abs(all_k) > 1)),
            "mean_abs_gradient_rad": float(np.mean(np.abs(np.concatenate((gx.ravel(), gy.ravel()))))),
        })

    gradients = np.concatenate(gradients)
    ambiguities = np.concatenate(ambiguities)
    phase_ptp = np.asarray([row["phase_ptp_rad"] for row in rows])
    figure, axes = plt.subplots(1, 4, figsize=(14.5, 3.6))
    axes[0].hist(gradients / np.pi, bins=np.linspace(0, 8, 81), color="#356aa0")
    axes[0].axvline(1, color="crimson", ls="--", label=r"$|\Delta\phi|=\pi$")
    axes[0].set(xlabel=r"Absolute phase gradient $|\Delta\phi|/\pi$", ylabel="Sampled edges",
                title="Phase-gradient distribution")
    axes[0].legend(frameon=False, fontsize=8)

    values, counts = np.unique(np.clip(ambiguities, -4, 4), return_counts=True)
    labels = ["≤-4" if value == -4 else "≥4" if value == 4 else str(int(value)) for value in values]
    axes[1].bar(labels, counts / counts.sum(), color="#d17a22")
    axes[1].set(xlabel="Integer edge ambiguity K", ylabel="Edge fraction",
                title="Ambiguity classes")
    axes[1].set_yscale("log")

    axes[2].hist(phase_ptp, bins=35, color="#4b9b69")
    axes[2].set(xlabel="Scene peak-to-peak phase (rad)", ylabel="Scenes",
                title="Deformation/topography amplitude")

    axes[3].axis("off")
    axes[3].text(0.5, 0.58, "Coherence unavailable", ha="center", va="center",
                 fontsize=13, fontweight="bold")
    axes[3].text(0.5, 0.38,
                 "The InSAR-DLPU training pairs contain\nwrapped and absolute phase only.\n"
                 "The all-one loader array is a placeholder\nand is not plotted as sensor data.",
                 ha="center", va="center", fontsize=9)
    axes[3].set_title("Training coherence")
    for axis in axes[:3]:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    _save(figure, args.out, "fig_training_distribution")

    difficulties = np.asarray([row["nonzero_ambiguity_fraction"] for row in rows])
    quantiles = [0.1, 0.5, 0.9]
    chosen = [int(np.argmin(np.abs(difficulties - np.quantile(difficulties, q)))) for q in quantiles]
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 9.5))
    for row_index, (record_index, quantile) in enumerate(zip(chosen, quantiles)):
        record = rows[record_index]
        name = record["name"]
        wrapped = dataset_module._load_mat_array(os.path.join(wrapped_dir, name), "input")
        absolute = dataset_module._load_mat_array(os.path.join(absolute_dir, name), "output")
        _, _, kx, ky = _edge_statistics(wrapped, absolute)
        kmag = np.zeros_like(wrapped, dtype=np.float32)
        kmag[:, :-1] = np.maximum(kmag[:, :-1], np.abs(kx))
        kmag[:-1, :] = np.maximum(kmag[:-1, :], np.abs(ky))
        images = [(wrapped, "twilight", "Wrapped phase"),
                  (absolute - absolute.mean(), "viridis", "Reference absolute phase"),
                  (kmag, "magma", "Incident |K|")]
        for column, (image, cmap, title) in enumerate(images):
            artist = axes[row_index, column].imshow(image, cmap=cmap)
            axes[row_index, column].set_title(title if row_index == 0 else "")
            axes[row_index, column].axis("off")
            figure.colorbar(artist, ax=axes[row_index, column], fraction=0.046)
        axes[row_index, 0].text(
            -0.08, 0.5,
            f"{int(quantile*100)}th percentile\ncut density={record['nonzero_ambiguity_fraction']:.2%}",
            transform=axes[row_index, 0].transAxes, ha="right", va="center",
            rotation=90, fontsize=9,
        )
    figure.suptitle("Representative InSAR-DLPU training pairs selected by ambiguity density", y=0.995)
    figure.tight_layout()
    _save(figure, args.out, "fig_training_examples")

    with open(os.path.join(args.out, "training_scene_statistics.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "sampled_scenes": n,
        "selection": "evenly spaced indices over the 30,000-scene training split",
        "sampled_edges": int(len(ambiguities)),
        "coherence_available": False,
        "nonzero_ambiguity_fraction_sampled": float(np.mean(ambiguities != 0)),
        "outside_three_class_fraction_sampled": float(np.mean(np.abs(ambiguities) > 1)),
        "gradient_over_pi_fraction_sampled": float(np.mean(gradients > np.pi)),
        "phase_ptp_median_rad": float(np.median(phase_ptp)),
        "phase_ptp_p95_rad": float(np.quantile(phase_ptp, 0.95)),
        "representative_files": [rows[index]["name"] for index in chosen],
    }
    with open(os.path.join(args.out, "training_distribution_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
