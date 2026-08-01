"""Generate the manuscript figures that aren't already in outputs/.

Existing (from the unwrap/eval runs): outputs/{ridgecrest,amatrice,gorkha,kumamoto}_compare.png,
outputs/gps_scatter.png, outputs/amatrice_gps_scatter.png.

This script writes the quantitative summary plots (benchmark bars, per-event high-
coherence agreement, combined-cost ablation, differentiable-cost loss curve, and a
simple method schematic) into outputs/ for inclusion in the manuscript.

    python -m scripts.make_figures
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 200})


def benchmark_bar():
    methods = ["LS", "Quality-guided", "Goldstein", "Unit-cost MCF", "SNAPHU",
               "Learned MCF", "GT-grad MCF"]
    rmse = [4.327, 4.079, 8.368, 2.315, 1.714, 0.895, 0.839]
    colors = ["#9e9e9e", "#9e9e9e", "#9e9e9e", "#6baed6", "#2171b5", "#d62728", "#7fbf7b"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar(range(len(methods)), rmse, color=colors)
    ax.axhline(0.756, ls="--", c="k", lw=0.8)
    ax.text(len(methods) - 1, 0.95, "GT-K floor 0.76", fontsize=7.5, ha="right")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("aligned RMSE (rad)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_benchmark.png", bbox_inches="tight"); plt.close(fig)


def highcoh_grouped():
    events = ["Ridgecrest", "Amatrice", "Gorkha", "Kumamoto"]
    data = {  # high-coherence agreement vs SNAPHU (rad)
        "Unit-cost MCF":  [57.29, 0.223, 0.012, 16.18],
        "Coherence cost": [2.252, 0.039, 0.000, 0.782],
        "Learned cost":   [0.780, 0.061, 0.000, 0.977],
        "Learned+coh":    [0.694, 0.039, 0.000, 0.570],
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(events)); w = 0.2
    for i, (k, v) in enumerate(data.items()):
        ax.bar(x + (i - 1.5) * w, np.clip(v, 1e-3, None), w, label=k)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(events, fontsize=8)
    ax.set_ylabel("high-coh agreement\nvs SNAPHU (rad, log)")
    ax.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_highcoh.png", bbox_inches="tight"); plt.close(fig)


def combined_ablation():
    events = ["Ridgecrest", "Amatrice", "Gorkha", "Kumamoto"]
    learned = [0.780, 0.061, 0.001, 0.977]
    coh = [2.252, 0.039, 0.001, 0.782]
    comb = [0.694, 0.039, 0.001, 0.570]
    x = np.arange(len(events)); w = 0.25
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(x - w, learned, w, label="Learned", color="#d62728")
    ax.bar(x, coh, w, label="Coherence", color="#6baed6")
    ax.bar(x + w, comb, w, label="Learned + coherence", color="#2171b5")
    ax.set_xticks(x); ax.set_xticklabels(events, fontsize=8)
    ax.set_ylabel("high-coh agreement (rad)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_combined.png", bbox_inches="tight"); plt.close(fig)


def diff_loss():
    f = f"{OUT}/train_diff_loss.txt"
    if os.path.exists(f):
        h = np.loadtxt(f); xs = np.arange(1, len(h) + 1)
    else:  # the logged per-25-step running means (fallback)
        xs = np.arange(25, 301, 25)
        h = np.array([1.795, 1.716, 1.637, 1.581, 2.020, 2.013, 1.864,
                      1.898, 1.902, 1.729, 1.373, 1.946])
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(xs, h, lw=1.0, color="#d62728")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("unwrapping RMSE (rad)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_diffloss.png", bbox_inches="tight"); plt.close(fig)


def schematic():
    fig, ax = plt.subplots(figsize=(6.8, 1.7)); ax.axis("off")
    boxes = [("Wrapped\nphase $\\psi$", 0.02), ("Swin + LoRA\nedge head", 0.20),
             ("Per-edge\ncosts $-\\log P(k_e)$", 0.40), ("Min-cost-flow\nsolver", 0.61),
             ("Unwrapped $\\hat\\phi$\n+ uncertainty", 0.81)]
    for i, (t, xpos) in enumerate(boxes):
        c = "#d6e8f5" if i in (1, 2) else ("#e8f5e9" if i == 3 else "#f0f0f0")
        ax.add_patch(plt.Rectangle((xpos, 0.3), 0.16, 0.4, fc=c, ec="k", lw=1))
        ax.text(xpos + 0.08, 0.5, t, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(xpos + 0.185, 0.5), xytext=(xpos + 0.16, 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_pipeline.png"); plt.close(fig)


if __name__ == "__main__":
    benchmark_bar(); highcoh_grouped(); combined_ablation(); diff_loss(); schematic()
    print("wrote: fig_benchmark, fig_highcoh, fig_combined, fig_diffloss, fig_pipeline (in outputs/)")
