"""PDF figures from the completed paired stress-test CSVs (no solver rerun).

Run from the repository root with ``python -m scripts.plot_paper_figures``.
The mismatch figure uses per-scene rows, never means from the summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "learned-cost MCF",
    "SNAPHU true coherence",
    "SNAPHU uniform coherence",
    "unit-cost MCF",
)
REGIMES = ("homogeneous", "heterogeneous", "decorrelation_patches", "mixed")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A5195")
LABELS = ("Learned MCF", "SNAPHU known coherence", "SNAPHU uniform coherence", "Unit MCF")


def _read_mismatch(path: Path, manifest_path: Path):
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("protocol") != 3 or manifest.get("split") != "test":
        raise ValueError("Expected a completed protocol-v3 test manifest")
    requested_regimes = tuple(manifest["regimes"])
    severities = tuple(sorted(float(v) for v in manifest["severities"]))
    if set(requested_regimes) != set(REGIMES):
        raise ValueError(f"Expected all four paper regimes; got {requested_regimes}")
    values = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"protocol", "split", "scene_index", "regime", "severity", "method", "rmse"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing per-scene columns: {required - set(reader.fieldnames or [])}")
        for row in reader:
            if int(row["protocol"]) != 3 or row["split"] != "test":
                raise ValueError("Mixed protocol or split in per-scene CSV")
            key = (row["regime"], float(row["severity"]), row["method"])
            scene = int(row["scene_index"])
            if scene in values[key]:
                raise ValueError(f"Duplicate scene row: {key}, scene {scene}")
            values[key][scene] = float(row["rmse"])
    expected = None
    for regime in REGIMES:
        for severity in severities:
            for method in METHODS:
                ids = set(values[(regime, severity, method)])
                if not ids:
                    raise ValueError(f"Empty condition: {(regime, severity, method)}")
                if expected is None:
                    expected = ids
                elif ids != expected:
                    raise ValueError("Condition scene IDs differ; complete/resume the sweep before plotting")
    print(f"Validated {len(expected)} paired scenes x {len(REGIMES)} regimes x "
          f"{len(severities)} severities x {len(METHODS)} methods")
    return values, severities, len(expected)


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _save(fig, path: Path):
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata={"Creator": "plot_paper_figures.py"})
    plt.close(fig)
    print(f"wrote {path}")


def _tail_figure(values, severities, n, path):
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.8), sharex=True, sharey="row", layout="constrained")
    for col, regime in enumerate(REGIMES):
        for row, (percentile, label) in enumerate(((50, "Median aligned RMSE (rad)"),
                                                   (95, "95th-percentile aligned RMSE (rad)"))):
            ax = axes[row, col]
            for method, color, display in zip(METHODS, COLORS, LABELS):
                y = [np.percentile(list(values[(regime, s, method)].values()), percentile)
                     for s in severities]
                ax.plot(severities, y, "o-", color=color, lw=1.8, ms=4, label=display)
            ax.grid(axis="y", alpha=0.25)
            ax.set_xticks(severities)
            if row == 0:
                ax.set_title(regime.replace("_", " "))
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Mismatch severity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.055),
               ncol=4, frameon=False)
    fig.suptitle(f"Paired synthetic stress test (n={n} scenes)", fontsize=11)
    _save(fig, path)


def _paired_figure(values, severities, n, path, bootstrap, seed):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.6), sharey=True, layout="constrained")
    for ax, regime in zip(axes, REGIMES):
        means, lows, highs = [], [], []
        for severity in severities:
            learned = values[(regime, severity, METHODS[0])]
            snaphu = values[(regime, severity, METHODS[1])]
            ids = sorted(learned)
            delta = np.asarray([learned[i] - snaphu[i] for i in ids], dtype=np.float64)
            means.append(float(delta.mean()))
            # Paired resampling: the same scene ID supplies both method values.
            samples = np.empty(bootstrap)
            for j in range(bootstrap):
                samples[j] = delta[rng.integers(0, len(delta), len(delta))].mean()
            lo, hi = np.percentile(samples, [2.5, 97.5])
            lows.append(means[-1] - lo)
            highs.append(hi - means[-1])
        ax.errorbar(severities, means, yerr=[lows, highs], fmt="o-", lw=1.6,
                    capsize=3, color=COLORS[0])
        ax.axhline(0, color="0.35", ls="--", lw=0.9)
        ax.set_title(regime.replace("_", " "))
        ax.set_xlabel("Mismatch severity")
        ax.set_xticks(severities)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean paired RMSE difference (rad)\nLearned MCF - known-coherence SNAPHU")
    fig.suptitle(f"Scene-paired mean differences with 95% bootstrap intervals (n={n})", fontsize=11)
    _save(fig, path)


def _ablation_figure(path_in: Path, path_out: Path):
    with path_in.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"method", "rmse", "slip", "n"}.issubset(rows[0]):
        raise ValueError("Invalid solver-ablation CSV")
    names = [r["method"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate ablation method names")
    labels = [name.replace(" learned cost ", " learned\ncost ").replace(" + ", " + ")
              for name in names]
    rmse = np.asarray([float(r["rmse"]) for r in rows])
    slip = np.asarray([100 * float(r["slip"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.6), layout="constrained")
    y = np.arange(len(rows))
    bar_colors = [COLORS[0] if name == "signed learned cost + MCF" else "#8A8F98" for name in names]
    for ax, data, xlabel in zip(axes, (rmse, slip), ("Aligned RMSE (rad)", "Cycle-slip pixels (%)")):
        ax.barh(y, data, color=bar_colors, height=0.65)
        ax.invert_yaxis()
        ax.set_yticks(y, labels if ax is axes[0] else [""] * len(rows))
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(data) * 1.18)
        for yi, val in enumerate(data):
            ax.text(val + max(data) * 0.02, yi, f"{val:.2f}", va="center", fontsize=8)
    fig.suptitle(f"Solver/cost ablation (n={rows[0]['n']} scenes)", fontsize=11)
    _save(fig, path_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatch-dir", type=Path, required=True,
                        help="Directory containing the stress-test manifest and per-scene CSV")
    parser.add_argument("--ablation-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    values, severities, n = _read_mismatch(
        args.mismatch_dir / "cost_mismatch_test_per_scene.csv",
        args.mismatch_dir / "cost_mismatch_test_manifest.json")
    _tail_figure(values, severities, n, args.out_dir / "fig_mismatch_tails.pdf")
    _paired_figure(values, severities, n, args.out_dir / "fig_mismatch_paired.pdf",
                   args.bootstrap, args.seed)
    if args.ablation_csv is not None:
        _ablation_figure(args.ablation_csv, args.out_dir / "fig_solver_ablation.pdf")


if __name__ == "__main__":
    main()
