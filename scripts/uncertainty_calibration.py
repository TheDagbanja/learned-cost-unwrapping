"""Evaluate synthetic-domain error ranking and probability calibration.

The input is ``calib.npz`` from :mod:`scripts.dump_fig_arrays`.  Calibration is
fit on complete scenes from the first half of the file and evaluated on complete
scenes from the second half.  This split is mandatory: randomly splitting pixels
would leak scene-specific structure and overstate calibration quality.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def _scene_split(data, pixels_per_scene: int):
    n = len(data["entropy"])
    if "scene_index" in data:
        scene = np.asarray(data["scene_index"])
        unique = np.unique(scene)
        cut = len(unique) // 2
        calibration = np.isin(scene, unique[:cut])
        return calibration, ~calibration, int(cut), int(len(unique) - cut)
    if n % pixels_per_scene:
        raise ValueError(
            "calib.npz has no scene_index and its length is not divisible by pixels_per_scene"
        )
    n_scenes = n // pixels_per_scene
    cut = n_scenes // 2
    boundary = cut * pixels_per_scene
    calibration = np.zeros(n, dtype=bool)
    calibration[:boundary] = True
    return calibration, ~calibration, cut, n_scenes - cut


def _sample(mask: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) <= limit:
        return indices
    return np.sort(rng.choice(indices, size=limit, replace=False))


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    records, total = [], len(y)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if not np.any(selected):
            continue
        confidence = float(probability[selected].mean())
        observed = float(y[selected].mean())
        count = int(selected.sum())
        ece += count / total * abs(confidence - observed)
        records.append({"low": low, "high": high, "confidence": confidence,
                        "observed": observed, "count": count})
    return float(ece), records


def _coverage_table(y: np.ndarray, probability: np.ndarray, raw_score: np.ndarray,
                    coverages) -> list[dict]:
    rows = []
    prevalence = float(y.mean())
    for coverage in coverages:
        # Isotonic regression creates tied probabilities.  Break ties with the
        # original entropy ranking so the requested coverage is attained exactly.
        order = np.lexsort((raw_score, probability))
        retained = np.zeros(len(y), dtype=bool)
        retained[order[:int(round(float(coverage) * len(y)))]] = True
        flagged = ~retained
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        fn = int(np.sum(retained & (y == 1)))
        rows.append({
            "retained_coverage": float(retained.mean()),
            "flagged_fraction": float(flagged.mean()),
            "retained_error_rate": float(y[retained].mean()) if np.any(retained) else float("nan"),
            "relative_risk": float(y[retained].mean() / prevalence) if prevalence and np.any(retained) else float("nan"),
            "flag_precision": tp / max(tp + fp, 1),
            "error_recall": tp / max(tp + fn, 1),
            "threshold": float(probability[order[max(int(round(float(coverage) * len(y))) - 1, 0)]]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="results/uncertainty")
    parser.add_argument("--pixels-per-scene", type=int, default=256 * 256)
    parser.add_argument("--fit-pixels", type=int, default=750_000)
    parser.add_argument("--eval-pixels", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.input, allow_pickle=False)
    required = {"entropy", "is_error", "grad_proxy"}
    if not required.issubset(data.files):
        raise ValueError(f"{args.input} must contain {sorted(required)}")
    calibration_mask, evaluation_mask, n_cal_scenes, n_eval_scenes = _scene_split(
        data, args.pixels_per_scene
    )
    rng = np.random.default_rng(args.seed)
    fit_indices = _sample(calibration_mask, args.fit_pixels, rng)
    eval_indices = _sample(evaluation_mask, args.eval_pixels, rng)

    entropy = np.asarray(data["entropy"], dtype=np.float64)
    labels = np.asarray(data["is_error"], dtype=np.uint8)
    proxy = np.asarray(data["grad_proxy"], dtype=np.float64)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(entropy[fit_indices], labels[fit_indices])
    probability = calibrator.predict(entropy[eval_indices])
    y = labels[eval_indices]
    score_entropy = entropy[eval_indices]
    score_proxy = proxy[eval_indices]

    ece, reliability = _ece(y, probability)
    metrics = {
        "scope": "synthetic-domain risk ranking and held-out-scene calibration",
        "calibration_scenes": n_cal_scenes,
        "evaluation_scenes": n_eval_scenes,
        "fit_pixels": int(len(fit_indices)),
        "evaluation_pixels": int(len(eval_indices)),
        "error_prevalence": float(y.mean()),
        "entropy_auroc": float(roc_auc_score(y, score_entropy)),
        "entropy_average_precision": float(average_precision_score(y, score_entropy)),
        "gradient_proxy_auroc": float(roc_auc_score(y, score_proxy)),
        "gradient_proxy_average_precision": float(average_precision_score(y, score_proxy)),
        "isotonic_brier": float(brier_score_loss(y, probability)),
        "isotonic_ece_15": ece,
    }
    coverages = _coverage_table(y, probability, score_entropy,
                                (0.50, 0.70, 0.80, 0.90, 0.95))

    with open(os.path.join(args.out, "uncertainty_metrics.json"), "w") as stream:
        json.dump({"metrics": metrics, "coverage": coverages, "reliability": reliability},
                  stream, indent=2)
    with open(os.path.join(args.out, "uncertainty_coverage.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverages[0]))
        writer.writeheader(); writer.writerows(coverages)

    precision_e, recall_e, _ = precision_recall_curve(y, score_entropy)
    precision_g, recall_g, _ = precision_recall_curve(y, score_proxy)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].plot(recall_e, precision_e, label=f"Entropy (AP={metrics['entropy_average_precision']:.3f})")
    axes[0].plot(recall_g, precision_g, label=f"Gradient proxy (AP={metrics['gradient_proxy_average_precision']:.3f})")
    axes[0].axhline(metrics["error_prevalence"], color="0.5", ls="--", label="Prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Error detection")
    axes[0].legend(frameon=False, fontsize=8)

    # Suppress nearly empty equal-width bins in the plotted line; all bins,
    # including sparse high-probability bins, remain in the JSON audit record.
    plotted_reliability = [row for row in reliability if row["count"] >= 100]
    observed = [row["observed"] for row in plotted_reliability]
    confidence = [row["confidence"] for row in plotted_reliability]
    axes[1].plot([0, 1], [0, 1], color="0.5", ls="--")
    axes[1].plot(confidence, observed, "o-", label=f"ECE={ece:.3f}")
    axes[1].set(xlabel="Predicted error probability", ylabel="Observed error frequency",
                title="Held-out reliability", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(frameon=False)

    cov = np.asarray([row["retained_coverage"] for row in coverages])
    risk = np.asarray([row["retained_error_rate"] for row in coverages])
    axes[2].plot(cov, risk, "o-")
    axes[2].axhline(metrics["error_prevalence"], color="0.5", ls="--", label="No abstention")
    axes[2].set(xlabel="Retained pixel coverage", ylabel="Cycle-error rate",
                title="Risk-coverage")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(os.path.join(args.out, f"fig_uncertainty_calibration.{extension}"),
                       dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
