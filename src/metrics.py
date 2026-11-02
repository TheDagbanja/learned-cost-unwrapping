"""Shared evaluation metrics for auditable phase-unwrapping experiments.

The manuscript separates integer-cycle mistakes from within-cycle error.  This
module is the single implementation used by the benchmark, ablation, runtime,
and leave-one-domain-out scripts so that every reported row is comparable.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np

from . import mcf


def aligned_residual(pred: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the residual after removing the unobservable global phase offset."""
    residual = np.asarray(pred, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return residual - residual.mean()


def residue_balance_violations(
    pred: np.ndarray,
    wrapped: np.ndarray,
    wrapped_residues: np.ndarray | None = None,
) -> int:
    """Count plaquettes whose implied integer field violates residue balance."""
    pred = np.asarray(pred, dtype=np.float64)
    wrapped = np.asarray(wrapped, dtype=np.float64)
    if wrapped_residues is None:
        wrapped_residues = mcf.residues(wrapped)
    nx = np.round(
        ((pred[:, 1:] - pred[:, :-1]) - mcf.wrap(wrapped[:, 1:] - wrapped[:, :-1]))
        / mcf.TWO_PI
    )
    ny = np.round(
        ((pred[1:, :] - pred[:-1, :]) - mcf.wrap(wrapped[1:, :] - wrapped[:-1, :]))
        / mcf.TWO_PI
    )
    defect = nx[:-1, :] + ny[:, 1:] - nx[1:, :] - ny[:, :-1] + wrapped_residues
    return int(np.count_nonzero(defect))


def decomposition_metrics(
    pred: np.ndarray,
    reference: np.ndarray,
    wrapped: np.ndarray,
    wrapped_residues: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute aligned RMSE, cycle-slip rate, slip-free RMSE, and balance defects."""
    residual = aligned_residual(pred, reference)
    cycles = np.round(residual / mcf.TWO_PI)
    slip_free = residual - mcf.TWO_PI * cycles
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "slip": float(np.mean(cycles != 0)),
        "eps": float(np.sqrt(np.mean(slip_free**2))),
        "residues": float(residue_balance_violations(pred, wrapped, wrapped_residues)),
    }


def tail_summary(values: Iterable[float]) -> Dict[str, float]:
    """Summarize central performance and the catastrophic-error tail."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("tail_summary requires at least one value")
    q95 = float(np.quantile(array, 0.95))
    tail = array[array >= q95]
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": q95,
        "p99": float(np.quantile(array, 0.99)),
        "cvar95": float(tail.mean()),
        "max": float(array.max()),
        "n": float(array.size),
    }


def summarize_method_rows(rows: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    """Aggregate per-scene metric rows for a manuscript table."""
    rows = list(rows)
    if not rows:
        raise ValueError("summarize_method_rows requires at least one row")
    rmse = np.asarray([float(row["rmse"]) for row in rows], dtype=np.float64)
    out = tail_summary(rmse)
    out["cat_rmse_2pi"] = float(np.mean(rmse >= mcf.TWO_PI))
    for key in ("slip", "eps", "residues", "runtime_s"):
        present = [float(row[key]) for row in rows if key in row and row[key] not in (None, "")]
        if present:
            out[key] = float(np.mean(present))
            if key == "slip":
                out["cat_slip_005"] = float(np.mean(np.asarray(present) >= 0.05))
            if key == "runtime_s":
                out["runtime_median_s"] = float(np.median(present))
                out["runtime_p95_s"] = float(np.quantile(present, 0.95))
    return out
