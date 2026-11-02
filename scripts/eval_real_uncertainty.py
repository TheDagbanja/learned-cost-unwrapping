"""Evaluate entropy ranking on the four real Sentinel-1 event products.

The analysis deliberately reports ranking, not transferred probability
calibration.  At image scale, the endpoint is integer-cycle disagreement among
the declared solver-based products.  At GPS stations, uncertainty is compared
with the absolute reference-aligned LOS residual.  The highest-residual
quintile is the predeclared station-level tail, and the reported review budget
is the highest-entropy quintile.

Inputs are declared in a JSON manifest; see
``configs/real_uncertainty_events.json``.  All rasters are reprojected onto the
uncertainty raster grid before comparison, so full-scene SNAPHU products and
cropped learned/DENet products can be mixed without index assumptions.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from scripts.gps_validate import (
    LAMBDA_C_BAND,
    load_offsets,
    sample_at,
    sample_patch_mean,
)


TWO_PI = 2.0 * np.pi
REVIEW_FRACTIONS = (0.05, 0.10, 0.20)


def _find(product: str, tag: str) -> str:
    hits = glob.glob(os.path.join(product, f"*{tag}.tif"))
    if not hits:
        raise FileNotFoundError(f"no *{tag}.tif in {product!r}")
    return hits[0]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_fraction(values: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    """Flag exactly ceil(fraction*N) valid cells with the largest values."""
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("fixed-coverage ranking has no valid cells")
    count = min(len(indices), max(1, int(math.ceil(float(fraction) * len(indices)))))
    order = np.argsort(values.ravel()[indices], kind="stable")
    flagged = np.zeros(values.size, dtype=bool)
    flagged[indices[order[-count:]]] = True
    return flagged.reshape(values.shape)


def _capture(target: np.ndarray, flagged: np.ndarray) -> float | None:
    total = int(np.sum(target))
    return None if total == 0 else float(np.sum(target & flagged) / total)


def _cycle_disagreement(
    arrays: dict[str, np.ndarray], valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return pixels where any solver differs by an integer cycle.

    Each product is aligned to the first solver by its median continuous phase
    offset before rounding the residual to cycles.  This removes the arbitrary
    global phase constant without suppressing regional cycle errors.
    """
    if len(arrays) < 2:
        raise ValueError("cycle disagreement requires at least two solver products")
    common = valid.copy()
    for array in arrays.values():
        common &= np.isfinite(array)
    reference_name = next(iter(arrays))
    reference = arrays[reference_name]
    disagreement = np.zeros(reference.shape, dtype=bool)
    offsets = {reference_name: 0.0}
    for name, array in list(arrays.items())[1:]:
        delta = array[common] - reference[common]
        offset = float(np.median(delta))
        offsets[name] = offset
        cycles = np.zeros(reference.shape, dtype=np.int32)
        cycles[common] = np.rint((array[common] - reference[common] - offset) / TWO_PI).astype(np.int32)
        disagreement |= cycles != 0
    return disagreement & common, common, offsets


def _read_on_grid(path: str, reference_path: str, *, nearest: bool) -> np.ndarray:
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(reference_path) as reference, rasterio.open(path) as source:
        if (
            source.crs == reference.crs
            and source.transform == reference.transform
            and source.width == reference.width
            and source.height == reference.height
        ):
            return source.read(1).astype(np.float64)
        destination = np.full((reference.height, reference.width), np.nan, dtype=np.float64)
        reproject(
            source=source.read(1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            src_nodata=source.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
        return destination


def _gps_metrics(event: dict) -> dict:
    gps_paths = [str(path) for path in event.get("gps", [])]
    if not gps_paths:
        return {
            "gps_n": None,
            "gps_residual_entropy_spearman": None,
            "gps_residual_entropy_p": None,
            "gps_upper_quintile_n": None,
            "gps_upper_quintile_capture_at_20pct": None,
        }

    import rasterio
    from scipy.stats import spearmanr

    product = str(event["product"])
    with ExitStack() as stack:
        src_theta = stack.enter_context(rasterio.open(_find(product, "_lv_theta")))
        src_phi = stack.enter_context(rasterio.open(_find(product, "_lv_phi")))
        src_snaphu = stack.enter_context(rasterio.open(_find(product, "_unw_phase")))
        src_coherence = stack.enter_context(rasterio.open(_find(product, "_corr")))
        src_ours = stack.enter_context(rasterio.open(str(event["ours"])))
        src_uncertainty = stack.enter_context(rasterio.open(str(event["uncertainty"])))

        records = []
        factor = 4.0 * np.pi / float(event.get("wavelength", LAMBDA_C_BAND))
        patch_win = int(event.get("patch_win", 2))
        for station in load_offsets(gps_paths):
            lon, lat = station["lon"], station["lat"]
            theta = sample_at(src_theta, lon, lat)
            phi = sample_at(src_phi, lon, lat)
            snaphu = sample_at(src_snaphu, lon, lat)
            ours = sample_at(src_ours, lon, lat)
            entropy = sample_patch_mean(src_uncertainty, lon, lat, win=patch_win)
            coherence = sample_patch_mean(src_coherence, lon, lat, win=patch_win)
            values = (theta, phi, snaphu, ours, entropy, coherence)
            if any(value is None or not np.isfinite(value) for value in values):
                continue
            if theta == 0 or snaphu == 0 or ours == 0:
                continue
            le = np.cos(theta) * np.cos(phi)
            ln = np.cos(theta) * np.sin(phi)
            lu = np.sin(theta)
            los = station["de"] * le + station["dn"] * ln + station["du"] * lu
            records.append({
                "station": station["station"],
                "gps": -factor * los,
                "snaphu": snaphu,
                "ours": ours,
                "entropy": entropy,
                "coherence": coherence,
            })

    minimum = float(event.get("min_coherence", 0.5))
    records = [row for row in records if row["coherence"] >= minimum]
    if len(records) < 3:
        raise ValueError(f"{event['name']}: fewer than three gated GPS stations")
    gps = np.asarray([row["gps"] for row in records], dtype=np.float64)
    snaphu = np.asarray([row["snaphu"] for row in records], dtype=np.float64)
    ours = np.asarray([row["ours"] for row in records], dtype=np.float64)
    entropy = np.asarray([row["entropy"] for row in records], dtype=np.float64)
    if np.corrcoef(snaphu, gps)[0, 1] < 0:
        gps = -gps
    residual = ours - gps
    residual -= residual.mean()
    absolute_residual = np.abs(residual)
    statistic = spearmanr(entropy, absolute_residual)
    rho = float(statistic.statistic)
    pvalue = float(statistic.pvalue)
    outliers = _top_fraction(absolute_residual, np.ones(len(records), dtype=bool), 0.20)
    reviewed = _top_fraction(entropy, np.ones(len(records), dtype=bool), 0.20)
    return {
        "gps_n": len(records),
        "gps_residual_entropy_spearman": rho if np.isfinite(rho) else None,
        "gps_residual_entropy_p": pvalue if np.isfinite(pvalue) else None,
        "gps_upper_quintile_n": int(np.sum(outliers)),
        "gps_upper_quintile_capture_at_20pct": _capture(outliers, reviewed),
    }


def _resolve_solvers(event: dict) -> dict[str, str]:
    solvers = {}
    for name, path in event["solvers"].items():
        solvers[name] = _find(str(event["product"]), "_unw_phase") if path == "auto:snaphu" else str(path)
    return solvers


def _event_metrics(event: dict) -> dict:
    uncertainty_path = str(event["uncertainty"])
    entropy = _read_on_grid(uncertainty_path, uncertainty_path, nearest=False)
    coherence_path = _find(str(event["product"]), "_corr")
    coherence = _read_on_grid(coherence_path, uncertainty_path, nearest=False)
    solver_paths = _resolve_solvers(event)
    arrays = {
        name: _read_on_grid(path, uncertainty_path, nearest=True)
        for name, path in solver_paths.items()
    }
    valid = np.isfinite(entropy) & np.isfinite(coherence) & (coherence > 0)
    disagreement, valid, offsets = _cycle_disagreement(arrays, valid)
    if not np.any(valid):
        raise ValueError(f"{event['name']}: solver rasters have no common valid pixels")
    result = {
        "event": str(event["name"]),
        "n_solvers": len(arrays),
        "solvers": list(arrays),
        "solver_offsets_rad": offsets,
        "valid_pixels": int(np.sum(valid)),
        "cycle_disagreement_pixels": int(np.sum(disagreement)),
        "cycle_disagreement_prevalence": float(np.mean(disagreement[valid])),
    }
    for fraction in REVIEW_FRACTIONS:
        label = f"{int(round(100 * fraction)):02d}"
        flagged = _top_fraction(entropy, valid, fraction)
        retained = valid & ~flagged
        result[f"cycle_disagreement_capture_at_{label}pct"] = _capture(disagreement, flagged)
        result[f"retained_disagreement_at_{100-int(round(100*fraction)):02d}pct"] = (
            float(np.mean(disagreement[retained])) if np.any(retained) else None
        )
    result.update(_gps_metrics(event))
    return result


def _flat_row(result: dict) -> dict:
    return {key: value for key, value in result.items() if not isinstance(value, (dict, list))}


def _format_percent(value) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}\\%"


def _write_latex(results: list[dict], path: Path) -> None:
    by_name = {row["event"]: row for row in results}
    order = ("Ridgecrest", "Amatrice", "Gorkha", "Kumamoto")
    rows = [by_name[name] for name in order]
    lines = [
        "GPS residual--entropy rank correlation & "
        + " & ".join("n/a" if row["gps_residual_entropy_spearman"] is None else f"{row['gps_residual_entropy_spearman']:.3f}" for row in rows)
        + r" \\",
        "Upper-quintile GPS residuals captured at 20\\,\\% review & "
        + " & ".join(_format_percent(row["gps_upper_quintile_capture_at_20pct"]) for row in rows)
        + r" \\",
        "Cycle disagreement captured at 10\\,\\% review & "
        + " & ".join(_format_percent(row["cycle_disagreement_capture_at_10pct"]) for row in rows)
        + r" \\",
        "Cycle disagreement captured at 20\\,\\% review & "
        + " & ".join(_format_percent(row["cycle_disagreement_capture_at_20pct"]) for row in rows)
        + r" \\",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/real_uncertainty_events.json")
    parser.add_argument("--out", default="results/real_uncertainty")
    args = parser.parse_args()
    with open(args.manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    events = manifest.get("events", [])
    if [event.get("name") for event in events] != ["Ridgecrest", "Amatrice", "Gorkha", "Kumamoto"]:
        raise ValueError("manifest events must be ordered Ridgecrest, Amatrice, Gorkha, Kumamoto")
    for event in events:
        if len(event.get("solvers", {})) < 3:
            raise ValueError(f"{event['name']}: declare at least three solver-based products")

    results = [_event_metrics(event) for event in events]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "real_uncertainty_events.csv"
    flat = [_flat_row(row) for row in results]
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    payload = {
        "protocol_version": 1,
        "definitions": {
            "cycle_disagreement": "any declared solver differs from the first by a nonzero integer cycle after median global-offset alignment",
            "gps_rank_endpoint": "Spearman correlation between 5x5 mean entropy and absolute reference-aligned LOS residual",
            "gps_tail": "highest-residual 20% of gated stations",
            "review": "exact highest-entropy fraction, with stable tie handling",
            "calibration_claim": "ranking only; no real-domain probability calibration",
        },
        "manifest": {"path": str(Path(args.manifest).resolve()), "sha256": _sha256(args.manifest)},
        "git_commit": git_commit,
        "results": results,
    }
    json_path = out / "real_uncertainty_events.json"
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    latex_path = out / "real_uncertainty_table_rows.tex"
    _write_latex(results, latex_path)
    print(f"wrote {csv_path}\nwrote {json_path}\nwrote {latex_path}")


if __name__ == "__main__":
    main()
