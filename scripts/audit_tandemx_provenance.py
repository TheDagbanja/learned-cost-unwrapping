"""Create a reproducible provenance and content audit for InSAR-DLPU TanDEM-X."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src import dataset as dataset_module, mcf


DATASET_REPOSITORY = "https://github.com/zhoulifan/InSAR-DLPU"
DATASET_DOI = "https://doi.org/10.1109/MGRS.2024.3359691"
PUFORMER_DOI = "https://doi.org/10.1109/JSTARS.2026.3677162"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--out", default="results/tandemx_provenance")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root = dataset_module.resolve_dlpu_root(args.root)
    wrapped_dir = os.path.join(root, "test_wrapped_real")
    reference_dir = os.path.join(root, "test_absolute_real")
    wrapped_files = sorted(Path(wrapped_dir).glob("*.mat"))
    reference_files = sorted(Path(reference_dir).glob("*.mat"))
    if [path.name for path in wrapped_files] != [path.name for path in reference_files]:
        raise ValueError("Wrapped and reference TanDEM-X filenames do not match exactly")

    rows = []
    shapes = set()
    for wrapped_path, reference_path in zip(wrapped_files, reference_files):
        wrapped = dataset_module._load_mat_array(str(wrapped_path), "input")
        reference = dataset_module._load_mat_array(str(reference_path), "output")
        shapes.add((tuple(wrapped.shape), tuple(reference.shape)))
        circular = mcf.wrap(wrapped - mcf.wrap(reference))
        circular_offset = float(np.angle(np.mean(np.exp(1j * circular))))
        circular_aligned = mcf.wrap(circular - circular_offset)
        rows.append({
            "name": wrapped_path.name,
            "wrapped_sha256": _sha256(str(wrapped_path)),
            "reference_sha256": _sha256(str(reference_path)),
            "height": wrapped.shape[0], "width": wrapped.shape[1],
            "wrapped_min": float(wrapped.min()), "wrapped_max": float(wrapped.max()),
            "reference_ptp_rad": float(np.ptp(reference)),
            "wrapped_vs_reference_circular_rmse": float(np.sqrt(np.mean(circular**2))),
            "wrapped_vs_reference_aligned_circular_rmse": float(np.sqrt(np.mean(circular_aligned**2))),
            "wrapped_vs_reference_circular_mae": float(np.mean(np.abs(circular))),
            "wrapped_vs_reference_resultant_length": float(abs(np.mean(np.exp(1j * circular)))),
        })

    csv_path = os.path.join(args.out, "tandemx_reference_inventory.csv")
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "pairs": len(rows),
        "shapes": [[list(a), list(b)] for a, b in sorted(shapes)],
        "local_variables": {"wrapped": "input", "reference": "output"},
        "coherence_in_release": False,
        "geolocation_in_release": False,
        "sensor_metadata_in_release": False,
        "reference_construction_metadata_in_release": False,
        "mean_circular_rmse_rad": float(np.mean([row["wrapped_vs_reference_circular_rmse"] for row in rows])),
        "mean_aligned_circular_rmse_rad": float(np.mean([row["wrapped_vs_reference_aligned_circular_rmse"] for row in rows])),
        "mean_resultant_length": float(np.mean([row["wrapped_vs_reference_resultant_length"] for row in rows])),
        "median_reference_ptp_rad": float(np.median([row["reference_ptp_rad"] for row in rows])),
    }
    with open(os.path.join(args.out, "tandemx_reference_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)

    report = f"""# TanDEM-X reference provenance audit

## Finding

The 100 InSAR-DLPU TanDEM-X targets should be described as **DEM-derived reference
absolute phase**, not as independently verified physical ground truth.  The available
evidence does not indicate that the targets were produced by SNAPHU or another phase
unwrapper: the later PUformer paper by the dataset authors identifies the displayed
TanDEM-X reference as obtained from SRTM DEM.  However, the released `.mat` files do
not include the geolocation, baseline, incidence geometry, phase-to-height conversion,
coregistration, filtering, uncertainty, or manual-correction record needed to reproduce
or independently validate that conversion.

## Source trail

1. The [official InSAR-DLPU repository]({DATASET_REPOSITORY}) describes 100 real
   TanDEM-X interferograms and paired absolute-phase patches, and labels the latter
   ground truth.  It does not document their construction.
2. The dataset article ([DOI 10.1109/MGRS.2024.3359691]({DATASET_DOI})) establishes
   the benchmark and its SRTM basis but the public abstract and repository do not
   supply scene-level provenance.
3. The later paper by the dataset authors, PUformer
   ([DOI 10.1109/JSTARS.2026.3677162]({PUFORMER_DOI})), states in the Fig. 8 caption
   that the TanDEM-X absolute phase is obtained from SRTM DEM.  This is the clearest
   published provenance statement located in this audit.

## Local release audit

- Pair count: {summary['pairs']}.
- Array shapes: {summary['shapes']}.
- Each wrapped file contains only the MATLAB variable `input`; each paired target
  contains only `output`.
- The release contains no paired coherence arrays, geolocation, acquisition metadata,
  sensor geometry, or per-scene reference-generation log.
- Mean circular discrepancy after removing the per-scene global circular offset between
  the released wrapped observation and the wrapped DEM-derived reference:
  {summary['mean_aligned_circular_rmse_rad']:.3f} rad (mean circular resultant length
  {summary['mean_resultant_length']:.3f}).  This confirms
  that the observation is not merely an exact numerical rewrap of the target and is
  consistent with sensor/noise and processing differences, but it does not establish
  physical accuracy.
- Median target peak-to-peak range: {summary['median_reference_ptp_rad']:.2f} rad.
- File-level SHA-256 hashes and per-scene diagnostics are recorded in
  `tandemx_reference_inventory.csv`.

## Consequence for evaluation and wording

The subset can measure agreement with a DEM-derived topographic reference.  It cannot,
by itself, resolve which phase-unwrapping result is physically correct where the methods
and reference disagree.  Tables should use `Reference RMSE`, the text should use
`TanDEM-X/SRTM reference subset`, and claims of real-data superiority should rely on
independent measurements or be withdrawn.  The Sentinel-1 GPS experiment remains an
independent check, but its current precision supports comparability among the
solver-based methods rather than superiority of one method.
"""
    with open(os.path.join(args.out, "TANDEMX_PROVENANCE_AUDIT.md"), "w", encoding="utf-8") as stream:
        stream.write(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
