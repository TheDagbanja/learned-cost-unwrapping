"""Reproduce the manuscript's GPS validation numbers directly from the
``*_stations.csv`` files written by ``scripts/gps_validate.py``.

This is an independent, turnkey re-derivation of Table V (and the Amatrice /
Kumamoto tie values in Sec. V-F): it re-applies the coherence gate and the
aligned-RMSE metric to the committed per-station CSVs, so a coauthor or reviewer
can confirm the station counts and per-method residuals without rerunning the
unwrappers or touching the rasters. The CSV already stores the sign-calibrated
GPS phase (gps_validate applies the SNAPHU-correlation flip before writing), so
no re-calibration is needed here.

    python -m scripts.reproduce_gps_tables \\
        outputs/gps_ridgecrest_v2_gated_stations.csv \\
        outputs/gps_amatrice_gated_stations.csv \\
        outputs/gps_kumamoto_big_stations.csv \\
        --min-coherence 0.5

Aligned RMSE per method = std of (pred - gps) over the common, gated stations
(one global reference constant removed), matching gps_validate.resid_rmse.
"""

from __future__ import annotations

import argparse
import csv
import math

LAMBDA_C_BAND = 0.055465763          # m; rad -> cm factor below
CM_PER_RAD = LAMBDA_C_BAND / (4 * math.pi) * 100


def _load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty")
    method_cols = [c for c in rows[0] if c.endswith("_phase") and c != "gps_phase"]
    has_coh = "coherence" in rows[0]
    return rows, method_cols, has_coh


def _fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _aligned_rmse(pred, gps):
    d = [p - g for p, g in zip(pred, gps)]
    m = sum(d) / len(d)
    return math.sqrt(sum((x - m) ** 2 for x in d) / len(d))


def reproduce(path, min_coh):
    rows, method_cols, has_coh = _load(path)
    methods = ["snaphu_phase"] + [c for c in method_cols if c != "snaphu_phase"]

    kept = []
    dropped_gate = []
    for r in rows:
        vals = {c: _fnum(r.get(c)) for c in methods}
        gps = _fnum(r.get("gps_phase"))
        if gps is None or any(v is None for v in vals.values()):
            continue                                   # not common to all methods
        if has_coh and min_coh > 0:
            coh = _fnum(r.get("coherence"))
            if coh is None or coh < min_coh:
                dropped_gate.append((r["station"], coh))
                continue
        kept.append((r["station"], gps, vals))

    n = len(kept)
    print(f"\n=== {path} ===")
    if has_coh and min_coh > 0:
        drop = ", ".join(f"{s}({c:.2f})" for s, c in dropped_gate) or "none"
        print(f"coherence gate >= {min_coh:.2f}: dropped {len(dropped_gate)} common station(s): {drop}")
    elif not has_coh and min_coh > 0:
        print(f"(no coherence column in CSV; gate not applied)")
    print(f"common stations scored: n = {n}")
    print(f"{'method':16s}{'RMSE (rad)':>12}{'RMSE (cm)':>11}")
    gps = [g for _, g, _ in kept]
    table = []
    for c in methods:
        pred = [v[c] for _, _, v in kept]
        rmse = _aligned_rmse(pred, gps)
        table.append((c.replace("_phase", ""), rmse))
    for name, rmse in sorted(table, key=lambda x: x[1]):
        print(f"{name:16s}{rmse:12.3f}{rmse * CM_PER_RAD:11.1f}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="one or more *_stations.csv files")
    ap.add_argument("--min-coherence", type=float, default=0.5)
    args = ap.parse_args()
    for path in args.csv:
        reproduce(path, args.min_coherence)


if __name__ == "__main__":
    main()
