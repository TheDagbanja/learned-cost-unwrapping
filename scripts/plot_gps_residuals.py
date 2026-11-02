"""Per-station GPS residual plot -- a clearer view of the solver-based tie than
the raw scatter.

The raw InSAR-vs-GPS scatter hides the three-way tie because the tied markers
(ours / SNAPHU / DENet) sit on top of one another and, at small signal, all
points cluster near the origin. This plot instead shows the *aligned residual*
(InSAR - GPS, one global constant removed) per station, one series per method:
the solver-based methods hug the zero line and overlap, while the pure deep
networks spike away from it. That makes the tie, and the separation from pure
DL, immediately legible -- suitable as a supplementary figure for Amatrice and
the expanded (large-ROI) Kumamoto validation.

    python -m scripts.plot_gps_residuals \\
        outputs/gps_amatrice_gated_stations.csv \\
        --min-coherence 0.5 --title "Amatrice" --out outputs/gps_amatrice_residuals

Reads the same CSV schema as scripts.reproduce_gps_tables; writes PNG + PDF.
"""

from __future__ import annotations

import argparse
import csv
import math

LAMBDA_C_BAND = 0.055465763
CM_PER_RAD = LAMBDA_C_BAND / (4 * math.pi) * 100

# Wong palette, matching scripts/paper_figures.py.
C = {"snaphu": "#E69F00", "ours": "#0072B2", "denet": "#56B4E9",
     "dl": "#D55E00", "other": "#999999"}
SOLVER = {"snaphu", "ours", "denet"}


def _fnum(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--min-coherence", type=float, default=0.5)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="outputs/gps_residuals")
    args = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    methods = [c.replace("_phase", "") for c in rows[0]
               if c.endswith("_phase") and c != "gps_phase"]
    has_coh = "coherence" in rows[0]

    kept = []
    for r in rows:
        gps = _fnum(r.get("gps_phase"))
        vals = {m: _fnum(r.get(m + "_phase")) for m in methods}
        if gps is None or any(v is None for v in vals.values()):
            continue
        if has_coh and args.min_coherence > 0:
            coh = _fnum(r.get("coherence"))
            if coh is None or coh < args.min_coherence:
                continue
        kept.append((r["station"], gps, vals))

    kept.sort(key=lambda x: x[1])                       # sort by GPS LOS phase
    stations = [k[0] for k in kept]
    gps = np.array([k[1] for k in kept])
    x = np.arange(len(kept))

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    ax.axhline(0, color="k", lw=0.6, zorder=0)
    for m in methods:
        pred = np.array([k[2][m] for k in kept])
        d = pred - gps
        d = d - d.mean()                                # aligned residual (rad -> cm)
        fam = m if m in SOLVER else ("dl" if m not in ("gt_grad",) else "other")
        col = C.get(fam, C["other"])
        solver = m in SOLVER
        ax.plot(x, d * CM_PER_RAD, marker="o" if solver else "s",
                ms=4 if solver else 3, lw=1.2 if solver else 0.0,
                color=col, alpha=0.95 if solver else 0.7,
                label=m + (" (solver)" if solver else ""), zorder=3 if solver else 2)
    ax.set_xticks(x); ax.set_xticklabels(stations, rotation=90, fontsize=5)
    ax.set_ylabel("aligned residual vs GPS (cm)")
    ax.set_xlabel("station (sorted by GPS LOS phase)")
    if args.title:
        ax.set_title(f"{args.title}: per-station GPS residual  (n={len(kept)})")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print(f"wrote {args.out}.png (+ .pdf)   n={len(kept)} stations")


if __name__ == "__main__":
    main()
