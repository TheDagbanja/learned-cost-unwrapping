"""Turn ``outputs/dl_events_summary.csv`` (written by ``unwrap_geotiff_dl.py``)
into Table IV's ``DL (range)``/``DL slip`` cells: per-event min-max across the
six per-pixel DL baselines, high-coherence (gamma>0.7) column, plus the
across-all-events range for the table's footnote.

    python -m scripts.summarize_dl_events --csv outputs/dl_events_summary.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/dl_events_summary.csv")
    ap.add_argument("--methods", default="dlpu_cnn,phasenet2,gradient_net,"
                                         "attention_unet,deeplabv3plus,unetpp",
                    help="comma-separated method names expected per event")
    args = ap.parse_args()
    expected = set(args.methods.split(","))

    rows = defaultdict(list)
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows[r["event"]].append(r)

    print(f"{'event':<14s}{'n':>3s}  {'agree_hi range (rad)':<24s}{'slip_hi range (%)':<20s}")
    all_agree, all_slip = [], []
    for event, rs in rows.items():
        have = {r["method"] for r in rs}
        missing = expected - have
        agree = [float(r["agree_hi_rad"]) for r in rs]
        slip = [float(r["slip_hi_pct"]) for r in rs]
        all_agree += agree
        all_slip += slip
        flag = f"  MISSING: {sorted(missing)}" if missing else ""
        print(f"{event:<14s}{len(rs):>3d}  "
              f"{min(agree):.2f}--{max(agree):.2f}{'':<14s}"
              f"{min(slip):.1f}--{max(slip):.1f}{'':<10s}{flag}")

    if all_agree:
        print(f"\nAcross all events: {min(all_agree):.1f}--{max(all_agree):.1f} rad, "
              f"{min(all_slip):.1f}--{max(all_slip):.1f} % slip "
              f"(for the Table IV footnote)")

    print("\nLaTeX row fragments (paste into Table IV, replacing \\TODO{--}):")
    for event, rs in rows.items():
        agree = [float(r["agree_hi_rad"]) for r in rs]
        slip = [float(r["slip_hi_pct"]) for r in rs]
        print(f"  {event:<12s}: {min(agree):.2f}--{max(agree):.2f} & "
              f"{min(slip):.0f}--{max(slip):.0f}\\,\\%")


if __name__ == "__main__":
    main()
