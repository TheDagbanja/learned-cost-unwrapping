"""Consolidate strict LODO benchmark outputs into manuscript-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DOMAINS = ("fault", "volcanic", "subsidence", "atmosphere")
METHODS = {
    "signed_learned_mcf": "learned signed cost + MCF",
    "unsigned_learned_mcf": "learned unsigned cost + MCF",
    "unit_mcf": "unit-cost MCF",
}


def _cell(row: dict[str, str]) -> str:
    """Format mean / median / P95 / CVaR95 for the manuscript table."""
    return " / ".join(f"{float(row[key]):.3f}" for key in ("mean", "median", "p95", "cvar95"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("results/lodo"))
    args = parser.parse_args()
    rows = []
    for domain in DOMAINS:
        path = args.root / f"holdout_{domain}" / "fair_benchmark_test.csv"
        if not path.is_file():
            raise FileNotFoundError(f"LODO result missing: {path}")
        with open(path, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({"held_out_domain": domain, **row})

    out = args.root / "lodo_summary.csv"
    fields = ["held_out_domain", *[key for key in rows[0] if key != "held_out_domain"]]
    with open(out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table_rows, selections = [], {}
    for domain in DOMAINS:
        domain_rows = [row for row in rows if row["held_out_domain"] == domain]
        by_method = {row["method"]: row for row in domain_rows}
        missing = [name for name in METHODS.values() if name not in by_method]
        if missing:
            raise ValueError(f"{domain} is missing required methods: {missing}")
        snaphu = [row for row in domain_rows if row["method"].startswith("SNAPHU")]
        if not snaphu:
            raise ValueError(f"{domain} has no SNAPHU sensitivity rows")
        best = min(snaphu, key=lambda row: float(row["mean"]))
        selections[domain] = {
            "selection_rule": "minimum held-out-fold mean within the declared sensitivity grid",
            "method": best["method"],
            "row": best,
        }
        table_rows.append({
            "held_out_domain": domain,
            **{key: _cell(by_method[name]) for key, name in METHODS.items()},
            "best_audited_snaphu": _cell(best),
        })

    table_path = args.root / "lodo_table.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    selection_path = args.root / "lodo_snaphu_selection.json"
    with open(selection_path, "w", encoding="utf-8") as stream:
        json.dump(selections, stream, indent=2, sort_keys=True)
    print(f"wrote {out}\nwrote {table_path}\nwrote {selection_path}")


if __name__ == "__main__":
    main()
