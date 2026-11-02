"""Freeze a SNAPHU configuration on calibration data and retrieve its test row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration_summary")
    parser.add_argument("test_summary")
    parser.add_argument("--metric", choices=("mean", "cvar95", "p95"), default="mean")
    parser.add_argument(
        "--additional-summary", action="append", default=[], metavar="LABEL=CSV",
        help="retrieve the already-frozen method from another evaluation summary",
    )
    parser.add_argument("--out", default="results/fair_benchmark/snaphu_selection.json")
    args = parser.parse_args()

    calibration = [row for row in _rows(args.calibration_summary) if row["method"].startswith("SNAPHU")]
    test = {row["method"]: row for row in _rows(args.test_summary)}
    if not calibration:
        raise ValueError("Calibration summary has no SNAPHU rows")
    selected = min(calibration, key=lambda row: float(row[args.metric]))
    if selected["method"] not in test:
        raise ValueError(f"Selected method is absent from test summary: {selected['method']}")

    payload = {
        "selection_rule": f"minimum calibration {args.metric}",
        "selected_method": selected["method"],
        "selected_information": selected["information"],
        "selected_cost": selected["snaphu_cost"],
        "selected_nlooks": float(selected["nlooks"]),
        "calibration_row": selected,
        "test_row": test[selected["method"]],
        "sources": {
            "calibration_summary": {
                "path": str(Path(args.calibration_summary).resolve()),
                "sha256": _sha256(args.calibration_summary),
            },
            "test_summary": {
                "path": str(Path(args.test_summary).resolve()),
                "sha256": _sha256(args.test_summary),
            },
        },
    }
    additional_rows = {}
    for specification in args.additional_summary:
        if "=" not in specification:
            raise ValueError("--additional-summary expects LABEL=CSV")
        label, path = specification.split("=", 1)
        rows = {row["method"]: row for row in _rows(path)}
        if selected["method"] not in rows:
            raise ValueError(f"Selected method is absent from {label}: {selected['method']}")
        additional_rows[label] = {
            "row": rows[selected["method"]],
            "source": {"path": str(Path(path).resolve()), "sha256": _sha256(path)},
        }
    if additional_rows:
        payload["additional_evaluations"] = additional_rows
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
