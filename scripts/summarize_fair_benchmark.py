"""Regenerate an audited fair-benchmark summary from existing per-scene rows."""

from __future__ import annotations

import argparse
import os

from scripts.eval_fair_benchmark import _write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("per_scene_csv")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    output = args.out or args.per_scene_csv.replace("_per_scene.csv", ".csv")
    if output == args.per_scene_csv:
        raise ValueError("--out is required when the input does not end in _per_scene.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    _write_summary(args.per_scene_csv, output)
    print(output)


if __name__ == "__main__":
    main()
