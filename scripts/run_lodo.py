"""Run the strict leave-one-domain-out (LODO) training/evaluation matrix.

Each run trains on three primary mechanisms and evaluates only on the fourth.
The synthetic generator guarantees that atmospheric fringes are not leaked into
the fault, volcanic, or subsidence training domains.  Evaluation is delegated
to the same auditable benchmark used by the main comparison.

Examples
--------
Inspect all commands without launching training::

    python -m scripts.run_lodo --dry-run

Launch all four train/evaluate runs on a GPU server::

    python -m scripts.run_lodo --mode all --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DOMAINS = ("fault", "volcanic", "subsidence", "atmosphere")


def _hydra_list(values) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def _run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _train_complete(path: Path, args, held_out: str, checkpoint: Path) -> bool:
    if not path.is_file() or not checkpoint.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "held_out_domain": held_out,
        "epochs": args.epochs,
        "epoch_size": args.epoch_size,
        "val_size": args.val_size,
        "checkpoint": str(checkpoint),
    }
    return payload == expected


def _train_command(args, held_out: str, checkpoint_dir: Path) -> list[str]:
    train_domains = [domain for domain in DOMAINS if domain != held_out]
    return [
        sys.executable, "-m", "src.train",
        "regime=pretrain",
        f"device={args.device}",
        "data.source=synthetic",
        "data.synthetic.domain_protocol=true",
        f"data.synthetic.domain_types={_hydra_list(train_domains)}",
        f"data.synthetic.domain_weights={_hydra_list([1.0] * len(train_domains))}",
        "data.synthetic.forced_domain=null",
        f"data.epoch_size={args.epoch_size}",
        f"data.val_size={args.val_size}",
        f"train.epochs={args.epochs}",
        "model.decoder.head_type=edge",
        "model.decoder.out_channels=6",
        "logging.backend=none",
        f"checkpoint.dir={checkpoint_dir.as_posix()}",
        "checkpoint.resume=null",
    ]


def _eval_command(args, held_out: str, checkpoint_dir: Path, result_dir: Path) -> list[str]:
    checkpoint = checkpoint_dir / "pretrain_last.pt"
    return [
        sys.executable, "-m", "scripts.eval_fair_benchmark",
        f"device={args.device}",
        "data.source=synthetic",
        "data.synthetic.domain_protocol=true",
        f"data.synthetic.domain_types={_hydra_list([held_out])}",
        "data.synthetic.domain_weights=[1.0]",
        f"data.synthetic.forced_domain={held_out}",
        f"data.val_size={args.test_size}",
        "eval.split=test",
        "eval.max_batches=null",
        f"checkpoint.resume={checkpoint.as_posix()}",
        f"ext.out_dir={result_dir.as_posix()}",
        "ext.fair_methods=[learned_signed,learned_unsigned,unit_mcf,snaphu]",
        "ext.fair_snaphu_sources=[supplied]",
        "ext.fair_snaphu_costs=[defo,smooth]",
        f"ext.fair_snaphu_nlooks={_hydra_list(args.nlooks)}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval", "all"), default="all")
    parser.add_argument(
        "--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS),
        help="held-out folds to run (default: all four); useful for one fold per GPU",
    )
    parser.add_argument("--device", default="cuda")
    # Match the 50-epoch budget used by the main learned-cost checkpoint.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--epoch-size", type=int, default=30000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--nlooks", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/lodo"))
    parser.add_argument("--result-root", type=Path, default=Path("results/lodo"))
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="reuse an existing fold checkpoint/result instead of overwriting it",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = {
        "protocol_version": 1,
        "domains": list(args.domains),
        "definition": (
            "Train on three mutually exclusive primary mechanisms and evaluate on "
            "the held-out fourth; topography and flat-earth phase are common nuisances."
        ),
        "settings": vars(args) | {
            "checkpoint_root": str(args.checkpoint_root),
            "result_root": str(args.result_root),
        },
    }
    if not args.dry_run:
        args.result_root.mkdir(parents=True, exist_ok=True)
        suffix = "" if set(args.domains) == set(DOMAINS) else "_" + "-".join(args.domains)
        with open(args.result_root / f"lodo_protocol{suffix}.json", "w", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2, sort_keys=True)

    for held_out in args.domains:
        checkpoint_dir = args.checkpoint_root / f"holdout_{held_out}"
        result_dir = args.result_root / f"holdout_{held_out}"
        if args.mode in ("train", "all"):
            checkpoint = checkpoint_dir / "pretrain_last.pt"
            completion = checkpoint_dir / "lodo_train_complete.json"
            if args.skip_existing and _train_complete(completion, args, held_out, checkpoint):
                print(f"reuse {checkpoint}", flush=True)
            else:
                _run(_train_command(args, held_out, checkpoint_dir), args.dry_run)
                if not args.dry_run:
                    completion.write_text(json.dumps({
                        "held_out_domain": held_out,
                        "epochs": args.epochs,
                        "epoch_size": args.epoch_size,
                        "val_size": args.val_size,
                        "checkpoint": str(checkpoint),
                    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.mode in ("eval", "all"):
            if not args.dry_run and not (checkpoint_dir / "pretrain_last.pt").is_file():
                raise FileNotFoundError(
                    f"Missing {checkpoint_dir / 'pretrain_last.pt'}; run --mode train first"
                )
            # The benchmark itself is row-resumable; always invoke it so a
            # partial per-scene CSV is completed rather than mistaken for done.
            _run(_eval_command(args, held_out, checkpoint_dir, result_dir), args.dry_run)

    if args.mode in ("eval", "all") and set(args.domains) == set(DOMAINS):
        command = [sys.executable, "-m", "scripts.summarize_lodo", str(args.result_root)]
        _run(command, args.dry_run)


if __name__ == "__main__":
    main()
