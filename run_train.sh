#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Remote (Linux + GPU) launcher for pretraining / adaptation.
# Usage:
#   ./run_train.sh                       # defaults from configs/default.yaml
#   ./run_train.sh data.batch_size=16 train.lr=1e-3 device=cuda:0
# Any argument is forwarded verbatim to Hydra as a config override.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve repo root regardless of where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Make 'src' importable as a package (python -m src.train).
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Reasonable thread defaults; override in the environment if needed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# export CUDA_VISIBLE_DEVICES=0          # pin a GPU if desired
# export WANDB_MODE=offline              # uncomment to log without network

echo "[run_train] python: $(command -v python)"
echo "[run_train] overrides: $*"

python -m src.train "$@"
