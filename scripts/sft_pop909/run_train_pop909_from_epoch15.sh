#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULTS_FILE="${DEFAULTS_FILE:-$ROOT_DIR/scripts/sft_pop909/train_pop909_from_epoch15.defaults.sh}"
export DEFAULTS_FILE

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif [[ -x "/data/home/yuanxin/RT-accompanimentV2/.venv/bin/python" ]]; then
    PYTHON_BIN="/data/home/yuanxin/RT-accompanimentV2/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
  export PYTHON_BIN
fi

exec "$ROOT_DIR/scripts/run_train_experiment.sh" "$@"
