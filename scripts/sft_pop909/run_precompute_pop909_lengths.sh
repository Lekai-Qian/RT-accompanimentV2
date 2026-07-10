#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif [[ -x "/data/home/yuanxin/RT-accompanimentV2/.venv/bin/python" ]]; then
    PYTHON_BIN="/data/home/yuanxin/RT-accompanimentV2/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DATA_DIR="${DATA_DIR:-/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/train}"
WORKERS="${WORKERS:-24}"
OVERWRITE="${OVERWRITE:-false}"

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/sft_pop909/precompute_lengths.py"
  --data-dir "$DATA_DIR"
  --workers "$WORKERS"
)

if [[ "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" || "$OVERWRITE" == "1" ]]; then
  CMD+=(--overwrite)
elif [[ "$OVERWRITE" != "false" && "$OVERWRITE" != "FALSE" && "$OVERWRITE" != "0" ]]; then
  echo "Invalid OVERWRITE=$OVERWRITE (expect true/false)" >&2
  exit 1
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
