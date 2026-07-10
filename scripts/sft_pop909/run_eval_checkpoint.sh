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

CKPT="${CKPT:-}"
DATA_DIR="${DATA_DIR:-/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/test}"
MODE="${MODE:-test}"
TEST_SPLIT_RATIO="${TEST_SPLIT_RATIO:-1.0}"
RANDOM_SEED="${RANDOM_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-0}"
DEVICE="${DEVICE:-cuda}"
OUT_JSON="${OUT_JSON:-}"

if [[ -z "$CKPT" ]]; then
  echo "CKPT is required" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/sft_pop909/eval_checkpoint.py"
  --ckpt "$CKPT"
  --data-dir "$DATA_DIR"
  --mode "$MODE"
  --test-split-ratio "$TEST_SPLIT_RATIO"
  --random-seed "$RANDOM_SEED"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --max-batches "$MAX_BATCHES"
  --device "$DEVICE"
)

if [[ -n "$OUT_JSON" ]]; then
  CMD+=(--out-json "$OUT_JSON")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
