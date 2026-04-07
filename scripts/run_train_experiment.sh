#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEFAULTS_FILE="${DEFAULTS_FILE:-}"

if [[ -z "$DEFAULTS_FILE" ]]; then
  echo "Please provide DEFAULTS_FILE, for example:" >&2
  echo "  DEFAULTS_FILE=$ROOT_DIR/scripts/train_initdrop8.env ./scripts/run_train_experiment.sh" >&2
  exit 1
fi

if [[ -f "$DEFAULTS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULTS_FILE"
else
  echo "DEFAULTS_FILE not found: $DEFAULTS_FILE" >&2
  exit 1
fi

CKPT="${CKPT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
ACC_DROP_PROB="${ACC_DROP_PROB:-}"
DROP_INITIAL_BEATS="${DROP_INITIAL_BEATS:-}"
DROP_INITIAL_BEATS_PROB="${DROP_INITIAL_BEATS_PROB:-}"
POS_SHIFT_MAX="${POS_SHIFT_MAX:-}"
DATA_DIR="${DATA_DIR:-}"
PRINT_CONFIG="${PRINT_CONFIG:-true}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

if [[ -n "$CKPT" && "$CKPT" != /* ]]; then
  CKPT="$ROOT_DIR/$CKPT"
fi
if [[ -n "$OUTPUT_ROOT" && "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
fi
if [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi
if [[ -n "$DATA_DIR" && "$DATA_DIR" != /* ]]; then
  DATA_DIR="$ROOT_DIR/$DATA_DIR"
fi

CMD=("$PYTHON_BIN" "$ROOT_DIR/train.py")

if [[ -n "$CKPT" ]]; then
  CMD+=(--ckpt "$CKPT")
fi
if [[ -n "$EXPERIMENT_NAME" ]]; then
  CMD+=(--experiment-name "$EXPERIMENT_NAME")
fi
if [[ -n "$OUTPUT_ROOT" ]]; then
  CMD+=(--output-root "$OUTPUT_ROOT")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$NUM_EPOCHS" ]]; then
  CMD+=(--num-epochs "$NUM_EPOCHS")
fi
if [[ -n "$ACC_DROP_PROB" ]]; then
  CMD+=(--acc-drop-prob "$ACC_DROP_PROB")
fi
if [[ -n "$DROP_INITIAL_BEATS" ]]; then
  CMD+=(--drop-initial-beats "$DROP_INITIAL_BEATS")
fi
if [[ -n "$DROP_INITIAL_BEATS_PROB" ]]; then
  CMD+=(--drop-initial-beats-prob "$DROP_INITIAL_BEATS_PROB")
fi
if [[ -n "$POS_SHIFT_MAX" ]]; then
  CMD+=(--pos-shift-max "$POS_SHIFT_MAX")
fi
if [[ -n "$DATA_DIR" ]]; then
  CMD+=(--data-dir "$DATA_DIR")
fi

if [[ "$PRINT_CONFIG" == "true" || "$PRINT_CONFIG" == "TRUE" || "$PRINT_CONFIG" == "1" ]]; then
  CMD+=(--print-config)
elif [[ "$PRINT_CONFIG" != "false" && "$PRINT_CONFIG" != "FALSE" && "$PRINT_CONFIG" != "0" ]]; then
  echo "Invalid PRINT_CONFIG=$PRINT_CONFIG (expect true/false)" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" || "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
elif [[ "$DRY_RUN" != "false" && "$DRY_RUN" != "FALSE" && "$DRY_RUN" != "0" ]]; then
  echo "Invalid DRY_RUN=$DRY_RUN (expect true/false)" >&2
  exit 1
fi

CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
