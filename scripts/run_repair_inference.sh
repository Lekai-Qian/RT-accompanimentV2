#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEFAULTS_FILE="${DEFAULTS_FILE:-$ROOT_DIR/scripts/repair_inference.defaults.env}"

if [[ -f "$DEFAULTS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULTS_FILE"
fi

JOB="${JOB:-repair_jobs/example_job.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-repair_outputs}"
DEVICE="${DEVICE:-auto}"
USE_FP16="${USE_FP16:-false}"
SEED="${SEED:-}"
CKPT="${CKPT:-}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_K="${TOP_K:-}"
TOP_P="${TOP_P:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"

CMD=("$PYTHON_BIN" "$ROOT_DIR/repair_inference.py" --job "$JOB" --output-root "$OUTPUT_ROOT" --device "$DEVICE")

if [[ "$USE_FP16" == "true" || "$USE_FP16" == "TRUE" || "$USE_FP16" == "1" ]]; then
  CMD+=(--use-fp16)
elif [[ "$USE_FP16" == "false" || "$USE_FP16" == "FALSE" || "$USE_FP16" == "0" ]]; then
  :
else
  echo "Invalid USE_FP16=$USE_FP16 (expect true/false)" >&2
  exit 1
fi

if [[ -n "$SEED" ]]; then
  CMD+=(--seed "$SEED")
fi
if [[ -n "$CKPT" ]]; then
  CMD+=(--ckpt "$CKPT")
fi
if [[ -n "$TEMPERATURE" ]]; then
  CMD+=(--temperature "$TEMPERATURE")
fi
if [[ -n "$TOP_K" ]]; then
  CMD+=(--top-k "$TOP_K")
fi
if [[ -n "$TOP_P" ]]; then
  CMD+=(--top-p "$TOP_P")
fi
if [[ -n "$REPETITION_PENALTY" ]]; then
  CMD+=(--repetition-penalty "$REPETITION_PENALTY")
fi

CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
