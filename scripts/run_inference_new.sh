#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEFAULTS_FILE="${DEFAULTS_FILE:-$ROOT_DIR/scripts/inference_new.defaults.env}"

if [[ -f "$DEFAULTS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULTS_FILE"
fi

CKPT="${CKPT:-}"
CKPT_GLOB="${CKPT_GLOB:-}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SEED="${SEED:-20260317}"
OUTPUT_DIR="${OUTPUT_DIR:-generated_samples_repro}"
DATASET_MODE="${DATASET_MODE:-test}"
SAMPLING_MODE="${SAMPLING_MODE:-with_replacement}"
GT_PREFIX_BEATS="${GT_PREFIX_BEATS:-12}"
EXPORT_GT_MIDI="${EXPORT_GT_MIDI:-true}"

if [[ -z "$CKPT" && -n "$CKPT_GLOB" ]]; then
  CKPT_PATTERN="$CKPT_GLOB"
  if [[ "$CKPT_PATTERN" != /* ]]; then
    CKPT_PATTERN="$ROOT_DIR/$CKPT_PATTERN"
  fi
  shopt -s nullglob
  MATCHED_CKPTS=($CKPT_PATTERN)
  shopt -u nullglob
  if [[ ${#MATCHED_CKPTS[@]} -eq 0 ]]; then
    echo "No checkpoints matched CKPT_GLOB=$CKPT_GLOB" >&2
    exit 1
  fi
  CKPT="$(ls -1t "${MATCHED_CKPTS[@]}" | head -n 1)"
fi

if [[ -z "$CKPT" ]]; then
  CKPT="checkpoints-drop-0.2/epoch_3_0311_1354/model.safetensors"
fi

if [[ "$CKPT" != /* ]]; then
  CKPT="$ROOT_DIR/$CKPT"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/inference_new.py"
  --ckpt "$CKPT"
  --num-samples "$NUM_SAMPLES"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  --dataset-mode "$DATASET_MODE"
  --sampling-mode "$SAMPLING_MODE"
  --gt-prefix-beats "$GT_PREFIX_BEATS"
)

if [[ "$EXPORT_GT_MIDI" == "true" || "$EXPORT_GT_MIDI" == "TRUE" || "$EXPORT_GT_MIDI" == "1" ]]; then
  CMD+=(--export-gt-midi)
elif [[ "$EXPORT_GT_MIDI" == "false" || "$EXPORT_GT_MIDI" == "FALSE" || "$EXPORT_GT_MIDI" == "0" ]]; then
  CMD+=(--no-export-gt-midi)
else
  echo "Invalid EXPORT_GT_MIDI=$EXPORT_GT_MIDI (expect true/false)" >&2
  exit 1
fi

# 支持临时追加/覆盖参数，例如:
# bash scripts/run_inference_new.sh --num-samples 5 --seed 1
CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
