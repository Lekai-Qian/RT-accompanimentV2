#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEFAULTS_FILE="${DEFAULTS_FILE:-$ROOT_DIR/scripts/interval_rollout_inference.defaults.env}"

if [[ -f "$DEFAULTS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULTS_FILE"
fi

CKPT="${CKPT:-}"
DEVICE="${DEVICE:-auto}"
USE_FP16="${USE_FP16:-false}"
DATA_DIR="${DATA_DIR:-}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
SEED="${SEED:-42}"
INDICES_FILE="${INDICES_FILE:-}"
SAMPLING_MODE="${SAMPLING_MODE:-with_replacement}"
DATASET_MODE="${DATASET_MODE:-test}"
DATASET_SEED="${DATASET_SEED:-42}"
TEST_SPLIT_RATIO="${TEST_SPLIT_RATIO:-0.10}"
CACHE_LENGTHS="${CACHE_LENGTHS:-false}"
GT_PREFIX_BEATS="${GT_PREFIX_BEATS:-0}"
INFERENCE_INTERVAL="${INFERENCE_INTERVAL:-1}"
GENERATION_LENGTH="${GENERATION_LENGTH:-1}"
CATCH_UP_PROB="${CATCH_UP_PROB:-1.0}"
FALLBACK_POLICY="${FALLBACK_POLICY:-hold_last}"
MAX_TICKS="${MAX_TICKS:-}"
PER_SAMPLE_TIMEOUT_SEC="${PER_SAMPLE_TIMEOUT_SEC:-0}"
PER_SAMPLE_TIMEOUT_FACTOR="${PER_SAMPLE_TIMEOUT_FACTOR:-2.0}"
OUTPUT_DIR="${OUTPUT_DIR:-local_tmp/generated_samples_interval_rollout}"
EXPORT_GT_MIDI="${EXPORT_GT_MIDI:-true}"
TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_K="${TOP_K:-10}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

if [[ -z "$CKPT" ]]; then
  echo "CKPT is required" >&2
  exit 1
fi

if [[ "$CKPT" != /* ]]; then
  CKPT="$ROOT_DIR/$CKPT"
fi
if [[ -n "$DATA_DIR" && "$DATA_DIR" != /* ]]; then
  DATA_DIR="$ROOT_DIR/$DATA_DIR"
fi
if [[ -n "$INDICES_FILE" && "$INDICES_FILE" != /* ]]; then
  INDICES_FILE="$ROOT_DIR/$INDICES_FILE"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/Inference/interval_rollout_inference.py"
  --ckpt "$CKPT"
  --device "$DEVICE"
  --num-samples "$NUM_SAMPLES"
  --seed "$SEED"
  --sampling-mode "$SAMPLING_MODE"
  --dataset-mode "$DATASET_MODE"
  --dataset-seed "$DATASET_SEED"
  --test-split-ratio "$TEST_SPLIT_RATIO"
  --gt-prefix-beats "$GT_PREFIX_BEATS"
  --inference-interval "$INFERENCE_INTERVAL"
  --generation-length "$GENERATION_LENGTH"
  --catch-up-prob "$CATCH_UP_PROB"
  --fallback-policy "$FALLBACK_POLICY"
  --output-dir "$OUTPUT_DIR"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --top-p "$TOP_P"
  --repetition-penalty "$REPETITION_PENALTY"
)

if [[ -n "$DATA_DIR" ]]; then
  CMD+=(--data-dir "$DATA_DIR")
fi
if [[ -n "$INDICES_FILE" ]]; then
  CMD+=(--indices-file "$INDICES_FILE")
fi
if [[ -n "$MAX_TICKS" ]]; then
  CMD+=(--max-ticks "$MAX_TICKS")
fi
if [[ -n "$PER_SAMPLE_TIMEOUT_SEC" ]]; then
  CMD+=(--per-sample-timeout-sec "$PER_SAMPLE_TIMEOUT_SEC")
fi
if [[ -n "$PER_SAMPLE_TIMEOUT_FACTOR" ]]; then
  CMD+=(--per-sample-timeout-factor "$PER_SAMPLE_TIMEOUT_FACTOR")
fi

if [[ "$USE_FP16" == "true" || "$USE_FP16" == "TRUE" || "$USE_FP16" == "1" ]]; then
  CMD+=(--use-fp16)
elif [[ "$USE_FP16" != "false" && "$USE_FP16" != "FALSE" && "$USE_FP16" != "0" ]]; then
  echo "Invalid USE_FP16=$USE_FP16 (expect true/false)" >&2
  exit 1
fi

if [[ "$CACHE_LENGTHS" == "true" || "$CACHE_LENGTHS" == "TRUE" || "$CACHE_LENGTHS" == "1" ]]; then
  CMD+=(--cache-lengths)
elif [[ "$CACHE_LENGTHS" != "false" && "$CACHE_LENGTHS" != "FALSE" && "$CACHE_LENGTHS" != "0" ]]; then
  echo "Invalid CACHE_LENGTHS=$CACHE_LENGTHS (expect true/false)" >&2
  exit 1
fi

if [[ "$EXPORT_GT_MIDI" == "true" || "$EXPORT_GT_MIDI" == "TRUE" || "$EXPORT_GT_MIDI" == "1" ]]; then
  CMD+=(--export-gt-midi)
elif [[ "$EXPORT_GT_MIDI" == "false" || "$EXPORT_GT_MIDI" == "FALSE" || "$EXPORT_GT_MIDI" == "0" ]]; then
  CMD+=(--no-export-gt-midi)
else
  echo "Invalid EXPORT_GT_MIDI=$EXPORT_GT_MIDI (expect true/false)" >&2
  exit 1
fi

CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
