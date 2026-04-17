#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

# Preserve caller-provided overrides before sourcing defaults.
OVERRIDE_INDICES_FILE="${INDICES_FILE-}"
OVERRIDE_OUTPUT_ROOT="${OUTPUT_ROOT-}"
OVERRIDE_CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT-}"
OVERRIDE_PROMPT_BEATS="${PROMPT_BEATS-}"
OVERRIDE_MODEL_FILTER="${MODEL_FILTER-}"
OVERRIDE_MAX_FAMILIES="${MAX_FAMILIES-}"
OVERRIDE_DRY_RUN="${DRY_RUN-}"
OVERRIDE_DEVICE="${DEVICE-}"
OVERRIDE_USE_FP16="${USE_FP16-}"
OVERRIDE_DATA_DIR="${DATA_DIR-}"
OVERRIDE_SEED="${SEED-}"
OVERRIDE_SAMPLING_MODE="${SAMPLING_MODE-}"
OVERRIDE_DATASET_MODE="${DATASET_MODE-}"
OVERRIDE_DATASET_SEED="${DATASET_SEED-}"
OVERRIDE_TEST_SPLIT_RATIO="${TEST_SPLIT_RATIO-}"
OVERRIDE_CACHE_LENGTHS="${CACHE_LENGTHS-}"
OVERRIDE_EXPORT_GT_MIDI="${EXPORT_GT_MIDI-}"
OVERRIDE_TEMPERATURE="${TEMPERATURE-}"
OVERRIDE_TOP_K="${TOP_K-}"
OVERRIDE_TOP_P="${TOP_P-}"
OVERRIDE_REPETITION_PENALTY="${REPETITION_PENALTY-}"

# Reuse the current prompt-variant defaults so ordinary and mel-only runs share
# the same sampling / decoding settings unless explicitly overridden here.
DEFAULTS_FILE="${DEFAULTS_FILE:-$ROOT_DIR/scripts/no_acc_prompt_inference.defaults.env}"
if [[ -f "$DEFAULTS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULTS_FILE"
fi

INDICES_FILE="${OVERRIDE_INDICES_FILE:-${INDICES_FILE:-$ROOT_DIR/offline-0407-common19/selected_indices.json}}"
OUTPUT_ROOT="${OVERRIDE_OUTPUT_ROOT:-${OUTPUT_ROOT:-$ROOT_DIR/offline-0408-prompt-matrix}}"
CHECKPOINTS_ROOT="${OVERRIDE_CHECKPOINTS_ROOT:-${CHECKPOINTS_ROOT:-$ROOT_DIR}}"
PROMPT_BEATS="${OVERRIDE_PROMPT_BEATS:-${PROMPT_BEATS:-8}}"
MODEL_FILTER="${OVERRIDE_MODEL_FILTER:-${MODEL_FILTER:-}}"
MAX_FAMILIES="${OVERRIDE_MAX_FAMILIES:-${MAX_FAMILIES:-0}}"
DRY_RUN="${OVERRIDE_DRY_RUN:-${DRY_RUN:-0}}"

DEVICE="${OVERRIDE_DEVICE:-${DEVICE:-auto}}"
GPU_DEVICES="${GPU_DEVICES:-4,5}"
TIMEOUT_MODELS="${TIMEOUT_MODELS:-initdrop8}"
PER_SAMPLE_TIMEOUT_SEC="${PER_SAMPLE_TIMEOUT_SEC:-180}"
USE_FP16="${OVERRIDE_USE_FP16:-${USE_FP16:-false}}"
DATA_DIR="${OVERRIDE_DATA_DIR:-${DATA_DIR:-}}"
SEED="${OVERRIDE_SEED:-${SEED:-42}}"
SAMPLING_MODE="${OVERRIDE_SAMPLING_MODE:-${SAMPLING_MODE:-with_replacement}}"
DATASET_MODE="${OVERRIDE_DATASET_MODE:-${DATASET_MODE:-test}}"
DATASET_SEED="${OVERRIDE_DATASET_SEED:-${DATASET_SEED:-42}}"
TEST_SPLIT_RATIO="${OVERRIDE_TEST_SPLIT_RATIO:-${TEST_SPLIT_RATIO:-0.10}}"
CACHE_LENGTHS="${OVERRIDE_CACHE_LENGTHS:-${CACHE_LENGTHS:-false}}"
EXPORT_GT_MIDI="${OVERRIDE_EXPORT_GT_MIDI:-${EXPORT_GT_MIDI:-true}}"
TEMPERATURE="${OVERRIDE_TEMPERATURE:-${TEMPERATURE:-1.1}}"
TOP_K="${OVERRIDE_TOP_K:-${TOP_K:-10}}"
TOP_P="${OVERRIDE_TOP_P:-${TOP_P:-0.95}}"
REPETITION_PENALTY="${OVERRIDE_REPETITION_PENALTY:-${REPETITION_PENALTY:-1.0}}"

if [[ ! -f "$INDICES_FILE" ]]; then
  echo "Indices file not found: $INDICES_FILE" >&2
  exit 1
fi

if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
fi
mkdir -p "$OUTPUT_ROOT"

NUM_SAMPLES="$("$PYTHON_BIN" - "$INDICES_FILE" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8").strip()
if path.suffix.lower() == ".json":
    payload = json.loads(text)
    print(len(payload))
else:
    print(len([line for line in text.splitlines() if line.strip()]))
PY
)"

bool_flag() {
  local name="$1"
  local value="$2"
  local true_flag="$3"
  local false_flag="$4"
  if [[ "$value" == "true" || "$value" == "TRUE" || "$value" == "1" ]]; then
    printf '%s\n' "$true_flag"
  elif [[ "$value" == "false" || "$value" == "FALSE" || "$value" == "0" ]]; then
    printf '%s\n' "$false_flag"
  else
    echo "Invalid $name=$value (expect true/false)" >&2
    exit 1
  fi
}

run_or_echo() {
  echo "Running: $*"
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    return 0
  fi
  "$@"
}

csv_contains_exact() {
  local needle="$1"
  local haystack="$2"
  local item
  IFS=',' read -r -a items <<< "$haystack"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

build_common_args() {
  local -n out_ref="$1"
  out_ref=(
    --device "$DEVICE"
    --num-samples "$NUM_SAMPLES"
    --seed "$SEED"
    --indices-file "$INDICES_FILE"
    --sampling-mode "$SAMPLING_MODE"
    --dataset-mode "$DATASET_MODE"
    --dataset-seed "$DATASET_SEED"
    --test-split-ratio "$TEST_SPLIT_RATIO"
    --temperature "$TEMPERATURE"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --repetition-penalty "$REPETITION_PENALTY"
  )
  if [[ -n "$DATA_DIR" ]]; then
    out_ref+=(--data-dir "$DATA_DIR")
  fi
  local fp16_flag
  fp16_flag="$(bool_flag USE_FP16 "$USE_FP16" --use-fp16 "")"
  if [[ -n "$fp16_flag" ]]; then
    out_ref+=("$fp16_flag")
  fi
  local cache_flag
  cache_flag="$(bool_flag CACHE_LENGTHS "$CACHE_LENGTHS" --cache-lengths "")"
  if [[ -n "$cache_flag" ]]; then
    out_ref+=("$cache_flag")
  fi
  local gt_flag
  gt_flag="$(bool_flag EXPORT_GT_MIDI "$EXPORT_GT_MIDI" --export-gt-midi --no-export-gt-midi)"
  out_ref+=("$gt_flag")
}

run_single_mode_with_timeout() {
  local gpu_id="$1"
  local model_name="$2"
  local mode_name="$3"
  local output_dir="$4"
  shift 4
  local -a base_cmd=("$@")

  mkdir -p "$output_dir"
  local tmp_root="$output_dir/.timeout_tmp"
  rm -rf "$tmp_root"
  mkdir -p "$tmp_root"
  local failures_file="$output_dir/failures.jsonl"
  : > "$failures_file"

  mapfile -t sample_indices < <("$PYTHON_BIN" - "$INDICES_FILE" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8").strip()
if path.suffix.lower() == ".json":
    payload = json.loads(text)
else:
    payload = [int(line.strip()) for line in text.splitlines() if line.strip()]
for idx in payload:
    print(int(idx))
PY
)

  local sample_idx=0
  local idx_value=""
  for idx_value in "${sample_indices[@]}"; do
    local sample_dir="$tmp_root/sample_${sample_idx}"
    local sample_indices_file="$sample_dir/index.json"
    mkdir -p "$sample_dir"
    printf '[%s]\n' "$idx_value" > "$sample_indices_file"

    echo "[timeout-guard] model=$model_name mode=$mode_name sample=${sample_idx} dataset_index=${idx_value} timeout=${PER_SAMPLE_TIMEOUT_SEC}s"
    if env CUDA_VISIBLE_DEVICES="$gpu_id" timeout --signal=TERM "${PER_SAMPLE_TIMEOUT_SEC}s" \
      "${base_cmd[@]}" --num-samples 1 --indices-file "$sample_indices_file" --output-dir "$sample_dir"; then
      find "$sample_dir" -maxdepth 1 -type f -exec mv {} "$output_dir"/ \;
    else
      status=$?
      printf '{"sample_id":%d,"dataset_index":%d,"mode":"%s","timeout_sec":%d,"exit_code":%d}\n' \
        "$sample_idx" "$idx_value" "$mode_name" "$PER_SAMPLE_TIMEOUT_SEC" "$status" >> "$failures_file"
    fi
    rm -rf "$sample_dir"
    sample_idx=$((sample_idx + 1))
  done

  rm -rf "$tmp_root"
}

run_model_group() {
  local gpu_id="$1"
  local model_name="$2"
  local ckpt="$3"
  local slug="$4"

  echo "Model: $model_name"
  echo "Checkpoint: $ckpt"
  echo "Assigned GPU: $gpu_id"
  local use_timeout=0
  if [[ "$PER_SAMPLE_TIMEOUT_SEC" =~ ^[0-9]+$ ]] && [[ "$PER_SAMPLE_TIMEOUT_SEC" -gt 0 ]] && csv_contains_exact "$model_name" "$TIMEOUT_MODELS"; then
    use_timeout=1
    echo "Timeout guard: enabled (${PER_SAMPLE_TIMEOUT_SEC}s per sample)"
  fi

  local -a prefix_cmd=()
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    prefix_cmd=()
  else
    prefix_cmd=(env CUDA_VISIBLE_DEVICES="$gpu_id")
  fi

  local gtprompt_output="$OUTPUT_ROOT/generated_samples-${slug}-gtprompt${PROMPT_BEATS}"
  local melprompt_output="$OUTPUT_ROOT/generated_samples-${slug}-melprompt${PROMPT_BEATS}"
  local noprompt_output="$OUTPUT_ROOT/generated_samples-${slug}-noprompt0"

  local -a gtprompt_cmd=("$PYTHON_BIN" "$ROOT_DIR/inference_new.py"
    --ckpt "$ckpt"
    "${COMMON_ARGS[@]}"
    --gt-prefix-beats "$PROMPT_BEATS"
    --output-dir "$gtprompt_output")
  local -a melprompt_cmd=("$PYTHON_BIN" "$ROOT_DIR/Inference/no_acc_prompt_inference.py"
    --ckpt "$ckpt"
    "${COMMON_ARGS[@]}"
    --prompt-beats "$PROMPT_BEATS"
    --prompt-mode mel_only
    --output-dir "$melprompt_output")
  local -a noprompt_cmd=("$PYTHON_BIN" "$ROOT_DIR/inference_new.py"
    --ckpt "$ckpt"
    "${COMMON_ARGS[@]}"
    --gt-prefix-beats 0
    --output-dir "$noprompt_output")

  if [[ "$use_timeout" -eq 1 && "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" ]]; then
    run_single_mode_with_timeout "$gpu_id" "$model_name" "gtprompt${PROMPT_BEATS}" "$gtprompt_output" "${gtprompt_cmd[@]}"
    run_single_mode_with_timeout "$gpu_id" "$model_name" "melprompt${PROMPT_BEATS}" "$melprompt_output" "${melprompt_cmd[@]}"
    run_single_mode_with_timeout "$gpu_id" "$model_name" "noprompt0" "$noprompt_output" "${noprompt_cmd[@]}"
  else
    run_or_echo "${prefix_cmd[@]}" "${gtprompt_cmd[@]}"
    run_or_echo "${prefix_cmd[@]}" "${melprompt_cmd[@]}"
    run_or_echo "${prefix_cmd[@]}" "${noprompt_cmd[@]}"
  fi

  echo
}

declare -a MODEL_NAMES=(
  "base"
  "initdrop2"
  "initdrop2p0p3"
  "initdrop8"
  "initdrop8p0p3"
  "initdrop8p0p3_melloss"
)

declare -A MODEL_CKPTS=(
  ["base"]="$ROOT_DIR/checkpoints-resume/epoch_15_0307_1858/model.safetensors"
  ["initdrop2"]="$ROOT_DIR/checkpoints-initial-drop-acc-2/epoch_3_0315_2029/model.safetensors"
  ["initdrop2p0p3"]="$ROOT_DIR/checkpoints/music_transformer_initdrop2p0p3_0316_0833/epoch_3_0316_1434/model.safetensors"
  ["initdrop8"]="$ROOT_DIR/checkpoints-teacher/music_transformer_initdrop8_0407_1138/epoch_3_0407_1740/model.safetensors"
  ["initdrop8p0p3"]="$ROOT_DIR/checkpoints-teacher/music_transformer_initdrop8p0p3_0407_1138/epoch_3_0407_1740/model.safetensors"
  ["initdrop8p0p3_melloss"]="$ROOT_DIR/checkpoints-teacher/music_transformer_initdrop8p0p3_melloss_0407_2000/epoch_3_0408_0205/model.safetensors"
)

SELECTED_MODELS=()
for model_name in "${MODEL_NAMES[@]}"; do
  if [[ -n "$MODEL_FILTER" && "$model_name" != *"$MODEL_FILTER"* ]]; then
    continue
  fi
  SELECTED_MODELS+=("$model_name")
done

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
  echo "No models matched MODEL_FILTER=$MODEL_FILTER" >&2
  exit 1
fi

if [[ "$MAX_FAMILIES" =~ ^[0-9]+$ ]] && [[ "$MAX_FAMILIES" -gt 0 ]] && [[ ${#SELECTED_MODELS[@]} -gt "$MAX_FAMILIES" ]]; then
  SELECTED_MODELS=("${SELECTED_MODELS[@]:0:$MAX_FAMILIES}")
fi

echo "Selected models:"
for model_name in "${SELECTED_MODELS[@]}"; do
  printf '  %s -> %s\n' "$model_name" "${MODEL_CKPTS[$model_name]}"
done
echo

COMMON_ARGS=()
build_common_args COMMON_ARGS

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "GPU_DEVICES is empty" >&2
  exit 1
fi

VALID_MODELS=()
for model_name in "${SELECTED_MODELS[@]}"; do
  ckpt="${MODEL_CKPTS[$model_name]}"
  if [[ ! -f "$ckpt" ]]; then
    echo "Skipping $model_name (missing checkpoint: $ckpt)" >&2
    continue
  fi
  VALID_MODELS+=("$model_name")
done

for ((i=0; i<${#VALID_MODELS[@]}; i+=${#GPU_LIST[@]})); do
  pids=()
  for ((j=0; j<${#GPU_LIST[@]}; j++)); do
    model_idx=$((i + j))
    if [[ $model_idx -ge ${#VALID_MODELS[@]} ]]; then
      continue
    fi
    model_name="${VALID_MODELS[$model_idx]}"
    ckpt="${MODEL_CKPTS[$model_name]}"
    slug="$model_name"
    gpu_id="${GPU_LIST[$j]}"

    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
      run_model_group "$gpu_id" "$model_name" "$ckpt" "$slug"
    else
      (
        run_model_group "$gpu_id" "$model_name" "$ckpt" "$slug"
      ) &
      pids+=("$!")
    fi
  done

  if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
  fi
done

echo "Done. Output root: $OUTPUT_ROOT"
