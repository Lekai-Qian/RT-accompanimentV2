#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/offline-0416-user-melody-prompt-matrix}"
GPU_DEVICES="${GPU_DEVICES:-4,5}"
DEVICE="${DEVICE:-auto}"
PROMPT_BEATS="${PROMPT_BEATS:-8}"
SEEDS="${SEEDS:-42,43,44,45,46}"
DRY_RUN="${DRY_RUN:-0}"
USE_FP16="${USE_FP16:-false}"
DATA_DIR="${DATA_DIR:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
MODE_FILTER="${MODE_FILTER:-}"
EXPORT_DATASET_GT_MIDI="${EXPORT_DATASET_GT_MIDI:-false}"
PIECE_SPECS="${PIECE_SPECS:-}"

TEMPERATURE="${TEMPERATURE:-1.1}"
TOP_K="${TOP_K:-10}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

mkdir -p "$OUTPUT_ROOT"

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

declare -a PIECE_ARGS=()
if [[ -n "$PIECE_SPECS" ]]; then
  IFS=',' read -r -a PIECE_SPEC_LIST <<< "$PIECE_SPECS"
  for piece_spec in "${PIECE_SPEC_LIST[@]}"; do
    piece_spec="${piece_spec#${piece_spec%%[![:space:]]*}}"
    piece_spec="${piece_spec%${piece_spec##*[![:space:]]}}"
    [[ -z "$piece_spec" ]] && continue
    PIECE_ARGS+=(--piece "$piece_spec")
  done
else
  PIECE_ARGS=(
    --piece "3387831:mariage:$ROOT_DIR/user_midi_recording_data/user_midi_recording/mariage_20260409-114245_3387831/performance_melody.mid"
    --piece "6217163:periodic:$ROOT_DIR/user_midi_recording_data/user_midi_recording/periodic_20260409-114946_6217163/performance_melody.mid"
  )
fi

declare -a MODEL_NAMES=(
  "base"
  "initdrop8"
  "initdrop8p0p3"
  "initdrop8p0p3_melloss"
)

declare -A MODEL_CKPTS=(
  ["base"]="$ROOT_DIR/checkpoints-resume/epoch_15_0307_1858/model.safetensors"
  ["initdrop8"]="$ROOT_DIR/checkpoints-teacher/music_transformer_initdrop8_0407_1138/epoch_3_0407_1740/model.safetensors"
  ["initdrop8p0p3"]="$ROOT_DIR/checkpoints-teacher/music_transformer_initdrop8p0p3_0407_1138/epoch_3_0407_1740/model.safetensors"
  ["initdrop8p0p3_melloss"]="$ROOT_DIR/checkpoints-teacher/music_transformer_initdrop8p0p3_melloss_0407_2000/epoch_3_0408_0205/model.safetensors"
)

declare -a PROMPT_MODES=(
  "with_prompt"
  "mel_only_prompt"
  "no_prompt"
)

declare -A MODE_SLUGS=(
  ["with_prompt"]="withprompt${PROMPT_BEATS}"
  ["mel_only_prompt"]="melprompt${PROMPT_BEATS}"
  ["no_prompt"]="noprompt0"
)

SELECTED_MODELS=()
for model_name in "${MODEL_NAMES[@]}"; do
  if [[ -n "$MODEL_FILTER" && "$model_name" != *"$MODEL_FILTER"* ]]; then
    continue
  fi
  ckpt="${MODEL_CKPTS[$model_name]}"
  if [[ ! -f "$ckpt" ]]; then
    echo "Skipping missing checkpoint for $model_name: $ckpt" >&2
    continue
  fi
  SELECTED_MODELS+=("$model_name")
done

SELECTED_MODES=()
for mode in "${PROMPT_MODES[@]}"; do
  if [[ -n "$MODE_FILTER" && "$mode" != *"$MODE_FILTER"* ]]; then
    continue
  fi
  SELECTED_MODES+=("$mode")
done

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
  echo "No selected models" >&2
  exit 1
fi
if [[ ${#SELECTED_MODES[@]} -eq 0 ]]; then
  echo "No selected modes" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "GPU_DEVICES is empty" >&2
  exit 1
fi

USE_FP16_FLAG="$(bool_flag USE_FP16 "$USE_FP16" --use-fp16 "")"
GT_EXPORT_FLAG="$(bool_flag EXPORT_DATASET_GT_MIDI "$EXPORT_DATASET_GT_MIDI" --export-dataset-gt-midi --no-export-dataset-gt-midi)"

build_cmd() {
  local model_name="$1"
  local mode="$2"
  local output_dir="$3"

  local -a cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/Inference/user_melody_prompt_inference.py"
    --ckpt "${MODEL_CKPTS[$model_name]}"
    --model-name "$model_name"
    --device "$DEVICE"
    --prompt-mode "$mode"
    --prompt-beats "$PROMPT_BEATS"
    --seeds "$SEEDS"
    --output-dir "$output_dir"
    --temperature "$TEMPERATURE"
    --top-k "$TOP_K"
    --top-p "$TOP_P"
    --repetition-penalty "$REPETITION_PENALTY"
    "$GT_EXPORT_FLAG"
    "${PIECE_ARGS[@]}"
  )
  if [[ -n "$USE_FP16_FLAG" ]]; then
    cmd+=("$USE_FP16_FLAG")
  fi
  if [[ -n "$DATA_DIR" ]]; then
    cmd+=(--data-dir "$DATA_DIR")
  fi
  printf '%q ' "${cmd[@]}"
}

echo "Output root: $OUTPUT_ROOT"
echo "GPU devices: $GPU_DEVICES"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Modes: ${SELECTED_MODES[*]}"
echo "Seeds: $SEEDS"
echo "Piece specs: ${PIECE_ARGS[*]}"
echo

jobs=()
for model_name in "${SELECTED_MODELS[@]}"; do
  for mode in "${SELECTED_MODES[@]}"; do
    slug="${MODE_SLUGS[$mode]}"
    output_dir="$OUTPUT_ROOT/generated_samples-${model_name}-${slug}"
    jobs+=("$model_name|$mode|$output_dir")
    echo "Planned: model=$model_name mode=$mode output=$output_dir"
  done
done
echo

run_job() {
  local gpu_id="$1"
  local spec="$2"
  local model_name mode output_dir
  IFS='|' read -r model_name mode output_dir <<< "$spec"
  mkdir -p "$output_dir"
  echo "Running on GPU $gpu_id: model=$model_name mode=$mode"
  local cmd_str
  cmd_str="$(build_cmd "$model_name" "$mode" "$output_dir")"
  echo "$cmd_str"
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    return 0
  fi
  eval "CUDA_VISIBLE_DEVICES=$gpu_id $cmd_str"
}

for ((i=0; i<${#jobs[@]}; i+=${#GPU_LIST[@]})); do
  pids=()
  for ((j=0; j<${#GPU_LIST[@]}; j++)); do
    idx=$((i + j))
    if [[ $idx -ge ${#jobs[@]} ]]; then
      continue
    fi
    gpu_id="${GPU_LIST[$j]}"
    if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
      run_job "$gpu_id" "${jobs[$idx]}"
    else
      (
        run_job "$gpu_id" "${jobs[$idx]}"
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
