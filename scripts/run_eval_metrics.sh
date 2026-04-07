#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEFAULTS_FILE="${DEFAULTS_FILE:-$ROOT_DIR/scripts/eval_metrics.defaults.env}"

if [[ -f "$DEFAULTS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULTS_FILE"
fi

EVAL_MODE="${EVAL_MODE:-auto}"
AUTO_DIR="${AUTO_DIR:-}"
REFS_DIR="${REFS_DIR:-}"
GENS_DIR="${GENS_DIR:-}"
MATCH_GT_SUFFIX="${MATCH_GT_SUFFIX:-false}"
GT_SUFFIX="${GT_SUFFIX:-_GT}"
TOL="${TOL:-0.05}"
COMPUTE_FMD="${COMPUTE_FMD:-false}"
FMD_BOOTSTRAP_ITERS="${FMD_BOOTSTRAP_ITERS:-300}"
FMD_CI_LEVEL="${FMD_CI_LEVEL:-0.95}"
FMD_BOOTSTRAP_SEED="${FMD_BOOTSTRAP_SEED:-1234}"
OUT="${OUT:-}"

CMD=("$PYTHON_BIN" "$ROOT_DIR/eval/compute_midi_metrics.py")

if [[ "$EVAL_MODE" == "auto" ]]; then
  if [[ -z "$AUTO_DIR" ]]; then
    echo "AUTO_DIR 为空，请在 defaults.env 设置" >&2
    exit 1
  fi
  CMD+=(--auto-dir "$AUTO_DIR")
elif [[ "$EVAL_MODE" == "pair" ]]; then
  if [[ -z "$REFS_DIR" || -z "$GENS_DIR" ]]; then
    echo "pair 模式需要 REFS_DIR 和 GENS_DIR" >&2
    exit 1
  fi
  CMD+=(--refs "$REFS_DIR" --gens "$GENS_DIR")

  if [[ "$MATCH_GT_SUFFIX" == "true" || "$MATCH_GT_SUFFIX" == "TRUE" || "$MATCH_GT_SUFFIX" == "1" ]]; then
    CMD+=(--match-gt-suffix)
  elif [[ "$MATCH_GT_SUFFIX" == "false" || "$MATCH_GT_SUFFIX" == "FALSE" || "$MATCH_GT_SUFFIX" == "0" ]]; then
    :
  else
    echo "Invalid MATCH_GT_SUFFIX=$MATCH_GT_SUFFIX (expect true/false)" >&2
    exit 1
  fi
else
  echo "Invalid EVAL_MODE=$EVAL_MODE (expect auto/pair)" >&2
  exit 1
fi

CMD+=(--gt-suffix "$GT_SUFFIX" --tol "$TOL")

if [[ "$COMPUTE_FMD" == "true" || "$COMPUTE_FMD" == "TRUE" || "$COMPUTE_FMD" == "1" ]]; then
  CMD+=(--compute-fmd --fmd-bootstrap-iters "$FMD_BOOTSTRAP_ITERS" --fmd-ci-level "$FMD_CI_LEVEL" --fmd-bootstrap-seed "$FMD_BOOTSTRAP_SEED")
elif [[ "$COMPUTE_FMD" == "false" || "$COMPUTE_FMD" == "FALSE" || "$COMPUTE_FMD" == "0" ]]; then
  CMD+=(--no-compute-fmd)
else
  echo "Invalid COMPUTE_FMD=$COMPUTE_FMD (expect true/false)" >&2
  exit 1
fi

# OUT 为空时不传 --out，保留 compute_midi_metrics.py 的默认行为
if [[ -n "$OUT" ]]; then
  CMD+=(--out "$OUT")
fi

# 支持命令行临时覆盖参数
CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
