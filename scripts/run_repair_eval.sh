#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEFAULTS_FILE="${DEFAULTS_FILE:-$ROOT_DIR/scripts/repair_eval.defaults.env}"

if [[ -f "$DEFAULTS_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$DEFAULTS_FILE"
fi

REPAIR_DIR="${REPAIR_DIR:-repair_outputs}"
OUT="${OUT:-repair_eval.csv}"
TOL="${TOL:-0.05}"
SIG_DIGITS="${SIG_DIGITS:-4}"

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/eval/compute_repair_metrics.py"
  --repair-dir "$REPAIR_DIR"
  --out "$OUT"
  --tol "$TOL"
  --sig-digits "$SIG_DIGITS"
)

CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
