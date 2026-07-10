CKPT="${CKPT:-/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors}"
DATA_DIR="${DATA_DIR:-/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/sft_pop909_compatible/checkpoints}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pop909_compatible_from_epoch15}"

# Keep the old sequence format. These are existing train.py arguments only.
NUM_EPOCHS="${NUM_EPOCHS:-5}"
ACC_DROP_PROB="${ACC_DROP_PROB:-0}"
DROP_INITIAL_BEATS="${DROP_INITIAL_BEATS:-0}"
DROP_INITIAL_BEATS_PROB="${DROP_INITIAL_BEATS_PROB:-0}"
POS_SHIFT_MAX="${POS_SHIFT_MAX:-0}"

PRINT_CONFIG="${PRINT_CONFIG:-true}"
DRY_RUN="${DRY_RUN:-false}"
