# POP909 Compatible SFT Pipeline

This folder is intentionally a thin wrapper around the existing RT-accompanimentV2
training stack.

Hard constraints:

- Do not modify `PianoDataset.py`, `my_tokenizer.py`, `model.py`, or the model
  sequence format for this experiment.
- Training and evaluation must keep using `PianoMusicTokenizer.build_training_sequence()`.
- POP909 changes only the NPZ data source, not the tokenization contract seen by
  the epoch-15 base checkpoint.

## Data

Default POP909-SFT NPZ directory:

```text
/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/train
```

The dataset must contain old-compatible 4-channel NPZ files with:

```text
measure_0, measure_1, ..., metadata
```

## Pipeline

1. Precompute length cache with the existing `get_length.py` logic:

```bash
bash scripts/sft_pop909/run_precompute_pop909_lengths.sh
```

For test-set evaluation, precompute the test directory too:

```bash
DATA_DIR=/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/test \
bash scripts/sft_pop909/run_precompute_pop909_lengths.sh
```

2. Evaluate the base epoch-15 checkpoint on POP909-SFT test:

```bash
CKPT=/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors \
DATA_DIR=/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/test \
TEST_SPLIT_RATIO=1.0 \
OUT_JSON=experiments/sft_pop909_compatible/eval/base_epoch15_pop909_test.json \
bash scripts/sft_pop909/run_eval_checkpoint.sh
```

3. Finetune from the base epoch-15 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/sft_pop909/run_train_pop909_from_epoch15.sh
```

The default train settings are tracked in:

```text
scripts/sft_pop909/train_pop909_from_epoch15.defaults.sh
```

4. Evaluate the finetuned checkpoint on the same POP909-SFT test split:

```bash
CKPT=/path/to/finetuned/model.safetensors \
DATA_DIR=/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/test \
TEST_SPLIT_RATIO=1.0 \
OUT_JSON=experiments/sft_pop909_compatible/eval/finetune_pop909_test.json \
bash scripts/sft_pop909/run_eval_checkpoint.sh
```

5. Evaluate base and finetuned checkpoints on the original dataset using the same
   script and old-compatible tokenizer path:

```bash
DATA_DIR=/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore
```

Only results produced by this compatible path should be compared against the old
base training/eval losses.
