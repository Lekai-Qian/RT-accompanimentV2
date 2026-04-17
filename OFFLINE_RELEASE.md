# Offline Inference Minimal Release

This branch contains the minimal code needed to run offline accompaniment
inference, prompt-condition experiments, optional interval rollout simulation,
and eval-acc-style JSD metrics. It intentionally excludes checkpoints, datasets,
MIDI recordings, generated outputs, and listening bundles.

## What Is Included

Core model/data/tokenizer files:

- `config.py`
- `model.py`
- `PianoDataset.py`
- `my_tokenizer.py`
- `Token2Midi.py`

Inference entrypoints:

- `inference_new.py`: ordinary offline inference with `--gt-prefix-beats`
- `Inference/no_acc_prompt_inference.py`: prompt matrix modes using dataset melody
- `Inference/user_melody_prompt_inference.py`: prompt matrix modes using an external melody MIDI for all beats
- `Inference/interval_rollout_inference.py`: fixed-interval rollout simulation

Evaluation:

- `eval_acc.py`: canonical accompaniment JSD implementation
- `eval/eval_acc_jsd_bridge.py`: bridge used by the CSV metric script
- `eval/compute_midi_metrics.py`: batch metric CSV generation

## Checkpoints

Checkpoints are not committed to git. Download the needed files separately, for
example from the project Hugging Face repo:

<https://huggingface.co/chenyuanxin/RT-accompanimentV2-checkpoints>

Then pass checkpoint paths through `--ckpt` or by editing the shell runner envs.

## Data Layout

The scripts expect the dataset directory to contain `.npz` files with the same
format used during training. You can pass the dataset root with:

```bash
DATA_DIR=/path/to/allxml_npz_dual_track_optimized_no_underscore \
  bash scripts/run_inference_new.sh
```

## Ordinary Offline Inference

```bash
python inference_new.py \
  --ckpt /path/to/model.safetensors \
  --data-dir /path/to/npz_dataset \
  --num-samples 20 \
  --gt-prefix-beats 8 \
  --output-dir offline_outputs/generated_samples
```

Use `--gt-prefix-beats 0` for no prompt from beat 0.

## Prompt Matrix With Dataset Melody

```bash
python Inference/no_acc_prompt_inference.py \
  --ckpt /path/to/model.safetensors \
  --data-dir /path/to/npz_dataset \
  --indices-file /path/to/selected_indices.json \
  --prompt-mode mel_only \
  --prompt-beats 8 \
  --output-dir offline_outputs/generated_samples_mel_only
```

Prompt modes:

- `gt`: first `N` beats use dataset GT accompaniment plus dataset melody
- `mel_only`: first `N` beats use empty accompaniment plus dataset melody
- `no_prompt`: generate accompaniment from beat 0

## Prompt Matrix With External Melody MIDI

Use this when the melody condition comes from a separate MIDI recording, while
optional accompaniment prompt still comes from the dataset piece.

```bash
python Inference/user_melody_prompt_inference.py \
  --ckpt /path/to/model.safetensors \
  --model-name initdrop8p0p3 \
  --data-dir /path/to/npz_dataset \
  --prompt-mode with_prompt \
  --prompt-beats 8 \
  --seeds 42,43,44,45,46 \
  --piece 3387831:mariage:/path/to/performance_melody.mid \
  --output-dir offline_outputs/user_melody_with_prompt
```

Prompt modes:

- `with_prompt`: first `N` beats use dataset GT accompaniment; melody always comes from the external MIDI
- `mel_only_prompt`: first `N` beats use empty accompaniment; melody always comes from the external MIDI
- `no_prompt`: accompaniment is generated from beat 0; melody always comes from the external MIDI


The batch helper accepts external melody pieces through `PIECE_SPECS`:

```bash
PIECE_SPECS="3387831:mariage:/path/to/melody.mid,6217163:periodic:/path/to/melody.mid" \
  bash scripts/run_user_melody_prompt_matrix.sh
```

Each entry is `piece_id:label:melody_midi_path`. The `piece_id` must match a
`.npz` file in `DATA_DIR`.

## Eval-Acc-Style JSD

```bash
python eval/compute_midi_metrics.py \
  --input-dir offline_outputs/generated_samples \
  --output-csv offline_outputs/eval_results.csv
```

The reported pitch/onset JSD values are computed through `eval_acc.py` semantics.

## Excluded From This Branch

Do not commit these to the release branch:

- `checkpoints*`, `*.safetensors`, `*.pt`, `*.pth`
- `out/`, `offline-*`, `generated_samples*`, `listening_bundles/`
- user recording zips or extracted user MIDI data
- `hf_upload/`, `local_tmp/`, `__pycache__/`
