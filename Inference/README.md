# Offline Inference Entrypoints

This directory contains focused inference scripts for offline experiments. The
root `inference_new.py` remains the ordinary GT-prefix baseline runner.

## Prompt Variants With Dataset Melody

Entry:

- `Inference/no_acc_prompt_inference.py`
- `scripts/run_no_acc_prompt_inference.sh`

This script uses the dataset melody as the conditioning melody and changes only
the accompaniment prompt behavior.

Prompt modes:

- `gt`: first `N` beats inject dataset GT accompaniment; melody is dataset melody
- `mel_only`: first `N` beats inject `EMPTY` accompaniment; melody is dataset melody
- `no_prompt`: no accompaniment prompt; accompaniment is generated from beat 0

After the prompt region, accompaniment is generated autoregressively and melody
is always injected from the dataset.

## Prompt Variants With External Melody MIDI

Entry:

- `Inference/user_melody_prompt_inference.py`
- `scripts/run_user_melody_prompt_matrix.sh`

This script uses the dataset only for metadata and optional accompaniment prompt.
The melody condition for all beats comes from an external MIDI file.

Prompt modes:

- `with_prompt`: first `N` beats inject dataset GT accompaniment; melody is external MIDI
- `mel_only_prompt`: first `N` beats inject `EMPTY` accompaniment; melody is external MIDI
- `no_prompt`: no accompaniment prompt; melody is external MIDI from beat 0

The external MIDI is converted on its own beat grid, then aligned to the target
dataset piece grid.

## Interval Rollout / Realtime-Style Simulation

Entry:

- `Inference/interval_rollout_inference.py`
- `scripts/run_interval_rollout_inference.sh`

Purpose:

- simulate a system that runs inference every `I` beats
- generate `G` internal beats per successful inference call
- use `catch_up_prob` to simulate missed inference calls
- on missed calls, use a fallback policy such as `empty` or `hold_last`

Important options:

- `--gt-prefix-beats`
- `--inference-interval`
- `--generation-length`
- `--catch-up-prob`
- `--fallback-policy`

Current convention:

- `hold_last` repeats the last consumed accompaniment segment when a catch-up attempt fails
- the implementation prevents two consecutive catch-up failures when that option is enabled
- generated melody is internal rollout state and is not consumed into the final output

## Deprecated Entrypoints

- root `inference.py`: legacy offline inference glue; keep for reference only
- root `inference_special.py`: older experimental entrypoint, superseded by focused scripts under `Inference/`
