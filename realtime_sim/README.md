# Realtime Simulation

This folder is for the realtime-style inference experiment.

The goal is to study why the model quality drops in a realtime system compared with offline inference.

We focus on two effects:

1. prompt / prefix availability
2. rollout length inside one inference call

In particular, this folder follows the clarified system definition:

- offline baseline is approximately `I=1, G=1`
- realtime may use `G > 1`
- only **odd** `G` is meaningful
- model-generated `melody` is **not used as output**
- generated `melody` only exists as an internal rollout state and is discarded afterward

So:

- `G=1` means: `acc`
- `G=3` means: `acc -> mel -> acc`
- `G=5` means: `acc -> mel -> acc -> mel -> acc`

Only the generated `acc` segments are consumed by the target system.

The generated `mel` segments are internal-only and discarded.

## Folder Purpose

This folder is meant to keep the realtime experiment separate from:

- standard offline inference
- repair inference
- augmentation presentation notes

This helps us keep the experimental assumptions clear.

## Suggested Next Step

Implement a dedicated `inference_realtime_sim.py` here with:

- tick timer
- `inference_interval`
- odd-only `generation_length`
- output buffer
- fallback / backup path
- detailed logging of:
  - deadline misses
  - fallback usage
  - generated acc count
  - discarded mel count

