# Realtime Simulation TODO

## Phase 1

- create `inference_realtime_sim.py`
- add CLI args:
  - `--inference-interval`
  - `--generation-length`
  - `--fallback-policy`
- enforce odd-only `generation_length`
- log:
  - total ticks
  - inference calls
  - usable acc outputs
  - discarded mel outputs
  - fallback count

## Phase 2

- connect to current model checkpoint loading
- connect to current tokenizer / schedule logic
- simulate output buffer
- write generated accompaniment MIDI
- write manifest with per-run realtime stats

## Phase 3

- compare:
  - `I=1,G=1`
  - `I=1,G=3`
  - `I=1,G=5`
- then vary `I`
- add evaluation integration

## Notes

- generated melody is internal-only and discarded
- only acc output should be evaluated
- even `G` is intentionally excluded
