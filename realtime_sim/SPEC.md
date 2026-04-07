# Realtime Simulation Spec

## 1. Definitions

### 1.1 Tick

The system has a global timer.

Every `+1` on the timer is treated as one `tick`.

### 1.2 Inference Interval

`Inference Interval = I`

The system performs model inference once every `I` ticks.

So if `I = 1`, inference happens at every tick.
If `I = 3`, inference happens every 3 ticks.

### 1.3 Generation Length

`Generation Length = G`

This is **not** the number of output accompaniment beats directly.

Instead, it is the number of alternating internal rollout segments inside one inference call.

The rollout always starts from `acc`.

Examples:

- `G=1` -> `acc`
- `G=2` -> `acc, mel`
- `G=3` -> `acc, mel, acc`
- `G=4` -> `acc, mel, acc, mel`
- `G=5` -> `acc, mel, acc, mel, acc`

## 2. Important Constraint

Only **odd** `G` is meaningful for the target system.

Why:

- even `G` ends at `mel`
- but generated `mel` is not used by the system output
- so even `G` does not produce an additional usable `acc`

Therefore the main experiment space should be:

- `G=1`
- `G=3`
- `G=5`

## 3. Melody Handling

Generated `melody` is **discarded**.

This is a key assumption.

The model-generated melody is only used to roll the internal autoregressive state forward.

It is **not** written to the final system output.

So the experiment is not asking:

"Can the model generate good melody?"

It is asking:

"If the model is forced to generate and discard melody internally, does that damage the following accompaniment generation?"

## 4. Offline vs Realtime

### 4.1 Offline Baseline

Offline inference is approximately:

- `I=1`
- `G=1`

and melody is injected in the normal offline pipeline.

### 4.2 Realtime Target System

Realtime rollout may require:

- sparse inference calls
- longer internal rollout chains
- fallback when inference is late

So quality degradation may come from:

- weaker prompt availability
- longer internal rollout
- more accumulated model error
- system-level timing constraints

## 5. Buffer Semantics

At each inference call, the model produces one rollout block.

From that block:

- generated `acc` segments are pushed into the output buffer
- generated `mel` segments are discarded after serving their internal rollout role

Then the playback side consumes one usable `acc` unit per tick.

## 6. Backup / Fallback

If inference cannot provide usable output in time, the system should use a backup policy.

Candidate fallback policies:

- silence
- hold previous accompaniment
- empty beat token

Initial recommended fallback:

- silence / empty accompaniment

because it is the simplest to analyze.

## 7. Main Hypothesis

The main hypothesis is:

> The model performs well in offline conditions close to `I=1, G=1`, but quality drops when realtime rollout requires larger odd `G`, because the model must internally generate discarded melody segments that were not directly trained as a generation target.

## 8. Recommended First Experiments

1. `I=1, G=1`
2. `I=1, G=3`
3. `I=1, G=5`

Then:

1. `I=2, G=3`
2. `I=3, G=3`

This separates:

- rollout-length effects
- scheduling-frequency effects

