# Training Augmentations For Accompaniment Generation

## 1. Motivation

Autoregressive accompaniment generation suffers from exposure bias:

- During training, the model always sees correct historical accompaniment.
- During inference, one wrong beat can affect all later beats.
- Errors accumulate because later generation depends on previous generated accompaniment.

The augmentations below are designed to reduce over-reliance on accompaniment history and improve robustness.

---

## 2. Encoding Method

### 2.1 Beat-Level Sequence

The whole sequence can be written compactly as:

```text
([BEAT][acc ... TRK_ACC][mel ... TRK_MEL]) x num_beats
```

That is, each beat contributes one accompaniment segment and one melody segment:

```text
[BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
```

Normal training beat:

```text
input:  [BEAT][acc_gt ... TRK_ACC][mel ... TRK_MEL]
label:  [PAD] [acc_gt ... TRK_ACC][PAD  PAD       ]
```

Interpretation:

- `BEAT` is a structure token.
- `acc_gt ... TRK_ACC` is the prediction target.
- `mel ... TRK_MEL` is condition input.

Relevant code:

- [my_tokenizer.py](/data/home/yuanxin/RT-accompanimentV2/my_tokenizer.py#L456)
- [PianoDataset.py](/data/home/yuanxin/RT-accompanimentV2/PianoDataset.py#L138)

---

## 3. Shift

### 3.1 Idea

Shift augmentation randomly offsets the relative position markers inside every beat.

Instead of always making `BEAT` the strict local anchor, we sample a shift `Delta` and move the internal position encoding by that amount.

### 3.2 Why

- Reduce overfitting to one fixed local alignment pattern.
- Make the model less sensitive to rigid position anchoring.

### 3.3 Implementation

One random shift is sampled per song:

```text
Delta in {0, ..., pos_shift_max}
```

Then all beats in that song use the same shift.

So this is not a per-beat random jitter. It is a per-song global offset.

In code, when `pos_shift_max > 0`, we sample:

```text
Delta ~ Uniform({0, 1, ..., pos_shift_max})
```

and pass the same `Delta` into `_encode_measures(...)`, so all beats in this song share the same local position offset.

### 3.4 Example

For presentation, it is easier to use a compact notation such as `acc(0)` to mean "an accompaniment token located at local position 0".

Below I use a simplified example with consecutive local positions `0, 1, 2`, because it is easier to read during presentation.

```text
before shift:
[BEAT][acc(0)][acc(1)][acc(2)][TRK_ACC][mel(0)][mel(1)][mel(2)][TRK_MEL]
```

If we sample `Delta = 1`, then the same beat becomes:

```text
after shift:
[BEAT][acc(1)][acc(2)][acc(3)][TRK_ACC][mel(1)][mel(2)][mel(3)][TRK_MEL]
```

The note content is unchanged. Only the local position indices are shifted.

In the actual implementation, this is realized by shifting the position markers attached to acc and mel tokens. Because the real encoding is sparse, actual examples may also skip positions such as `0, 2` or `1, 4`.

So the model no longer sees `BEAT -> pos0` as a rigid pattern for every beat.

Relevant code:

- [my_tokenizer.py](/data/home/yuanxin/RT-accompanimentV2/my_tokenizer.py#L489)
- [config.py](/data/home/yuanxin/RT-accompanimentV2/config.py#L62)

---

## 4. Drop

### 4.1 Idea

Randomly remove accompaniment context from some beats during training.

This is a beat-level operation:

- keep `BEAT`
- replace the whole accompaniment segment of that beat
- keep the melody segment unchanged

Normal beat:

```text
input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]
```

Dropped beat:

```text
input:  [BEAT][EMPTY][TRK_ACC][mel ... TRK_MEL]
label:  [PAD] [PAD  ][PAD    ][PAD  ...       ]
```

So the model still sees the beat boundary and the melody condition, but it no longer sees the ground-truth accompaniment for that beat.

### 4.2 Why

- The model cannot rely on complete accompaniment history all the time.
- It must continue generation using melody and broader context.

### 4.3 Important Property

Dropped accompaniment does not contribute to loss.

This means the model is not being trained to generate empty accompaniment.
Instead, it is trained to remain robust when accompaniment history is missing.

### 4.4 Implementation

Per beat:

```text
if random() < acc_drop_prob:
    drop accompaniment for this beat
```

So `drop` is not token-level deletion inside a beat. It is whole-segment replacement for that beat's accompaniment track.

Relevant code:

- [my_tokenizer.py](/data/home/yuanxin/RT-accompanimentV2/my_tokenizer.py#L519)
- [PianoDataset.py](/data/home/yuanxin/RT-accompanimentV2/PianoDataset.py#L144)
- [train.py](/data/home/yuanxin/RT-accompanimentV2/train.py#L69)
- [config.py](/data/home/yuanxin/RT-accompanimentV2/config.py#L49)

---

## 5. Init Drop

### 5.1 Idea

Instead of dropping accompaniment anywhere, deliberately drop the first `N` beats of the song.

This is still a beat-level operation, but it is applied to a fixed prefix region of the song.

Normal beat:

```text
input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]
```

Init-dropped beat:

```text
input:  [BEAT][EMPTY][TRK_ACC][mel ... TRK_MEL]
label:  [PAD] [PAD  ][PAD    ][PAD  ...       ]
```

For example, if `N = 2`, then the beginning of a song looks like:

```text
[beat 0] input:  [BEAT][EMPTY][TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [PAD  ][PAD    ][PAD  ...       ]

[beat 1] input:  [BEAT][EMPTY][TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [PAD  ][PAD    ][PAD  ...       ]

[beat 2] input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]
```

So the key difference from ordinary `drop` is not how one beat is modified, but which beats are selected.

### 5.2 Why

The beginning of generation is a cold-start setting:

- There is little or no generated accompaniment history.
- The model must start from melody and metadata.

Init drop explicitly trains this case.

### 5.3 Implementation

```text
if beat_idx < N:
    drop accompaniment
```

So the first `N` beats are always trained as missing-accompaniment beats.

Relevant code:

- [my_tokenizer.py](/data/home/yuanxin/RT-accompanimentV2/my_tokenizer.py#L496)
- [config.py](/data/home/yuanxin/RT-accompanimentV2/config.py#L54)

---

## 6. Probabilistic Init Drop

### 6.1 Idea

Always dropping the first `N` beats may be too strong, so we add a probability gate.

This keeps the same init-drop pattern, but does not apply it to every song.

Normal prefix beat:

```text
input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]
```

Probabilistic init-dropped prefix beat:

```text
input:  [BEAT][EMPTY][TRK_ACC][mel ... TRK_MEL]
label:  [PAD] [PAD  ][PAD    ][PAD  ...       ]
```

So there are two song-level cases.

Case 1: gate is on, so the prefix region is dropped:

```text
[beat 0] input:  [BEAT][EMPTY][TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [PAD  ][PAD    ][PAD  ...       ]

[beat 1] input:  [BEAT][EMPTY][TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [PAD  ][PAD    ][PAD  ...       ]

[beat 2] input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]
```

Case 2: gate is off, so the same prefix region stays normal:

```text
[beat 0] input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]

[beat 1] input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]

[beat 2] input:  [BEAT][acc ... TRK_ACC][mel ... TRK_MEL]
         label:  [PAD] [acc ... TRK_ACC][PAD  ...       ]
```

### 6.2 Why

- Keep the cold-start training signal.
- Avoid shifting the training distribution too aggressively.

### 6.3 Implementation

Per song:

```text
should_drop_initial = random() < drop_initial_beats_prob
```

Then only if `should_drop_initial` is true do we drop the first `N` beats:

```text
if should_drop_initial and beat_idx < N:
    drop accompaniment
```

So probabilistic init drop can be understood as:

```text
song level gate
    -> if on: apply init drop to the prefix region
    -> if off: keep the song in normal training mode
```

Relevant code:

- [my_tokenizer.py](/data/home/yuanxin/RT-accompanimentV2/my_tokenizer.py#L497)
- [config.py](/data/home/yuanxin/RT-accompanimentV2/config.py#L57)

---

## 7. Training Flow

Overall path:

```text
npz file
  -> PianoDataset.__getitem__
  -> sample pitch_shift
  -> forward acc_drop_prob / pos_shift_max / drop_initial_beats / drop_initial_beats_prob
  -> tokenizer.build_training_sequence(...)
  -> training sequence
```

Relevant code:

- [PianoDataset.py](/data/home/yuanxin/RT-accompanimentV2/PianoDataset.py#L130)
- [PianoDataset.py](/data/home/yuanxin/RT-accompanimentV2/PianoDataset.py#L138)
- [train.py](/data/home/yuanxin/RT-accompanimentV2/train.py#L62)

---

## 8. Test-Time Behavior

These are training-time augmentations only.

Inference does not randomly apply:

- beat-wise drop
- init drop
- shift augmentation

Inference uses the clean generation schedule:

- [my_tokenizer.py](/data/home/yuanxin/RT-accompanimentV2/my_tokenizer.py#L572)

---

## 9. Suggested Presentation Order

1. Base encoding method
2. Shift
3. Drop
4. Init drop
5. Probabilistic init drop

This order is natural because each later method is a direct extension of the previous design.

---

## 10. Experiment Names

For presentation, I use shorter experiment names than the raw output folder names:

- `Base` = `generated_samples-base`
- `Base (No Prefix)` = `generated_samples-base-no-prefix`
- `InitDrop p=0.3` = `generated_samples-initdrop2p0p3-0316_0833`
- `InitDrop p=0.3 (No Prefix)` = `generated_samples-initdrop2p0p3-0316_0833-with-no-prefix`
- `InitDrop Hard` = `generated_samples-initial-drop-acc-2`
- `InitDrop Hard (No Prefix)` = `generated_samples-initial-drop-acc-2-with-no-prefix`
- `Shift-3` = `generated_samples-shift-3`

This makes it easier to discuss results in the report while still keeping a clear mapping to the actual experiment directories.

---

## 11. Preliminary Result Interpretation

These observations are based on the current evaluation under `out/offline-0318/`.

At this stage, each run has only `20` evaluated pieces, so I treat the conclusions below as preliminary rather than final.

### 11.1 Main Takeaways

- `InitDrop p=0.3` is currently the most balanced setting.
- `Base` is a strong baseline.
- `No Prefix` consistently hurts performance.
- `InitDrop Hard` has decent local matching metrics, but its distribution-level behavior looks less stable.
- `Shift-3` does not look clearly better than the baseline in the current results.

### 11.2 Per-Run Interpretation

- `Base`
  - `onset_f1 = 0.7053`
  - `pitch_acc = 0.4661`
  - `note_density_gen = 6.604`, close to reference `6.702`
  - This is a solid baseline and already performs reasonably well.

- `Base (No Prefix)`
  - `onset_f1 = 0.6478`
  - `pitch_acc = 0.4214`
  - `note_density_gen = 5.867`, noticeably sparser than the reference
  - Compared with `Base`, removing prefix clearly hurts performance.

- `InitDrop p=0.3`
  - `onset_f1 = 0.7246`, the best among the current runs
  - `pitch_acc = 0.4586`
  - `note_density_gen = 6.699`, almost identical to reference `6.702`
  - This suggests that a moderate probabilistic init-drop gives a good robustness/performance balance.

- `InitDrop p=0.3 (No Prefix)`
  - `onset_f1 = 0.6351`
  - `pitch_acc = 0.4187`
  - `note_density_gen = 5.829`
  - It is still usable, but it drops noticeably compared with the prefix version.

- `InitDrop Hard`
  - `onset_f1 = 0.7154`
  - `pitch_acc = 0.4689`
  - `note_density_gen = 6.308`
  - Local matching metrics are fairly good, but FMD is much higher than `Base` and `InitDrop p=0.3`, so its overall generated distribution may be less natural.

- `InitDrop Hard (No Prefix)`
  - `onset_f1 = 0.5876`
  - `pitch_acc = 0.4094`
  - `note_density_gen = 7.195`
  - This is one of the weaker settings in the current comparison.

- `Shift-3`
  - `onset_f1 = 0.7061`
  - `pitch_acc = 0.4506`
  - `note_density_gen = 6.389`
  - Its local metrics are close to `Base`, but the current FMD estimate is very unstable.

### 11.3 About FMD

FMD is useful as a distribution-level signal, but with only `20` pieces per run it should be read carefully.

- `InitDrop p=0.3` has the lowest current FMD (`1.838`)
- `Base` is also low (`2.917`)
- `Shift-3` has a very large FMD (`118.2`) and a very wide confidence interval, so I would not draw a strong conclusion from it yet

So for now, I mainly use FMD to detect clearly suspicious runs, not to make fine-grained ranking claims.

### 11.4 Practical Conclusion For This Stage

If I summarize the current evidence in one sentence:

`InitDrop p=0.3` looks like the most promising setting so far, `Base` is a strong reference point, and removing prefix consistently makes generation harder.
