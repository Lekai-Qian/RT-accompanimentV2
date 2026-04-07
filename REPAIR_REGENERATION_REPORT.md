# Repair Regeneration For Accompaniment

## 1. Motivation

Even when the overall generated accompaniment is acceptable, some local segments may still sound wrong.

Typical cases:

- a bad bass pattern
- an awkward chord change
- a noisy fill-in
- a problematic opening phrase

In these cases, regenerating the whole piece is inefficient.
So the goal of repair is:

- keep the acceptable prefix
- identify the bad segment
- regenerate from that point onward

This is a local repair strategy rather than full-sequence regeneration.

---

## 2. Core Idea

The repair pipeline is based on **generated-prefix repair**.

That means:

- before the repair point, keep the accompaniment already generated in `before.mid`
- from the repair point onward, regenerate accompaniment with the model

This is different from the earlier GT-prefix version.

### 2.1 GT-Prefix Repair

```text
prefix: use GT accompaniment
suffix: regenerate
```

This is easy to implement, but it changes the music even before the marked bad region.

### 2.2 Generated-Prefix Repair

```text
prefix: use previously generated accompaniment from before.mid
suffix: regenerate
```

This better matches the real repair use case, because we want to preserve what was already acceptable.

---

## 3. Annotation Workflow

The workflow is:

1. listen to a generated MIDI
2. mark the bad region in seconds
3. convert that region to beat indices
4. keep the prefix before the first bad beat
5. regenerate from that beat to the end

We store repair annotations in `repair_jobs/*.json`.

Example schema:

```json
{
  "job_name": "base_no_prefix_piece12_25s_27s",
  "source_midi": "offline-0318/generated_samples-base-no-prefix/20260317_200010_12_5817834.mid",
  "source_gt_midi": "offline-0318/generated_samples-base-no-prefix/20260317_200010_12_5817834_GT.mid",
  "source_manifest": "offline-0318/generated_samples-base-no-prefix/20260317_200010_manifest.json",
  "sample_id": 12,
  "seed": 20260317,
  "bpm": 120,
  "bad_regions_sec": [
    {
      "t0": 25.0,
      "t1": 27.0,
      "note": "key change around 25-27s"
    }
  ]
}
```

---

## 4. Seconds To Beat Conversion

The user listens in seconds, but the model works in beat units.

So we convert:

```text
beat = time_sec * bpm / 60
```

Implementation:

```text
beat_start = floor(t0 * bpm / 60)
beat_end   = ceil(t1 * bpm / 60)
```

This means repair boundaries are beat-aligned, not exact second-aligned.

So if the bad segment is marked as `25s - 27s`, the actual regeneration starts from the corresponding beat boundary.

Relevant code:

- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L77)
- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L84)

---

## 5. Generated-Prefix Repair Pipeline

### 5.1 Step 1: Load Original Inference Metadata

The repair script reads:

- the original manifest
- the original generated MIDI
- the marked bad region

This lets us recover:

- dataset index
- seed
- checkpoint
- BPM
- original generation configuration

Relevant code:

- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L206)
- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L223)

### 5.2 Step 2: Build A Clean Generation Schedule

We first rebuild the generation plan with:

```text
gt_prefix_beats = 0
```

So the schedule itself starts as a fully generative schedule.

Relevant code:

- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L242)

### 5.3 Step 3: Parse The Generated Prefix From `before.mid`

We read the accompaniment track from `before.mid`, convert it back into pianoroll form, and then re-encode it into beat-level acc tokens.

Conceptually:

```text
before.mid
  -> accompaniment track
  -> beat-aligned sustain/onset pianoroll
  -> tokenizer encoding
  -> prefix acc beats
```

This is the key difference from GT-prefix repair.

Relevant code:

- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L112)
- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L127)

### 5.4 Step 4: Patch The Schedule

After we know `regen_start_beat`, we patch the schedule:

- beats before `regen_start_beat` become `inject_gt`, but the injected data is actually the parsed generated prefix
- beats from `regen_start_beat` onward stay as `generate`

So the effective logic is:

```text
if beat_idx < regen_start_beat:
    inject generated prefix beat
else:
    generate a new beat
```

Relevant code:

- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L171)
- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L252)

### 5.5 Step 5: Run Model Generation

The model then continues generation from the repaired boundary onward.

The result is exported as:

- `before.mid`
- `gt.mid`
- `repaired.mid`
- `repair_result.json`

Relevant code:

- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L288)
- [repair_inference.py](/data/home/yuanxin/RT-accompanimentV2/repair_inference.py#L318)

---

## 6. Why This Design

This design has several advantages:

- it preserves acceptable generated context
- it is closer to a real editing workflow
- it avoids replacing the prefix with unavailable GT information
- it lets us test whether local regeneration can improve a bad continuation

Compared with full regeneration, it is also more targeted.

---

## 7. Current Assumptions And Limitations

### 7.1 Beat-Aligned Repair

Repair starts at a beat boundary, not at an arbitrary time point.

So the repair region is approximate in time, but exact in beat index.

### 7.2 Prefix Is Recovered From MIDI

The current implementation reconstructs the prefix from `before.mid`.

So it depends on:

- correct track selection
- correct beat alignment
- correct MIDI-to-token re-encoding

### 7.3 Regeneration Goes To The End

The current version does not only repair a short middle window.

Instead:

```text
keep prefix
regenerate from repair point to the end
```

This is the simplest and most stable version.

### 7.4 Evaluation Is Still Needed

To judge whether repair helps, we still need paired comparison:

- before repair
- after repair

and ideally:

- automatic metrics
- listening-based evaluation

---

## 8. Current Example Jobs

Current example repair jobs include:

- [base_no_prefix_piece12_25s_27s.json](/data/home/yuanxin/RT-accompanimentV2/repair_jobs/base_no_prefix_piece12_25s_27s.json)
- [base_no_prefix_piece7_from_start.json](/data/home/yuanxin/RT-accompanimentV2/repair_jobs/base_no_prefix_piece7_from_start.json)

Default launcher:

- [run_repair_inference.sh](/data/home/yuanxin/RT-accompanimentV2/scripts/run_repair_inference.sh)
- [repair_inference.defaults.env](/data/home/yuanxin/RT-accompanimentV2/scripts/repair_inference.defaults.env)

Run with:

```bash
./scripts/run_repair_inference.sh
```

---

## 9. Practical Summary

If I summarize the repair mechanism in one sentence:

**mark a bad segment in seconds, convert it to beats, keep the already generated prefix, and regenerate accompaniment from that beat onward.**
