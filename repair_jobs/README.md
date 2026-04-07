# Repair Jobs

This folder stores manual repair annotations for bad accompaniment segments.

Why keep these JSON files separate from the generated MIDI folders:
- The original generation folders stay clean and remain the source of truth.
- Repair annotations are an extra experiment layer and are easier to manage centrally.
- We can re-run repair inference later without touching original outputs.

Recommended workflow:
1. Listen to a generated MIDI in `offline-0318/...`.
2. Create one JSON job file in `repair_jobs/`.
3. Mark bad regions in seconds.
4. A later `repair_inference` script will convert seconds to beat ranges and re-generate from the chosen point.

Minimal schema:

```json
{
  "job_name": "short-readable-name",
  "source_midi": "offline-0318/generated_samples-xxx/20260318_0001_piece.mid",
  "source_gt_midi": "offline-0318/generated_samples-xxx/20260318_0001_piece_GT.mid",
  "source_manifest": "offline-0318/generated_samples-xxx/20260318_0001_manifest.json",
  "sample_id": 0,
  "seed": 20260318,
  "bpm": 120,
  "bad_regions_sec": [
    {
      "t0": 23.4,
      "t1": 28.1,
      "note": "bass pattern sounds messy"
    }
  ]
}
```

Field notes:
- `source_midi`: the generated MIDI to repair.
- `source_gt_midi`: optional, useful for later comparison.
- `source_manifest`: optional but recommended, helps recover the exact inference run.
- `sample_id`: optional if it can be recovered from filename/manifest, but recommended.
- `seed`: optional but useful for reproducibility bookkeeping.
- `bpm`: required for converting seconds to beat index.
- `bad_regions_sec`: one or more bad regions in seconds.

Current convention:
- We store times in seconds first because that matches listening workflow.
- Later code will convert:
  `beat = time_sec * bpm / 60`
- Repair generation should normally keep the prefix before the first bad beat and regenerate from there.
