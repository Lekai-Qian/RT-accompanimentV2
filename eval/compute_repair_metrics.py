"""
compute_repair_metrics.py

Evaluate repair outputs by comparing:
- before.mid   vs gt.mid
- repaired.mid vs gt.mid

Expected folder layout (recursive):
  repair_outputs/<job_dir>/
    before.mid
    gt.mid
    repaired.mid
    repair_result.json

Outputs:
- detailed CSV with before/after/delta metrics
- summary CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from compute_midi_metrics import eval_pair, format_sig, build_summary


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_repair_runs(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    runs: List[Path] = []
    for dpath, _, files in os.walk(root):
        file_set = set(files)
        if {"before.mid", "gt.mid", "repaired.mid"}.issubset(file_set):
            runs.append(Path(dpath))
    return sorted(runs)


def region_summary(regions: List[Dict[str, Any]], key_start: str, key_end: str) -> str:
    if not regions:
        return ""
    parts = []
    for r in regions:
        if key_start in r and key_end in r:
            parts.append(f"{r[key_start]}-{r[key_end]}")
    return "; ".join(parts)


def maybe_note_density_gap(row: Dict[str, Any], prefix: str) -> float:
    ref = float(row[f"{prefix}note_density_ref"])
    gen = float(row[f"{prefix}note_density_gen"])
    return abs(gen - ref)


def evaluate_run(run_dir: Path, tol: float) -> Dict[str, Any]:
    before_mid = run_dir / "before.mid"
    gt_mid = run_dir / "gt.mid"
    repaired_mid = run_dir / "repaired.mid"
    result_json = run_dir / "repair_result.json"

    meta: Dict[str, Any] = {}
    if result_json.is_file():
        meta = load_json(result_json)

    before = eval_pair(str(gt_mid), str(before_mid), tol=tol)
    after = eval_pair(str(gt_mid), str(repaired_mid), tol=tol)

    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "job_name": meta.get("job_name", run_dir.name),
        "repair_mode": meta.get("repair_mode", ""),
        "sample_id": meta.get("source_item", {}).get("sample_id", ""),
        "dataset_index": meta.get("dataset_index", ""),
        "regen_start_beat": meta.get("regen_start_beat", ""),
        "regions_sec": region_summary(meta.get("regions_sec", []), "t0", "t1"),
        "regions_beats": region_summary(meta.get("regions_beats", []), "beat_start", "beat_end"),
        "source_midi": meta.get("source_midi", ""),
        "gt_mid": str(gt_mid),
        "before_mid": str(before_mid),
        "repaired_mid": str(repaired_mid),
    }

    for key, value in before.items():
        row[f"before_{key}"] = value
    for key, value in after.items():
        row[f"after_{key}"] = value

    delta_keys = [
        "onset_p",
        "onset_r",
        "onset_f1",
        "pitch_acc",
        "pitch_jsd",
        "onset_jsd",
        "matched",
        "gen_notes",
        "gen_dur",
        "note_density_gen",
    ]
    for key in delta_keys:
        row[f"delta_{key}"] = float(row[f"after_{key}"]) - float(row[f"before_{key}"])

    row["before_note_density_gap"] = maybe_note_density_gap(row, "before_")
    row["after_note_density_gap"] = maybe_note_density_gap(row, "after_")
    row["delta_note_density_gap"] = row["after_note_density_gap"] - row["before_note_density_gap"]

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate before/repaired MIDI pairs under a repair output folder.")
    parser.add_argument("--repair-dir", required=True, help="root directory containing repair output subfolders")
    parser.add_argument("--out", default="repair_eval.csv", help="output CSV path")
    parser.add_argument("--tol", type=float, default=0.05, help="onset tolerance in seconds")
    parser.add_argument("--sig-digits", type=int, default=4, help="significant digits in CSV output")
    args = parser.parse_args()

    if args.sig_digits <= 0:
        raise ValueError("--sig-digits must be > 0")

    runs = find_repair_runs(args.repair_dir)
    if not runs:
        raise FileNotFoundError(f"No repair runs found under: {args.repair_dir}")

    if args.out == "repair_eval.csv":
        last_dir = os.path.basename(os.path.normpath(args.repair_dir))
        out_dir = os.path.join("out", last_dir)
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, "repair_eval.csv")
    else:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    for run_dir in runs:
        print(f"Evaluating repair run: {run_dir}")
        rows.append(evaluate_run(run_dir, tol=args.tol))

    keys = [
        "run_dir",
        "job_name",
        "repair_mode",
        "sample_id",
        "dataset_index",
        "regen_start_beat",
        "regions_sec",
        "regions_beats",
        "source_midi",
        "gt_mid",
        "before_mid",
        "repaired_mid",
        "before_ref_notes",
        "before_gen_notes",
        "before_ref_dur",
        "before_gen_dur",
        "before_onset_p",
        "before_onset_r",
        "before_onset_f1",
        "before_pitch_acc",
        "before_pitch_jsd",
        "before_onset_jsd",
        "before_matched",
        "before_note_density_ref",
        "before_note_density_gen",
        "before_note_density_gap",
        "after_ref_notes",
        "after_gen_notes",
        "after_ref_dur",
        "after_gen_dur",
        "after_onset_p",
        "after_onset_r",
        "after_onset_f1",
        "after_pitch_acc",
        "after_pitch_jsd",
        "after_onset_jsd",
        "after_matched",
        "after_note_density_ref",
        "after_note_density_gen",
        "after_note_density_gap",
        "delta_gen_notes",
        "delta_gen_dur",
        "delta_onset_p",
        "delta_onset_r",
        "delta_onset_f1",
        "delta_pitch_acc",
        "delta_pitch_jsd",
        "delta_onset_jsd",
        "delta_matched",
        "delta_note_density_gen",
        "delta_note_density_gap",
    ]

    numeric_keys = {
        "sample_id",
        "dataset_index",
        "regen_start_beat",
        "before_ref_notes",
        "before_gen_notes",
        "before_ref_dur",
        "before_gen_dur",
        "before_onset_p",
        "before_onset_r",
        "before_onset_f1",
        "before_pitch_acc",
        "before_pitch_jsd",
        "before_onset_jsd",
        "before_matched",
        "before_note_density_ref",
        "before_note_density_gen",
        "before_note_density_gap",
        "after_ref_notes",
        "after_gen_notes",
        "after_ref_dur",
        "after_gen_dur",
        "after_onset_p",
        "after_onset_r",
        "after_onset_f1",
        "after_pitch_acc",
        "after_pitch_jsd",
        "after_onset_jsd",
        "after_matched",
        "after_note_density_ref",
        "after_note_density_gen",
        "after_note_density_gap",
        "delta_gen_notes",
        "delta_gen_dur",
        "delta_onset_p",
        "delta_onset_r",
        "delta_onset_f1",
        "delta_pitch_acc",
        "delta_pitch_jsd",
        "delta_onset_jsd",
        "delta_matched",
        "delta_note_density_gen",
        "delta_note_density_gap",
    }

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in keys:
                value = row.get(key, "")
                formatted[key] = format_sig(value, sig_digits=args.sig_digits) if key in numeric_keys else value
            writer.writerow(formatted)

    summary_keys = [
        "before_onset_p",
        "before_onset_r",
        "before_onset_f1",
        "before_pitch_acc",
        "before_pitch_jsd",
        "before_onset_jsd",
        "before_note_density_gap",
        "after_onset_p",
        "after_onset_r",
        "after_onset_f1",
        "after_pitch_acc",
        "after_pitch_jsd",
        "after_onset_jsd",
        "after_note_density_gap",
        "delta_onset_p",
        "delta_onset_r",
        "delta_onset_f1",
        "delta_pitch_acc",
        "delta_pitch_jsd",
        "delta_onset_jsd",
        "delta_note_density_gap",
    ]
    summary = build_summary(rows, summary_keys)

    summary_csv = args.out + ".summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "count", "mean", "std", "median", "min", "max"])
        for key in summary_keys:
            stat = summary[key]
            writer.writerow(
                [
                    key,
                    stat["count"],
                    format_sig(stat["mean"], sig_digits=args.sig_digits),
                    format_sig(stat["std"], sig_digits=args.sig_digits),
                    format_sig(stat["median"], sig_digits=args.sig_digits),
                    format_sig(stat["min"], sig_digits=args.sig_digits),
                    format_sig(stat["max"], sig_digits=args.sig_digits),
                ]
            )

    print(f"Wrote detail CSV to {args.out}")
    print(f"Wrote summary CSV to {summary_csv}")


if __name__ == "__main__":
    main()
