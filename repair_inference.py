"""
repair_inference.py — regenerate accompaniment from a marked bad region.

Workflow:
1. Read one repair job JSON from `repair_jobs/`.
2. Locate the original manifest item / dataset sample.
3. Convert bad region seconds to beat index.
4. Keep accompaniment prefix before the first bad beat.
5. Re-generate from that beat to the end.
6. Save before / gt / repaired / metadata into a separate output folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pretty_midi
import torch

from config import ModelConfig
from inference_new import load_model, prepare_generation, resolve_device, seed_everything
from my_tokenizer import GenerationStep
from PianoDataset import PianoDataset
from Token2Midi import MidiConverter


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, content: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def resolve_path(path_str: Optional[str], base_dir: Path) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def match_manifest_item(
    manifest: Dict[str, Any],
    source_midi: Optional[Path],
    sample_id: Optional[int],
) -> Dict[str, Any]:
    items = manifest.get("items", [])
    if sample_id is not None:
        for item in items:
            if int(item.get("sample_id", -1)) == int(sample_id):
                return item

    if source_midi is not None:
        source_name = source_midi.name
        for item in items:
            gen_midi_path = item.get("gen_midi_path")
            gt_midi_path = item.get("gt_midi_path")
            if gen_midi_path and Path(gen_midi_path).name == source_name:
                return item
            if gt_midi_path and Path(gt_midi_path).name == source_name:
                return item

    raise ValueError("Could not locate the source sample in manifest. Provide sample_id or matching source_midi.")


def seconds_to_beat_idx(t_sec: float, bpm: float, mode: str) -> int:
    raw = float(t_sec) * float(bpm) / 60.0
    if mode == "start":
        return max(0, int(raw // 1))
    return max(0, int(-(-raw // 1)))


def convert_regions_to_beats(regions_sec: List[Dict[str, Any]], bpm: float) -> List[Dict[str, Any]]:
    converted = []
    for region in regions_sec:
        t0 = float(region["t0"])
        t1 = float(region["t1"])
        b0 = seconds_to_beat_idx(t0, bpm, mode="start")
        b1 = seconds_to_beat_idx(t1, bpm, mode="end")
        converted.append(
            {
                "t0": t0,
                "t1": t1,
                "beat_start": b0,
                "beat_end": b1,
                "note": region.get("note", ""),
            }
        )
    return converted


def copy_if_exists(src: Optional[Path], dst: Path) -> Optional[str]:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def select_accompaniment_instrument(pm: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
    if not pm.instruments:
        raise ValueError("Source MIDI contains no instruments.")

    for inst in pm.instruments:
        name = (inst.name or "").strip().lower()
        if "accompaniment" in name or name == "acc":
            return inst

    if len(pm.instruments) >= 2:
        return pm.instruments[1]

    return pm.instruments[-1]


def generated_midi_to_acc_beats(
    midi_path: Path,
    tokenizer,
    bpm: float,
    num_beats: int,
    timesteps_per_beat: int = 4,
) -> List[torch.Tensor]:
    """Parse accompaniment from a generated MIDI and re-encode it as beat tokens."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    inst = select_accompaniment_instrument(pm)

    num_steps = max(1, int(num_beats * timesteps_per_beat))
    sec_per_step = 60.0 / float(bpm) / float(timesteps_per_beat)

    sustain = np.zeros((tokenizer.vocab.img_h, num_steps), dtype=np.float32)
    onset = np.zeros((tokenizer.vocab.img_h, num_steps), dtype=np.float32)

    for note in inst.notes:
        pitch_idx = int(note.pitch) - 21
        if not (0 <= pitch_idx < tokenizer.vocab.img_h):
            continue

        start_step = int(round(note.start / sec_per_step))
        end_step = int(round(note.end / sec_per_step))
        start_step = max(0, min(start_step, num_steps - 1))
        end_step = max(start_step + 1, min(end_step, num_steps))

        onset[pitch_idx, start_step] = 1.0
        sustain[pitch_idx, start_step:end_step] = 1.0

    acc_roll = np.stack([sustain, onset], axis=0)
    beats: List[torch.Tensor] = []

    for beat_idx in range(num_beats):
        s = beat_idx * timesteps_per_beat
        e = s + timesteps_per_beat
        beat = acc_roll[:, :, s:e]
        tokens_acc = tokenizer._codec.image_to_patch_tokens(beat, strict_mode=True)
        comp_acc = tokenizer.compress_tokens(tokens_acc, track_marker=tokenizer.vocab.track_marker_acc, pos_shift=0)
        beats.append(torch.tensor(comp_acc, dtype=torch.long))

    return beats


def patch_schedule_with_generated_prefix(
    schedule: List[GenerationStep],
    prefix_acc_beats: List[torch.Tensor],
    prefix_beats_count: int,
) -> List[GenerationStep]:
    patched: List[GenerationStep] = []
    beat_idx = 0

    for step in schedule:
        if step.action in ("generate", "inject_gt"):
            if beat_idx < prefix_beats_count:
                patched.append(GenerationStep("inject_gt", prefix_acc_beats[beat_idx]))
            else:
                patched.append(GenerationStep("generate"))
            beat_idx += 1
        else:
            patched.append(step)

    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair accompaniment by regenerating from a marked bad region.")
    parser.add_argument("--job", required=True, help="path to one repair job json")
    parser.add_argument("--output-root", default="local_tmp/repair_outputs", help="root directory for repair outputs")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--use-fp16", action="store_true", help="run model in fp16")
    parser.add_argument("--seed", type=int, default=None, help="override generation seed")
    parser.add_argument("--ckpt", type=str, default=None, help="override checkpoint path")
    parser.add_argument("--temperature", type=float, default=None, help="override temperature")
    parser.add_argument("--top-k", type=int, default=None, help="override top-k")
    parser.add_argument("--top-p", type=float, default=None, help="override top-p")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="override repetition penalty")
    args = parser.parse_args()

    job_path = Path(args.job).expanduser().resolve()
    if not job_path.is_file():
        raise FileNotFoundError(f"Repair job not found: {job_path}")

    job = load_json(str(job_path))
    job_dir = job_path.parent

    source_manifest = resolve_path(job.get("source_manifest"), job_dir)
    if source_manifest is None or not source_manifest.is_file():
        raise FileNotFoundError("Repair job must include a valid source_manifest.")
    manifest = load_json(str(source_manifest))

    source_midi = resolve_path(job.get("source_midi"), job_dir)
    source_gt_midi = resolve_path(job.get("source_gt_midi"), job_dir)
    sample_id = job.get("sample_id")
    item = match_manifest_item(manifest, source_midi=source_midi, sample_id=sample_id)

    manifest_dataset = manifest["dataset"]
    model_config = ModelConfig()
    dataset = PianoDataset(
        data_dir=manifest_dataset["data_dir"],
        config=model_config,
        cache_lengths=bool(manifest_dataset.get("cache_lengths", False)),
        mode=manifest_dataset["mode"],
        test_split_ratio=float(manifest_dataset["test_split_ratio"]),
        random_seed=int(manifest_dataset["dataset_seed"]),
    )

    dataset_index = int(item["dataset_index"])
    bpm = float(job.get("bpm") or item.get("metadata", {}).get("bpm") or 120.0)
    regions_sec = job.get("bad_regions_sec", [])
    if not regions_sec:
        raise ValueError("Repair job must contain non-empty bad_regions_sec.")
    regions_beats = convert_regions_to_beats(regions_sec, bpm=bpm)
    regen_start_beat = min(region["beat_start"] for region in regions_beats)

    prep = prepare_generation(dataset, dataset_index, gt_prefix_beats=0)
    if source_midi is None or not source_midi.is_file():
        raise FileNotFoundError("generated-prefix repair requires a valid source_midi.")

    prefix_acc_beats = generated_midi_to_acc_beats(
        midi_path=source_midi,
        tokenizer=dataset.tokenizer,
        bpm=bpm,
        num_beats=len(prep["acc_beats_gt"]),
    )
    schedule = patch_schedule_with_generated_prefix(
        prep["schedule"],
        prefix_acc_beats=prefix_acc_beats,
        prefix_beats_count=regen_start_beat,
    )

    generation_cfg = manifest.get("generation", {})
    seed = int(args.seed if args.seed is not None else job.get("seed", manifest.get("seed", 42)))
    ckpt = args.ckpt or job.get("ckpt") or manifest.get("ckpt")
    if not ckpt:
        raise ValueError("No checkpoint found. Provide --ckpt or include ckpt in job/manifest.")
    ckpt_path = resolve_path(ckpt, source_manifest.parent)
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    temperature = float(args.temperature if args.temperature is not None else generation_cfg.get("temperature", 1.1))
    top_k = int(args.top_k if args.top_k is not None else generation_cfg.get("top_k", 10))
    top_p = float(args.top_p if args.top_p is not None else generation_cfg.get("top_p", 0.95))
    repetition_penalty = float(
        args.repetition_penalty
        if args.repetition_penalty is not None
        else generation_cfg.get("repetition_penalty", 1.0)
    )
    use_fp16 = bool(args.use_fp16 or generation_cfg.get("use_fp16", False))

    seed_everything(seed)
    device = resolve_device(args.device)
    model = load_model(str(ckpt_path), model_config=model_config, device=device, use_fp16=use_fp16)

    generator = None
    if device.startswith("cuda") and torch.cuda.is_available():
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)

    acc_beats, generated_seq = model.generate_accompaniment(
        initial_tokens=prep["initial_tokens"],
        schedule=schedule,
        vocab=prep["vocab"],
        device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generator=generator,
    )

    output_root = Path(args.output_root)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = job.get("job_name") or job_path.stem
    job_output_dir = output_root / f"{run_timestamp}_{job_name}"
    job_output_dir.mkdir(parents=True, exist_ok=True)

    converter = MidiConverter(dataset.tokenizer)
    repaired_midi_path = job_output_dir / "repaired.mid"
    converter.beats_to_midi(
        mel_beats=prep["mel_beats"],
        acc_beats=acc_beats,
        tempo=prep["metadata"]["bpm"] or 120,
        save_path=str(repaired_midi_path),
    )

    before_copy = copy_if_exists(source_midi, job_output_dir / "before.mid")
    gt_copy = copy_if_exists(source_gt_midi, job_output_dir / "gt.mid")

    result = {
        "job_path": str(job_path),
        "job_name": job_name,
        "source_manifest": str(source_manifest),
        "source_midi": str(source_midi) if source_midi else None,
        "source_gt_midi": str(source_gt_midi) if source_gt_midi else None,
        "source_item": item,
        "repair_mode": "generated_prefix",
        "dataset_index": dataset_index,
        "regen_start_beat": regen_start_beat,
        "regions_sec": regions_sec,
        "regions_beats": regions_beats,
        "seed": seed,
        "device": device,
        "ckpt": str(ckpt_path),
        "generation": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "use_fp16": use_fp16,
        },
        "output": {
            "before_mid": before_copy,
            "gt_mid": gt_copy,
            "repaired_mid": str(repaired_midi_path),
            "generated_tokens": int(generated_seq.numel()),
        },
        "prefix_source": "source_midi_accompaniment",
    }
    save_json(job_output_dir / "repair_result.json", result)

    print(f"Repair complete: {job_name}")
    print(f"regen_start_beat={regen_start_beat}")
    print(f"output_dir={job_output_dir}")


if __name__ == "__main__":
    main()
