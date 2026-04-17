"""
DEPRECATED experimental entrypoint.

inference_special.py originally bundled multiple stress-test ideas together:
1. overwrite the front acc prompt to simulate without-prompt behavior
2. optionally generate melody instead of injecting GT melody

These focused modes are now being split into dedicated scripts under
`Inference/` for cleaner experiment control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from config import ModelConfig, TrainingConfig
from inference_new import (
    load_model,
    prepare_generation,
    resolve_device,
    save_json,
    seed_everything,
    select_indices,
)
from my_tokenizer import GenerationStep
from PianoDataset import PianoDataset
from Token2Midi import MidiConverter


def classify_injected_step(step: GenerationStep, vocab) -> str:
    if step.data is None:
        return "unknown"

    data = step.data.cpu().tolist()
    if len(data) == 1:
        tok = data[0]
        if tok == vocab.bar_token_id:
            return "bar"
        if tok == vocab.beat_marker:
            return "beat"
        return "scalar"

    last_tok = data[-1]
    if last_tok == vocab.track_marker_acc:
        return "acc"
    if last_tok == vocab.track_marker_mel:
        return "mel"
    return "unknown"


def build_special_schedule(base_schedule: List[GenerationStep], vocab, acc_prefix_mode: str, melody_mode: str):
    schedule: List[GenerationStep] = []

    for step in base_schedule:
        if step.action == "inject_gt":
            if acc_prefix_mode == "gt":
                schedule.append(GenerationStep("inject_gt", step.data))
            elif acc_prefix_mode == "none":
                schedule.append(GenerationStep("generate_acc"))
            else:
                raise ValueError(f"Unsupported acc_prefix_mode: {acc_prefix_mode}")

        elif step.action == "generate":
            schedule.append(GenerationStep("generate_acc"))

        elif step.action == "inject":
            kind = classify_injected_step(step, vocab)
            if kind == "mel":
                if melody_mode == "inject":
                    schedule.append(step)
                elif melody_mode == "generate":
                    schedule.append(GenerationStep("generate_mel"))
                else:
                    raise ValueError(f"Unsupported melody_mode: {melody_mode}")
            else:
                schedule.append(step)

        else:
            raise ValueError(f"Unsupported base schedule action: {step.action}")

    return schedule


def summarize_schedule(schedule: List[GenerationStep]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for step in schedule:
        counts[step.action] = counts.get(step.action, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Special inference for prompt/melody stress tests")

    parser.add_argument("--ckpt", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--use-fp16", action="store_true", help="run model in FP16")

    parser.add_argument("--data-dir", type=str, default=None, help="dataset root; default from TrainingConfig")
    parser.add_argument("--num-samples", type=int, default=20, help="number of sampled pieces")
    parser.add_argument("--seed", type=int, default=42, help="sampling / generation seed")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["with_replacement", "without_replacement"],
        default="with_replacement",
    )
    parser.add_argument("--dataset-mode", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--test-split-ratio", type=float, default=0.10)
    parser.add_argument("--cache-lengths", action="store_true")

    parser.add_argument("--gt-prefix-beats", type=int, default=12)
    parser.add_argument("--acc-prefix-mode", type=str, choices=["gt", "none"], default="gt")
    parser.add_argument("--melody-mode", type=str, choices=["inject", "generate"], default="inject")

    parser.add_argument("--output-dir", type=str, default="local_tmp/generated_samples_special")
    parser.add_argument("--export-gt-midi", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    args = parser.parse_args()

    ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.num_samples <= 0:
        raise ValueError("num-samples must be > 0")

    seed_everything(args.seed)
    device = resolve_device(args.device)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    data_dir = args.data_dir or train_config.data_dir

    dataset = PianoDataset(
        data_dir=data_dir,
        config=model_config,
        cache_lengths=args.cache_lengths,
        mode=args.dataset_mode,
        test_split_ratio=args.test_split_ratio,
        random_seed=args.dataset_seed,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    indices = select_indices(
        population_size=len(dataset),
        num_samples=args.num_samples,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
    )

    model = load_model(
        model_path=ckpt_path,
        model_config=model_config,
        device=device,
        use_fp16=args.use_fp16,
    )
    converter = MidiConverter(dataset.tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest_items = []
    for sample_id, idx in enumerate(indices):
        prep = prepare_generation(dataset, idx, gt_prefix_beats=args.gt_prefix_beats)
        special_schedule = build_special_schedule(
            base_schedule=prep["schedule"],
            vocab=prep["vocab"],
            acc_prefix_mode=args.acc_prefix_mode,
            melody_mode=args.melody_mode,
        )

        generator = None
        if device.startswith("cuda") and torch.cuda.is_available():
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
        generator.manual_seed(args.seed)

        acc_beats, mel_beats, generated_seq = model.generate_by_schedule(
            initial_tokens=prep["initial_tokens"],
            schedule=special_schedule,
            vocab=prep["vocab"],
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            generator=generator,
        )

        stem = Path(prep["gt_path"]).stem
        gen_midi_path = output_dir / f"{run_timestamp}_{sample_id}_{stem}.mid"
        converter.beats_to_midi(
            mel_beats=mel_beats,
            acc_beats=acc_beats,
            tempo=prep["metadata"]["bpm"] or 120,
            save_path=str(gen_midi_path),
        )

        gt_midi_path = None
        if args.export_gt_midi:
            gt_midi_path = output_dir / f"{run_timestamp}_{sample_id}_{stem}_GT.mid"
            converter.gt_to_midi(prep["gt_path"], str(gt_midi_path))

        manifest_items.append(
            {
                "sample_id": int(sample_id),
                "dataset_index": int(idx),
                "file_name": prep["file_name"],
                "gt_path": prep["gt_path"],
                "gt_midi_path": str(gt_midi_path) if gt_midi_path else None,
                "gen_midi_path": str(gen_midi_path),
                "metadata": prep["metadata"],
                "schedule_counts": summarize_schedule(special_schedule),
                "num_acc_beats": int(len(acc_beats)),
                "num_mel_beats": int(len(mel_beats)),
                "generated_tokens": int(generated_seq.numel()),
            }
        )

        print(
            f"[{sample_id + 1}/{len(indices)}] done | idx={idx} | file={prep['file_name']} | "
            f"acc_prefix_mode={args.acc_prefix_mode} | melody_mode={args.melody_mode}"
        )

    fingerprint_input = "\n".join(f"{x['dataset_index']}|{x['file_name']}" for x in manifest_items)
    fingerprint = hashlib.sha1(fingerprint_input.encode("utf-8")).hexdigest()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "device": device,
        "ckpt": ckpt_path,
        "num_samples": int(args.num_samples),
        "sampling_mode": args.sampling_mode,
        "dataset": {
            "data_dir": data_dir,
            "mode": args.dataset_mode,
            "dataset_seed": int(args.dataset_seed),
            "test_split_ratio": float(args.test_split_ratio),
            "cache_lengths": bool(args.cache_lengths),
            "pool_size": int(len(dataset)),
        },
        "generation": {
            "gt_prefix_beats": int(args.gt_prefix_beats),
            "acc_prefix_mode": args.acc_prefix_mode,
            "melody_mode": args.melody_mode,
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "use_fp16": bool(args.use_fp16),
            "export_gt_midi": bool(args.export_gt_midi),
        },
        "selection_fingerprint_sha1": fingerprint,
        "items": manifest_items,
    }
    save_json(output_dir / f"{run_timestamp}_manifest.json", manifest)


if __name__ == "__main__":
    main()
