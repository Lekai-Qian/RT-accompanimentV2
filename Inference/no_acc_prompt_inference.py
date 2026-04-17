from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ModelConfig, TrainingConfig
from inference_new import (
    load_indices_file,
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


PROMPT_MODES = ("gt", "mel_only", "no_prompt")


def resolve_repo_path(path_str: str | None) -> str | None:
    if not path_str:
        return path_str
    expanded = os.path.expanduser(path_str)
    if os.path.isabs(expanded):
        return expanded
    return str(ROOT_DIR / expanded)


def extract_beat_plans(prep: Dict) -> List[Dict]:
    """
    Convert the standard generation schedule (built with gt_prefix_beats=0) into
    beat-level plans.

    Each beat plan contains:
      - prefix_tokens: [bar?] + [beat_marker]
      - mel_tokens: the GT melody segment for this beat
    """
    schedule = prep["schedule"]
    vocab = prep["vocab"]
    beat_plans: List[Dict] = []
    pending_prefix: List[torch.Tensor] = []

    for step in schedule:
        if step.action != "inject" or step.data is None:
            continue

        data = step.data.cpu().tolist()
        if len(data) == 1 and data[0] in (vocab.bar_token_id, vocab.beat_marker):
            pending_prefix.append(step.data.clone())
        elif len(data) > 0 and data[-1] == vocab.track_marker_mel:
            beat_plans.append(
                {
                    "prefix_tokens": [tok.clone() for tok in pending_prefix],
                    "mel_tokens": step.data.clone(),
                }
            )
            pending_prefix = []

    return beat_plans


def build_prompt_schedule(prep: Dict, prompt_beats: int, prompt_mode: str) -> List[GenerationStep]:
    """
    Build a clean beat-level schedule with explicit semantics.

    Modes:
      - gt:        first N beats use GT acc prompt + GT mel prompt
      - mel_only:  first N beats use EMPTY acc + GT mel prompt
      - no_prompt: no GT prompt at all; generation starts from beat 0

    After the prompt region, all beats are generated in the standard way:
      prefix -> generate_acc -> inject mel
    """
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    vocab = prep["vocab"]
    empty_acc = torch.tensor([vocab.empty_marker, vocab.track_marker_acc], dtype=torch.long)
    beat_plans = extract_beat_plans(prep)
    acc_gt = prep["acc_beats_gt"]

    schedule: List[GenerationStep] = []
    effective_prompt_beats = 0 if prompt_mode == "no_prompt" else max(0, prompt_beats)

    for beat_idx, beat in enumerate(beat_plans):
        for tok in beat["prefix_tokens"]:
            schedule.append(GenerationStep("inject", tok.clone()))

        if beat_idx < effective_prompt_beats:
            if prompt_mode == "gt":
                schedule.append(GenerationStep("inject", acc_gt[beat_idx].clone()))
            elif prompt_mode == "mel_only":
                schedule.append(GenerationStep("inject", empty_acc.clone()))
            else:
                schedule.append(GenerationStep("generate_acc"))
        else:
            schedule.append(GenerationStep("generate_acc"))

        schedule.append(GenerationStep("inject", beat["mel_tokens"].clone()))

    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prompt-variant inference. Use this script only for prompt semantics experiments: "
            "GT prompt, melody-only prompt, or no prompt from the beginning."
        )
    )
    parser.add_argument("--ckpt", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--use-fp16", action="store_true", help="run model in FP16")

    parser.add_argument("--data-dir", type=str, default=None, help="dataset root; default from TrainingConfig")
    parser.add_argument("--num-samples", type=int, default=20, help="number of sampled pieces")
    parser.add_argument("--seed", type=int, default=42, help="sampling / generation seed")
    parser.add_argument("--indices-file", type=str, default=None, help="explicit dataset indices file")
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

    parser.add_argument("--prompt-beats", type=int, default=8, help="prompt length in beats")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=list(PROMPT_MODES),
        default="mel_only",
        help="gt = GT acc+mel prompt, mel_only = only mel prompt, no_prompt = start from beginning",
    )

    parser.add_argument("--output-dir", type=str, default="local_tmp/generated_samples_prompt_variants")
    parser.add_argument("--export-gt-midi", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    args = parser.parse_args()

    ckpt_path = resolve_repo_path(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.num_samples <= 0:
        raise ValueError("num-samples must be > 0")
    if args.prompt_beats < 0:
        raise ValueError("prompt-beats must be >= 0")

    seed_everything(args.seed)
    device = resolve_device(args.device)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    data_dir = resolve_repo_path(args.data_dir or train_config.data_dir)

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

    if args.indices_file:
        indices_file = resolve_repo_path(args.indices_file)
        indices = load_indices_file(indices_file)
        if args.num_samples != len(indices):
            print(
                f"indices-file provides {len(indices)} samples; overriding num-samples={args.num_samples}",
                flush=True,
            )
    else:
        indices_file = None
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

    output_dir = Path(resolve_repo_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest_items = []
    for sample_id, idx in enumerate(indices):
        # Always build from the clean no-GT-prefix base schedule, then explicitly patch prompt semantics.
        prep = prepare_generation(dataset, idx, gt_prefix_beats=0)
        schedule = build_prompt_schedule(prep=prep, prompt_beats=args.prompt_beats, prompt_mode=args.prompt_mode)

        generator = torch.Generator(device=device) if device.startswith("cuda") and torch.cuda.is_available() else torch.Generator()
        generator.manual_seed(args.seed + sample_id)

        acc_beats, mel_beats, generated_seq = model.generate_by_schedule(
            initial_tokens=prep["initial_tokens"],
            schedule=schedule,
            vocab=prep["vocab"],
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            verbose=False,
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

        effective_prompt_beats = 0 if args.prompt_mode == "no_prompt" else int(args.prompt_beats)
        manifest_items.append(
            {
                "sample_id": int(sample_id),
                "dataset_index": int(idx),
                "file_name": prep["file_name"],
                "gt_path": prep["gt_path"],
                "gt_midi_path": str(gt_midi_path) if gt_midi_path else None,
                "gen_midi_path": str(gen_midi_path),
                "metadata": prep["metadata"],
                "prompt_beats": effective_prompt_beats,
                "prompt_mode": args.prompt_mode,
                "generated_tokens": int(generated_seq.numel()),
                "num_acc_beats": int(len(acc_beats)),
                "num_mel_beats": int(len(mel_beats)),
            }
        )

        print(
            f"[{sample_id + 1}/{len(indices)}] done | idx={idx} | file={prep['file_name']} | "
            f"prompt_mode={args.prompt_mode} | prompt_beats={effective_prompt_beats}"
        )

    fingerprint_input = "\n".join(f"{x['dataset_index']}|{x['file_name']}" for x in manifest_items)
    fingerprint = hashlib.sha1(fingerprint_input.encode("utf-8")).hexdigest()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "device": device,
        "ckpt": ckpt_path,
        "num_samples": int(args.num_samples),
        "resolved_num_samples": int(len(indices)),
        "sampling_mode": args.sampling_mode,
        "indices_file": indices_file,
        "dataset": {
            "data_dir": data_dir,
            "mode": args.dataset_mode,
            "dataset_seed": int(args.dataset_seed),
            "test_split_ratio": float(args.test_split_ratio),
            "cache_lengths": bool(args.cache_lengths),
            "pool_size": int(len(dataset)),
        },
        "prompt_variant": {
            "prompt_mode": args.prompt_mode,
            "prompt_beats": 0 if args.prompt_mode == "no_prompt" else int(args.prompt_beats),
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
