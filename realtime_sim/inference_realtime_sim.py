"""
inference_realtime_sim.py

Minimal realtime-style simulation for accompaniment generation.

Key assumptions:
- each tick consumes one usable acc beat
- inference is triggered every `I` ticks
- one inference call internally rolls out `G` alternating segments starting from acc
- only odd G is allowed
- generated melody is internal-only and discarded from final output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ModelConfig, TrainingConfig
from inference_new import load_model, prepare_generation, resolve_device, save_json, seed_everything, select_indices
from my_tokenizer import GenerationStep
from PianoDataset import PianoDataset
from Token2Midi import MidiConverter


def extract_beat_plans(schedule: List[GenerationStep], vocab) -> List[Dict]:
    beat_plans: List[Dict] = []
    pending_prefix: List[torch.Tensor] = []

    for step in schedule:
        if step.action != "inject" or step.data is None:
            continue

        data = step.data.cpu().tolist()
        if len(data) == 1 and data[0] == vocab.bar_token_id:
            pending_prefix.append(step.data.clone())
        elif len(data) == 1 and data[0] == vocab.beat_marker:
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


def build_rollout_schedule(
    beat_plans: List[Dict],
    start_beat: int,
    generation_length: int,
) -> Tuple[List[GenerationStep], int]:
    schedule: List[GenerationStep] = []
    segments_used = 0
    beat_idx = start_beat
    generated_acc_count = 0

    while segments_used < generation_length and beat_idx < len(beat_plans):
        for tok in beat_plans[beat_idx]["prefix_tokens"]:
            schedule.append(GenerationStep("inject", tok.clone()))

        schedule.append(GenerationStep("generate_acc"))
        generated_acc_count += 1
        segments_used += 1
        if segments_used >= generation_length:
            break

        schedule.append(GenerationStep("generate_mel"))
        segments_used += 1
        beat_idx += 1

    return schedule, generated_acc_count


def append_completed_beat_context(
    context_tokens: torch.Tensor,
    beat_plan: Dict,
    acc_tokens: List[int],
) -> torch.Tensor:
    parts = [context_tokens]
    for tok in beat_plan["prefix_tokens"]:
        parts.append(tok.clone())
    parts.append(torch.tensor(acc_tokens, dtype=torch.long))
    parts.append(beat_plan["mel_tokens"].clone())
    return torch.cat(parts)


def fallback_acc_tokens(vocab, policy: str, last_acc: List[int] | None) -> List[int]:
    empty = [vocab.empty_marker, vocab.track_marker_acc]
    if policy == "empty":
        return empty
    if policy == "hold_last":
        return list(last_acc) if last_acc is not None else empty
    raise ValueError(f"Unsupported fallback policy: {policy}")


def simulate_one_piece(
    model,
    prep: Dict,
    gt_prefix_beats: int,
    inference_interval: int,
    generation_length: int,
    fallback_policy: str,
    max_ticks: int | None,
    device: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
):
    vocab = prep["vocab"]
    beat_plans = extract_beat_plans(prep["schedule"], vocab)
    total_beats = len(beat_plans)

    context_tokens = prep["initial_tokens"].clone()
    consumed_acc_beats: List[List[int]] = []
    consumed_mel_beats: List[List[int]] = []
    last_acc: List[int] | None = None

    warmup_beats = min(max(gt_prefix_beats, 0), total_beats)
    for beat_idx in range(warmup_beats):
        acc_tokens = prep["acc_beats_gt"][beat_idx].cpu().tolist()
        context_tokens = append_completed_beat_context(context_tokens, beat_plans[beat_idx], acc_tokens)
        consumed_acc_beats.append(acc_tokens)
        consumed_mel_beats.append(beat_plans[beat_idx]["mel_tokens"].cpu().tolist())
        last_acc = acc_tokens

    play_cursor = warmup_beats
    timer = 0
    buffer: List[List[int]] = []
    stats = {
        "total_beats": total_beats,
        "warmup_beats": warmup_beats,
        "simulated_ticks": 0,
        "inference_calls": 0,
        "interval_hits": 0,
        "interval_skips_with_buffer": 0,
        "fallback_count": 0,
        "generated_acc_count": 0,
        "discarded_mel_count": 0,
        "call_logs": [],
    }

    while play_cursor < total_beats:
        if max_ticks is not None and stats["simulated_ticks"] >= max_ticks:
            break

        if timer % inference_interval == 0:
            stats["interval_hits"] += 1
            if len(buffer) == 0:
                rollout_schedule, expected_acc = build_rollout_schedule(
                    beat_plans=beat_plans,
                    start_beat=play_cursor,
                    generation_length=generation_length,
                )
                if rollout_schedule:
                    generator = None
                    if device.startswith("cuda") and torch.cuda.is_available():
                        generator = torch.Generator(device=device)
                    else:
                        generator = torch.Generator()

                    seed_value = int(timer + play_cursor + total_beats)
                    generator.manual_seed(seed_value)
                    acc_beats, mel_beats, _ = model.generate_by_schedule(
                        initial_tokens=context_tokens,
                        schedule=rollout_schedule,
                        vocab=vocab,
                        device=device,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        verbose=False,
                        generator=generator,
                    )
                    buffer.extend([list(x) for x in acc_beats])
                    stats["inference_calls"] += 1
                    stats["generated_acc_count"] += len(acc_beats)
                    stats["discarded_mel_count"] += len(mel_beats)
                    stats["call_logs"].append(
                        {
                            "tick": timer,
                            "start_beat": play_cursor,
                            "expected_acc": expected_acc,
                            "generated_acc": len(acc_beats),
                            "generated_mel": len(mel_beats),
                            "buffer_after_call": len(buffer),
                        }
                    )
            else:
                stats["interval_skips_with_buffer"] += 1

        if buffer:
            acc_tokens = buffer.pop(0)
            source = "generated"
        else:
            acc_tokens = fallback_acc_tokens(vocab, fallback_policy, last_acc)
            source = "fallback"
            stats["fallback_count"] += 1

        context_tokens = append_completed_beat_context(context_tokens, beat_plans[play_cursor], acc_tokens)
        consumed_acc_beats.append(acc_tokens)
        consumed_mel_beats.append(beat_plans[play_cursor]["mel_tokens"].cpu().tolist())
        last_acc = acc_tokens
        play_cursor += 1
        timer += 1
        stats["simulated_ticks"] += 1

        if stats["call_logs"]:
            stats["call_logs"][-1].setdefault("consumed_sources", []).append(source)

    stats["final_buffer_remaining"] = len(buffer)
    stats["completed_beats"] = len(consumed_acc_beats)
    stats["stopped_early"] = bool(max_ticks is not None and play_cursor < total_beats)
    return consumed_acc_beats, consumed_mel_beats, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime-style simulation for accompaniment generation")
    parser.add_argument("--ckpt", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--use-fp16", action="store_true", help="run model in FP16")

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
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

    parser.add_argument("--gt-prefix-beats", type=int, default=0)
    parser.add_argument("--inference-interval", type=int, default=1)
    parser.add_argument("--generation-length", type=int, default=1)
    parser.add_argument("--fallback-policy", type=str, choices=["empty", "hold_last"], default="empty")
    parser.add_argument("--max-ticks", type=int, default=None, help="optional limit for quick smoke tests")

    parser.add_argument("--output-dir", type=str, default="local_tmp/generated_samples_realtime_sim")
    parser.add_argument("--export-gt-midi", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("num-samples must be > 0")
    if args.inference_interval <= 0:
        raise ValueError("inference-interval must be > 0")
    if args.generation_length <= 0 or args.generation_length % 2 == 0:
        raise ValueError("generation-length must be a positive odd integer")
    if args.max_ticks is not None and args.max_ticks <= 0:
        raise ValueError("max-ticks must be > 0 when provided")

    ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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
        prep = prepare_generation(dataset, idx, gt_prefix_beats=0)
        acc_beats, mel_beats, stats = simulate_one_piece(
            model=model,
            prep=prep,
            gt_prefix_beats=args.gt_prefix_beats,
            inference_interval=args.inference_interval,
            generation_length=args.generation_length,
            fallback_policy=args.fallback_policy,
            max_ticks=args.max_ticks,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
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
                "stats": stats,
            }
        )

        print(
            f"[{sample_id + 1}/{len(indices)}] done | idx={idx} | file={prep['file_name']} | "
            f"I={args.inference_interval} | G={args.generation_length} | fallback={args.fallback_policy}"
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
        "realtime_sim": {
            "gt_prefix_beats": int(args.gt_prefix_beats),
            "inference_interval": int(args.inference_interval),
            "generation_length": int(args.generation_length),
            "fallback_policy": args.fallback_policy,
            "max_ticks": int(args.max_ticks) if args.max_ticks is not None else None,
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
