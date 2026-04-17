from __future__ import annotations

import argparse
import gc
import hashlib
import os
import signal
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
from inference_new import load_indices_file
from my_tokenizer import GenerationStep
from PianoDataset import PianoDataset
from Token2Midi import MidiConverter


def resolve_repo_path(path_str: str | None) -> str | None:
    if not path_str:
        return path_str
    expanded = os.path.expanduser(path_str)
    if os.path.isabs(expanded):
        return expanded
    return str(ROOT_DIR / expanded)


def extract_beat_plans(schedule: List[GenerationStep], vocab) -> List[Dict]:
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


def required_acc_before_next_interval(play_cursor: int, total_beats: int, inference_interval: int) -> int:
    return max(0, min(inference_interval, total_beats - play_cursor))


class SampleTimeoutError(RuntimeError):
    pass


def _timeout_handler(signum, frame):
    raise SampleTimeoutError("per-sample timeout exceeded")


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def cleanup_after_oom() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def cleanup_after_timeout() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def resolve_sample_timeout_sec(
    fixed_timeout_sec: int,
    timeout_factor: float,
    piece_length_sec: float,
) -> int:
    if fixed_timeout_sec > 0:
        return int(fixed_timeout_sec)
    if timeout_factor <= 0 or piece_length_sec <= 0:
        return 0
    return max(1, int(round(timeout_factor * piece_length_sec)))


def simulate_one_piece(
    model,
    prep: Dict,
    gt_prefix_beats: int,
    inference_interval: int,
    generation_length: int,
    catch_up_prob: float,
    fallback_policy: str,
    max_ticks: int | None,
    device: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    rng_seed: int,
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
    rng = torch.Generator()
    rng.manual_seed(rng_seed)
    previous_interval_missed = False

    stats = {
        "total_beats": total_beats,
        "warmup_beats": warmup_beats,
        "simulated_ticks": 0,
        "scheduled_calls": 0,
        "successful_calls": 0,
        "missed_calls": 0,
        "fallback_count": 0,
        "generated_acc_count": 0,
        "discarded_mel_count": 0,
        "buffer_shortfall_count": 0,
        "forced_catch_up_count": 0,
        "call_logs": [],
    }

    while play_cursor < total_beats:
        if max_ticks is not None and stats["simulated_ticks"] >= max_ticks:
            break

        if timer % inference_interval == 0:
            stats["scheduled_calls"] += 1
            random_catch_up = torch.rand(1, generator=rng).item() < catch_up_prob
            force_catch_up = previous_interval_missed
            should_catch_up = force_catch_up or random_catch_up
            interval_log = {
                "tick": timer,
                "start_beat": play_cursor,
                "buffer_before_call": len(buffer),
                "random_catch_up": bool(random_catch_up),
                "forced_catch_up": bool(force_catch_up),
                "should_catch_up": bool(should_catch_up),
            }

            if should_catch_up:
                if force_catch_up:
                    stats["forced_catch_up_count"] += 1
                rollout_schedule, expected_acc = build_rollout_schedule(
                    beat_plans=beat_plans,
                    start_beat=play_cursor,
                    generation_length=generation_length,
                )
                interval_log["expected_acc"] = expected_acc
                if rollout_schedule:
                    generator = torch.Generator(device=device) if device.startswith("cuda") and torch.cuda.is_available() else torch.Generator()
                    generator.manual_seed(int(rng_seed + timer + play_cursor + total_beats))
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
                    buffer = [list(x) for x in acc_beats]
                    stats["successful_calls"] += 1
                    stats["generated_acc_count"] += len(acc_beats)
                    stats["discarded_mel_count"] += len(mel_beats)
                    interval_log["generated_acc"] = len(acc_beats)
                    interval_log["generated_mel"] = len(mel_beats)
                else:
                    buffer = []
                    stats["successful_calls"] += 1
                    interval_log["generated_acc"] = 0
                    interval_log["generated_mel"] = 0
                previous_interval_missed = False
            else:
                stats["missed_calls"] += 1
                previous_interval_missed = True
                if fallback_policy == "empty":
                    buffer = []
                interval_log["missed_reason"] = "probabilistic_skip"
                interval_log["buffer_preserved_on_miss"] = len(buffer)

            needed = required_acc_before_next_interval(play_cursor, total_beats, inference_interval)
            interval_log["required_before_next_interval"] = needed
            interval_log["buffer_after_call"] = len(buffer)
            interval_log["caught_up"] = len(buffer) >= needed
            if not interval_log["caught_up"]:
                stats["buffer_shortfall_count"] += 1
            stats["call_logs"].append(interval_log)

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

    stats["empirical_catch_up_rate"] = (
        stats["successful_calls"] / stats["scheduled_calls"] if stats["scheduled_calls"] > 0 else 0.0
    )
    stats["buffer_coverage_rate"] = (
        sum(1 for x in stats["call_logs"] if x.get("caught_up")) / len(stats["call_logs"])
        if stats["call_logs"] else 0.0
    )
    stats["final_buffer_remaining"] = len(buffer)
    stats["completed_beats"] = len(consumed_acc_beats)
    stats["stopped_early"] = bool(max_ticks is not None and play_cursor < total_beats)
    return consumed_acc_beats, consumed_mel_beats, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probabilistic interval-rollout inference: every m beats, inference may or may not catch up."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--use-fp16", action="store_true", help="run model in FP16")

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices-file", type=str, default=None, help="explicit dataset indices file (json list or one integer per line)")
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
    parser.add_argument("--catch-up-prob", type=float, default=1.0)
    parser.add_argument("--fallback-policy", type=str, choices=["empty", "hold_last"], default="empty")
    parser.add_argument("--max-ticks", type=int, default=None, help="optional limit for quick smoke tests")
    parser.add_argument("--per-sample-timeout-sec", type=int, default=0, help="timeout per sample; <=0 disables")
    parser.add_argument(
        "--per-sample-timeout-factor",
        type=float,
        default=2.0,
        help="if per-sample-timeout-sec <= 0, use timeout_factor * original piece length in seconds",
    )

    parser.add_argument("--output-dir", type=str, default="local_tmp/generated_samples_interval_rollout")
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
    if not (0.0 <= args.catch_up_prob <= 1.0):
        raise ValueError("catch-up-prob must be in [0, 1]")
    if args.max_ticks is not None and args.max_ticks <= 0:
        raise ValueError("max-ticks must be > 0 when provided")
    if args.per_sample_timeout_sec < 0:
        raise ValueError("per-sample-timeout-sec must be >= 0")
    if args.per_sample_timeout_factor < 0:
        raise ValueError("per-sample-timeout-factor must be >= 0")

    ckpt_path = resolve_repo_path(args.ckpt)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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
    failed_items = []
    for sample_id, idx in enumerate(indices):
        prep = prepare_generation(dataset, idx, gt_prefix_beats=0)
        stem = Path(prep["gt_path"]).stem
        sample_timeout_sec = resolve_sample_timeout_sec(
            fixed_timeout_sec=int(args.per_sample_timeout_sec),
            timeout_factor=float(args.per_sample_timeout_factor),
            piece_length_sec=float(prep["metadata"].get("piece_length_sec", 0.0) or 0.0),
        )
        try:
            if sample_timeout_sec > 0:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(int(sample_timeout_sec))
            acc_beats, mel_beats, stats = simulate_one_piece(
                model=model,
                prep=prep,
                gt_prefix_beats=args.gt_prefix_beats,
                inference_interval=args.inference_interval,
                generation_length=args.generation_length,
                catch_up_prob=args.catch_up_prob,
                fallback_policy=args.fallback_policy,
                max_ticks=args.max_ticks,
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                rng_seed=args.seed + sample_id,
            )

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
                    "sample_timeout_sec": int(sample_timeout_sec),
                }
            )

            print(
                f"[{sample_id + 1}/{len(indices)}] done | idx={idx} | file={prep['file_name']} | "
                f"interval={args.inference_interval} | G={args.generation_length} | catch_up_prob={args.catch_up_prob:.2f} | "
                f"empirical={stats['empirical_catch_up_rate']:.3f}"
            )
        except Exception as exc:
            if isinstance(exc, SampleTimeoutError):
                cleanup_after_timeout()
                failed = {
                    "sample_id": int(sample_id),
                    "dataset_index": int(idx),
                    "file_name": prep["file_name"],
                    "gt_path": prep["gt_path"],
                    "metadata": prep["metadata"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failure_reason": "timeout",
                    "timeout_sec": int(sample_timeout_sec),
                    "interval": int(args.inference_interval),
                    "generation_length": int(args.generation_length),
                    "catch_up_prob": float(args.catch_up_prob),
                }
                failed_items.append(failed)
                print(
                    f"[{sample_id + 1}/{len(indices)}] TIMEOUT | idx={idx} | file={prep['file_name']} | "
                    f"interval={args.inference_interval} | G={args.generation_length} | "
                    f"catch_up_prob={args.catch_up_prob:.2f} | timeout={sample_timeout_sec}s"
                )
            elif is_cuda_oom(exc):
                cleanup_after_oom()
                failed = {
                    "sample_id": int(sample_id),
                    "dataset_index": int(idx),
                    "file_name": prep["file_name"],
                    "gt_path": prep["gt_path"],
                    "metadata": prep["metadata"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failure_reason": "cuda_oom",
                    "interval": int(args.inference_interval),
                    "generation_length": int(args.generation_length),
                    "catch_up_prob": float(args.catch_up_prob),
                }
                failed_items.append(failed)
                print(
                    f"[{sample_id + 1}/{len(indices)}] OOM | idx={idx} | file={prep['file_name']} | "
                    f"interval={args.inference_interval} | G={args.generation_length} | catch_up_prob={args.catch_up_prob:.2f}"
                )
            else:
                raise
        finally:
            if sample_timeout_sec > 0:
                signal.alarm(0)

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
        "interval_rollout": {
            "gt_prefix_beats": int(args.gt_prefix_beats),
            "inference_interval": int(args.inference_interval),
            "generation_length": int(args.generation_length),
            "catch_up_prob": float(args.catch_up_prob),
            "fallback_policy": args.fallback_policy,
            "max_ticks": int(args.max_ticks) if args.max_ticks is not None else None,
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "use_fp16": bool(args.use_fp16),
            "export_gt_midi": bool(args.export_gt_midi),
            "per_sample_timeout_sec": int(args.per_sample_timeout_sec),
            "per_sample_timeout_factor": float(args.per_sample_timeout_factor),
        },
        "selection_fingerprint_sha1": fingerprint,
        "items": manifest_items,
        "failed_items": failed_items,
    }
    save_json(output_dir / f"{run_timestamp}_manifest.json", manifest)


if __name__ == "__main__":
    main()
