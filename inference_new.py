"""
inference_new.py — 可复现推理脚本

数据流:
  加载数据 -> 按 seed 可复现抽样 -> tokenizer 构建 schedule -> model 生成 -> MIDI 输出

说明:
  1) 默认执行完整 inference（含模型生成）
  2) 可用 --prepare-only 仅准备计划并记录 manifest（不跑模型）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import safetensors.torch
import torch
from transformers import LlamaConfig

from config import ModelConfig, TrainingConfig
from model import PianoLLaMA
from PianoDataset import PianoDataset
from Token2Midi import MidiConverter


def seed_everything(seed: int) -> None:
    """固定随机性来源，保证抽样可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device_arg


def setup_model_configs_llama(model_config: ModelConfig) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_token_id=model_config.pad_token_id,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        rope_theta=model_config.rope_theta,
        attention_dropout=model_config.dropout,
        use_cache=True,
        initializer_range=0.02,
    )


def load_model(model_path: str, model_config: ModelConfig, device: str = "cuda", use_fp16: bool = False) -> PianoLLaMA:
    token_config = setup_model_configs_llama(model_config)
    model = PianoLLaMA(token_config)

    weights = safetensors.torch.load_file(model_path)
    model.load_state_dict(weights, strict=True)

    if use_fp16 and torch.cuda.is_available():
        model = model.half()

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} params, {'FP16' if use_fp16 else 'FP32'}")
    return model


def select_indices(population_size: int, num_samples: int, seed: int, sampling_mode: str) -> List[int]:
    """按给定 seed 抽样索引。"""
    rng = np.random.default_rng(seed)

    if sampling_mode == "with_replacement":
        return rng.integers(0, population_size, size=num_samples).tolist()

    if num_samples > population_size:
        raise ValueError(
            f"without_replacement 模式下 num_samples={num_samples} 不能大于数据池大小 {population_size}"
        )
    return rng.choice(population_size, size=num_samples, replace=False).tolist()


def prepare_generation(dataset: PianoDataset, condition_idx: int, gt_prefix_beats: int = 12) -> Dict:
    """从 dataset 加载曲目并构建 generation 计划。"""
    tokenizer = dataset.tokenizer
    file_name = dataset.data_files[condition_idx]
    file_path = os.path.join(dataset.root_dir, file_name)

    save_dict = np.load(file_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()
    measures = [save_dict[f"measure_{i}"] for i in range(metadata["num_measures"])]

    gen_data = tokenizer.build_generation_schedule(
        measures=measures,
        metadata=metadata,
        gt_prefix_beats=gt_prefix_beats,
    )

    ts_idx = int(metadata.get("time_signature_idx", 4))
    if ts_idx == 9:
        ts_idx = 4

    return {
        "dataset_index": int(condition_idx),
        "file_name": file_name,
        "gt_path": file_path,
        "initial_tokens": gen_data["initial_tokens"],
        "schedule": gen_data["schedule"],
        "vocab": tokenizer.vocab,
        "mel_beats": gen_data["mel_beats"],
        "acc_beats_gt": gen_data["acc_beats_gt"],
        "metadata": {
            "time_signature_idx": ts_idx,
            "bpm": int(metadata.get("bpm", 120) or 120),
            "num_measures": int(metadata["num_measures"]),
        },
    }


def save_json(path: Path, content: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="可复现 inference")

    parser.add_argument("--ckpt", type=str, default=None, help="模型 checkpoint 路径")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--use-fp16", action="store_true", help="模型推理使用 FP16")

    parser.add_argument("--data-dir", type=str, default=None, help="数据目录；默认读取 TrainingConfig")
    parser.add_argument("--num-samples", type=int, default=50, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        choices=["with_replacement", "without_replacement"],
        default="with_replacement",
        help="抽样方式",
    )

    parser.add_argument("--dataset-mode", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--test-split-ratio", type=float, default=0.10)
    parser.add_argument("--cache-lengths", action="store_true", help="是否读取长度缓存")

    parser.add_argument("--gt-prefix-beats", type=int, default=12)
    parser.add_argument("--output-dir", type=str, default="generated_samples_repro")
    parser.add_argument(
        "--export-gt-midi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否导出 GT MIDI",
    )
    parser.add_argument(
        "--prepare-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅准备计划与 manifest，不执行模型生成",
    )

    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("num-samples 必须 > 0")

    if not args.prepare_only and not args.ckpt:
        raise ValueError("完整 inference 需要 --ckpt；若只准备计划请加 --prepare-only")

    ckpt_path = None
    if args.ckpt:
        ckpt_path = os.path.abspath(os.path.expanduser(args.ckpt))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--ckpt 指定文件不存在: {ckpt_path}")

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
        raise RuntimeError("数据集为空，无法抽样")

    indices = select_indices(
        population_size=len(dataset),
        num_samples=args.num_samples,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
    )

    model = None
    if not args.prepare_only:
        model = load_model(
            model_path=ckpt_path,
            model_config=model_config,
            device=device,
            use_fp16=args.use_fp16,
        )

    output_dir = Path(args.output_dir)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    converter = MidiConverter(dataset.tokenizer)

    manifest_items = []
    for sample_id, idx in enumerate(indices):
        prep = prepare_generation(dataset, idx, gt_prefix_beats=args.gt_prefix_beats)
        stem = Path(prep["gt_path"]).stem

        gt_midi_path = None
        if args.export_gt_midi:
            gt_midi_path = output_dir / f"{run_timestamp}_{sample_id}_{stem}_GT.mid"
            converter.gt_to_midi(prep["gt_path"], str(gt_midi_path))

        gen_midi_path = None
        generated_tokens = 0
        if model is not None:
            acc_beats, generated_seq = model.generate_accompaniment(
                initial_tokens=prep["initial_tokens"],
                schedule=prep["schedule"],
                vocab=prep["vocab"],
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            generated_tokens = int(generated_seq.numel())

            tempo = prep["metadata"]["bpm"] or 120
            gen_midi_path = output_dir / f"{run_timestamp}_{sample_id}_{stem}.mid"
            converter.beats_to_midi(
                mel_beats=prep["mel_beats"],
                acc_beats=acc_beats,
                tempo=tempo,
                save_path=str(gen_midi_path),
            )

        manifest_items.append(
            {
                "sample_id": int(sample_id),
                "dataset_index": int(idx),
                "file_name": prep["file_name"],
                "gt_path": prep["gt_path"],
                "gt_midi_path": str(gt_midi_path) if gt_midi_path else None,
                "gen_midi_path": str(gen_midi_path) if gen_midi_path else None,
                "metadata": prep["metadata"],
                "num_schedule_steps": int(len(prep["schedule"])),
                "num_mel_beats": int(len(prep["mel_beats"])),
                "generated_tokens": generated_tokens,
            }
        )

        print(f"[{sample_id + 1}/{len(indices)}] done | idx={idx} | file={prep['file_name']}")

    fingerprint_input = "\n".join(f"{x['dataset_index']}|{x['file_name']}" for x in manifest_items)
    fingerprint = hashlib.sha1(fingerprint_input.encode("utf-8")).hexdigest()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "device": device,
        "ckpt": ckpt_path,
        "prepare_only": bool(args.prepare_only),
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

    print("\n完成：inference 结果已生成")
    print(f"输出目录: {output_dir}")
    print(f"样本数量: {len(manifest_items)}")
    print(f"抽样指纹: {fingerprint}")


if __name__ == "__main__":
    main()
