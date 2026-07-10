#!/usr/bin/env python
"""Evaluate a checkpoint through the existing RT dataset/tokenizer/model path."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import safetensors.torch
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import LlamaConfig


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from config import ModelConfig  # noqa: E402
from PianoDataset import DataCollatorForVariableLengthLM, PianoDataset  # noqa: E402
from model import PianoLLaMA  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate old-compatible RT checkpoint loss.")
    parser.add_argument("--ckpt", required=True, help="Path to model.safetensors.")
    parser.add_argument("--data-dir", required=True, help="NPZ dataset directory.")
    parser.add_argument("--mode", choices=["train", "test"], default="test")
    parser.add_argument("--test-split-ratio", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-json", default="", help="Optional path for JSON summary.")
    return parser.parse_args()


def create_llama_config(model_config: ModelConfig) -> LlamaConfig:
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
    )


def main():
    args = parse_args()
    ckpt = os.path.abspath(os.path.expanduser(args.ckpt))
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    torch.manual_seed(args.random_seed)
    model_config = ModelConfig()

    dataset = PianoDataset(
        data_dir,
        config=model_config,
        cache_lengths=True,
        mode=args.mode,
        test_split_ratio=args.test_split_ratio,
        random_seed=args.random_seed,
    )
    collator = DataCollatorForVariableLengthLM(model_config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=args.device == "cuda",
    )

    llama_config = create_llama_config(model_config)
    model = PianoLLaMA(llama_config)
    weights = safetensors.torch.load_file(ckpt)
    missing, unexpected = model.load_state_dict(weights, strict=False)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_labeled_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="eval")):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"],
            )
            total_loss += float(outputs.loss.item())
            total_batches += 1
            total_labeled_tokens += int((batch["labels"] != -100).sum().item())

    if total_batches == 0:
        raise RuntimeError("No batches evaluated.")

    avg_loss = total_loss / total_batches
    result = {
        "checkpoint": ckpt,
        "data_dir": data_dir,
        "mode": args.mode,
        "test_split_ratio": args.test_split_ratio,
        "random_seed": args.random_seed,
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "num_batches": total_batches,
        "total_labeled_tokens": total_labeled_tokens,
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss) if avg_loss < 50 else float("inf"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "tokenizer_path": "PianoMusicTokenizer.build_training_sequence",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = ROOT_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
