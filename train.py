"""
训练主程序
负责初始化数据集、模型和训练器，然后启动训练流程。

使用方法:
  单GPU训练: python train.py
  继续训练并覆盖增强参数:
      python train.py --ckpt checkpoints-resume/.../model.safetensors --drop-initial-beats 8 --drop-initial-beats-prob 1.0
  仅打印解析后的配置:
      python train.py --print-config --dry-run
"""

import argparse
import json
import os
from dataclasses import asdict
from typing import Optional

# 允许外部脚本覆盖 GPU 选择；未设置时保持原先默认卡 3
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import safetensors.torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig

from config import TrainingConfig, ModelConfig
from PianoDataset import BucketBatchSampler, DataCollatorForVariableLengthLM, PianoDataset
from trainer import TransformerTrainer
from model import PianoLLaMA


def parse_args():
    parser = argparse.ArgumentParser(description="Train accompaniment model with configurable experiment presets.")
    parser.add_argument("--ckpt", type=str, default=None, help="optional checkpoint to continue from")
    parser.add_argument("--experiment-name", type=str, default=None, help="override experiment name")
    parser.add_argument("--output-root", type=str, default=None, help="override output root directory")
    parser.add_argument("--output-dir", type=str, default=None, help="override full output directory")
    parser.add_argument("--num-epochs", type=int, default=None, help="override number of epochs")
    parser.add_argument("--acc-drop-prob", type=float, default=None, help="override per-beat acc drop probability")
    parser.add_argument("--drop-initial-beats", type=int, default=None, help="override number of initial beats to drop")
    parser.add_argument(
        "--drop-initial-beats-prob",
        type=float,
        default=None,
        help="override probability of applying init drop",
    )
    parser.add_argument("--pos-shift-max", type=int, default=None, help="override maximum per-song position shift")
    parser.add_argument("--data-dir", type=str, default=None, help="override training data directory")
    parser.add_argument("--print-config", action="store_true", help="print resolved config before training")
    parser.add_argument("--dry-run", action="store_true", help="resolve config and exit without training")
    return parser.parse_args()


def create_model_config(model_config: ModelConfig) -> LlamaConfig:
    """根据模型配置创建LLaMA配置。"""
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


def build_training_config(args) -> TrainingConfig:
    overrides = {}
    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.acc_drop_prob is not None:
        overrides["acc_drop_prob"] = args.acc_drop_prob
    if args.drop_initial_beats is not None:
        overrides["drop_initial_beats"] = args.drop_initial_beats
    if args.drop_initial_beats_prob is not None:
        overrides["drop_initial_beats_prob"] = args.drop_initial_beats_prob
    if args.pos_shift_max is not None:
        overrides["pos_shift_max"] = args.pos_shift_max
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    return TrainingConfig(**overrides)


def resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    if not checkpoint_path:
        return None
    path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def print_resolved_config(train_config: TrainingConfig, checkpoint_path: Optional[str]) -> None:
    payload = {
        "checkpoint": checkpoint_path,
        "training": asdict(train_config),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def save_run_metadata(train_config: TrainingConfig, checkpoint_path: Optional[str], cli_args) -> None:
    os.makedirs(train_config.output_dir, exist_ok=True)
    payload = {
        "checkpoint": checkpoint_path,
        "training": asdict(train_config),
        "cli_args": vars(cli_args),
    }
    out_path = os.path.join(train_config.output_dir, "run_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def create_datasets(train_config: TrainingConfig, model_config: ModelConfig, use_length_aware: bool):
    """创建训练集和测试集。"""
    train_dataset = PianoDataset(
        train_config.data_dir,
        config=model_config,
        cache_lengths=use_length_aware,
        mode="train",
        test_split_ratio=train_config.test_split_ratio,
        random_seed=train_config.random_seed,
        acc_drop_prob=train_config.acc_drop_prob,
        pos_shift_max=train_config.pos_shift_max,
        drop_initial_beats=train_config.drop_initial_beats,
        drop_initial_beats_prob=train_config.drop_initial_beats_prob,
    )

    test_dataset = None
    if train_config.use_test_set:
        test_dataset = PianoDataset(
            train_config.data_dir,
            config=model_config,
            cache_lengths=use_length_aware,
            mode="test",
            test_split_ratio=train_config.test_split_ratio,
            random_seed=train_config.random_seed,
        )
        print(f"训练集大小: {len(train_dataset)} 个样本")
        print(f"测试集大小: {len(test_dataset)} 个样本")

    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    test_dataset,
    train_config: TrainingConfig,
    model_config: ModelConfig,
    use_length_aware: bool,
    bucket_size: int,
):
    """创建数据加载器。"""
    collator = DataCollatorForVariableLengthLM(model_config)

    if use_length_aware:
        batch_sampler = BucketBatchSampler(
            train_dataset, batch_size=train_config.train_batch_size, bucket_size=bucket_size, shuffle=True
        )
        train_dataloader = DataLoader(
            train_dataset, batch_sampler=batch_sampler, num_workers=32, collate_fn=collator, pin_memory=True
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.train_batch_size,
            shuffle=True,
            num_workers=32,
            collate_fn=collator,
            pin_memory=True,
        )

    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config.test_batch_size,
            shuffle=False,
            num_workers=32,
            collate_fn=collator,
            pin_memory=True,
        )

    return train_dataloader, test_dataloader


def initialize_model(llama_config: LlamaConfig, checkpoint_path: Optional[str] = None) -> PianoLLaMA:
    """初始化模型并加载预训练权重。"""
    model = PianoLLaMA(llama_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    if checkpoint_path:
        print(f"正在加载预训练权重: {checkpoint_path}")
        weights = safetensors.torch.load_file(checkpoint_path)
        model.load_state_dict(weights, strict=False)
        print("预训练权重加载完成")

    return model


def main():
    """主函数：协调整个训练流程。"""
    args = parse_args()
    train_config = build_training_config(args)
    model_config = ModelConfig()
    checkpoint_path = resolve_checkpoint_path(args.ckpt)

    if args.print_config or args.dry_run:
        print_resolved_config(train_config, checkpoint_path)

    if args.dry_run:
        return

    use_length_aware_batching = True
    bucket_size = train_config.train_batch_size * 100

    llama_config = create_model_config(model_config)

    train_dataset, test_dataset = create_datasets(train_config, model_config, use_length_aware_batching)

    print("创建数据加载器")
    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset, test_dataset, train_config, model_config, use_length_aware_batching, bucket_size
    )

    print("初始化模型")
    model = initialize_model(llama_config, checkpoint_path)

    save_run_metadata(train_config, checkpoint_path, args)

    trainer = TransformerTrainer(
        config=train_config, model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader
    )

    print("\n开始训练...")
    trainer.train()
    print("\n训练完成!")


if __name__ == "__main__":
    main()
