"""
训练主程序
负责初始化数据集、模型和训练器，然后启动训练流程

使用方法:
  单GPU训练: python train.py
  多GPU训练: accelerate launch --multi_gpu --num_processes=3 train.py
  后台运行: nohup accelerate launch --multi_gpu --num_processes=2 train.py > training.log 2>&1 &
"""

import os

# 设置可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import safetensors.torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig

from config import TrainingConfig, ModelConfig
from PianoDataset import BucketBatchSampler, DataCollatorForVariableLengthLM, PianoDataset
from trainer import TransformerTrainer
from model import PianoLLaMA


def create_model_config(model_config: ModelConfig) -> LlamaConfig:
    """根据模型配置创建LLaMA配置

    Args:
        model_config: 自定义的模型配置对象

    Returns:
        LlamaConfig: transformers库的LLaMA配置对象
    """
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


def create_datasets(train_config: TrainingConfig, model_config: ModelConfig, use_length_aware: bool):
    """创建训练集和测试集

    Args:
        train_config: 训练配置
        model_config: 模型配置
        use_length_aware: 是否使用长度感知batching

    Returns:
        tuple: (训练集, 测试集或None)
    """
    # 创建训练数据集
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

    # 创建测试数据集（如果启用）
    test_dataset = None
    if train_config.use_test_set:
        test_dataset = PianoDataset(
            train_config.data_dir,
            config=model_config,
            cache_lengths=use_length_aware,
            mode="test",
            test_split_ratio=train_config.test_split_ratio,
            random_seed=train_config.random_seed,
            # 测试集不做 drop，保持评估一致性
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
    """创建数据加载器

    Args:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        train_config: 训练配置
        model_config: 模型配置
        use_length_aware: 是否使用长度感知batching
        bucket_size: bucket的大小

    Returns:
        tuple: (训练集dataloader, 测试集dataloader或None)
    """
    collator = DataCollatorForVariableLengthLM(model_config)

    # 创建训练集dataloader
    if use_length_aware:
        # 使用长度感知的采样器
        batch_sampler = BucketBatchSampler(
            train_dataset, batch_size=train_config.train_batch_size, bucket_size=bucket_size, shuffle=True
        )
        train_dataloader = DataLoader(
            train_dataset, batch_sampler=batch_sampler, num_workers=32, collate_fn=collator, pin_memory=True
        )
    else:
        # 传统的随机batching
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.train_batch_size,
            shuffle=True,
            num_workers=32,
            collate_fn=collator,
            pin_memory=True,
        )

    # 创建测试集dataloader
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


def initialize_model(llama_config: LlamaConfig, checkpoint_path: str = None) -> PianoLLaMA:
    """初始化模型并加载预训练权重

    Args:
        llama_config: LLaMA配置
        checkpoint_path: 预训练权重的路径（可选）

    Returns:
        PianoLLaMA: 初始化好的模型
    """
    model = PianoLLaMA(llama_config)

    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 加载预训练权重（如果提供）
    if checkpoint_path:
        print(f"正在加载预训练权重: {checkpoint_path}")
        weights = safetensors.torch.load_file(checkpoint_path)
        model.load_state_dict(weights, strict=False)
        print("预训练权重加载完成")

    return model


def main():
    """主函数：协调整个训练流程"""
    # 加载配置
    train_config = TrainingConfig()
    model_config = ModelConfig()

    # 数据加载配置
    use_length_aware_batching = True  # 是否使用长度感知batching
    bucket_size = train_config.train_batch_size * 100  # 每个bucket包含的样本数

    # 创建模型配置
    llama_config = create_model_config(model_config)

    # 创建数据集
    train_dataset, test_dataset = create_datasets(train_config, model_config, use_length_aware_batching)

    print("创建数据加载器")
    train_dataloader, test_dataloader = create_dataloaders(
        train_dataset, test_dataset, train_config, model_config, use_length_aware_batching, bucket_size
    )

    print("初始化模型")
    checkpoint_path = "checkpoints-resume/epoch_15_0307_1858/model.safetensors"
    model = initialize_model(llama_config, checkpoint_path)

    # 初始化训练器
    trainer = TransformerTrainer(
        config=train_config, model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()
    print("\n训练完成!")


if __name__ == "__main__":
    main()
