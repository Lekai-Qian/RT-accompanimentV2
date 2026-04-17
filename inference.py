"""
DEPRECATED / LEGACY ENTRYPOINT.

inference.py — 推理编排层

职责: 加载数据 → tokenizer 构建 schedule → model 生成 → MidiConverter 输出
是较早的离线推理胶水代码，保留主要用于参考和向后兼容。

正式离线实验建议改用 `inference_new.py`。
"""

from pathlib import Path
from datetime import datetime
import os
import torch
import numpy as np
import safetensors.torch
from transformers import LlamaConfig

from config import ModelConfig
from model import PianoLLaMA
from Token2Midi import MidiConverter
from PianoDataset import PianoDataset

device = "cuda:5" if torch.cuda.is_available() else "cpu"


# ==================== 模型加载 ====================


def setup_model_configs_llama(model_config: ModelConfig):
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


def load_model(model_path, model_config, device="cuda", use_fp16=False):
    token_config = setup_model_configs_llama(model_config)
    model = PianoLLaMA(token_config)

    if model_path:
        weights = safetensors.torch.load_file(model_path)
        model.load_state_dict(weights, strict=True)

    if use_fp16 and torch.cuda.is_available():
        model = model.half()

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} params, {'FP16' if use_fp16 else 'FP32'}")
    return model


# ==================== 数据准备（从 model.py 移出） ====================


def prepare_generation(dataset, condition_idx, gt_prefix_beats=12):
    """
    从 dataset 加载一首曲子并构建生成计划。

    Returns:
        dict: initial_tokens, schedule, vocab, mel_beats, gt_path, metadata
    """
    tokenizer = dataset.tokenizer
    file_path = os.path.join(dataset.root_dir, dataset.data_files[condition_idx])
    save_dict = np.load(file_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()
    measures = [save_dict[f"measure_{i}"] for i in range(metadata["num_measures"])]

    gen_data = tokenizer.build_generation_schedule(
        measures=measures,
        metadata=metadata,
        gt_prefix_beats=gt_prefix_beats,
    )

    ts_idx = metadata["time_signature_idx"]
    if ts_idx == 9:
        ts_idx = 4

    return {
        "initial_tokens": gen_data["initial_tokens"],
        "schedule": gen_data["schedule"],
        "vocab": tokenizer.vocab,
        "mel_beats": gen_data["mel_beats"],
        "gt_path": file_path,
        "metadata": {
            "time_signature_idx": ts_idx,
            "bpm": metadata["bpm"],
            "num_measures": metadata["num_measures"],
        },
    }


# ==================== 批量生成 ====================


def batch_generate(
    model, dataset, converter, output_dir="generated_samples", num_samples=50, gt_prefix_beats=12, **gen_kwargs
):
    """
    批量生成并保存 MIDI。

    数据流: dataset → prepare_generation → model.generate → converter.beats_to_midi
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_samples):
        idx = torch.randint(0, len(dataset), (1,)).item()

        # 1. 准备数据 + schedule
        prep = prepare_generation(dataset, idx, gt_prefix_beats)

        # 2. 模型生成（纯推理）
        acc_beats, _ = model.generate_accompaniment(
            initial_tokens=prep["initial_tokens"],
            schedule=prep["schedule"],
            vocab=prep["vocab"],
            device=device,
            **gen_kwargs,
        )

        # 3. 保存 MIDI
        gt_name = Path(prep["gt_path"]).stem
        tempo = prep["metadata"]["bpm"] or 120

        converter.gt_to_midi(prep["gt_path"], f"{output_dir}/{timestamp}_{i}_{gt_name}_GT.mid")

        converter.beats_to_midi(
            mel_beats=prep["mel_beats"],
            acc_beats=acc_beats,
            tempo=tempo,
            save_path=f"{output_dir}/{timestamp}_{i}_{gt_name}.mid",
        )

        print(f"[{i + 1}/{num_samples}] done")


# ==================== 入口 ====================

if __name__ == "__main__":
    model_config = ModelConfig()

    dataset = PianoDataset(
        "/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore",
        config=model_config,
        cache_lengths=False,
    )
    print(f"Using device: {device}")

    # model = load_model(
    #     model_path="checkpoints-resume/epoch_15_0307_1858/model.safetensors",
    #     model_config=model_config,
    #     device=device,
    # )
    model = load_model(
        model_path="checkpoints-drop-0.2/epoch_3_0311_1354/model.safetensors",
        model_config=model_config,
        device=device,
    )

    converter = MidiConverter(dataset.tokenizer)

    batch_generate(
        model,
        dataset,
        converter,
        # output_dir="generated_samples_1030-shift3",
        output_dir="generated_samples_0314-drop-02-no-prefix",
        num_samples=200,
        # gt_prefix_beats=12,
        gt_prefix_beats=0,
        temperature=1.1,
        top_k=10,
        top_p=0.95,
        repetition_penalty=1.0,
    )
