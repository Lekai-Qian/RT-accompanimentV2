# PianoDataset.py — 瘦数据集层
# 只负责: 文件 I/O、数据增强、截断、batching
# 所有 tokenization 逻辑委托给 PianoMusicTokenizer

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import pickle
from typing import List, Dict
from dataclasses import dataclass
from my_tokenizer import PianoMusicTokenizer


class PianoDataset(Dataset):
    """支持长度感知的钢琴音乐数据集"""

    def __init__(
        self,
        data_dir,
        config,
        cache_lengths=True,
        mode="train",
        test_split_ratio=0.05,
        random_seed=42,
        acc_drop_prob=0.0,
        pos_shift_max=0,
        drop_initial_beats=0,
        drop_initial_beats_prob=1.0,
        include_melody_loss=False,
    ):
        """
        Args:
            data_dir: 数据目录
            config: ModelConfig 实例
            cache_lengths: 是否使用长度缓存
            mode: 'train' 或 'test'
            test_split_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.root_dir = data_dir
        self.max_seq_len = config.train_cutoff_len
        self.mode = mode
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed
        self.acc_drop_prob = acc_drop_prob
        self.pos_shift_max = pos_shift_max
        self.drop_initial_beats = drop_initial_beats
        self.drop_initial_beats_prob = drop_initial_beats_prob
        self.include_melody_loss = include_melody_loss

        # 唯一的 tokenizer 入口
        self.tokenizer = PianoMusicTokenizer(config=config)

        self.data_files = [f for f in os.listdir(self.root_dir) if f.endswith(".npz")]
        print(f"找到 {len(self.data_files)} 个有效的npz文件")

        # 长度缓存
        cache_file = os.path.join(data_dir, ".lengths_cache.pkl")

        if cache_lengths and os.path.exists(cache_file):
            print("加载长度缓存...")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            if (
                cache_data["patch_h"] != self.tokenizer.vocab.default_patch_h
                or cache_data["patch_w"] != self.tokenizer.vocab.default_patch_w
            ):
                raise ValueError(
                    f"缓存的patch参数({cache_data['patch_h']}x{cache_data['patch_w']}) "
                    f"与配置({self.tokenizer.vocab.default_patch_h}x{self.tokenizer.vocab.default_patch_w})不匹配"
                )

            self.data_files = cache_data["data_files"]
            self.file_lengths = cache_data["lengths"]
            self.sorted_indices = cache_data["sorted_indices"]
            print(f"加载 {len(self.data_files)} 个文件的长度信息")

        elif cache_lengths:
            raise FileNotFoundError(f"长度缓存不存在: {cache_file}\n请先运行: python get_length.py")
        else:
            self.file_lengths = None
            self.sorted_indices = None
            print(f"找到 {len(self.data_files)} 个文件（未使用长度缓存）")

        self._split_train_test()

    def _split_train_test(self):
        """根据mode划分训练集和测试集"""
        total = len(self.data_files)
        np.random.seed(self.random_seed)
        indices = np.arange(total)
        np.random.shuffle(indices)

        test_size = int(total * self.test_split_ratio)
        train_size = total - test_size

        if self.mode == "train":
            sel = indices[:train_size]
            print(f"使用训练集: {len(sel)} 个文件 ({train_size}/{total})")
        elif self.mode == "test":
            sel = indices[train_size:]
            print(f"使用测试集: {len(sel)} 个文件 ({test_size}/{total})")
        else:
            raise ValueError(f"mode必须是'train'或'test'，当前为: {self.mode}")

        self.data_files = [self.data_files[i] for i in sel]
        if self.file_lengths is not None:
            self.file_lengths = [self.file_lengths[i] for i in sel]
            self.sorted_indices = sorted(range(len(self.file_lengths)), key=lambda i: self.file_lengths[i])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.data_files[idx])
        file_name = self.data_files[idx]

        # 1. 加载原始数据
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict["metadata"].item()
        num_measures = metadata["num_measures"]

        # 2. 根据文件名决定是否添加 BOS
        add_bos = True
        if "_" in file_name:
            suffix = file_name.split("_")[-1].replace(".npz", "")
            if suffix != "1" and suffix.isdigit():
                add_bos = False

        # 3. 数据增强: 音高移调
        pitch_shift = 0
        if np.random.random() < 0.7:
            pitch_shift = np.random.randint(-5, 6)

        # 4. 收集小节
        measures = [save_dict[f"measure_{i}"] for i in range(num_measures)]

        # 5. tokenize（全部委托给 tokenizer）
        input_ids, labels = self.tokenizer.build_training_sequence(
            measures=measures,
            metadata=metadata,
            add_bos=add_bos,
            pitch_shift=pitch_shift,
            acc_drop_prob=self.acc_drop_prob,
            pos_shift_max=self.pos_shift_max,
            drop_initial_beats=self.drop_initial_beats,
            drop_initial_beats_prob=self.drop_initial_beats_prob,
            include_melody_loss=self.include_melody_loss,
        )

        # 6. 截断
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class BucketBatchSampler(Sampler):
    """长度感知的批采样器"""

    def __init__(self, dataset, batch_size=16, bucket_size=100, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        if dataset.sorted_indices is None:
            raise ValueError("Dataset需要启用cache_lengths=True")

        self._create_buckets()

    def _create_buckets(self):
        self.buckets = []
        sorted_indices = self.dataset.sorted_indices
        for i in range(0, len(sorted_indices), self.bucket_size):
            self.buckets.append(sorted_indices[i : i + self.bucket_size])
        print(f"创建了 {len(self.buckets)} 个长度buckets")

    def __iter__(self):
        if self.shuffle:
            bucket_order = np.random.permutation(len(self.buckets))
        else:
            bucket_order = range(len(self.buckets))

        for bucket_idx in bucket_order:
            bucket = self.buckets[bucket_idx].copy()
            if self.shuffle:
                np.random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) > 0:
                    yield batch

    def __len__(self):
        return sum(len(b) for b in self.buckets) // self.batch_size


@dataclass
class DataCollatorForVariableLengthLM:
    """数据整理器，支持动态padding"""

    def __init__(self, config):
        self.pad_token_id = config.pad_token_id
        self.max_length = config.train_cutoff_len

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        batch = {"input_ids": [], "labels": [], "attention_mask": []}

        for f in features:
            ids = f["input_ids"]
            lbl = f["labels"]
            seq_len = len(ids)

            if seq_len < max_len:
                pad_len = max_len - seq_len
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
                mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            else:
                mask = torch.ones(seq_len, dtype=torch.long)

            batch["input_ids"].append(ids)
            batch["labels"].append(lbl)
            batch["attention_mask"].append(mask)

        return {k: torch.stack(v) for k, v in batch.items()}
