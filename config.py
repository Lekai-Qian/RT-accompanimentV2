from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


def _format_exp_value(value: float) -> str:
    """将数值压缩成适合实验名的短字符串。"""
    text = f"{value:g}"
    return text.replace(".", "p")


@dataclass
class TrainingConfig:
    """训练配置类"""

    # 基础配置
    time: str = field(default_factory=lambda: datetime.now().strftime("%m%d_%H%M"))
    experiment_name: str = ""  # 留空时根据关键增强参数自动生成
    output_root: str = "./checkpoints"
    output_dir: str = ""  # 留空时自动绑定到 experiment_name
    num_epochs: int = 4
    save_model_epochs: int = 1
    train_batch_size: int = 16
    gradient_accumulation_steps: int = 32  # effective batch = 16×32=512
    lr_warmup_steps: int = 200
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"  # H200 原生支持 bf16，更稳定

    # 学习率配置
    learning_rate: float = 5e-5

    # 日志配置
    log: bool = True
    log_every_n_steps: int = 20
    tensorboard_log_dir: str = "./logs"
    tensorboard_log_name: str = ""  # 留空时自动绑定到 experiment_name
    # standard (192K files): allxml_npz_dual_track_optimized_no_underscore
    # augmented (342K files): allxml_npz_dual_track_optimized
    data_dir: str = "/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore"
    save_steps: int = -1

    # 测试集配置
    use_test_set: bool = True  # 是否使用测试集
    test_split_ratio: float = 0.10  # 测试集划分比例（5%作为测试集）
    test_frequency: float = 0.10  # 测试频率（每0.10个epoch测试一次）
    test_batch_size: int = 4  # 测试时的batch size
    test_save_results: bool = True  # 是否保存测试结果到tensorboard

    # Drop Accompaniment 训练增强
    # acc_drop_prob: 以此概率随机 drop 任意 beat 的 acc（per-beat）
    # 0.0 = 关闭（默认），推荐值 0.1~0.2
    acc_drop_prob: float = 0

    # Drop Initial Beats 训练增强
    # drop_initial_beats: drop 每首曲子前 N 个 beat 的 acc（强制"冷启动"）
    # 0 = 关闭（默认）
    drop_initial_beats: int = 0
    # drop_initial_beats_prob: 应用概率
    # 0.0 = 关闭（默认），1.0 = 总是 drop 前 N beat，0.5 = 50% 概率 drop 前 N beat
    drop_initial_beats_prob: float = 0.0

    # Position Shift 训练增强
    # 每首曲子随机采样 Δ ∈ {0,...,pos_shift_max}，整体偏移 beat 内的位置标记
    # 使 BEAT token 不再是严格的"位置0"锚点，增强泛化
    # 0 = 关闭（默认），推荐值 3
    pos_shift_max: int = 0

    # Melody Loss 训练选项
    # False = mel 仍作为条件输入，不额外加入监督
    # True  = mel token 也作为预测目标加入 loss
    include_melody_loss: bool = False
    random_seed: int = 42  # 数据集划分的随机种子

    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = self._build_experiment_name()

        run_name = f"{self.experiment_name}_{self.time}"
        if not self.tensorboard_log_name:
            self.tensorboard_log_name = run_name
        if not self.output_dir:
            self.output_dir = f"{self.output_root}/{run_name}"

    def _build_experiment_name(self) -> str:
        """根据最关键的训练增强参数生成稳定、易读的实验名。"""
        parts = ["music_transformer"]

        if self.pos_shift_max > 0:
            parts.append(f"shift{self.pos_shift_max}")

        if self.acc_drop_prob > 0:
            parts.append(f"accdrop{_format_exp_value(self.acc_drop_prob)}")

        if self.drop_initial_beats > 0 and self.drop_initial_beats_prob > 0:
            part = f"initdrop{self.drop_initial_beats}"
            if self.drop_initial_beats_prob < 1.0:
                part += f"p{_format_exp_value(self.drop_initial_beats_prob)}"
            parts.append(part)

        if self.include_melody_loss:
            parts.append("melloss")

        return "_".join(parts)


@dataclass
class ModelConfig:
    """模型架构配置"""

    # Token-level GPT配置
    vocab_size: int = 268
    hidden_size: int = 768  #
    num_hidden_layers: int = 18
    num_attention_heads: int = 6
    intermediate_size: int = 3072
    max_position_embeddings: int = 3500
    markoffset = 81  # 位置标记起始ID
    measures_length = 88  # 每个小节的最大长度（单位：位置标记数）
    patch_h = 1  # 音高方向的patch大小
    patch_w = 4  # 时间方向的patch大小

    # Token IDs
    track_marker_acc_id: int = 170  # acc 轨结束标记
    track_marker_mel_id: int = 171  # mel 轨结束标记
    beat_marker_id: int = 172  # 拍分隔符
    bar_token_id: int = 255
    eos_token_id: int = 256
    bos_token_id: int = 257
    pad_token_id: int = 258
    time_sig_offset_id: int = 259
    bpm_offset_id: int = 264
    train_cutoff_len = 2048  # 训练时的截断长度
    rope_theta: float = 10000.0  # RoPE base
    dropout = 0.1

    # 拍号 191: 3/4 192: 4/4
    # 和弦作为小节bar  190
