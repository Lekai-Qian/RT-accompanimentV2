"""
PianoMusicTokenizer - 钢琴音乐完整编解码系统

三层架构：
  Layer 1 (PatchCodec):       Piano roll <-> patch token matrix (三进制编码)
  Layer 2 (Beat encoding):    Measure (4,88,t) -> 逐拍压缩 token 列表
  Layer 3 (SequenceBuilder):  Measures + metadata -> 训练/推理就绪的 token 序列

序列格式 (beat-marker):
  [BOS][TS][BPM] [bar][beat][acc...track_acc][mel...track_mel] [beat]... [bar]... [EOS]
"""

import numpy as np
import torch
from typing import Optional, Union, Tuple, List, Dict
from dataclasses import dataclass


# ============================================================================
#  Vocabulary — 所有 token ID 的唯一定义处
# ============================================================================


@dataclass(frozen=True)
class Vocabulary:
    """
    集中管理所有 token ID，消除 magic number。
    frozen=True 防止运行时意外修改。
    """

    # --- Patch codec 标记 ---
    empty_marker: int = 169
    track_marker_acc: int = 170  # acc 轨结束标记
    track_marker_mel: int = 171  # mel 轨结束标记
    beat_marker: int = 172  # 拍分隔符

    # --- Codec 参数 ---
    marker_offset: int = 81
    measures_length: int = 88
    img_h: int = 88

    # --- 序列级 token ---
    bar_token_id: int = 255
    eos_token_id: int = 256
    bos_token_id: int = 257
    pad_token_id: int = 258
    time_sig_offset_id: int = 259
    bpm_offset_id: int = 264  # = 259 + 5

    # --- BPM 分桶阈值 ---
    bpm_slow_threshold: int = 90
    bpm_fast_threshold: int = 200

    # --- Patch 默认尺寸 ---
    default_patch_h: int = 1
    default_patch_w: int = 4

    @classmethod
    def from_config(cls, config) -> "Vocabulary":
        """从 ModelConfig 创建 Vocabulary，用 config 值覆盖默认值。"""
        return cls(
            track_marker_acc=config.track_marker_acc_id,
            track_marker_mel=config.track_marker_mel_id,
            beat_marker=config.beat_marker_id,
            bar_token_id=config.bar_token_id,
            eos_token_id=config.eos_token_id,
            bos_token_id=config.bos_token_id,
            pad_token_id=config.pad_token_id,
            time_sig_offset_id=config.time_sig_offset_id,
            bpm_offset_id=config.bpm_offset_id,
            marker_offset=config.markoffset,
            measures_length=config.measures_length,
            default_patch_h=config.patch_h,
            default_patch_w=config.patch_w,
        )


@dataclass
class GenerationStep:
    """生成计划中的单个步骤。"""

    action: str  # "inject" = 注入已知tokens, "inject_gt" = 注入GT, "generate" = 自回归生成
    data: Optional[torch.Tensor] = None


# ============================================================================
#  PatchCodec — Layer 1: piano roll <-> patch token matrix
# ============================================================================


class PatchCodec:
    """
    底层 patch 编解码器。
    处理 (sustain, onset) 双通道 piano roll 与三进制 patch token 矩阵之间的转换。
    """

    def __init__(
        self,
        patch_h: int = 1,
        patch_w: int = 4,
        img_h: int = 88,
    ):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.img_h = img_h

        self.patch_size = patch_h * patch_w
        self.powers_3 = 3 ** np.arange(self.patch_size - 1, -1, -1)

        # strict 模式的特殊 token 替换规则
        self.special_token_ids = [
            13,
            12,
            59,
            31,
            64,
            11,
            55,
            73,
            37,
            30,
            28,
            5,
            15,
            46,
            16,
            17,
            10,
            14,
            32,
            19,
            3,
            9,
            1,
            57,
            4,
        ]
        self.replacement_ids = [0, 67, 7, 40, 63]

    # ---- 编码 ----

    def image_to_patch_tokens(
        self,
        image: Union[np.ndarray, torch.Tensor],
        strict_mode: bool = True,
    ) -> np.ndarray:
        """
        双通道 piano roll → patch token 矩阵（三进制编码）。

        Args:
            image: (2, 88, t) — ch0: sustain, ch1: onset
            strict_mode: 是否替换特殊 token

        Returns:
            tokens: (num_time_patches, num_pitch_patches)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

        sustain = image[0].copy()
        onset = image[1].copy()
        onset[sustain == 0] = 0

        img_h, img_w = sustain.shape

        padding_w = (self.patch_w - img_w % self.patch_w) % self.patch_w
        if padding_w > 0:
            sustain = np.pad(sustain, ((0, 0), (0, padding_w)), constant_values=0)
            onset = np.pad(onset, ((0, 0), (0, padding_w)), constant_values=0)
            img_w = sustain.shape[1]

        n_rows = img_h // self.patch_h
        n_cols = img_w // self.patch_w

        sustain_p = self._reshape_to_patches(sustain, n_rows, n_cols)
        onset_p = self._reshape_to_patches(onset, n_rows, n_cols)

        combined = sustain_p.astype(np.int64) + onset_p.astype(np.int64)
        tokens = np.dot(combined, self.powers_3)

        if strict_mode:
            tokens = self._replace_special_tokens(tokens)

        return tokens

    def _reshape_to_patches(self, ch: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
        p = ch.reshape(n_rows, self.patch_h, n_cols, self.patch_w)
        p = p.transpose(2, 0, 1, 3)
        return p.reshape(n_cols, n_rows, self.patch_size)

    def _replace_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
        mask = np.isin(tokens, self.special_token_ids)
        if np.any(mask):
            tokens = tokens.copy()
            tokens[mask] = np.random.choice(self.replacement_ids, size=int(np.sum(mask)))
        return tokens

    # ---- 解码 ----

    def patch_tokens_to_image(self, tokens: np.ndarray) -> np.ndarray:
        """
        token 矩阵 → 双通道 piano roll (2, 88, t)。
        """
        n_cols, n_rows = tokens.shape

        combined = np.zeros((n_cols, n_rows, self.patch_size), dtype=np.int64)
        tmp = tokens.copy()
        for i in range(self.patch_size):
            combined[:, :, i] = tmp // self.powers_3[i]
            tmp = tmp % self.powers_3[i]

        sustain_p = (combined >= 1).astype(np.float32)
        onset_p = (combined == 2).astype(np.float32)

        sustain_ch = self._patches_to_channel(sustain_p, n_cols, n_rows)
        onset_ch = self._patches_to_channel(onset_p, n_cols, n_rows)

        return np.stack([sustain_ch, onset_ch], axis=0)

    def _patches_to_channel(self, patches: np.ndarray, n_cols: int, n_rows: int) -> np.ndarray:
        p = patches.reshape(n_cols, n_rows, self.patch_h, self.patch_w)
        p = p.transpose(1, 2, 0, 3)
        return p.reshape(self.img_h, n_cols * self.patch_w)


# ============================================================================
#  PianoMusicTokenizer — 完整的音乐 tokenizer
# ============================================================================


class PianoMusicTokenizer:
    """
    钢琴音乐的完整 tokenizer。

    组合 PatchCodec + Vocabulary，提供从原始 piano roll 到训练就绪
    token 序列的全部编解码功能。这是所有 tokenization 操作的唯一入口。

    序列格式:
      [BOS][TS][BPM] [bar][beat][acc...track_acc][mel...track_mel] [beat]... [bar]... [EOS]
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        config=None,
    ):
        if vocab is None and config is not None:
            self.vocab = Vocabulary.from_config(config)
        elif vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary()

        self._codec = PatchCodec(
            patch_h=self.vocab.default_patch_h,
            patch_w=self.vocab.default_patch_w,
            img_h=self.vocab.img_h,
        )

    # ===================== Layer 1: 压缩 / 解压 =====================

    def compress_tokens(
        self,
        token_matrix: np.ndarray,
        track_marker: int,
        pos_shift: int = 0,
    ) -> np.ndarray:
        """
        使用相对位置编码压缩 token 矩阵。

        Args:
            token_matrix: (num_measures, measures_length) 的 token 矩阵
            track_marker: 轨道结束标记 (track_marker_acc 或 track_marker_mel)
            pos_shift: 位置偏移量。将每拍的起始 prev 设为 -pos_shift，
                       使第一个位置标记变为 idx + pos_shift，
                       从而让 BEAT 不再是严格的"位置0"锚点。

        Returns:
            compressed: 压缩后的一维序列
        """
        v = self.vocab

        compressed_seqs = []
        for row in token_matrix:
            nz = np.where(row != 0)[0]
            if len(nz) == 0:
                compressed_seqs.append(np.array([v.empty_marker, track_marker], dtype=np.int64))
            else:
                parts = []
                prev = -pos_shift  # 位置偏移：第一个相对距离 = idx - (-shift) = idx + shift
                for idx in nz:
                    parts.extend([v.marker_offset + (idx - prev), row[idx]])
                    prev = idx
                parts.append(track_marker)
                compressed_seqs.append(np.array(parts, dtype=np.int64))

        return np.concatenate(compressed_seqs)

    def decompress_tokens(
        self,
        compressed: Union[np.ndarray, list],
        track_marker_id: int,
    ) -> np.ndarray:
        """
        解压缩 token 序列 → token 矩阵。

        Args:
            compressed: 压缩 token 序列
            track_marker_id: 轨道结束标记 ID

        Returns:
            (num_measures, measures_length)
        """
        v = self.vocab
        if isinstance(compressed, list):
            compressed = np.array(compressed, dtype=np.int64)

        measures = []
        i = 0

        while i < len(compressed):
            row = np.zeros(v.measures_length, dtype=np.int64)
            abs_pos = 0
            while i < len(compressed):
                tok = compressed[i]
                i += 1
                if tok == track_marker_id:
                    break
                if i >= len(compressed):
                    break
                if tok == v.empty_marker:
                    i += 1
                    continue
                val = compressed[i]
                i += 1
                abs_pos += tok - v.marker_offset
                if 0 <= abs_pos < v.measures_length:
                    row[abs_pos] = val
            measures.append(row)

        return np.stack(measures, axis=0)

    # ===================== Layer 2: Beat 级编码 =====================

    def encode_measure(
        self,
        measure: np.ndarray,
        timesteps_per_beat: int = 4,
        pos_shift: int = 0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        将 4 通道 measure 编码为逐拍 (acc, mel) 压缩 token 对。

        Args:
            measure: (4, 88, t) — ch0:2 = mel(part0), ch2:4 = acc(part1)
            timesteps_per_beat: 每拍的时间步数
            pos_shift: 位置偏移量，传递给 compress_tokens

        Returns:
            beats: List[(acc_tensor, mel_tensor)] — 每元素对应一拍
        """
        v = self.vocab
        t = measure.shape[2]
        beat_len = timesteps_per_beat
        num_beats = (t + beat_len - 1) // beat_len

        beats = []

        for b in range(num_beats):
            s = b * beat_len
            e = min(s + beat_len, t)
            beat = measure[:, :, s:e]

            if e - s < beat_len:
                pad_w = beat_len - (e - s)
                beat = np.pad(beat, ((0, 0), (0, 0), (0, pad_w)), constant_values=0)

            # mel: channels 0-1
            tokens_mel = self._codec.image_to_patch_tokens(beat[:2], strict_mode=True)
            comp_mel = self.compress_tokens(tokens_mel, track_marker=v.track_marker_mel, pos_shift=pos_shift)

            # acc: channels 2-3
            tokens_acc = self._codec.image_to_patch_tokens(beat[2:], strict_mode=True)
            comp_acc = self.compress_tokens(tokens_acc, track_marker=v.track_marker_acc, pos_shift=pos_shift)

            beats.append(
                (
                    torch.tensor(comp_acc, dtype=torch.long),
                    torch.tensor(comp_mel, dtype=torch.long),
                )
            )

        return beats

    def encode_bpm(self, bpm) -> int:
        """BPM 值 → token ID（已包含 offset）。"""
        v = self.vocab
        if bpm is None:
            bucket = 3  # UNK
        else:
            bpm_int = int(bpm)
            if bpm_int < v.bpm_slow_threshold:
                bucket = 0
            elif bpm_int <= v.bpm_fast_threshold:
                bucket = 1
            else:
                bucket = 2
        return bucket + v.bpm_offset_id

    def encode_time_sig(self, time_sig_idx: int) -> int:
        """拍号索引 → token ID（已包含 offset，含 2/2 拍特殊映射）。"""
        if time_sig_idx == 9:
            time_sig_idx = 4
        return time_sig_idx + self.vocab.time_sig_offset_id

    # ===================== 内部: 编码所有小节 =====================

    def _encode_measures(
        self,
        measures: List[np.ndarray],
        metadata: dict,
        timesteps_per_beat: int = 4,
        pitch_shift: int = 0,
        pos_shift: int = 0,
    ) -> Tuple[int, int, bool, List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        编码所有小节，返回按小节分组的 (acc, mel) 拍对。

        Returns:
            (ts_token, bpm_token, is_continuation, measure_beats)
            measure_beats: List[List[(acc_tensor, mel_tensor)]]
                第一层按小节分组，第二层按拍分组
        """
        ts_token = self.encode_time_sig(metadata["time_signature_idx"])
        bpm_token = self.encode_bpm(metadata["bpm"])
        is_continuation = metadata.get("is_continuation", False)

        measure_beats = []
        for measure in measures:
            if pitch_shift != 0:
                measure = np.roll(measure, pitch_shift, axis=1)
                if pitch_shift > 0:
                    measure[:, :pitch_shift, :] = 0
                else:
                    measure[:, pitch_shift:, :] = 0

            beats = self.encode_measure(measure, timesteps_per_beat, pos_shift=pos_shift)
            measure_beats.append(beats)

        return ts_token, bpm_token, is_continuation, measure_beats

    # ===================== Layer 3: 完整序列构建 =====================

    def build_training_sequence(
        self,
        measures: List[np.ndarray],
        metadata: dict,
        add_bos: bool = True,
        timesteps_per_beat: int = 4,
        pitch_shift: int = 0,
        acc_drop_prob: float = 0.0,
        pos_shift_max: int = 0,
        drop_initial_beats: int = 0,
        drop_initial_beats_prob: float = 1.0,
        include_melody_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建完整的训练序列 (input_ids, labels)。

        序列格式:
          [BOS][TS][BPM] [bar][beat][acc...track_acc][mel...track_mel] [beat]... [bar]... [EOS]

        Labels: bar/beat_marker 部分用 pad_token_id 填充（不参与 loss），
                acc 部分作为预测目标。
                当 include_melody_loss=True 时，mel 部分也作为预测目标。
                当 acc_drop_prob > 0 时，随机将某些 beat 的 acc 替换为空，
                这些 beat 的 acc 也不参与 loss（强迫模型减少对 acc history 的依赖）。
                当 drop_initial_beats > 0 时，drop 每首曲子前 N 个 beat 的 acc（冷启动），
                由 drop_initial_beats_prob 控制是否应用（1.0=总是，0.5=50%概率）。
                当 pos_shift_max > 0 时，对每首曲子随机采样 Δ ∈ {0,...,pos_shift_max}，
                将该曲所有 beat 的 acc 和 mel 位置标记整体偏移 Δ，
                使 BEAT token 不再是严格的"位置0"锚点。

        Returns:
            (input_ids, labels): 均为 1D torch.long tensor。
        """
        v = self.vocab

        # 位置偏移增强：每首曲子固定一个 Δ（固定 per-song）
        pos_shift = int(np.random.randint(0, pos_shift_max + 1)) if pos_shift_max > 0 else 0

        ts_token, bpm_token, is_continuation, measure_beats = self._encode_measures(
            measures, metadata, timesteps_per_beat, pitch_shift, pos_shift=pos_shift
        )

        # Drop initial beats: 决定是否 drop 前 N 个 beat（per-song）
        should_drop_initial = False
        if drop_initial_beats > 0 and drop_initial_beats_prob > 0.0:
            should_drop_initial = np.random.random() < drop_initial_beats_prob

        empty_acc = torch.tensor([v.empty_marker, v.track_marker_acc], dtype=torch.long)

        inp_parts = []
        lbl_parts = []
        global_beat_idx = 0

        for beats in measure_beats:
            # [bar]
            bar = torch.tensor([v.bar_token_id], dtype=torch.long)
            inp_parts.append(bar)
            lbl_parts.append(torch.full_like(bar, v.pad_token_id))

            for acc, mel in beats:
                # [beat_marker] — 结构标记，不参与 loss
                bm = torch.tensor([v.beat_marker], dtype=torch.long)
                inp_parts.append(bm)
                lbl_parts.append(torch.full_like(bm, v.pad_token_id))

                # 判断是否 drop 当前 beat 的 acc
                drop_this_beat = False
                if should_drop_initial and global_beat_idx < drop_initial_beats:
                    drop_this_beat = True
                elif acc_drop_prob > 0.0 and np.random.random() < acc_drop_prob:
                    drop_this_beat = True

                if drop_this_beat:
                    inp_parts.append(empty_acc)
                    lbl_parts.append(torch.full_like(empty_acc, v.pad_token_id))
                else:
                    # [acc tokens] — 预测目标
                    inp_parts.append(acc)
                    lbl_parts.append(acc)

                # [mel tokens] — 条件输入；可选加入 loss
                inp_parts.append(mel)
                if include_melody_loss:
                    lbl_parts.append(mel)
                else:
                    lbl_parts.append(torch.full_like(mel, v.pad_token_id))

                global_beat_idx += 1

        measure_inp = torch.cat(inp_parts)
        measure_lbl = torch.cat(lbl_parts)

        # [BOS] + TS + BPM + content + [EOS]
        seq_inp = []
        seq_lbl = []

        if add_bos:
            bos = torch.tensor([v.bos_token_id], dtype=torch.long)
            seq_inp.append(bos)
            seq_lbl.append(torch.full_like(bos, v.pad_token_id))

        ts_t = torch.tensor([ts_token], dtype=torch.long)
        bp_t = torch.tensor([bpm_token], dtype=torch.long)
        seq_inp.extend([ts_t, bp_t])
        seq_lbl.extend(
            [
                torch.full_like(ts_t, v.pad_token_id),
                torch.full_like(bp_t, v.pad_token_id),
            ]
        )

        seq_inp.append(measure_inp)
        seq_lbl.append(measure_lbl)

        if not is_continuation:
            eos = torch.tensor([v.eos_token_id], dtype=torch.long)
            seq_inp.append(eos)
            seq_lbl.append(eos)

        return torch.cat(seq_inp), torch.cat(seq_lbl)

    def build_generation_schedule(
        self,
        measures: List[np.ndarray],
        metadata: dict,
        gt_prefix_beats: int = 0,
        timesteps_per_beat: int = 4,
    ) -> Dict:
        """
        构建推理生成计划。调度与执行分离：tokenizer 决定"做什么"，model 决定"怎么做"。

        Schedule 结构 (每小节):
          inject[bar] → { inject[beat] → generate/inject_gt acc → inject mel } × num_beats

        Returns:
            dict:
                'initial_tokens': [BOS, TS, BPM] tensor
                'schedule': List[GenerationStep]
                'mel_beats': List[torch.Tensor] — 所有 mel 拍数据（用于解码）
                'acc_beats_gt': List[torch.Tensor] — 所有 acc 拍 GT（用于参考）
        """
        v = self.vocab
        ts_token, bpm_token, _, measure_beats = self._encode_measures(measures, metadata, timesteps_per_beat)

        initial = torch.tensor([v.bos_token_id, ts_token, bpm_token], dtype=torch.long)

        steps = []
        mel_beats_all = []
        acc_beats_gt_all = []
        beat_idx = 0  # 全局拍计数，用于 GT 前缀判断

        for beats in measure_beats:
            # inject [bar]
            steps.append(GenerationStep("inject", torch.tensor([v.bar_token_id], dtype=torch.long)))

            for acc, mel in beats:
                # inject [beat_marker]
                steps.append(GenerationStep("inject", torch.tensor([v.beat_marker], dtype=torch.long)))

                # acc: GT 前缀 → inject_gt / 其他 → 自回归生成
                if beat_idx < gt_prefix_beats:
                    steps.append(GenerationStep("inject_gt", acc))
                else:
                    steps.append(GenerationStep("generate"))

                # mel: 始终注入
                steps.append(GenerationStep("inject", mel))

                mel_beats_all.append(mel)
                acc_beats_gt_all.append(acc)
                beat_idx += 1

        return {
            "initial_tokens": initial,
            "schedule": steps,
            "mel_beats": mel_beats_all,
            "acc_beats_gt": acc_beats_gt_all,
        }

    # ===================== 解码 =====================

    def decode_beats_to_pianoroll(
        self,
        beats_list: list,
        track_marker_id: int,
    ) -> np.ndarray:
        """
        beat token 列表 → piano roll (2, 88, t)。

        Args:
            beats_list: beat token 的列表（tensor / list / int 混合均可）
            track_marker_id: 轨道结束标记 ID（track_marker_acc 或 track_marker_mel）

        Returns:
            pianoroll: (2, 88, t)
        """
        v = self.vocab

        beat_images = []
        beat_width = getattr(v, "default_patch_w", 4)

        for beat in beats_list:
            if isinstance(beat, torch.Tensor):
                beat_tokens = beat.cpu().tolist()
            elif isinstance(beat, np.ndarray):
                beat_tokens = beat.tolist()
            elif isinstance(beat, list):
                beat_tokens = beat
            else:
                beat_tokens = [beat]

            # 只保留当前 beat 内的 patch / position / empty / track-end token。
            filtered = np.array([t for t in beat_tokens if t < v.beat_marker], dtype=np.int64)

            if len(filtered) == 0:
                beat_img = np.zeros((2, v.img_h, beat_width), dtype=np.float32)
            else:
                mat = self.decompress_tokens(filtered, track_marker_id=track_marker_id)
                beat_img = self._codec.patch_tokens_to_image(mat)
                if beat_img.shape[2] < beat_width:
                    beat_img = np.pad(
                        beat_img,
                        ((0, 0), (0, 0), (0, beat_width - beat_img.shape[2])),
                        mode="constant",
                    )
                elif beat_img.shape[2] > beat_width:
                    beat_img = beat_img[:, :, :beat_width]

            beat_images.append(beat_img)

        if not beat_images:
            return np.zeros((2, v.img_h, 0), dtype=np.float32)

        return np.concatenate(beat_images, axis=2)

    # ===================== 工具方法 =====================

    def estimate_sequence_length(
        self,
        measures: List[np.ndarray],
        timesteps_per_beat: int = 4,
    ) -> int:
        """
        估算 token 序列长度（用于长度缓存预计算）。

        格式: [BOS][TS][BPM] + {[bar] + num_beats × ([beat] + acc + mel)} × num_measures + [EOS]
        """
        total = 4  # BOS + time_sig + BPM + EOS

        for measure in measures:
            beats = self.encode_measure(measure, timesteps_per_beat)
            total += 1  # bar token
            for acc, mel in beats:
                total += 1  # beat_marker
                total += len(acc)
                total += len(mel)

        return total

    def get_config(self) -> dict:
        """返回配置字典。"""
        v = self.vocab
        return {
            "patch_h": v.default_patch_h,
            "patch_w": v.default_patch_w,
            "marker_offset": v.marker_offset,
            "measures_length": v.measures_length,
            "track_marker_acc": v.track_marker_acc,
            "track_marker_mel": v.track_marker_mel,
            "beat_marker": v.beat_marker,
            "empty_marker": v.empty_marker,
            "img_h": v.img_h,
        }

    def __repr__(self) -> str:
        v = self.vocab
        return (
            f"PianoMusicTokenizer("
            f"patch={v.default_patch_h}x{v.default_patch_w}, "
            f"img_h={v.img_h}, "
            f"marker_offset={v.marker_offset})"
        )


# ============================================================================
#  向后兼容别名
# ============================================================================

PianoRollTokenizer = PatchCodec


def image_to_patch_tokens_vectorized_strict(image, H=2, W=4):
    codec = PatchCodec(patch_h=H, patch_w=W)
    return codec.image_to_patch_tokens(image, strict_mode=True)


def patch_tokens_to_image_vectorized(tokens, H=2, W=4, img_h=88):
    codec = PatchCodec(patch_h=H, patch_w=W, img_h=img_h)
    return codec.patch_tokens_to_image(tokens)
