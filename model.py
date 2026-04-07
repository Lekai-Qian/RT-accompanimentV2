import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedModel
from typing import Optional, List, Tuple


class PianoLLaMA(PreTrainedModel):
    """基于LLaMA架构的Piano生成模型（纯模型层，不含 I/O）"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = LlamaForCausalLM(self.config)
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.model.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False if labels is not None else True,
        )

    # ==================== 采样工具 ====================

    def _sample_token(
        self,
        logits,
        generated,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        generator: Optional[torch.Generator] = None,
    ):
        logits = logits / temperature

        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    def _generate_one_beat(
        self,
        generated,
        vocab,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        generator: Optional[torch.Generator] = None,
    ):
        """自回归生成一个 acc beat，直到遇到 track_marker_acc。"""
        end_markers = {vocab.track_marker_acc, vocab.bar_token_id, vocab.beat_marker}
        beat_tokens = []
        past_kv = None

        for _ in range(200):
            outputs = self.model(
                input_ids=generated[:, -1:] if past_kv else generated,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values

            next_token = self._sample_token(
                outputs.logits[:, -1, :], generated,
                temperature, top_k, top_p, repetition_penalty,
                generator=generator,
            )

            generated = torch.cat([generated, next_token], dim=1)
            token_id = next_token.item()
            beat_tokens.append(token_id)

            if token_id in end_markers:
                break

        return beat_tokens, generated

    def _generate_until_markers(
        self,
        generated,
        end_markers,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        generator: Optional[torch.Generator] = None,
        max_steps: int = 200,
    ):
        """Autoregressively generate tokens until one of the end markers is produced."""
        tokens = []
        past_kv = None

        for _ in range(max_steps):
            outputs = self.model(
                input_ids=generated[:, -1:] if past_kv else generated,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values

            next_token = self._sample_token(
                outputs.logits[:, -1, :],
                generated,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                generator=generator,
            )

            generated = torch.cat([generated, next_token], dim=1)
            token_id = next_token.item()
            tokens.append(token_id)

            if token_id in end_markers:
                break

        return tokens, generated

    # ==================== 主生成方法 ====================

    @torch.no_grad()
    def generate_accompaniment(
        self,
        initial_tokens: torch.Tensor,
        schedule: list,
        vocab,
        device: str = 'cuda',
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        verbose: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """
        执行 schedule 生成伴奏。纯模型推理，不含 I/O。

        Args:
            initial_tokens: (seq_len,) tensor — [BOS, TS, BPM]
            schedule: List[GenerationStep] — tokenizer 预计算的生成计划
            vocab: Vocabulary 对象
            device / temperature / top_k / top_p / repetition_penalty: 采样参数

        Returns:
            (acc_beats, generated_sequence):
                acc_beats: List[List[int]] — 每拍生成/注入的 acc tokens
                generated_sequence: (1, total_len) tensor — 完整序列
        """
        self.eval()
        generated = initial_tokens.unsqueeze(0).to(device)
        acc_beats = []

        if verbose:
            n_gen = sum(1 for s in schedule if s.action == "generate")
            n_gt = sum(1 for s in schedule if s.action == "inject_gt")
            print(f"生成计划: {len(schedule)} 步, {n_gen} 步自回归, {n_gt} 步GT")

        for step in schedule:
            if step.action in ("inject", "inject_gt"):
                generated = torch.cat(
                    [generated, step.data.unsqueeze(0).to(device)], dim=1)
                if step.action == "inject_gt":
                    acc_beats.append(step.data.cpu().tolist())

            elif step.action == "generate":
                beat_tokens, generated = self._generate_one_beat(
                    generated, vocab,
                    temperature, top_k, top_p, repetition_penalty,
                    generator=generator,
                )
                acc_beats.append(beat_tokens)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return acc_beats, generated.cpu()

    @torch.no_grad()
    def generate_by_schedule(
        self,
        initial_tokens: torch.Tensor,
        schedule: list,
        vocab,
        device: str = 'cuda',
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        verbose: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[List[List[int]], List[List[int]], torch.Tensor]:
        """
        Generic schedule runner for experimental inference.

        Supported actions:
          - inject / inject_gt
          - generate_acc
          - generate_mel
        """
        self.eval()
        generated = initial_tokens.unsqueeze(0).to(device)
        acc_beats: List[List[int]] = []
        mel_beats: List[List[int]] = []

        if verbose:
            counts = {}
            for step in schedule:
                counts[step.action] = counts.get(step.action, 0) + 1
            print(f"实验生成计划: {counts}")

        for step in schedule:
            if step.action in ("inject", "inject_gt"):
                generated = torch.cat([generated, step.data.unsqueeze(0).to(device)], dim=1)

                data_list = step.data.cpu().tolist()
                if len(data_list) > 0:
                    last_tok = data_list[-1]
                    if last_tok == vocab.track_marker_acc:
                        acc_beats.append(data_list)
                    elif last_tok == vocab.track_marker_mel:
                        mel_beats.append(data_list)

            elif step.action == "generate_acc":
                beat_tokens, generated = self._generate_until_markers(
                    generated=generated,
                    end_markers={vocab.track_marker_acc, vocab.bar_token_id, vocab.beat_marker},
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generator=generator,
                )
                acc_beats.append(beat_tokens)

            elif step.action == "generate_mel":
                beat_tokens, generated = self._generate_until_markers(
                    generated=generated,
                    end_markers={vocab.track_marker_mel, vocab.bar_token_id, vocab.beat_marker},
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generator=generator,
                )
                mel_beats.append(beat_tokens)

            else:
                raise ValueError(f"Unsupported schedule action: {step.action}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return acc_beats, mel_beats, generated.cpu()
