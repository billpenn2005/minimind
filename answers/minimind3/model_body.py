"""minimind3 MiniMindModel（Transformer 解码器主体，镜像 minimind 实现，手写）。

职责：
- Embedding：token id → 向量（后续与 lm_head 权重绑定）；
- 堆叠 N 个 MiniMindBlock；
- 末尾一个 RMSNorm（最终归一化）；
- 预计算并持有 RoPE 的 cos/sin 表（非持久 buffer，不进 checkpoint）；
- 上抛每个 block 的 KV-Cache（presents），供 generate 增量续写。
"""
import torch
from torch import nn

from .attention import Attention
from .block import MiniMindBlock
from .config import MiniMindConfig
from .feed_forward import FeedForward
from .rms_norm import RMSNorm
from .rope import precompute_freqs_cis


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE 表：shape [max_position_embeddings, head_dim]；persistent=False 不进 checkpoint
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape
        # 兼容 transformers 新版传入的 cache 对象（如 Cache 子类）
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 历史长度 -> 本次位置的 RoPE 切片起点
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # meta-device 初始化时 buffer 可能丢失，防御性重算（transformers>=5 兼容）
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length],
        )

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        # MoE 辅助损失汇总（非 MoE 时为 0；对齐标准实现的返回结构）
        aux_loss = sum(
            (l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, FeedForward) and hasattr(l.mlp, "aux_loss")),
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss