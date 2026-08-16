"""minimind3 Attention（GQA 多头注意力，镜像 minimind 实现，手写）。

特性：
- GQA：num_key_value_heads < num_attention_heads，KV 头共享（省显存/内存）；
- QK-Norm：对 q/k 的每头向量做 RMSNorm（稳定训练）；
- RoPE 注入位置信息（表由外部传入）；
- KV-Cache：生成时增量计算，无需重算历史；
- 因果掩码 + 可选 attention_mask（padding）；
- 快路径：torch.nn.functional.scaled_dot_product_attention（SDPA）。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import MiniMindConfig
from .rms_norm import RMSNorm
from .rope import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头复制 n_rep 份以对齐 query 头数（GQA 的核心展开）。"""
    bsz, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bsz, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm：每个 head 的向量归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # SDPA 快路径开关（CPU/GPU 均可用，autocast 下自动融合）
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple,
        past_key_value: tuple | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        bsz, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 拆分到 (batch, seq, head, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 拼接历史 KV（生成时增量）
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 展开 KV 头并转置为 (batch, head, seq, head_dim)
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 快路径条件：seq_len>1、无 past 干扰、无掩码偏移
        if (
            self.flash
            and seq_len > 1
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                # 对当前步新增的 seq_len 列做上三角 -inf（保留历史的因果性）
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv