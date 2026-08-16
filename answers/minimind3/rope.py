"""minimind3 RoPE（旋转位置编码，镜像 minimind 实现，手写）。

对每个 head 的特征按"相邻两两一组"做旋转：频率随维度按几何级数递减，
使注意力内积只依赖相对位置 m-n，从而天然支持长文本外推。

实现约定（与 minimind 一致）：
- 预计算 cos/sin 表：形状 [end, dim]（两个相同半表拼接，适用于 head_dim）；
- 应用时：q_embed = q*cos + rotate_half(q)*sin（GPT-NeoX 风格）；
- 支持 YaRN 长上下文缩放（rope_scaling 非 None 时启用频率拉伸）。
"""
import math

import torch


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    """预计算位置频率的 cos/sin 表。

    频率：theta_i = base^(-2i/dim)，i = 0..dim/2-1（由外积张成 [end, dim/2]）。
    YaRN（rope_scaling 非 None）：低频段 ramp 到 base 频率，高频段缩放 1/factor。
    """
    # 半维度频率向量
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:  # YaRN: f'(i) = f(i) * ((1-gamma) + gamma/s)
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            # 哪个维度是被 beta 约束的频率（转为维度下标）
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))  # noqa: E731
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()          # [end, dim/2]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把后一半放到前一半并取负（对应两两一组旋转）。"""
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """对 q/k 应用旋转。cos/sin 形状 [seq, head_dim]（head_dim 维靠 unsqueeze 广播）。"""
    def embed(x, c, s):
        return ((x * c.unsqueeze(unsqueeze_dim)) + (rotate_half(x) * s.unsqueeze(unsqueeze_dim))).to(x.dtype)

    q_embed = embed(q, cos, sin)
    k_embed = embed(k, cos, sin)
    return q_embed, k_embed