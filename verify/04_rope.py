"""第 04 课验收：RoPE 预计算与旋转应用。"""
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from verify._common import set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@register("04.1 频率表形状与公式数值")
def check_freqs(args):
    from minimind3.rope import precompute_freqs_cis

    dim, end, base = 24, 64, 10000.0
    cos, sin = precompute_freqs_cis(dim=dim, end=end, rope_base=base)
    assert cos.shape == sin.shape == (end, dim)
    # 半表公式：cos[t, i] == cos(t * base^(-2i/dim))，i < dim/2
    t, i = 17, 3
    theta = base ** (-2 * i / dim)
    assert torch.allclose(cos[t, i], torch.tensor(math.cos(t * theta)), atol=1e-6)
    # 表的后半与前半相同（约定与 minimind 一致）
    assert torch.allclose(cos[:, dim // 2:], cos[:, : dim // 2], atol=1e-6)


@register("04.2 旋转保持模长")
def check_norm(args):
    from minimind3.rope import apply_rotary_pos_emb, precompute_freqs_cis

    set_seed(0)
    cos, sin = precompute_freqs_cis(dim=8, end=32, rope_base=10000.0)
    q = torch.randn(2, 3, 4, 8)  # (B, S, H, D)
    qe, _ = apply_rotary_pos_emb(q, q, cos[:3], sin[:3])
    assert torch.allclose(qe.norm(dim=-1), q.norm(dim=-1), atol=1e-5), "旋转不应改变模长"


@register("04.3 旋转等价于公式 q*cos + rot(q)*sin")
def check_formula(args):
    from minimind3.rope import apply_rotary_pos_emb, precompute_freqs_cis, rotate_half

    set_seed(1)
    cos, sin = precompute_freqs_cis(dim=8, end=8, rope_base=10000.0)
    q = torch.randn(1, 1, 2, 8)
    c, s = cos[:1].unsqueeze(1), sin[:1].unsqueeze(1)  # (1,1,1,8)
    qe, _ = apply_rotary_pos_emb(q, q, cos[:1], sin[:1])
    ref = q * c + rotate_half(q) * s
    assert torch.allclose(qe, ref, atol=1e-6)


@register("04.4 点积的相对位置不变性（核心性质）")
def check_relative(args):
    """同一 query 相对 key 的距离不变时 dot 不变：
    dot(R(q, m), R(k, n)) == dot(R(q, m+Δ), R(k, n+Δ))
    """
    from minimind3.rope import apply_rotary_pos_emb, precompute_freqs_cis

    set_seed(2)
    cos, sin = precompute_freqs_cis(dim=16, end=64, rope_base=10000.0)
    q = torch.randn(1, 1, 4, 16)
    k = torch.randn(1, 1, 4, 16)

    def dot_at(qk, m, n):
        dist = n - m
        qe, ke = apply_rotary_pos_emb(q, k, cos[m:m + 1], sin[m:m + 1])
        # 把 key 用 (m, dist) 结算：R(k, dist) 需与 query 同表
        q0, k0 = apply_rotary_pos_emb(q, k, cos[m:m + 1], sin[m:m + 1])
        q1, k1 = apply_rotary_pos_emb(q, k, cos[m + dist:m + dist + 1], sin[m + dist:m + dist + 1])
        # q at m vs k at m+dist
        d1 = (q0 * k1).sum(-1)
        d2 = (q1 * k0).sum(-1)
        return d1, d2

    # 平移 Δ=5 后点积不变
    m, n, delta = 3, 19, 5
    d1, d2 = dot_at(0, m, n)
    d1p, d2p = dot_at(0, m + delta, n + delta)
    assert torch.allclose(d1, d1p, atol=1e-5), "相对位置不变性被破坏"
    assert torch.allclose(d2, d2p, atol=1e-5)


@register("04.5 dtype 保持")
def check_dtype(args):
    from minimind3.rope import apply_rotary_pos_emb, precompute_freqs_cis

    cos, sin = precompute_freqs_cis(dim=8, end=8, rope_base=10000.0)
    q = torch.randn(1, 1, 2, 8, dtype=torch.float16)
    qe, ke = apply_rotary_pos_emb(q, q, cos[:1], sin[:1])
    assert qe.dtype == torch.float16


@register("04.6 YaRN 缩放生效")
def check_yarn(args):
    from minimind3.rope import precompute_freqs_cis

    dim = 32
    cos_plain, _ = precompute_freqs_cis(dim=dim, end=4096, rope_base=10000.0)
    yarn = {
        "beta_fast": 32, "beta_slow": 1, "factor": 16,
        "original_max_position_embeddings": 2048, "attention_factor": 1.0, "type": "yarn",
    }
    cos_yarn, _ = precompute_freqs_cis(dim=dim, end=4096, rope_base=10000.0, rope_scaling=yarn)
    assert cos_yarn.shape == cos_plain.shape and not torch.isnan(cos_yarn).any()
    # 高位频率被压低（缩放 1/factor），低频与 base 接近：整体应存在显著差异
    assert not torch.allclose(cos_yarn, cos_plain, atol=1e-3), "YaRN 未改变频率"