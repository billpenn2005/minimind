"""第 05 课验收：Attention（GQA、因果、KV-Cache、SDPA/手写等价）。"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minimind3.attention import Attention, repeat_kv  # noqa: E402
from minimind3.config import MiniMindConfig  # noqa: E402
from minimind3.rope import precompute_freqs_cis  # noqa: E402
from verify._common import TINY_CONFIG, set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _make(config_dict: dict, flash: bool = True) -> Attention:
    cfg = MiniMindConfig(**config_dict, flash_attn=flash)
    return Attention(cfg)


@register("05.1 投影形状与 GQA 头扩展")
def check_shapes(args):
    cfg = MiniMindConfig(**TINY_CONFIG)
    attn = Attention(cfg)
    assert attn.q_proj.weight.shape == (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    assert attn.k_proj.weight.shape == (kv_dim, cfg.hidden_size) != attn.q_proj.weight.shape
    assert attn.v_proj.weight.shape == (kv_dim, cfg.hidden_size)
    assert attn.o_proj.weight.shape == (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)

    # repeat_kv 行为
    x = torch.randn(2, 5, 2, 24)
    out = repeat_kv(x, 2)
    assert out.shape == (2, 5, 4, 24)
    assert torch.allclose(out[:, :, 0], x[:, :, 0]) and torch.allclose(out[:, :, 1], x[:, :, 1])
    assert torch.allclose(out[:, :, 2], x[:, :, 0]) and torch.allclose(out[:, :, 3], x[:, :, 1])
    assert repeat_kv(x, 1) is x


@register("05.2 因果性：位置 i 不依赖 j>i")
def check_causal(args):
    cfg = MiniMindConfig(**TINY_CONFIG)
    attn = Attention(cfg)
    cos, sin = precompute_freqs_cis(cfg.head_dim, end=cfg.max_position_embeddings, rope_base=cfg.rope_theta)
    x = torch.randn(1, 6, cfg.hidden_size)
    # 前向在图内：取第 0 个位置的输出
    out, _ = attn(x, (cos[:6], sin[:6]))
    target = out[0, 0].sum()
    grads = torch.autograd.grad(target, x, retain_graph=True)[0][0]  # 每个输入位置对输出的梯度
    assert torch.allclose(grads[1:], torch.zeros_like(grads[1:]), atol=1e-6), "位置0看到了未来位置"


@register("05.3 网络输出与手写参考一致（无 flash）")
def check_reference(args):
    """关闭 flash 走手写分支，并与纯 NumPy/Torch 参考实现逐元素对比。"""
    from minimind3.rms_norm import RMSNorm

    set_seed(0)
    cfg = MiniMindConfig(**TINY_CONFIG)
    attn = Attention(cfg)  # 用标准 config；flash 默认 True -> 手写参考不可比，这里强制走手写
    attn.flash = False
    cos, sin = precompute_freqs_cis(cfg.head_dim, end=512, rope_base=cfg.rope_theta)
    x = torch.randn(2, 5, cfg.hidden_size)
    out, _ = attn(x, (cos[:5], sin[:5]))

    # 手写参考：全量点积注意力
    with torch.no_grad():
        xq = attn.q_norm(attn.q_proj(x)).view(2, 5, cfg.num_attention_heads, cfg.head_dim)
        xk = attn.k_norm(attn.k_proj(x)).view(2, 5, cfg.num_key_value_heads, cfg.head_dim)
        xv = attn.v_proj(x).view(2, 5, cfg.num_key_value_heads, cfg.head_dim)

    # 用公式直接旋转
    def rotate_half(t):
        return torch.cat((-t[..., t.shape[-1] // 2:], t[..., : t.shape[-1] // 2]), -1)
    qr = xq * cos[:5].unsqueeze(1) + rotate_half(xq) * sin[:5].unsqueeze(1)
    kr = xk * cos[:5].unsqueeze(1) + rotate_half(xk) * sin[:5].unsqueeze(1)
    kr = repeat_kv(kr, cfg.num_attention_heads // cfg.num_key_value_heads).transpose(1, 2)
    vr = repeat_kv(xv, cfg.num_attention_heads // cfg.num_key_value_heads).transpose(1, 2)
    scores = (qr.transpose(1, 2) @ kr.transpose(-2, -1)) / (cfg.head_dim ** 0.5)
    seq = 5
    scores[:, :, :, -seq:] += torch.full((seq, seq), float("-inf")).triu(1)
    w = torch.softmax(scores.float(), dim=-1).type_as(xq)
    ref_out = (w @ vr).transpose(1, 2).reshape(2, 5, -1)
    ref_out = attn.o_proj(ref_out)
    assert torch.allclose(out, ref_out, atol=1e-4), f"max diff {(out - ref_out).abs().max().item():.2e}"


@register("05.4 KV-Cache 增量计算与全量一致")
def check_kv_cache(args):
    set_seed(1)
    cfg = MiniMindConfig(**TINY_CONFIG)
    attn = Attention(cfg)
    attn.eval()
    cos, sin = precompute_freqs_cis(cfg.head_dim, end=512, rope_base=cfg.rope_theta)

    # 全量：一次跑 6 个位置
    x_all = torch.randn(1, 6, cfg.hidden_size)
    with torch.no_grad():
        out_full, _ = attn(x_all, (cos[:6], sin[:6]))

        # 增量：先 3 位置，再续 1+2 位置
        past = None
        out_chunks = []
        for start, length in [(0, 3), (3, 1), (4, 2)]:
            x_chunk = x_all[:, start:start + length]
            pos = (cos[start:start + length], sin[start:start + length])
            o, past = attn(x_chunk, pos, past_key_value=past, use_cache=True)
            out_chunks.append(o)
    out_inc = torch.cat(out_chunks, dim=1)
    assert torch.allclose(out_full, out_inc, atol=1e-4), "KV-Cache 增量结果与全量不一致"


@register("05.5 attention_mask 忽略 padding 且不产生 NaN")
def check_mask(args):
    set_seed(2)
    cfg = MiniMindConfig(**TINY_CONFIG)
    attn = Attention(cfg)
    attn.flash = False
    cos, sin = precompute_freqs_cis(cfg.head_dim, end=512, rope_base=cfg.rope_theta)
    x = torch.randn(1, 4, cfg.hidden_size)
    mask = torch.tensor([[1, 1, 1, 0]])  # 末位是 padding
    out, _ = attn(x, (cos[:4], sin[:4]), attention_mask=mask)
    assert not torch.isnan(out).any()
    assert torch.allclose(out[:, :3], attn(x, (cos[:4], sin[:4]), attention_mask=torch.ones(1, 4))[0][:, :3], atol=1e-5)


@register("05.6 SDPA 快路径与手写路径等价")
def check_flash_eq(args):
    set_seed(3)
    cfg = MiniMindConfig(**TINY_CONFIG)
    attn_f = Attention(cfg)          # flash=True
    attn_m = Attention(cfg); attn_m.flash = False
    attn_m.load_state_dict(attn_f.state_dict())
    attn_f.eval(); attn_m.eval()
    cos, sin = precompute_freqs_cis(cfg.head_dim, end=512, rope_base=cfg.rope_theta)
    x = torch.randn(2, 7, cfg.hidden_size)
    with torch.no_grad():
        of, _ = attn_f(x, (cos[:7], sin[:7]))
        om, _ = attn_m(x, (cos[:7], sin[:7]))
    assert torch.allclose(of, om, atol=1e-4), f"SDPA 与手写路径差异 {(of - om).abs().max().item():.2e}"