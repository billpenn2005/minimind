"""第 07 课验收：MiniMindBlock（Pre-Norm 残差）。"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minimind3.block import MiniMindBlock  # noqa: E402
from minimind3.config import MiniMindConfig  # noqa: E402
from minimind3.rope import precompute_freqs_cis  # noqa: E402
from verify._common import TINY_CONFIG, set_seed  # noqa: E402

CHECKS = []


def register(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _make():
    cfg = MiniMindConfig(**TINY_CONFIG)
    block = MiniMindBlock(0, cfg)
    cos, sin = precompute_freqs_cis(cfg.head_dim, end=cfg.max_position_embeddings, rope_base=cfg.rope_theta)
    return cfg, block, (cos, sin)


@register("07.1 输出形状与子模块命名（对齐标准实现）")
def check_shapes_names(args):
    cfg, block, pos = _make()
    x = torch.randn(2, 9, cfg.hidden_size)
    y, kv = block(x, (pos[0][:9], pos[1][:9]), use_cache=True)
    assert y.shape == x.shape
    # 命名必须与 minimind 一致：后续权重转换按这些键名做映射
    assert hasattr(block, "self_attn") and hasattr(block, "input_layernorm")
    assert hasattr(block, "post_attention_layernorm") and hasattr(block, "mlp")
    assert kv is not None and isinstance(kv, tuple) and len(kv) == 2


@register("07.2 残差恒等：子模块输出归零时 block(x) == x")
def check_residual_identity(args):
    cfg, block, pos = _make()
    block.eval()
    x = torch.randn(1, 4, cfg.hidden_size)
    # 清零所有可学参数 -> attn/norm/mlp 输出 0（norm 不变式：weight=0 -> 0）
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    y, _ = block(x, (pos[0][:4], pos[1][:4]))
    assert torch.allclose(y, x, atol=1e-6), "残差连接缺失"


@register("07.3 Pre-Norm：block(x) == x + attn(ln1(x)) + mlp(ln2(...))")
def check_manual_expansion(args):
    set_seed(0)
    cfg, block, pos = _make()
    x = torch.randn(2, 5, cfg.hidden_size)
    y, _ = block(x, (pos[0][:5], pos[1][:5]))
    # 手动按 Pre-Norm 结构重算
    ln1 = block.input_layernorm(x)
    a, _ = block.self_attn(ln1, (pos[0][:5], pos[1][:5]))
    h1 = x + a
    h2 = h1 + block.mlp(block.post_attention_layernorm(h1))
    assert torch.allclose(y, h2, atol=1e-5)


@register("07.4 梯度经两条残差支路回传（输入侧梯度非零）")
def check_grad(args):
    set_seed(1)
    cfg, block, pos = _make()
    x = torch.randn(1, 3, cfg.hidden_size, requires_grad=True)
    y, _ = block(x, (pos[0][:3], pos[1][:3]))
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@register("07.5 KV-Cache 透传：use_cache=False 时返回 None")
def check_cache_flag(args):
    cfg, block, pos = _make()
    x = torch.randn(1, 3, cfg.hidden_size)
    _, kv = block(x, (pos[0][:3], pos[1][:3]), use_cache=False)
    assert kv is None